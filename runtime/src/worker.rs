//! `jitmo-runtime --engine-worker`: one model in this process (llama.dll, see engine.rs),
//! served with the part of llama-server's HTTP API the runtime uses. The runtime starts it
//! instead of llama-server for a model with `engine = "inproc"`, in the same Job Object, so
//! the memory cap, the pacer's CPU cap and kill-on-close work unchanged.
//!
//! Endpoints: `GET /health`, `POST /v1/chat/completions` (messages, max_tokens, temperature,
//! top_k, top_p, min_p, seed, stop, grammar, response_format, stream). The response carries
//! llama-server-style `timings` (`prompt_n` = prompt tokens read, `cache_n` = reused).
//! Closing the connection stops the generation (checked between tokens).

use crate::engine::{self, Finish, GenStats, LoadOptions, Model, SamplerOpts, Step};
use anyhow::{anyhow, bail, Context as _, Result};
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc as tmpsc;

pub struct WorkerArgs {
    pub lib_dir: PathBuf,
    pub model: PathBuf,
    pub name: String,
    pub port: u16,
    pub ctx: u32,
    pub threads: i32,
    pub ngl: i32,
    pub kv_type: String,
    pub flash_attn: String,
    pub no_repack: bool,
}

/// The `--engine-worker` command line (the runtime builds it; see pool.rs).
pub fn parse(args: &[String]) -> Result<WorkerArgs> {
    let mut w = WorkerArgs {
        lib_dir: PathBuf::new(),
        model: PathBuf::new(),
        name: "model".into(),
        port: 0,
        ctx: 4096,
        threads: 4,
        ngl: 0,
        kv_type: "f16".into(),
        flash_attn: "auto".into(),
        no_repack: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().cloned().ok_or_else(|| anyhow!("{a} needs a value"));
        match a.as_str() {
            "--lib-dir" => w.lib_dir = v()?.into(),
            "--model" => w.model = v()?.into(),
            "--name" => w.name = v()?,
            "--port" => w.port = v()?.parse()?,
            "--ctx" => w.ctx = v()?.parse()?,
            "--threads" => w.threads = v()?.parse()?,
            "--ngl" => {
                let s = v()?;
                w.ngl = match s.as_str() {
                    "all" | "auto" => -1,
                    n => n.parse().context("--ngl must be a number, all or auto")?,
                }
            }
            "--kv-type" => w.kv_type = v()?,
            "--flash-attn" => w.flash_attn = v()?,
            "--no-repack" => w.no_repack = true,
            other => bail!("unknown --engine-worker option {other}"),
        }
    }
    if w.model.as_os_str().is_empty() || w.lib_dir.as_os_str().is_empty() || w.port == 0 {
        bail!("--engine-worker needs --lib-dir, --model and --port");
    }
    Ok(w)
}

enum Event {
    Text(String),
    Done(GenStats),
    /// HTTP status and message.
    Error(u16, String),
}

struct Job {
    messages: Vec<(String, String)>,
    max_tokens: usize,
    grammar: Option<String>,
    sampler: SamplerOpts,
    stop: Vec<String>,
    cancel: Arc<AtomicBool>,
    events: tmpsc::UnboundedSender<Event>,
}

/// Sets the cancel flag when the request goes away (client closed the connection).
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Holds back text that may be the start of a stop string, and cuts at a stop string.
struct StopFilter {
    stop: Vec<String>,
    full: String,
    sent: usize,
}

impl StopFilter {
    /// Text that is safe to send now, and whether a stop string was hit.
    fn push(&mut self, piece: &str) -> (String, bool) {
        self.full.push_str(piece);
        if let Some(at) = self.stop.iter().filter_map(|s| self.full[self.sent..].find(s.as_str()).map(|i| self.sent + i)).min() {
            let out = self.full[self.sent..at].to_string();
            self.sent = at;
            self.full.truncate(at);
            return (out, true);
        }
        // Keep back the longest tail that is a prefix of a stop string.
        let mut hold = 0;
        for s in &self.stop {
            for k in (1..s.len().min(self.full.len() - self.sent + 1)).rev() {
                if s.is_char_boundary(k) && self.full.ends_with(&s[..k]) {
                    hold = hold.max(k);
                    break;
                }
            }
        }
        let end = self.full.len() - hold;
        let out = self.full[self.sent..end].to_string();
        self.sent = end;
        (out, false)
    }
    fn rest(&mut self) -> String {
        let out = self.full[self.sent..].to_string();
        self.sent = self.full.len();
        out
    }
}

fn engine_thread(mut model: Model, jobs: mpsc::Receiver<Job>) {
    for job in jobs {
        let prompt = match model.apply_template(&job.messages) {
            Ok(p) => p,
            Err(e) => {
                let _ = job.events.send(Event::Error(400, e.to_string()));
                continue;
            }
        };
        let mut filter = StopFilter { stop: job.stop.clone(), full: String::new(), sent: 0 };
        let tx = job.events.clone();
        let r = model.generate(
            &prompt,
            job.max_tokens,
            job.grammar.as_deref(),
            &job.sampler,
            &job.cancel,
            |piece| {
                let (out, hit) = filter.push(piece);
                if !out.is_empty() && tx.send(Event::Text(out)).is_err() {
                    return Step::Stop; // the request is gone
                }
                if hit {
                    Step::Stop
                } else {
                    Step::Continue
                }
            },
            |_| Duration::ZERO,
        );
        match r {
            Ok(stats) => {
                let rest = filter.rest();
                if !rest.is_empty() {
                    let _ = job.events.send(Event::Text(rest));
                }
                let _ = job.events.send(Event::Done(stats));
            }
            Err(e) => {
                let msg = e.to_string();
                let status = if msg.contains("tokens; the context holds") || msg.contains("grammar") { 400 } else { 500 };
                let _ = job.events.send(Event::Error(status, msg));
            }
        }
    }
}

struct App {
    name: String,
    n_ctx: usize,
    jobs: std::sync::Mutex<mpsc::Sender<Job>>,
}

fn error(status: u16, msg: impl Into<String>) -> Response {
    let s = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let kind = if status < 500 { "invalid_request_error" } else { "server_error" };
    (s, Json(json!({ "error": { "code": status, "message": msg.into(), "type": kind } }))).into_response()
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn text_of(content: &Value) -> Result<String, String> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => Ok(parts
            .iter()
            .filter(|p| p["type"] == "text")
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("")),
        _ => Err("message content must be a string or a list of text parts".into()),
    }
}

/// Request body -> job settings.
fn job_from(req: &Value, n_ctx: usize) -> Result<(Vec<(String, String)>, usize, Option<String>, SamplerOpts, Vec<String>), String> {
    let msgs = req["messages"].as_array().ok_or("'messages' must be a list")?;
    let mut messages = Vec::new();
    for m in msgs {
        let role = m["role"].as_str().ok_or("each message needs a 'role'")?.to_string();
        messages.push((role, text_of(&m["content"])?));
    }
    if messages.is_empty() {
        return Err("'messages' is empty".into());
    }
    let max = req.get("max_tokens").or_else(|| req.get("max_completion_tokens")).or_else(|| req.get("n_predict")).and_then(Value::as_i64).unwrap_or(-1);
    let max_tokens = if max < 0 { n_ctx } else { max as usize };
    let grammar = match req.get("grammar").and_then(Value::as_str) {
        Some(g) if !g.is_empty() => Some(g.to_string()),
        _ => match req.get("response_format") {
            Some(rf) if !rf.is_null() => crate::json_grammar::from_response_format(rf)?,
            _ => match req.get("json_schema") {
                Some(s) if !s.is_null() => Some(crate::json_grammar::from_schema(s)?),
                _ => None,
            },
        },
    };
    let d = SamplerOpts::default();
    let f = |k: &str, def: f32| req.get(k).and_then(Value::as_f64).map_or(def, |x| x as f32);
    let sampler = SamplerOpts {
        temperature: f("temperature", d.temperature),
        top_k: req.get("top_k").and_then(Value::as_i64).map_or(d.top_k, |x| x as i32),
        top_p: f("top_p", d.top_p),
        min_p: f("min_p", d.min_p),
        seed: req.get("seed").and_then(Value::as_u64).map_or(d.seed, |x| x as u32),
    };
    let stop = match req.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string).collect(),
        _ => Vec::new(),
    };
    Ok((messages, max_tokens, grammar, sampler, stop))
}

fn timings(s: &GenStats) -> Value {
    json!({
        "cache_n": s.cached_tokens,
        "prompt_n": s.prompt_tokens - s.cached_tokens,
        "prompt_ms": s.prompt_ms,
        "predicted_n": s.tokens,
        "predicted_ms": s.gen_ms,
        "paused_ms": s.paused_ms,
    })
}

fn finish_reason(f: Finish) -> &'static str {
    match f {
        Finish::Length => "length",
        _ => "stop",
    }
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn chat(State(app): State<Arc<App>>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error(400, format!("body is not valid JSON: {e}")),
    };
    let (messages, max_tokens, grammar, sampler, stop) = match job_from(&req, app.n_ctx) {
        Ok(j) => j,
        Err(e) => return error(400, e),
    };
    let stream = req["stream"].as_bool().unwrap_or(false);
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancel.clone());
    let (tx, mut rx) = tmpsc::unbounded_channel();
    let job = Job { messages, max_tokens, grammar, sampler, stop, cancel, events: tx };
    if app.jobs.lock().unwrap_or_else(|p| p.into_inner()).send(job).is_err() {
        return error(500, "the engine thread is gone");
    }
    let id = format!("chatcmpl-{}", now());
    let model = app.name.clone();

    if !stream {
        let mut text = String::new();
        loop {
            match rx.recv().await {
                Some(Event::Text(t)) => text.push_str(&t),
                Some(Event::Done(s)) => {
                    drop(guard);
                    return Json(json!({
                        "id": id, "object": "chat.completion", "created": now(), "model": model,
                        "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": finish_reason(s.finish) }],
                        "usage": { "prompt_tokens": s.prompt_tokens, "completion_tokens": s.tokens, "total_tokens": s.prompt_tokens + s.tokens },
                        "timings": timings(&s),
                    }))
                    .into_response();
                }
                Some(Event::Error(status, msg)) => return error(status, msg),
                None => return error(500, "the engine stopped"),
            }
        }
    }

    // Stream: the body owns the guard, so a closed connection cancels the generation.
    let created = now();
    let chunk = move |delta: Value, finish: Option<&str>, extra: Option<(&str, Value)>| {
        let mut v = json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
        if let Some((k, x)) = extra {
            v[k] = x;
        }
        Bytes::from(format!("data: {v}\n\n"))
    };
    let first = chunk(json!({ "role": "assistant", "content": null }), None, None);
    let s = async_stream(rx, guard, chunk, first);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(s))
        .unwrap_or_else(|_| error(500, "cannot build the stream"))
}

type Chunk = Box<dyn Fn(Value, Option<&str>, Option<(&str, Value)>) -> Bytes + Send>;

fn async_stream(
    rx: tmpsc::UnboundedReceiver<Event>,
    guard: CancelOnDrop,
    chunk: impl Fn(Value, Option<&str>, Option<(&str, Value)>) -> Bytes + Send + 'static,
    first: Bytes,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> + Send {
    struct St {
        rx: tmpsc::UnboundedReceiver<Event>,
        _guard: CancelOnDrop,
        chunk: Chunk,
        first: Option<Bytes>,
        done: bool,
    }
    let st = St { rx, _guard: guard, chunk: Box::new(chunk), first: Some(first), done: false };
    futures_util::stream::unfold(st, |mut st| async move {
        if let Some(f) = st.first.take() {
            return Some((Ok(f), st));
        }
        if st.done {
            return None;
        }
        let out = match st.rx.recv().await {
            Some(Event::Text(t)) => (st.chunk)(json!({ "content": t }), None, None),
            Some(Event::Done(s)) => {
                st.done = true;
                let mut b = (st.chunk)(json!({}), Some(finish_reason(s.finish)), Some(("timings", timings(&s)))).to_vec();
                b.extend_from_slice(b"data: [DONE]\n\n");
                Bytes::from(b)
            }
            Some(Event::Error(status, msg)) => {
                st.done = true;
                Bytes::from(format!("data: {}\n\ndata: [DONE]\n\n", json!({ "error": { "code": status, "message": msg } })))
            }
            None => return None,
        };
        Some((Ok(out), st))
    })
}

/// Load the model, then serve until the process is killed (the runtime owns its life).
pub async fn run(args: &[String]) -> Result<()> {
    let w = parse(args)?;
    let kv_type = engine::kv_type_id(&w.kv_type).ok_or_else(|| anyhow!("unknown kv type {}", w.kv_type))?;
    let flash_attn = match w.flash_attn.as_str() {
        "on" => 1,
        "off" => 0,
        _ => -1,
    };
    let lib = engine::load_static(&w.lib_dir)?;
    eprintln!("engine-worker: {}", lib.system_info());
    let opts = LoadOptions { n_gpu_layers: w.ngl, ctx: w.ctx, threads: w.threads, no_repack: w.no_repack, kv_type, flash_attn };
    let model = Model::load(lib, &w.model, &opts)?;
    let n_ctx = model.n_ctx();
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new().name("engine".into()).spawn(move || engine_thread(model, rx))?;
    let app = Arc::new(App { name: w.name.clone(), n_ctx, jobs: std::sync::Mutex::new(tx) });
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", w.port)).await.with_context(|| format!("cannot listen on port {}", w.port))?;
    eprintln!("engine-worker: model '{}' ready, ctx {n_ctx}, listening on 127.0.0.1:{}", w.name, w.port);
    axum::serve(listener, router).await.context("server error")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_strings_cut_and_hold_back() {
        let mut f = StopFilter { stop: vec!["</s>".into()], full: String::new(), sent: 0 };
        assert_eq!(f.push("Hello <"), ("Hello ".into(), false), "'<' may start the stop string");
        assert_eq!(f.push("/"), (String::new(), false));
        assert_eq!(f.push("s> more"), (String::new(), true));
        assert_eq!(f.full, "Hello ");
        let mut g = StopFilter { stop: vec!["END".into()], full: String::new(), sent: 0 };
        assert_eq!(g.push("abc E"), ("abc ".into(), false));
        assert_eq!(g.push("x"), ("Ex".into(), false), "not a stop string after all");
        assert_eq!(g.rest(), "");
    }

    #[test]
    fn request_fields() {
        let r = json!({"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":5,"temperature":0,"stop":"x",
                       "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}}});
        let (m, max, g, s, stop) = job_from(&r, 100).unwrap();
        assert_eq!((m[0].1.as_str(), max, s.temperature, stop), ("hi", 5, 0.0, vec!["x".to_string()]));
        assert!(g.unwrap().contains("\\\"a\\\""));
        assert!(job_from(&json!({"messages":[]}), 10).is_err());
        assert_eq!(job_from(&json!({"messages":[{"role":"user","content":"x"}]}), 10).unwrap().1, 10);
    }
}
