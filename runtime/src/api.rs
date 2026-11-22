//! The HTTP API: OpenAI-compatible chat, embeddings and models, chains, request cancel,
//! health/stats, admin load/unload.
//!
//! Every generation request (chat, chain, embeddings) gets a request id: the client's own
//! `X-Jitmo-Request-Id`, or a new one. It is returned in the `X-Jitmo-Request-Id` response
//! header (also on errors) and `DELETE /v1/requests/{id}` cancels it.
//! `X-Jitmo-Priority: interactive | background` (default interactive) orders the queue.

use crate::chain::{self, Chain, ChainReport, RunInput};
use crate::config::{Config, ModelKind};
use crate::error::Error;
use crate::llama::{self, SseTracker};
use crate::pool::{ModelHandle, Phase, Pool, Slot};
use crate::sched::{cancellable, Priority, RequestGuard};
use crate::stats::{mb, RequestTiming, UnloadReason};
use crate::win;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures_util::Stream;
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const REQUEST_ID: &str = "x-jitmo-request-id";
pub const PRIORITY: &str = "x-jitmo-priority";

pub struct App {
    pub cfg: Arc<Config>,
    pub pool: Arc<Pool>,
    pub chains: Vec<Chain>,
}

impl App {
    fn chain(&self, name: &str) -> Result<&Chain, Error> {
        self.chains.iter().find(|c| c.name == name).ok_or_else(|| {
            let known: Vec<&str> = self.chains.iter().map(|c| c.name.as_str()).collect();
            Error::chain_not_found(format!("no chain named '{name}'; configured chains: {known:?}"))
        })
    }

    /// Register a request from its headers (id, priority).
    fn begin(&self, headers: &HeaderMap, kind: &'static str, model: &str) -> Result<RequestGuard, Error> {
        let priority = match headers.get(PRIORITY) {
            None => Priority::Interactive,
            Some(v) => v.to_str().ok().and_then(Priority::parse).ok_or_else(|| {
                Error::bad_request("X-Jitmo-Priority must be 'interactive' or 'background'")
            })?,
        };
        let wanted = match headers.get(REQUEST_ID) {
            None => None,
            Some(v) => Some(v.to_str().map_err(|_| Error::bad_request("X-Jitmo-Request-Id is not text"))?),
        };
        self.pool.requests.begin(wanted, kind, model, priority)
    }
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/chains", get(list_chains))
        .route("/v1/chains/{name}/run", post(run_chain))
        .route("/v1/requests", get(list_requests))
        .route("/v1/requests/{id}", delete(cancel_request))
        .route("/admin/load", post(admin_load))
        .route("/admin/unload", post(admin_unload))
        .with_state(app)
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = json!({ "error": { "message": self.message, "type": self.code, "code": self.code } });
        (status, Json(body)).into_response()
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn with_id(mut resp: Response, id: &str) -> Response {
    if let Ok(v) = HeaderValue::from_str(id) {
        resp.headers_mut().insert(REQUEST_ID, v);
    }
    resp
}

fn parse_body(body: &Bytes) -> Result<Value, Error> {
    let v: Value = serde_json::from_slice(body).map_err(|e| Error::bad_request(format!("body is not valid JSON: {e}")))?;
    if !v.is_object() {
        return Err(Error::bad_request("body must be a JSON object"));
    }
    Ok(v)
}

/// Count a finished request (and a cancel) in the stats.
fn finish<T>(app: &App, r: &Result<T, Error>) {
    match r {
        Err(e) if e.code == "request_cancelled" => {
            app.pool.stats.cancelled();
            app.pool.stats.request_done(false);
        }
        r => app.pool.stats.request_done(r.is_ok()),
    }
}

// ---------- health / stats ----------

fn model_brief(app: &App, m: &ModelHandle) -> Value {
    json!({ "name": m.name(), "kind": m.cfg.kind.as_str(), "state": m.phase().as_str(), "active": m.active(), "pinned": m.cfg.pinned,
            "last_used_secs_ago": m.last_used_secs(), "child_working_set_mb": m.child_snapshot(&app.pool.stats).map(|c| c.working_set_mb) })
}

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    let d = app.pool.default_model();
    let child = d.child_snapshot(&app.pool.stats);
    let c = app.pool.stats.counters();
    let sys = win::memory_status();
    Json(json!({
        "status": "ok",
        "model": d.name(),
        "model_state": d.phase().as_str(),
        "child_working_set_mb": child.as_ref().map(|c| c.working_set_mb),
        "child_private_mb": child.as_ref().map(|c| c.private_mb),
        "models": app.pool.models.iter().map(|m| model_brief(&app, m)).collect::<Vec<_>>(),
        "system_free_mb": sys.map(|m| mb(m.avail)),
        "requests_served": c.requests_served,
        "requests_in_flight": app.pool.requests.list().len(),
        "last_tokens_per_sec": c.last_tokens_per_sec,
        "pacer_capped": app.pool.pacer.capped(),
        "uptime_secs": app.pool.stats.uptime_secs(),
    }))
}

fn model_stats(app: &App, m: &ModelHandle) -> Value {
    let c = &m.cfg;
    let p = &m.plan;
    let l = &p.load;
    json!({
        "name": c.name, "kind": c.kind.as_str(), "state": m.phase().as_str(), "pinned": c.pinned,
        "path": c.path.display().to_string(), "file_mb": mb(l.file_bytes),
        "ctx_size": c.ctx_size, "threads": m.threads, "kv_cache_type": c.kv_cache_type, "flash_attn": c.flash_attn,
        "mmap": c.mmap, "no_repack": c.no_repack, "cache_ram_mb": c.cache_ram_mb, "ctx_checkpoints": c.ctx_checkpoints,
        "pooling": c.pooling,
        "adapters": m.adapters.iter().enumerate().map(|(i, a)| json!({ "id": i, "path": a.display().to_string() })).collect::<Vec<_>>(),
        "active": m.active(), "unload_pending": m.unload_pending(),
        "idle_secs": (m.idle_secs() * 10.0).round() / 10.0, "last_used_secs_ago": m.last_used_secs(),
        "idle_unload_secs": if c.pinned { 0 } else { m.idle_unload_secs },
        "memory_cap_mb": p.cap_mb(), "memory_cap_source": p.source,
        "memory_estimate_mb": p.estimate.as_ref().map(|e| json!({
            "kv": mb(e.kv_bytes), "recurrent": mb(e.recurrent_bytes), "compute": mb(e.compute_bytes),
            "cache_ram": mb(e.cache_ram_bytes), "checkpoints": mb(e.checkpoint_bytes), "overhead": mb(e.overhead_bytes),
            "total": mb(e.total_bytes), "kv_layers": e.kv_layers, "swa_layers": e.swa_layers, "arch": e.arch,
            "load_overhead": mb(e.load_overhead_bytes), "load_private": mb(e.load_private_bytes), "weights_private": e.weights_private,
        })),
        "load_estimate_mb": {
            "total": mb(l.total_bytes), "private": mb(l.private_bytes), "mapped": mb(l.mapped_bytes), "repack": mb(l.repack_bytes),
            "on_disk": l.on_disk.iter().map(|(n, b)| json!({ "tensor": n, "mb": mb(*b) })).collect::<Vec<_>>(), "source": l.source,
        },
        "counters": m.counters(),
        "child": m.child_snapshot(&app.pool.stats),
    })
}

async fn stats(State(app): State<Arc<App>>) -> Json<Value> {
    let cfg = &app.cfg;
    let pool = &app.pool;
    let d = pool.default_model();
    let sys = win::memory_status();
    let sched = pool.sched.snapshot();
    Json(json!({
        "uptime_secs": pool.stats.uptime_secs(),
        "models": pool.models.iter().map(|m| model_stats(&app, m)).collect::<Vec<_>>(),
        "pool": {
            "budget_mb": cfg.budget.pool_budget_mb,
            "resident_estimate_mb": mb(pool.resident_estimate()),
            "loaded": pool.models.iter().filter(|m| m.phase() == Phase::Loaded).map(|m| m.name()).collect::<Vec<_>>(),
            "evictions": pool.evictions(),
        },
        "queue": { "generating": sched.running > 0, "running": sched.running, "waiting": sched.waiting,
                   "waiting_interactive": sched.waiting_interactive, "waiting_background": sched.waiting_background,
                   "max_concurrent": sched.max_concurrent, "max_queue": sched.max_queue },
        "requests": pool.requests.list(),
        "system": {
            "total_mb": sys.map(|m| mb(m.total)),
            "free_mb": sys.map(|m| mb(m.avail)),
            "memory_load_pct": sys.map(|m| m.load_pct),
            "low_memory_signal": pool.low_memory_signal(),
        },
        "budget": {
            "reserve_free_mb": cfg.budget.reserve_free_mb,
            "pool_budget_mb": cfg.budget.pool_budget_mb,
            "pressure_free_mb": cfg.budget.pressure_free_mb,
            "critical_free_mb": cfg.budget.critical_free_mb,
            "idle_unload_secs": cfg.budget.idle_unload_secs,
            // v0.1 fields, for the default model:
            "memory_cap_mb": d.plan.cap_mb(),
            "memory_cap_source": d.plan.source,
            "load_estimate_mb": d.plan.load_bytes() / (1024 * 1024),
            "unload_pending": d.unload_pending(),
            "idle_secs": (d.idle_secs() * 10.0).round() / 10.0,
        },
        "pacer": pool.pacer.snapshot(pool.models.iter().any(|m| m.phase() == Phase::Loaded), d.threads),
        "counters": pool.stats.counters(),
        // v0.1 fields, for the default model:
        "model_state": d.phase().as_str(),
        "model": { "name": d.name(), "threads": d.threads },
        "child": d.child_snapshot(&pool.stats),
    }))
}

// ---------- models ----------

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let created = now_secs();
    let mut data: Vec<Value> = app
        .pool
        .models
        .iter()
        .map(|m| json!({ "id": m.name(), "object": "model", "created": created, "owned_by": "jitmo-runtime", "kind": m.cfg.kind.as_str() }))
        .collect();
    for c in &app.chains {
        data.push(json!({ "id": format!("chain:{}", c.name), "object": "model", "created": created, "owned_by": "jitmo-runtime", "kind": "chain" }));
    }
    Json(json!({ "object": "list", "data": data }))
}

// ---------- chat completions ----------

async fn chat_completions(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    let req = match parse_body(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    let model = req["model"].as_str().map(str::to_string);
    let label = model.clone().unwrap_or_else(|| app.pool.default_model().name().to_string());
    let guard = match app.begin(&headers, "chat", &label) {
        Ok(g) => g,
        Err(e) => return e.into_response(),
    };
    let id = guard.id.clone();
    let resp = match chat_inner(&app, req, model, guard).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    };
    with_id(resp, &id)
}

async fn chat_inner(app: &Arc<App>, mut req: Value, model: Option<String>, guard: RequestGuard) -> Result<Response, Error> {
    let started = Instant::now();
    if !req["messages"].is_array() {
        return Err(Error::bad_request("'messages' must be an array"));
    }
    let stream = req["stream"].as_bool().unwrap_or(false);
    if let Some(name) = model.as_deref().and_then(|m| m.strip_prefix("chain:")) {
        let r = cancellable(&guard.token, &guard.id, chain_as_chat(app, name, &req, stream, &guard)).await;
        finish(app, &r);
        return r;
    }
    let m = app.pool.model(model.as_deref(), Some(ModelKind::Chat))?;
    if let Some(k) = &m.cfg.chat_template_kwargs {
        if req.get("chat_template_kwargs").is_none() {
            req["chat_template_kwargs"] = k.clone();
        }
    }
    llama::full_lora(&mut req, m.adapters.len())?;
    let work = async {
        let slot = app.pool.acquire(&m, &guard).await?;
        let resp = llama::post_json(&app.pool.http, slot.port, "/v1/chat/completions", &req).await?;
        Ok((slot, resp))
    };
    let (slot, resp) = match cancellable(&guard.token, &guard.id, work).await {
        Ok(x) => x,
        Err(e) => {
            finish(app, &Err::<(), _>(e.clone()));
            return Err(e);
        }
    };
    let status = resp.status();

    if stream && status.is_success() {
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static("text/event-stream"));
        let token = guard.token.clone();
        let cancelled: Pin<Box<dyn Future<Output = ()> + Send>> = Box::pin(async move { token.cancelled().await });
        let body = GuardedStream {
            inner: Some(Box::pin(resp.into_body().into_data_stream())),
            cancelled,
            guard: StreamGuard { app: app.clone(), slot: Some(slot), tracker: SseTracker::default(), started, broken: false, cancelled: false, by_runtime: false, req: Some(guard) },
        };
        return Ok(Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(body))
            .expect("valid response"));
    }

    let r = cancellable(&guard.token, &guard.id, llama::collect(resp)).await;
    let (status, bytes) = match r {
        Ok(x) => x,
        Err(e) => {
            finish(app, &Err::<(), _>(e.clone()));
            return Err(e);
        }
    };
    if status.is_success() {
        let engine = serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| llama::engine_timings(&v)).unwrap_or_default();
        let timing = RequestTiming::new("chat", slot.model.name().to_string(), slot.queue_ms, slot.load_ms, &engine, None, ms_since(started))
            .with_ids(&guard.id, slot.model.name());
        app.pool.pacer.after_generation(&timing);
        app.pool.stats.record(timing);
        slot.model.count_generation();
    }
    app.pool.stats.request_done(status.is_success());
    drop(slot);
    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(bytes))
        .expect("valid response"))
}

/// The streamed body: llama-server's SSE, until it ends, the client goes away (the body is
/// dropped), or the request is cancelled (then one error event and `[DONE]` are sent). In every
/// case the upstream connection is closed, which makes llama-server stop the generation.
struct GuardedStream {
    inner: Option<Pin<Box<dyn Stream<Item = Result<Bytes, hyper::Error>> + Send>>>,
    cancelled: Pin<Box<dyn Future<Output = ()> + Send>>,
    guard: StreamGuard,
}

impl Stream for GuardedStream {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.inner.is_none() {
            return Poll::Ready(None);
        }
        if this.cancelled.as_mut().poll(cx).is_ready() {
            this.inner = None; // closes the connection to llama-server
            this.guard.cancelled = true;
            let req = this.guard.req.as_ref();
            let id = req.map_or("", |r| r.id.as_str());
            let e = req.and_then(|r| r.token.reason()).unwrap_or_else(|| Error::cancelled(format!("request {id} was cancelled")));
            this.guard.by_runtime = e.code != "request_cancelled";
            let ev = json!({ "error": { "message": e.message, "type": e.code, "code": e.code } });
            return Poll::Ready(Some(Ok(Bytes::from(format!("data: {ev}\n\ndata: [DONE]\n\n")))));
        }
        let inner = this.inner.as_mut().expect("checked above");
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                this.guard.tracker.feed(&bytes);
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.guard.broken = true;
                this.inner = None;
                Poll::Ready(Some(Err(std::io::Error::other(e))))
            }
            Poll::Ready(None) => {
                this.inner = None;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Lives as long as a streamed response: holds the slot and the request id, then records the timing.
struct StreamGuard {
    app: Arc<App>,
    slot: Option<Slot>,
    tracker: SseTracker,
    started: Instant,
    broken: bool,
    cancelled: bool,
    /// Cancelled by the runtime (memory pressure), not by the client.
    by_runtime: bool,
    req: Option<RequestGuard>,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        let Some(slot) = self.slot.take() else { return };
        let complete = !self.broken && !self.cancelled && self.tracker.timings.is_some();
        let id = self.req.as_ref().map(|r| r.id.clone()).unwrap_or_default();
        if complete {
            let engine = self.tracker.timings.clone().unwrap_or_default();
            let first = self.tracker.first_token_at.map(|t| (t - self.started).as_secs_f64() * 1000.0);
            let timing = RequestTiming::new(
                "chat_stream",
                slot.model.name().to_string(),
                slot.queue_ms,
                slot.load_ms,
                &engine,
                first,
                ms_since(self.started),
            )
            .with_ids(&id, slot.model.name());
            self.app.pool.pacer.after_generation(&timing);
            self.app.pool.stats.record(timing);
            slot.model.count_generation();
        } else if !self.tracker.done {
            let why = match (self.cancelled, self.by_runtime) {
                (true, true) => "stopped for memory pressure",
                (true, false) => "cancelled",
                _ => "client went away or engine error",
            };
            tracing::info!("request {id}: stream ended early ({why}); the connection to llama-server is closed");
        }
        if self.cancelled && !self.by_runtime {
            self.app.pool.stats.cancelled();
        }
        self.app.pool.stats.request_done(complete);
        drop(slot);
        self.req.take();
    }
}

/// Text of the last user message (string content, or the text parts of an array).
fn last_user_text(req: &Value) -> Option<String> {
    let msg = req["messages"].as_array()?.iter().rev().find(|m| m["role"] == "user")?;
    match &msg["content"] {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let text: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
            Some(text.join("\n"))
        }
        _ => None,
    }
}

fn object_field(v: &Value, key: &str) -> Result<Map<String, Value>, Error> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(m)) => Ok(m.clone()),
        Some(_) => Err(Error::bad_request(format!("'{key}' must be an object"))),
    }
}

/// `model: "chain:<name>"` on /v1/chat/completions: the last user message is the chain input,
/// `vars` (optional) the variables, the last step's output is the answer. With `stream: true`
/// the whole answer comes in one chunk.
async fn chain_as_chat(app: &Arc<App>, name: &str, req: &Value, stream: bool, guard: &RequestGuard) -> Result<Response, Error> {
    let chain = app.chain(name)?;
    let input = last_user_text(req).ok_or_else(|| Error::bad_request("a chain needs at least one user message"))?;
    let vars = object_field(req, "vars")?;
    let system = Map::new();
    let report = chain::run(&app.pool, chain, RunInput { input: &input, vars: &vars, system: &system }, guard).await?;
    let id = format!("chatcmpl-chain-{}", now_secs());
    let model = format!("chain:{name}");
    let created = now_secs();
    let prompt_tokens: u64 = report.steps.iter().map(|s| s.timing.prompt_tokens).sum();
    let completion_tokens: u64 = report.steps.iter().map(|s| s.timing.completion_tokens).sum();
    let finish = report.steps.last().and_then(|s| s.finish_reason.clone()).unwrap_or_else(|| "stop".into());
    if stream {
        let first = json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{ "index": 0, "delta": { "role": "assistant", "content": report.output }, "finish_reason": null }] });
        let last = json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": finish }],
            "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens } });
        let sse = format!("data: {first}\n\ndata: {last}\n\ndata: [DONE]\n\n");
        return Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from(sse))
            .expect("valid response"));
    }
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": report.output }, "finish_reason": finish }],
        "usage": { "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens },
        "chain": report,
    });
    Ok(Json(body).into_response())
}

// ---------- embeddings ----------

async fn embeddings(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    let req = match parse_body(&body) {
        Ok(r) => r,
        Err(e) => return e.into_response(),
    };
    let model = req["model"].as_str().map(str::to_string);
    let m = match app.pool.model(model.as_deref(), Some(ModelKind::Embedding)) {
        Ok(m) => m,
        Err(e) => return e.into_response(),
    };
    let guard = match app.begin(&headers, "embedding", m.name()) {
        Ok(g) => g,
        Err(e) => return e.into_response(),
    };
    let id = guard.id.clone();
    let r = cancellable(&guard.token, &guard.id, embeddings_inner(&app, &m, req, &guard)).await;
    finish(&app, &r);
    with_id(r.unwrap_or_else(|e| e.into_response()), &id)
}

async fn embeddings_inner(app: &Arc<App>, m: &Arc<ModelHandle>, mut req: Value, guard: &RequestGuard) -> Result<Response, Error> {
    let started = Instant::now();
    match &req["input"] {
        Value::String(_) => {}
        Value::Array(a) if !a.is_empty() => {}
        _ => return Err(Error::bad_request("'input' must be a string or a non-empty array")),
    }
    req["model"] = json!(m.name());
    let slot = app.pool.acquire(m, guard).await?;
    let resp = llama::post_json(&app.pool.http, slot.port, "/v1/embeddings", &req).await?;
    let (status, bytes) = llama::collect(resp).await?;
    if status.is_success() {
        let n = serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| v["usage"]["prompt_tokens"].as_u64()).unwrap_or(0);
        let engine = llama::EngineTimings { prompt_n: n, ..Default::default() };
        let timing = RequestTiming::new("embedding", m.name().to_string(), slot.queue_ms, slot.load_ms, &engine, None, ms_since(started))
            .with_ids(&guard.id, m.name());
        app.pool.stats.record(timing);
        m.count_generation();
    } else if status == StatusCode::NOT_IMPLEMENTED || status == StatusCode::NOT_FOUND {
        return Err(Error::engine(format!("llama-server for '{}' refused embeddings ({status})", m.name())));
    }
    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(bytes))
        .expect("valid response"))
}

// ---------- chains ----------

async fn list_chains(State(app): State<Arc<App>>) -> Json<Value> {
    let chains: Vec<Value> = app
        .chains
        .iter()
        .map(|c| {
            json!({
                "name": c.name,
                "description": c.description,
                "vars": c.var_names,
                "steps": c.steps.iter().map(|s| json!({
                    "name": s.name,
                    "model": s.model_name,
                    "json_schema": s.json_schema.is_some(),
                    "lora": s.lora_path.as_ref().map(|p| json!({ "path": p, "id": s.lora.map(|l| l.0), "scale": s.lora.map(|l| l.1) })),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    Json(json!({ "chains": chains }))
}

async fn run_chain(State(app): State<Arc<App>>, Path(name): Path<String>, headers: HeaderMap, body: Bytes) -> Response {
    let chain = match app.chain(&name) {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    let guard = match app.begin(&headers, "chain", &format!("chain:{name}")) {
        Ok(g) => g,
        Err(e) => return e.into_response(),
    };
    let id = guard.id.clone();
    let work = async {
        let req = parse_body(&body)
            .map_err(|e| Error::bad_request(format!("expected {{\"input\": \"...\", \"vars\": {{...}}, \"system\": {{...}}}}: {}", e.message)))?;
        let input = match &req["input"] {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            _ => return Err(Error::bad_request("'input' must be a string")),
        };
        let vars = object_field(&req, "vars")?;
        let system = object_field(&req, "system")?;
        if req.get("input").is_none() && vars.is_empty() {
            return Err(Error::bad_request("send 'input' and/or 'vars'"));
        }
        chain::run(&app.pool, chain, RunInput { input: &input, vars: &vars, system: &system }, &guard).await
    };
    let r: Result<ChainReport, Error> = cancellable(&guard.token, &guard.id, work).await;
    finish(&app, &r);
    with_id(r.map(|rep| Json(rep).into_response()).unwrap_or_else(|e| e.into_response()), &id)
}

// ---------- requests ----------

async fn list_requests(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({ "requests": app.pool.requests.list() }))
}

async fn cancel_request(State(app): State<Arc<App>>, Path(id): Path<String>) -> Result<Json<Value>, Error> {
    match app.pool.requests.cancel(&id) {
        Some(state) => {
            tracing::info!("request {id} cancelled ({state})");
            Ok(Json(json!({ "id": id, "cancelled": true, "state": state })))
        }
        None => Err(Error::request_not_found(format!("no request '{id}' is queued or running (it may have finished)"))),
    }
}

// ---------- admin ----------

#[derive(Deserialize, Default)]
struct ModelBody {
    model: Option<String>,
}

/// Warm-up: load a model (default: the default model) and return when it is ready.
async fn admin_load(State(app): State<Arc<App>>, body: Bytes) -> Result<Json<Value>, Error> {
    let b: ModelBody = if body.iter().all(|c| c.is_ascii_whitespace()) {
        ModelBody::default()
    } else {
        serde_json::from_slice(&body).map_err(|e| Error::bad_request(format!("expected {{\"model\": \"...\"}} or no body: {e}")))?
    };
    let m = match &b.model {
        Some(n) => app.pool.model(Some(n), None)?,
        None => app.pool.default_model().clone(),
    };
    let was_loaded = m.phase() == Phase::Loaded;
    let t0 = Instant::now();
    let (hold, _port, load_ms) = app.pool.use_model(&m, None).await?;
    drop(hold);
    Ok(Json(json!({
        "model": m.name(), "loaded": true, "was_loaded": was_loaded && load_ms == 0.0,
        "load_ms": (load_ms * 10.0).round() / 10.0, "total_ms": (ms_since(t0) * 10.0).round() / 10.0,
        "model_state": m.phase().as_str(),
    })))
}

#[derive(Deserialize)]
struct UnloadQuery {
    #[serde(default)]
    force: bool,
    model: Option<String>,
}

/// Unload one model (`?model=`) or all. Waits for running generations unless `?force=true`.
async fn admin_unload(State(app): State<Arc<App>>, Query(q): Query<UnloadQuery>) -> Result<Json<Value>, Error> {
    let targets: Vec<Arc<ModelHandle>> = match &q.model {
        Some(n) => vec![app.pool.model(Some(n), None)?],
        None => app.pool.models.clone(),
    };
    let mut unloaded = Vec::new();
    for m in &targets {
        let was = if q.force { app.pool.unload(m, UnloadReason::Admin).await } else { app.pool.unload_when_free(m, UnloadReason::Admin).await };
        if was {
            unloaded.push(m.name().to_string());
        }
    }
    Ok(Json(json!({ "unloaded": !unloaded.is_empty(), "models": unloaded, "model_state": app.pool.default_model().phase().as_str() })))
}

