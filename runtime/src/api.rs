//! The HTTP API: OpenAI-compatible chat + models, chains, health/stats, admin.

use crate::chain::{self, Chain, ChainReport};
use crate::config::Config;
use crate::error::Error;
use crate::llama::{self, SseTracker};
use crate::stats::{mb, RequestTiming, UnloadReason};
use crate::supervisor::{Slot, Supervisor};
use crate::win;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use http_body_util::BodyExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct App {
    pub cfg: Arc<Config>,
    pub sup: Arc<Supervisor>,
    pub chains: Vec<Chain>,
}

impl App {
    fn chain(&self, name: &str) -> Result<&Chain, Error> {
        self.chains.iter().find(|c| c.name == name).ok_or_else(|| {
            let known: Vec<&str> = self.chains.iter().map(|c| c.name.as_str()).collect();
            Error::chain_not_found(format!("no chain named '{name}'; configured chains: {known:?}"))
        })
    }
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chains", get(list_chains))
        .route("/v1/chains/{name}/run", post(run_chain))
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

// ---------- health / stats ----------

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    let child = app.sup.child_snapshot();
    let c = app.sup.stats.counters();
    let sys = win::memory_status();
    Json(json!({
        "status": "ok",
        "model": app.cfg.model.name,
        "model_state": app.sup.phase().as_str(),
        "child_working_set_mb": child.as_ref().map(|c| c.working_set_mb),
        "child_private_mb": child.as_ref().map(|c| c.private_mb),
        "system_free_mb": sys.map(|m| mb(m.avail)),
        "requests_served": c.requests_served,
        "last_tokens_per_sec": c.last_tokens_per_sec,
        "pacer_capped": app.sup.pacer.capped(),
        "uptime_secs": app.sup.stats.uptime_secs(),
    }))
}

async fn stats(State(app): State<Arc<App>>) -> Json<Value> {
    let cfg = &app.cfg;
    let sup = &app.sup;
    let sys = win::memory_status();
    Json(json!({
        "uptime_secs": sup.stats.uptime_secs(),
        "model_state": sup.phase().as_str(),
        "model": {
            "name": cfg.model.name,
            "path": cfg.model.path.display().to_string(),
            "ctx_size": cfg.model.ctx_size,
            "threads": sup.threads(),
            "kv_cache_type": cfg.model.kv_cache_type,
            "flash_attn": cfg.model.flash_attn,
            "mmap": cfg.model.mmap,
            "no_repack": cfg.model.no_repack,
            "cache_ram_mb": cfg.model.cache_ram_mb,
            "ctx_checkpoints": cfg.model.ctx_checkpoints,
            "adapters": sup.adapters().iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        },
        "child": sup.child_snapshot(),
        "system": {
            "total_mb": sys.map(|m| mb(m.total)),
            "free_mb": sys.map(|m| mb(m.avail)),
            "memory_load_pct": sys.map(|m| m.load_pct),
            "low_memory_signal": sup.low_memory_signal(),
        },
        "budget": {
            "memory_cap_mb": sup.plan.cap_mb(),
            "memory_cap_source": sup.plan.source,
            "memory_estimate_mb": sup.plan.estimate.as_ref().map(|e| json!({
                "kv": mb(e.kv_bytes), "recurrent": mb(e.recurrent_bytes), "compute": mb(e.compute_bytes),
                "cache_ram": mb(e.cache_ram_bytes), "checkpoints": mb(e.checkpoint_bytes), "overhead": mb(e.overhead_bytes),
                "total": mb(e.total_bytes), "kv_layers": e.kv_layers, "swa_layers": e.swa_layers, "arch": e.arch,
            })),
            "reserve_free_mb": cfg.budget.reserve_free_mb,
            "load_estimate_mb": crate::budget::load_estimate_bytes(cfg, &sup.plan) / (1024 * 1024),
            "pressure_free_mb": cfg.budget.pressure_free_mb,
            "critical_free_mb": cfg.budget.critical_free_mb,
            "unload_pending": sup.unload_pending(),
            "idle_unload_secs": cfg.budget.idle_unload_secs,
            "idle_secs": (sup.idle_secs() * 10.0).round() / 10.0,
        },
        "pacer": sup.pacer.snapshot(sup.phase() == crate::supervisor::Phase::Loaded),
        "queue": { "generating": sup.generating(), "waiting": sup.waiting(), "max_queue": cfg.server.max_queue },
        "counters": sup.stats.counters(),
    }))
}

// ---------- models ----------

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let created = now_secs();
    let mut data = vec![json!({ "id": app.cfg.model.name, "object": "model", "created": created, "owned_by": "jitmo-runtime" })];
    for c in &app.chains {
        data.push(json!({ "id": format!("chain:{}", c.name), "object": "model", "created": created, "owned_by": "jitmo-runtime" }));
    }
    Json(json!({ "object": "list", "data": data }))
}

// ---------- chat completions ----------

async fn chat_completions(State(app): State<Arc<App>>, body: Bytes) -> Result<Response, Error> {
    let started = Instant::now();
    let mut req: Value = serde_json::from_slice(&body).map_err(|e| Error::bad_request(format!("body is not valid JSON: {e}")))?;
    if !req.is_object() {
        return Err(Error::bad_request("body must be a JSON object"));
    }
    if !req["messages"].is_array() {
        return Err(Error::bad_request("'messages' must be an array"));
    }
    let model = req["model"].as_str().unwrap_or(&app.cfg.model.name).to_string();
    let stream = req["stream"].as_bool().unwrap_or(false);

    if let Some(name) = model.strip_prefix("chain:") {
        return chain_as_chat(&app, name, &req, stream).await;
    }
    if model != app.cfg.model.name {
        return Err(Error::model_not_found(format!(
            "unknown model '{model}'; use '{}' or 'chain:<name>' (see GET /v1/models)",
            app.cfg.model.name
        )));
    }
    if let Some(k) = &app.cfg.model.chat_template_kwargs {
        if req.get("chat_template_kwargs").is_none() {
            req["chat_template_kwargs"] = k.clone();
        }
    }

    let slot = match app.sup.acquire().await {
        Ok(s) => s,
        Err(e) => {
            app.sup.stats.request_done(false);
            return Err(e);
        }
    };
    let resp = match llama::post_json(&app.sup.http, slot.port, "/v1/chat/completions", &req).await {
        Ok(r) => r,
        Err(e) => {
            app.sup.stats.request_done(false);
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
        let mut guard = StreamGuard { app: app.clone(), slot: Some(slot), tracker: SseTracker::default(), started, broken: false };
        let body_stream = resp.into_body().into_data_stream().map(move |chunk| {
            match &chunk {
                Ok(bytes) => guard.tracker.feed(bytes),
                Err(_) => guard.broken = true,
            }
            chunk
        });
        return Ok(Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(body_stream))
            .expect("valid response"));
    }

    let (status, bytes) = match llama::collect(resp).await {
        Ok(x) => x,
        Err(e) => {
            app.sup.stats.request_done(false);
            return Err(e);
        }
    };
    if status.is_success() {
        let engine = serde_json::from_slice::<Value>(&bytes).ok().and_then(|v| llama::engine_timings(&v)).unwrap_or_default();
        let timing = RequestTiming::new("chat", model, slot.queue_ms, slot.load_ms, &engine, None, ms_since(started));
        app.sup.pacer.after_generation(&timing);
        app.sup.stats.record(timing);
    }
    app.sup.stats.request_done(status.is_success());
    drop(slot);
    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(bytes))
        .expect("valid response"))
}

/// Lives as long as a streamed response: holds the generation slot, then records the timing.
struct StreamGuard {
    app: Arc<App>,
    slot: Option<Slot>,
    tracker: SseTracker,
    started: Instant,
    broken: bool,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        let Some(slot) = self.slot.take() else { return };
        let complete = !self.broken && self.tracker.timings.is_some();
        if complete {
            let engine = self.tracker.timings.clone().unwrap_or_default();
            let first = self.tracker.first_token_at.map(|t| (t - self.started).as_secs_f64() * 1000.0);
            let timing = RequestTiming::new(
                "chat_stream",
                self.app.cfg.model.name.clone(),
                slot.queue_ms,
                slot.load_ms,
                &engine,
                first,
                ms_since(self.started),
            );
            self.app.sup.pacer.after_generation(&timing);
            self.app.sup.stats.record(timing);
        }
        self.app.sup.stats.request_done(complete);
        drop(slot);
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

async fn run_chain_counted(app: &App, chain: &Chain, input: &str) -> Result<ChainReport, Error> {
    let r = chain::run(&app.sup, &app.cfg, chain, input).await;
    app.sup.stats.request_done(r.is_ok());
    r
}

/// `model: "chain:<name>"` on /v1/chat/completions: the last user message is the chain input,
/// the last step's output is the answer. With `stream: true` the whole answer comes in one chunk.
async fn chain_as_chat(app: &Arc<App>, name: &str, req: &Value, stream: bool) -> Result<Response, Error> {
    let chain = app.chain(name)?;
    let input = last_user_text(req).ok_or_else(|| Error::bad_request("a chain needs at least one user message"))?;
    let report = run_chain_counted(app, chain, &input).await?;
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

// ---------- chains ----------

async fn list_chains(State(app): State<Arc<App>>) -> Json<Value> {
    let chains: Vec<Value> = app
        .chains
        .iter()
        .map(|c| {
            json!({
                "name": c.name,
                "description": c.description,
                "steps": c.steps.iter().map(|s| json!({
                    "name": s.name,
                    "json_schema": s.json_schema.is_some(),
                    "lora": s.lora_path.as_ref().map(|p| json!({ "path": p, "scale": s.lora.map(|l| l.1) })),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    Json(json!({ "chains": chains }))
}

#[derive(Deserialize)]
struct ChainRunRequest {
    input: String,
}

async fn run_chain(State(app): State<Arc<App>>, Path(name): Path<String>, body: Bytes) -> Result<Json<ChainReport>, Error> {
    let chain = app.chain(&name)?;
    let req: ChainRunRequest =
        serde_json::from_slice(&body).map_err(|e| Error::bad_request(format!("expected {{\"input\": \"...\"}}: {e}")))?;
    Ok(Json(run_chain_counted(&app, chain, &req.input).await?))
}

// ---------- admin ----------

#[derive(Deserialize)]
struct UnloadQuery {
    #[serde(default)]
    force: bool,
}

async fn admin_unload(State(app): State<Arc<App>>, Query(q): Query<UnloadQuery>) -> Json<Value> {
    let was_loaded = if q.force {
        app.sup.unload(UnloadReason::Admin).await
    } else {
        app.sup.unload_when_free(UnloadReason::Admin).await
    };
    Json(json!({ "unloaded": was_loaded, "model_state": app.sup.phase().as_str() }))
}
