//! Counters and per-request timings, kept in memory for `/health` and `/stats`.

use crate::llama::EngineTimings;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;

const RECENT: usize = 20;

/// Timing of one generation (one chat request, or one chain step).
#[derive(Debug, Clone, Serialize)]
pub struct RequestTiming {
    /// `chat`, `chat_stream`, `chain_step` or `embedding`.
    pub kind: &'static str,
    /// The request's id (`X-Jitmo-Request-Id`).
    pub request_id: String,
    /// The model that ran it.
    pub model: String,
    /// Model name, or `chain/step`.
    pub label: String,
    /// Time spent waiting for the one generation slot.
    pub queue_ms: f64,
    /// Time spent starting the model for this request (0 if it was already loaded).
    pub load_ms: f64,
    /// From request arrival to the first token. Measured for streams; for non-stream
    /// requests it is queue + load + the engine's prompt time.
    pub first_token_ms: Option<f64>,
    pub prompt_tokens: u64,
    pub prompt_ms: f64,
    pub prompt_tokens_per_sec: Option<f64>,
    pub completion_tokens: u64,
    pub generation_ms: f64,
    pub tokens_per_sec: Option<f64>,
    /// Wall time of the whole request inside the runtime.
    pub total_ms: f64,
}

impl RequestTiming {
    pub fn new(
        kind: &'static str,
        label: String,
        queue_ms: f64,
        load_ms: f64,
        engine: &EngineTimings,
        first_token_ms: Option<f64>,
        total_ms: f64,
    ) -> RequestTiming {
        let rate = |n: u64, ms: f64| (ms > 0.0 && n > 0).then(|| round2(n as f64 * 1000.0 / ms));
        RequestTiming {
            kind,
            request_id: String::new(),
            model: String::new(),
            label,
            queue_ms: round2(queue_ms),
            load_ms: round2(load_ms),
            first_token_ms: first_token_ms
                .or_else(|| (engine.prompt_ms > 0.0).then(|| queue_ms + load_ms + engine.prompt_ms))
                .map(round2),
            prompt_tokens: engine.prompt_n,
            prompt_ms: round2(engine.prompt_ms),
            prompt_tokens_per_sec: rate(engine.prompt_n, engine.prompt_ms),
            completion_tokens: engine.predicted_n,
            generation_ms: round2(engine.predicted_ms),
            tokens_per_sec: rate(engine.predicted_n, engine.predicted_ms),
            total_ms: round2(total_ms),
        }
    }
}

impl RequestTiming {
    pub fn with_ids(mut self, request_id: &str, model: &str) -> RequestTiming {
        self.request_id = request_id.to_string();
        self.model = model.to_string();
        self
    }
}

pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

pub fn mb(bytes: u64) -> f64 {
    (bytes as f64 / 1048576.0 * 10.0).round() / 10.0
}

#[derive(Debug, Clone, Copy)]
pub enum UnloadReason {
    Idle,
    MemoryPressure,
    MemoryCritical,
    /// Made room for another model (pool budget or free RAM).
    Evicted,
    Admin,
    Crashed,
    Shutdown,
}

impl UnloadReason {
    pub fn as_str(self) -> &'static str {
        match self {
            UnloadReason::Idle => "idle",
            UnloadReason::MemoryPressure => "memory_pressure",
            UnloadReason::MemoryCritical => "memory_critical",
            UnloadReason::Evicted => "evicted",
            UnloadReason::Admin => "admin",
            UnloadReason::Crashed => "crashed",
            UnloadReason::Shutdown => "shutdown",
        }
    }
}

#[derive(Default)]
struct Inner {
    served: u64,
    failed: u64,
    refused_memory: u64,
    refused_queue: u64,
    cancelled: u64,
    retried_after_crash: u64,
    loads: u64,
    load_failures: u64,
    last_load_ms: Option<f64>,
    unloads: BTreeMap<&'static str, u64>,
    last: Option<RequestTiming>,
    recent: VecDeque<RequestTiming>,
    peak_ws: u64,
    peak_private: u64,
    peak_job_commit: u64,
}

pub struct Stats {
    started: Instant,
    inner: Mutex<Inner>,
}

/// Everything `/stats` reports about counters (the children's live memory comes from the pool).
#[derive(Debug, Clone, Serialize)]
pub struct Counters {
    pub requests_served: u64,
    pub requests_failed: u64,
    pub refused_insufficient_memory: u64,
    pub refused_queue_full: u64,
    /// Requests ended by `DELETE /v1/requests/{id}`.
    pub cancelled: u64,
    /// Requests sent again because their model process died under them.
    pub retried_after_crash: u64,
    pub loads: u64,
    pub load_failures: u64,
    pub last_load_ms: Option<f64>,
    pub unloads: BTreeMap<&'static str, u64>,
    /// Highest child memory seen over all loads, in MB.
    pub peak_child_working_set_mb: f64,
    pub peak_child_private_mb: f64,
    pub peak_job_commit_mb: f64,
    pub last_tokens_per_sec: Option<f64>,
    pub last: Option<RequestTiming>,
    pub recent: Vec<RequestTiming>,
}

impl Stats {
    pub fn new() -> Stats {
        Stats { started: Instant::now(), inner: Mutex::new(Inner::default()) }
    }

    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        f(&mut g)
    }

    pub fn uptime_secs(&self) -> f64 {
        round2(self.started.elapsed().as_secs_f64())
    }

    pub fn request_done(&self, ok: bool) {
        self.with(|i| if ok { i.served += 1 } else { i.failed += 1 });
    }
    pub fn refused_memory(&self) {
        self.with(|i| i.refused_memory += 1);
    }
    pub fn refused_queue(&self) {
        self.with(|i| i.refused_queue += 1);
    }
    pub fn cancelled(&self) {
        self.with(|i| i.cancelled += 1);
    }
    pub fn retried_after_crash(&self) {
        self.with(|i| i.retried_after_crash += 1);
    }
    pub fn loaded(&self, ms: f64) {
        self.with(|i| {
            i.loads += 1;
            i.last_load_ms = Some(round2(ms));
        });
    }
    pub fn load_failed(&self) {
        self.with(|i| i.load_failures += 1);
    }
    pub fn unloaded(&self, reason: UnloadReason) {
        self.with(|i| *i.unloads.entry(reason.as_str()).or_default() += 1);
    }
    pub fn child_peaks(&self, ws: u64, private: u64, job_commit: u64) {
        self.with(|i| {
            i.peak_ws = i.peak_ws.max(ws);
            i.peak_private = i.peak_private.max(private);
            i.peak_job_commit = i.peak_job_commit.max(job_commit);
        });
    }
    pub fn record(&self, t: RequestTiming) {
        self.with(|i| {
            if i.recent.len() == RECENT {
                i.recent.pop_front();
            }
            i.recent.push_back(t.clone());
            i.last = Some(t);
        });
    }

    pub fn counters(&self) -> Counters {
        self.with(|i| Counters {
            requests_served: i.served,
            requests_failed: i.failed,
            refused_insufficient_memory: i.refused_memory,
            refused_queue_full: i.refused_queue,
            cancelled: i.cancelled,
            retried_after_crash: i.retried_after_crash,
            loads: i.loads,
            load_failures: i.load_failures,
            last_load_ms: i.last_load_ms,
            unloads: i.unloads.clone(),
            peak_child_working_set_mb: mb(i.peak_ws),
            peak_child_private_mb: mb(i.peak_private),
            peak_job_commit_mb: mb(i.peak_job_commit),
            last_tokens_per_sec: i.last.as_ref().and_then(|t| t.tokens_per_sec),
            last: i.last.clone(),
            recent: i.recent.iter().cloned().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_come_from_engine_timings() {
        let e = EngineTimings { prompt_n: 100, prompt_ms: 500.0, predicted_n: 20, predicted_ms: 2000.0 };
        let t = RequestTiming::new("chat", "m".into(), 10.0, 0.0, &e, None, 2600.0);
        assert_eq!(t.prompt_tokens_per_sec, Some(200.0));
        assert_eq!(t.tokens_per_sec, Some(10.0));
        assert_eq!(t.first_token_ms, Some(510.0));
        let s = Stats::new();
        s.record(t);
        assert_eq!(s.counters().last_tokens_per_sec, Some(10.0));
    }
}
