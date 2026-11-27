//! The model pool: one llama-server child per loaded model, each in its own Job Object
//! (commit cap + kill-on-close, so no child can outlive the runtime).
//!
//! - Models load on demand (or with `POST /admin/load`), on a private 127.0.0.1 port.
//! - Loads are serialized. Before a load the pool checks the pool budget (sum of the load
//!   estimates of loaded models) and free RAM; if either is short it evicts the least recently
//!   used model that is loaded, not pinned and not in use, and checks again.
//! - A model in use (loading for a request, or generating) holds a read lock on it; eviction
//!   and idle/pressure unloads need the write lock, so they never kill a running generation.
//!   Only critical memory and `?force=true` do.
//! - Generation slots (`server.max_concurrent`, default 1) are shared by all models: see `sched.rs`.
//! - A watchdog unloads idle models, handles memory pressure per model, and notices children
//!   that died. One pacer caps every loaded child's CPU while the user is busy.

use crate::budget::{self, MemPlan, MemoryAction};
use crate::config::{Config, ModelConfig, ModelKind};
use crate::error::Error;
use crate::llama::{self, HttpClient};
use crate::pacer::{self, Pacer};
use crate::sched::{CancelToken, Permit, RequestGuard, Requests, Scheduler};
use crate::stats::{mb, Stats, UnloadReason};
use crate::win::{self, Job, LowMemorySignal, ProcessMemory, HANDLE};
use hyper::StatusCode;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedRwLockReadGuard, RwLock};
use tracing::{info, warn};

const MB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Unloaded = 0,
    Loading = 1,
    Loaded = 2,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Unloaded => "unloaded",
            Phase::Loading => "loading",
            Phase::Loaded => "loaded",
        }
    }
}

struct Running {
    child: tokio::process::Child,
    job: Arc<Job>,
    pid: u32,
    port: u16,
}

/// What `/stats` may read about a running child without touching the child itself.
#[derive(Clone)]
struct ChildInfo {
    pid: u32,
    port: u16,
    loaded_at: Instant,
    load_ms: f64,
    job: Arc<Job>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChildSnapshot {
    pub pid: u32,
    pub port: u16,
    pub loaded_secs: f64,
    pub load_ms: f64,
    pub working_set_mb: f64,
    pub private_mb: f64,
    pub peak_working_set_mb: f64,
    pub peak_private_mb: f64,
    /// Peak commit as the Job Object counts it (this is what the memory cap limits).
    pub job_peak_commit_mb: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelCounters {
    pub loads: u64,
    pub load_failures: u64,
    pub evictions: u64,
    pub last_load_ms: Option<f64>,
    pub generations: u64,
}

/// One configured model and its child (when loaded).
pub struct ModelHandle {
    pub cfg: ModelConfig,
    pub plan: MemPlan,
    pub threads: u32,
    pub adapters: Vec<PathBuf>,
    pub idle_unload_secs: u64,
    state: tokio::sync::Mutex<Option<Running>>,
    info: Mutex<Option<ChildInfo>>,
    phase: AtomicU8,
    use_lock: Arc<RwLock<()>>,
    active: AtomicUsize,
    last_used: Mutex<Option<Instant>>,
    last_activity: Mutex<Instant>,
    /// Memory pressure was seen while it was in use: unload before its next use.
    pending_unload: AtomicBool,
    /// When `pending_unload` was set (for `budget.pressure_finish_max_ms`).
    pending_since: Mutex<Option<Instant>>,
    /// Cancel tokens of the requests that use the model now (by use id).
    users: Mutex<Vec<(u64, CancelToken)>>,
    counters: Mutex<ModelCounters>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl ModelHandle {
    fn new(cfg: &Config, m: &ModelConfig) -> ModelHandle {
        ModelHandle {
            plan: MemPlan::new(m, &cfg.budget),
            threads: budget::threads(m),
            adapters: cfg.adapters_for(&m.name),
            idle_unload_secs: m.idle_unload_secs.unwrap_or(cfg.budget.idle_unload_secs),
            cfg: m.clone(),
            state: tokio::sync::Mutex::new(None),
            info: Mutex::new(None),
            phase: AtomicU8::new(Phase::Unloaded as u8),
            use_lock: Arc::new(RwLock::new(())),
            active: AtomicUsize::new(0),
            last_used: Mutex::new(None),
            last_activity: Mutex::new(Instant::now()),
            pending_unload: AtomicBool::new(false),
            pending_since: Mutex::new(None),
            users: Mutex::new(Vec::new()),
            counters: Mutex::new(ModelCounters::default()),
        }
    }

    pub fn name(&self) -> &str {
        &self.cfg.name
    }

    pub fn phase(&self) -> Phase {
        match self.phase.load(Ordering::SeqCst) {
            1 => Phase::Loading,
            2 => Phase::Loaded,
            _ => Phase::Unloaded,
        }
    }

    fn set_phase(&self, p: Phase) {
        self.phase.store(p as u8, Ordering::SeqCst);
    }

    /// Requests using this model now (loading it for themselves, or generating).
    pub fn active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    pub fn unload_pending(&self) -> bool {
        self.pending_unload.load(Ordering::SeqCst)
    }

    fn touch(&self) {
        let now = Instant::now();
        *lock(&self.last_activity) = now;
        *lock(&self.last_used) = Some(now);
    }

    /// Seconds since the model was last used or loaded.
    pub fn idle_secs(&self) -> f64 {
        lock(&self.last_activity).elapsed().as_secs_f64()
    }

    /// Seconds since the model was last used by a request, or None if never.
    pub fn last_used_secs(&self) -> Option<f64> {
        lock(&self.last_used).map(|t| (t.elapsed().as_secs_f64() * 10.0).round() / 10.0)
    }

    pub fn counters(&self) -> ModelCounters {
        lock(&self.counters).clone()
    }

    pub fn count_generation(&self) {
        lock(&self.counters).generations += 1;
    }

    fn child(&self) -> Option<pacer::Child> {
        lock(&self.info).as_ref().map(|i| pacer::Child { pid: i.pid, job: i.job.clone(), threads: self.threads })
    }

    /// Live memory of the child, if it is loaded.
    pub fn child_snapshot(&self, stats: &Stats) -> Option<ChildSnapshot> {
        let info = lock(&self.info).clone()?;
        let mem = win::process_memory_by_pid(info.pid).unwrap_or_default();
        record_peaks(stats, &mem, &info.job);
        Some(ChildSnapshot {
            pid: info.pid,
            port: info.port,
            loaded_secs: (info.loaded_at.elapsed().as_secs_f64() * 10.0).round() / 10.0,
            load_ms: (info.load_ms * 10.0).round() / 10.0,
            working_set_mb: mb(mem.working_set),
            private_mb: mb(mem.private),
            peak_working_set_mb: mb(mem.peak_working_set),
            peak_private_mb: mb(mem.peak_private),
            job_peak_commit_mb: info.job.peak_process_commit().map(mb),
        })
    }
}

fn record_peaks(stats: &Stats, mem: &ProcessMemory, job: &Job) {
    stats.child_peaks(mem.peak_working_set, mem.peak_private, job.peak_process_commit().unwrap_or(0));
}

/// A request's hold on a model: it cannot be evicted or idle-unloaded while this lives.
pub struct ModelUse {
    _guard: OwnedRwLockReadGuard<()>,
    model: Arc<ModelHandle>,
    id: u64,
}

static NEXT_USE: AtomicU64 = AtomicU64::new(1);

impl Drop for ModelUse {
    fn drop(&mut self) {
        lock(&self.model.users).retain(|(id, _)| *id != self.id);
        self.model.touch();
        self.model.active.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The right to run one generation (or a chain step) on a loaded model.
pub struct Slot {
    pub model: Arc<ModelHandle>,
    _use: ModelUse,
    _permit: Permit,
    pub port: u16,
    pub queue_ms: f64,
    pub load_ms: f64,
}

pub struct Pool {
    cfg: Arc<Config>,
    pub models: Vec<Arc<ModelHandle>>,
    pub stats: Arc<Stats>,
    pub http: HttpClient,
    pub pacer: Pacer,
    pub sched: Arc<Scheduler>,
    pub requests: Arc<Requests>,
    load_lock: tokio::sync::Mutex<()>,
    low_memory: Option<LowMemorySignal>,
    evictions: AtomicU64,
}

impl Pool {
    pub fn new(cfg: Arc<Config>, stats: Arc<Stats>) -> Arc<Pool> {
        let low_memory = LowMemorySignal::new();
        if low_memory.is_none() {
            warn!("cannot create the Windows low-memory notification; using the free-RAM floor only");
        }
        let models = cfg.models.iter().map(|m| Arc::new(ModelHandle::new(&cfg, m))).collect();
        let low_priority = cfg.models.iter().all(|m| m.low_priority);
        Arc::new(Pool {
            pacer: Pacer::new(cfg.pacer.clone(), budget::logical_processors(), low_priority, cfg.budget.pressure_free_mb),
            sched: Scheduler::new(cfg.server.max_concurrent, cfg.server.max_queue, cfg.server.preempt_background),
            requests: Requests::new(),
            http: llama::client(),
            models,
            stats,
            load_lock: tokio::sync::Mutex::new(()),
            low_memory,
            evictions: AtomicU64::new(0),
            cfg,
        })
    }

    pub fn low_memory_signal(&self) -> bool {
        self.low_memory.as_ref().is_some_and(|s| s.is_low())
    }

    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::SeqCst)
    }

    /// Sum of the load estimates of the models that are loaded or loading, in bytes.
    pub fn resident_estimate(&self) -> u64 {
        self.models.iter().filter(|m| m.phase() != Phase::Unloaded).map(|m| m.plan.load_bytes()).sum()
    }

    pub fn default_model(&self) -> &Arc<ModelHandle> {
        self.models.iter().find(|m| m.cfg.kind == ModelKind::Chat).unwrap_or(&self.models[0])
    }

    /// The model a request names, or the first model of `kind` when it names none.
    pub fn model(&self, name: Option<&str>, kind: Option<ModelKind>) -> Result<Arc<ModelHandle>, Error> {
        let found = match name {
            Some(n) => self.models.iter().find(|m| m.cfg.name == n),
            None => self.models.iter().find(|m| kind.is_none_or(|k| m.cfg.kind == k)),
        };
        let Some(m) = found else {
            let known: Vec<&str> = self.models.iter().map(|m| m.name()).collect();
            return Err(match name {
                Some(n) => Error::model_not_found(format!("unknown model '{n}'; configured models: {known:?} (see GET /v1/models)")),
                None => Error::model_not_found(format!(
                    "no {} model is configured; configured models: {known:?}",
                    kind.map_or("", |k| k.as_str())
                )),
            });
        };
        if let Some(k) = kind {
            if m.cfg.kind != k {
                return Err(Error::wrong_kind(format!("model '{}' is {} model, not {} model", m.name(), article(m.cfg.kind), article(k))));
            }
        }
        Ok(m.clone())
    }

    /// Wait for a generation slot (priority queue), then make sure the model is loaded.
    pub async fn acquire(&self, m: &Arc<ModelHandle>, req: &RequestGuard) -> Result<Slot, Error> {
        let t0 = Instant::now();
        let permit = match self.sched.acquire(req.priority, Some(req.token.clone())).await {
            Ok(p) => p,
            Err(e) => {
                if e.code == "queue_full" {
                    self.stats.refused_queue();
                }
                return Err(e);
            }
        };
        let queue_ms = t0.elapsed().as_secs_f64() * 1000.0;
        req.set_running();
        let (u, port, load_ms) = self.use_model(m, Some(&req.token)).await?;
        Ok(Slot { model: m.clone(), _use: u, _permit: permit, port, queue_ms, load_ms })
    }

    /// Hold the model (load it if needed). Returns the hold, the child's port and the load time.
    /// `token` = the request's cancel token, so memory pressure can stop it (see `force_pressure_unload`).
    pub async fn use_model(&self, m: &Arc<ModelHandle>, token: Option<&CancelToken>) -> Result<(ModelUse, u16, f64), Error> {
        if m.pending_unload.load(Ordering::SeqCst) {
            let _w = m.use_lock.write().await;
            if m.pending_unload.swap(false, Ordering::SeqCst) {
                info!("unloading model '{}' for memory pressure before its next request", m.name());
                self.unload(m, UnloadReason::MemoryPressure).await;
            }
        }
        let guard = m.use_lock.clone().read_owned().await;
        m.active.fetch_add(1, Ordering::SeqCst);
        m.touch();
        let id = NEXT_USE.fetch_add(1, Ordering::Relaxed);
        if let Some(t) = token {
            lock(&m.users).push((id, t.clone()));
        }
        let u = ModelUse { _guard: guard, model: m.clone(), id };
        let (port, load_ms) = self.ensure_loaded(m).await?;
        Ok((u, port, load_ms))
    }

    /// Port of a running child, after removing a child that exited by itself.
    async fn running_port(&self, m: &ModelHandle) -> Option<u16> {
        let mut st = m.state.lock().await;
        if let Some(run) = st.as_mut() {
            if matches!(run.child.try_wait(), Ok(None)) {
                return Some(run.port);
            }
            warn!("llama-server of model '{}' had exited by itself; starting it again", m.name());
            let run = st.take().expect("checked above");
            self.finish_stop(m, run, UnloadReason::Crashed).await;
        }
        None
    }

    /// Returns the child's port and how long this request waited for the model to be ready
    /// (0 if it was already loaded).
    async fn ensure_loaded(&self, m: &Arc<ModelHandle>) -> Result<(u16, f64), Error> {
        if let Some(port) = self.running_port(m).await {
            return Ok((port, 0.0));
        }
        let t0 = Instant::now();
        let _l = self.load_lock.lock().await;
        if let Some(port) = self.running_port(m).await {
            // Another request (or /admin/load) loaded it while this one waited.
            return Ok((port, t0.elapsed().as_secs_f64() * 1000.0));
        }
        if let Err(why) = self.make_room(m).await {
            self.stats.refused_memory();
            warn!("{why}");
            return Err(Error::insufficient_memory(why));
        }
        m.set_phase(Phase::Loading);
        let mut st = m.state.lock().await;
        match self.start_child(m).await {
            Ok(run) => {
                let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let (port, pid) = (run.port, run.pid);
                *lock(&m.info) = Some(ChildInfo { pid, port, loaded_at: Instant::now(), load_ms, job: run.job.clone() });
                *st = Some(run);
                m.set_phase(Phase::Loaded);
                m.touch();
                self.stats.loaded(load_ms);
                {
                    let mut c = lock(&m.counters);
                    c.loads += 1;
                    c.last_load_ms = Some((load_ms * 10.0).round() / 10.0);
                }
                info!("model '{}' loaded in {:.0} ms (llama-server pid {pid}, port {port})", m.name(), load_ms);
                Ok((port, load_ms))
            }
            Err(e) => {
                m.set_phase(Phase::Unloaded);
                self.stats.load_failed();
                lock(&m.counters).load_failures += 1;
                warn!("model '{}' load failed: {}", m.name(), e.message);
                Err(e)
            }
        }
    }

    /// Evict least recently used idle, unpinned models until `m` fits the pool budget and the
    /// free-RAM reserve. Called with the load lock held.
    async fn make_room(&self, m: &Arc<ModelHandle>) -> Result<(), String> {
        let need = m.plan.load_bytes();
        let budget = self.cfg.budget.pool_budget_mb * MB;
        loop {
            let resident: u64 = self
                .models
                .iter()
                .filter(|o| !Arc::ptr_eq(o, m) && o.phase() != Phase::Unloaded)
                .map(|o| o.plan.load_bytes())
                .sum();
            let over_budget = budget > 0 && resident + need > budget;
            let ram = if over_budget { Ok(()) } else { budget::check_free_ram(need, self.cfg.budget.reserve_free_mb) };
            if !over_budget && ram.is_ok() {
                return Ok(());
            }
            let mut candidates: Vec<&Arc<ModelHandle>> = self
                .models
                .iter()
                .filter(|o| !Arc::ptr_eq(o, m) && o.phase() == Phase::Loaded && !o.cfg.pinned)
                .collect();
            candidates.sort_by_key(|o| *lock(&o.last_used));
            let mut evicted = false;
            for c in candidates {
                if let Ok(_w) = c.use_lock.clone().try_write_owned() {
                    info!(
                        "evicting model '{}' (least recently used, idle) to make room for '{}' ({})",
                        c.name(),
                        m.name(),
                        if over_budget { "pool budget" } else { "free RAM" }
                    );
                    if self.unload(c, UnloadReason::Evicted).await {
                        lock(&c.counters).evictions += 1;
                        self.evictions.fetch_add(1, Ordering::SeqCst);
                    }
                    evicted = true;
                    break;
                }
            }
            if !evicted {
                let others: Vec<String> = self
                    .models
                    .iter()
                    .filter(|o| !Arc::ptr_eq(o, m) && o.phase() != Phase::Unloaded)
                    .map(|o| {
                        let why = if o.cfg.pinned { "pinned" } else if o.active() > 0 { "in use" } else { "loading" };
                        format!("{} ({} MB, {why})", o.name(), o.plan.load_bytes() / MB)
                    })
                    .collect();
                let held = if others.is_empty() { String::new() } else { format!("; loaded and not evictable: {}", others.join(", ")) };
                return Err(if over_budget {
                    format!(
                        "model '{}' needs about {} MB, {} MB of the pool budget of {} MB is in use (budget.pool_budget_mb){held}",
                        m.name(),
                        need / MB,
                        resident / MB,
                        budget / MB
                    )
                } else {
                    format!(
                        "not enough free RAM to start model '{}': {}{held}. Close some apps and try again.",
                        m.name(),
                        ram.err().unwrap_or_default()
                    )
                });
            }
            // Give Windows a moment to account the freed memory before checking again.
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
    }

    async fn start_child(&self, h: &ModelHandle) -> Result<Running, Error> {
        let m = &h.cfg;
        let port = if m.port != 0 { m.port } else { free_port().map_err(|e| Error::load_failed(format!("no free port: {e}")))? };
        let args = llama_args(m, port, h.threads, &h.adapters);
        if let Some(c) = h.plan.cap_bytes {
            info!("model '{}': memory cap {} MB ({})", m.name, c / MB, h.plan.source);
        }
        info!("starting llama-server: {} {}", m.llama_server.display(), args.join(" "));
        let log_file = m.log_path();

        let mut cmd = tokio::process::Command::new(&m.llama_server);
        cmd.args(&args).stdin(Stdio::null()).kill_on_drop(true);
        let (out, err) = log_stdio(&log_file);
        cmd.stdout(out).stderr(err);
        let mut flags = win::CREATE_SUSPENDED | win::CREATE_NO_WINDOW;
        if m.low_priority {
            flags |= win::BELOW_NORMAL_PRIORITY_CLASS;
        }
        cmd.creation_flags(flags);
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::load_failed(format!("cannot start {}: {e}", m.llama_server.display())))?;
        let pid = child.id().unwrap_or(0);
        let cap = h.plan.cap_bytes;
        let job = match contain(&child, cap) {
            Ok(j) => Arc::new(j),
            Err(e) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(Error::load_failed(e));
            }
        };
        let mut run = Running { child, job, pid, port };

        let deadline = Instant::now() + Duration::from_secs(m.load_timeout_secs);
        loop {
            if let Ok(Some(status)) = run.child.try_wait() {
                let mut msg = format!("llama-server of model '{}' exited while loading ({status})", m.name);
                let tail = log_tail(&log_file, 8);
                if let Some(cap) = cap {
                    // One big allocation can fail while the peak is still well under the cap,
                    // so an allocation error in the log counts too.
                    let near_cap = run.job.peak_process_commit().is_some_and(|p| p as f64 >= cap as f64 * 0.97);
                    let lower = tail.to_lowercase();
                    let alloc_failed = lower.contains("alloc") || lower.contains("mem_buffer");
                    if near_cap || alloc_failed {
                        msg.push_str(&format!(
                            "; memory allocation failed under the memory cap of {} MB (memory_cap_mb, {}): raise the cap or lower ctx_size",
                            h.plan.cap_mb(),
                            h.plan.source
                        ));
                    }
                }
                if !tail.is_empty() {
                    msg.push_str(&format!(". Last lines of {}:\n{tail}", log_file.display()));
                }
                return Err(Error::load_failed(msg));
            }
            if llama::get_status(&self.http, port, "/health", Duration::from_secs(2)).await == Some(StatusCode::OK) {
                return Ok(run);
            }
            if Instant::now() > deadline {
                run.job.terminate();
                let _ = run.child.wait().await;
                return Err(Error::load_failed(format!(
                    "model '{}' was not ready after {} s (load_timeout_secs)",
                    m.name, m.load_timeout_secs
                )));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Kill the child now, even mid-generation. Returns false if it was not loaded.
    pub async fn unload(&self, m: &ModelHandle, reason: UnloadReason) -> bool {
        let mut st = m.state.lock().await;
        match st.take() {
            Some(run) => {
                self.finish_stop(m, run, reason).await;
                true
            }
            None => false,
        }
    }

    /// Wait until no request uses the model, then unload it.
    pub async fn unload_when_free(&self, m: &ModelHandle, reason: UnloadReason) -> bool {
        let _w = m.use_lock.write().await;
        self.unload(m, reason).await
    }

    async fn finish_stop(&self, m: &ModelHandle, mut run: Running, reason: UnloadReason) {
        let mem = win::process_memory_by_pid(run.pid).unwrap_or_default();
        run.job.terminate();
        let _ = run.child.start_kill();
        let _ = run.child.wait().await;
        record_peaks(&self.stats, &mem, &run.job);
        *lock(&m.info) = None;
        m.set_phase(Phase::Unloaded);
        m.pending_unload.store(false, Ordering::SeqCst);
        *lock(&m.pending_since) = None;
        self.stats.unloaded(reason);
        info!(
            "model '{}' unloaded ({}); child peak working set {:.0} MB, peak private {:.0} MB",
            m.name(),
            reason.as_str(),
            mb(mem.peak_working_set),
            mb(mem.peak_private)
        );
    }

    pub fn spawn_watchdog(self: &Arc<Self>) {
        let pool = self.clone();
        let every = Duration::from_millis(self.cfg.budget.watch_interval_ms);
        tokio::spawn(async move {
            let mut low_count = 0u32;
            let mut tick = tokio::time::interval(every);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                pool.watch_once(&mut low_count).await;
            }
        });
    }

    async fn watch_once(&self, low_count: &mut u32) {
        // 1. A child died by itself (crash, or it hit the memory cap).
        for m in &self.models {
            if let Ok(mut st) = m.state.try_lock() {
                let exited = st.as_mut().and_then(|r| r.child.try_wait().ok().flatten());
                if let Some(status) = exited {
                    let run = st.take().expect("child present");
                    let near_cap = m
                        .plan
                        .cap_bytes
                        .is_some_and(|cap| run.job.peak_process_commit().is_some_and(|p| p as f64 >= cap as f64 * 0.97));
                    warn!(
                        "llama-server of model '{}' exited by itself ({status}){}; the model is unloaded",
                        m.name(),
                        if near_cap { ", after reaching the memory cap" } else { "" }
                    );
                    self.finish_stop(m, run, UnloadReason::Crashed).await;
                }
            }
        }
        let loaded: Vec<&Arc<ModelHandle>> = self.models.iter().filter(|m| m.phase() == Phase::Loaded).collect();
        if loaded.is_empty() {
            *low_count = 0;
            return;
        }
        // 2. Memory. Pressure (two checks in a row): running generations may finish, then their
        //    models are unloaded. Critical (one check): unload every model at once.
        if let Some(mem) = win::memory_status() {
            let (low, signal) = budget::is_low(&self.cfg, mem.avail, self.low_memory.as_ref());
            *low_count = if low { *low_count + 1 } else { 0 };
            match budget::memory_action(&self.cfg, mem.avail, signal, *low_count) {
                MemoryAction::UnloadNow(why) => {
                    warn!("critical memory ({why}): unloading every model now; running generations fail");
                    for m in &loaded {
                        self.unload(m, UnloadReason::MemoryCritical).await;
                    }
                    *low_count = 0;
                    return;
                }
                MemoryAction::UnloadWhenFree(why) => {
                    let max_ms = self.cfg.budget.pressure_finish_max_ms;
                    for m in &loaded {
                        if let Ok(_w) = m.use_lock.clone().try_write_owned() {
                            warn!("memory pressure ({why}): unloading model '{}' now", m.name());
                            self.unload(m, UnloadReason::MemoryPressure).await;
                        } else if !m.pending_unload.swap(true, Ordering::SeqCst) {
                            *lock(&m.pending_since) = Some(Instant::now());
                            let limit = if max_ms > 0 { format!(" (at most {:.0} s)", max_ms as f64 / 1000.0) } else { String::new() };
                            warn!("memory pressure ({why}): model '{}' may finish its generation{limit}, then it is unloaded", m.name());
                        } else {
                            self.check_pressure_wait(m, true).await;
                        }
                    }
                    *low_count = 0;
                    return;
                }
                MemoryAction::None => {}
            }
        }
        let still_low = *low_count > 0;
        for m in &loaded {
            // Pressure seen earlier, the generation still runs: stop it if it ran too long.
            if m.phase() == Phase::Loaded && self.check_pressure_wait(m, still_low).await {
                continue;
            }
            // Pressure seen earlier; the generation has ended and nobody uses the model.
            if m.pending_unload.load(Ordering::SeqCst) {
                if let Ok(_w) = m.use_lock.clone().try_write_owned() {
                    if m.pending_unload.load(Ordering::SeqCst) {
                        info!("the generation ended: unloading model '{}' for memory pressure", m.name());
                        self.unload(m, UnloadReason::MemoryPressure).await;
                        continue;
                    }
                }
            }
            // 3. Idle (pinned models stay).
            let idle = m.idle_unload_secs;
            if idle > 0 && !m.cfg.pinned && m.idle_secs() > idle as f64 {
                if let Ok(_w) = m.use_lock.clone().try_write_owned() {
                    if m.idle_secs() > idle as f64 {
                        info!("model '{}': no requests for {idle} s, unloading it", m.name());
                        self.unload(m, UnloadReason::Idle).await;
                    }
                }
            }
        }
    }

    /// A model waits for its generation to end before a pressure unload. Past
    /// `budget.pressure_finish_max_ms`, and only while free RAM is still short, cancel the
    /// requests that use it (they get a `memory_pressure` error) and unload it now.
    /// Returns true if it unloaded the model.
    async fn check_pressure_wait(&self, m: &ModelHandle, still_low: bool) -> bool {
        let since = *lock(&m.pending_since);
        let waited = since.map(|t| t.elapsed());
        if !m.pending_unload.load(Ordering::SeqCst) || !budget::pressure_wait_over(waited, self.cfg.budget.pressure_finish_max_ms, still_low) {
            return false;
        }
        let secs = waited.unwrap_or_default().as_secs_f64();
        let users: Vec<CancelToken> = lock(&m.users).iter().map(|(_, t)| t.clone()).collect();
        warn!(
            "memory pressure: model '{}' waited {secs:.0} s for its generation (budget.pressure_finish_max_ms); cancelling {} request(s) and unloading it now",
            m.name(),
            users.len()
        );
        let e = Error::memory_pressure(format!(
            "the runtime stopped this generation to unload model '{}': free RAM stayed low for {secs:.0} s. Close some apps and try again",
            m.name()
        ));
        for t in &users {
            t.cancel_with(e.clone());
        }
        // Let the cancelled requests end (they drop their hold); then unload either way.
        let w = tokio::time::timeout(Duration::from_millis(500), m.use_lock.write()).await;
        self.unload(m, UnloadReason::MemoryPressure).await;
        m.pending_unload.store(false, Ordering::SeqCst);
        *lock(&m.pending_since) = None;
        drop(w);
        true
    }

    /// The pacer loop: one sample for the whole pool, the cap applies to every loaded child.
    pub fn spawn_pacer(self: &Arc<Self>) {
        let pool = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(pool.pacer.poll_interval());
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                let children: Vec<pacer::Child> =
                    pool.models.iter().filter(|m| m.phase() == Phase::Loaded).filter_map(|m| m.child()).collect();
                pool.pacer.tick(&children);
            }
        });
    }

    pub async fn shutdown(&self) {
        for m in &self.models {
            self.unload(m, UnloadReason::Shutdown).await;
        }
    }
}

fn article(k: ModelKind) -> &'static str {
    match k {
        ModelKind::Chat => "a chat",
        ModelKind::Embedding => "an embedding",
    }
}

/// The llama-server command line.
pub fn llama_args(m: &ModelConfig, port: u16, threads: u32, adapters: &[PathBuf]) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "-m".into(),
        m.path.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--alias".into(),
        m.name.clone(),
        "-c".into(),
        m.ctx_size.to_string(),
        "-t".into(),
        threads.to_string(),
        "--parallel".into(),
        "1".into(),
        "-ctk".into(),
        m.kv_cache_type.clone(),
        "-ctv".into(),
        m.kv_cache_type.clone(),
        "-fa".into(),
        m.flash_attn.clone(),
        "--no-webui".into(),
    ];
    if !m.mmap {
        // This llama.cpp build has no --no-mmap; load mode "none" reads the weights into private memory.
        a.push("-lm".into());
        a.push("none".into());
    }
    if m.no_repack {
        a.push("--no-repack".into());
    }
    // Both grow the child's private memory with use; the automatic memory cap includes them.
    a.push("--cache-ram".into());
    a.push(m.cache_ram_mb.to_string());
    a.push("--ctx-checkpoints".into());
    a.push(m.ctx_checkpoints.to_string());
    if m.kind == ModelKind::Embedding {
        // Embedding models read each input in one batch: batch = ubatch = ctx.
        a.push("--embeddings".into());
        a.push("-b".into());
        a.push(m.ubatch().to_string());
        a.push("-ub".into());
        a.push(m.ubatch().to_string());
        if let Some(p) = &m.pooling {
            a.push("--pooling".into());
            a.push(p.clone());
        }
    }
    for path in adapters {
        a.push("--lora".into());
        a.push(path.display().to_string());
    }
    if !adapters.is_empty() {
        // Adapters are loaded once with scale 0; each request (or chain step) sets its own scale.
        a.push("--lora-init-without-apply".into());
    }
    a.extend(m.extra_args.iter().cloned());
    a
}

/// The child was created suspended: put it in a job (cap + kill-on-close) before it runs a
/// single instruction, then let it run.
fn contain(child: &tokio::process::Child, cap: Option<u64>) -> Result<Job, String> {
    let handle = child.raw_handle().ok_or("llama-server exited at once")? as HANDLE;
    let job = Job::new(cap).map_err(|e| format!("cannot create a Job Object: {e}"))?;
    job.assign(handle).map_err(|e| format!("cannot put llama-server in the Job Object: {e}"))?;
    if !win::resume_process(handle) {
        return Err("cannot resume the llama-server process".into());
    }
    Ok(job)
}

fn free_port() -> std::io::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn log_stdio(path: &Path) -> (Stdio, Stdio) {
    if path.as_os_str().is_empty() {
        return (Stdio::null(), Stdio::null());
    }
    match std::fs::File::create(path).and_then(|f| Ok((f.try_clone()?, f))) {
        Ok((a, b)) => (Stdio::from(a), Stdio::from(b)),
        Err(e) => {
            warn!("cannot write llama-server log {}: {e}", path.display());
            (Stdio::null(), Stdio::null())
        }
    }
}

fn log_tail(path: &Path, lines: usize) -> String {
    let Ok(text) = std::fs::read(path) else { return String::new() };
    let text = String::from_utf8_lossy(&text);
    let all: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_args_carry_the_config() {
        let m = ModelConfig { mmap: false, no_repack: true, kv_cache_type: "q8_0".into(), extra_args: vec!["--cache-reuse".into(), "256".into()], ..Default::default() };
        let a = llama_args(&m, 5555, 3, &[PathBuf::from("a.gguf"), PathBuf::from("b.gguf")]);
        let s = a.join(" ");
        assert!(s.contains("--port 5555"));
        assert!(s.contains("-t 3"));
        assert!(s.contains("-c 4096"));
        assert!(s.contains("-ctk q8_0 -ctv q8_0"));
        assert!(s.contains("--parallel 1"));
        assert!(s.contains("-lm none") && s.contains("--no-repack"));
        assert!(s.contains("--cache-ram 256 --ctx-checkpoints 4"));
        assert!(s.contains("--lora a.gguf --lora b.gguf --lora-init-without-apply"));
        assert!(s.ends_with("--cache-reuse 256"));
        assert!(!s.contains("--embeddings"));
        let plain = llama_args(&ModelConfig { no_repack: false, ..Default::default() }, 1, 1, &[]).join(" ");
        assert!(!plain.contains("--lora") && !plain.contains("-lm none") && !plain.contains("--no-repack"));
        let e = ModelConfig { kind: ModelKind::Embedding, ctx_size: 512, pooling: Some("mean".into()), ..Default::default() };
        let s = llama_args(&e, 1, 2, &[]).join(" ");
        assert!(s.contains("--embeddings -b 512 -ub 512 --pooling mean"), "{s}");
    }

    fn pool_of(toml: &str) -> Arc<Pool> {
        Pool::new(Arc::new(Config::from_toml(toml).unwrap()), Arc::new(Stats::new()))
    }

    #[test]
    fn models_are_found_by_name_and_kind() {
        let p = pool_of("[[models]]\nname = \"chat\"\n[[models]]\nname = \"emb\"\nkind = \"embedding\"\n");
        assert_eq!(p.default_model().name(), "chat");
        assert_eq!(p.model(None, Some(ModelKind::Embedding)).unwrap().name(), "emb");
        assert_eq!(p.model(Some("emb"), None).unwrap().name(), "emb");
        assert_eq!(p.model(Some("emb"), Some(ModelKind::Chat)).err().unwrap().code, "wrong_model_kind");
        assert_eq!(p.model(Some("nope"), None).err().unwrap().status, 404);
        assert_eq!(p.resident_estimate(), 0);
    }

    #[tokio::test]
    async fn an_in_use_model_cannot_be_evicted() {
        let p = pool_of("[[models]]\nname = \"a\"\n[[models]]\nname = \"b\"\n");
        let a = p.models[0].clone();
        let held = a.use_lock.clone().read_owned().await;
        assert!(a.use_lock.clone().try_write_owned().is_err(), "eviction needs the write lock");
        drop(held);
        assert!(a.use_lock.clone().try_write_owned().is_ok());
        assert_eq!(a.idle_unload_secs, 90);
        assert!(a.last_used_secs().is_none());
    }

    /// A pressure unload past the time limit cancels the requests that hold the model.
    #[tokio::test]
    async fn a_model_use_registers_the_request_token() {
        let p = pool_of("[[models]]
name = \"a\"
");
        let a = p.models[0].clone();
        let t = CancelToken::default();
        let guard = a.use_lock.clone().read_owned().await;
        lock(&a.users).push((7, t.clone()));
        let u = ModelUse { _guard: guard, model: a.clone(), id: 7 };
        a.active.fetch_add(1, Ordering::SeqCst);
        // Not pending: nothing happens.
        assert!(!p.check_pressure_wait(&a, true).await);
        a.pending_unload.store(true, Ordering::SeqCst);
        *lock(&a.pending_since) = Some(Instant::now() - Duration::from_secs(20));
        assert!(!p.check_pressure_wait(&a, false).await, "RAM is fine again: no cancel");
        assert!(!t.is_cancelled());
        assert!(p.check_pressure_wait(&a, true).await, "20 s > 15 s and RAM still low");
        assert_eq!(t.reason().map(|e| e.code), Some("memory_pressure"));
        assert!(!a.pending_unload.load(Ordering::SeqCst) && lock(&a.pending_since).is_none());
        drop(u);
        assert!(lock(&a.users).is_empty(), "the use leaves the list when it ends");
    }
}
