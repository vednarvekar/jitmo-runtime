//! Supervisor of the one llama-server child that holds the base model.
//!
//! - Starts it lazily on the first request, on a private 127.0.0.1 port, inside a Job Object
//!   (commit cap + kill-on-close, so it can never outlive the runtime).
//! - One generation at a time: callers take the single `Slot`; others wait in a bounded queue.
//! - A watchdog unloads the model when it is idle, when the system is short of memory, and
//!   notices when the child dies by itself.

use crate::budget;
use crate::config::{Config, ModelConfig};
use crate::error::Error;
use crate::llama::{self, HttpClient};
use crate::pacer::Pacer;
use crate::stats::{mb, Stats, UnloadReason};
use crate::win::{self, Job, LowMemorySignal, ProcessMemory, HANDLE};
use hyper::StatusCode;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{info, warn};

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

/// What `/stats` may read about the running child without touching the child itself.
#[derive(Clone)]
struct ChildInfo {
    pid: u32,
    port: u16,
    loaded_at: Instant,
    load_ms: f64,
    job: Arc<Job>,
}

#[derive(Debug, Clone, serde::Serialize)]
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

pub struct Supervisor {
    cfg: Arc<Config>,
    pub stats: Arc<Stats>,
    pub http: HttpClient,
    pub pacer: Pacer,
    threads: u32,
    adapters: Vec<PathBuf>,
    state: tokio::sync::Mutex<Option<Running>>,
    info: Mutex<Option<ChildInfo>>,
    phase: AtomicU8,
    gate: Arc<Semaphore>,
    waiting: AtomicUsize,
    last_activity: Mutex<Instant>,
    low_memory: Option<LowMemorySignal>,
}

/// The right to run generations. Held for one request (or a whole chain). Dropping it lets
/// the next queued request run.
pub struct Slot {
    _permit: OwnedSemaphorePermit,
    sup: Arc<Supervisor>,
    pub port: u16,
    pub queue_ms: f64,
    pub load_ms: f64,
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.sup.touch();
    }
}

/// Decrements the waiting count even when a queued request is cancelled.
struct Waiting<'a>(&'a AtomicUsize);
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Supervisor {
    pub fn new(cfg: Arc<Config>, stats: Arc<Stats>) -> Arc<Supervisor> {
        let threads = budget::threads(&cfg);
        let adapters = cfg.adapters();
        let low_memory = LowMemorySignal::new();
        if low_memory.is_none() {
            warn!("cannot create the Windows low-memory notification; using the free-RAM floor only");
        }
        Arc::new(Supervisor {
            cfg,
            stats,
            http: llama::client(),
            pacer: Pacer,
            threads,
            adapters,
            state: tokio::sync::Mutex::new(None),
            info: Mutex::new(None),
            phase: AtomicU8::new(Phase::Unloaded as u8),
            gate: Arc::new(Semaphore::new(1)),
            waiting: AtomicUsize::new(0),
            last_activity: Mutex::new(Instant::now()),
            low_memory,
        })
    }

    pub fn threads(&self) -> u32 {
        self.threads
    }

    pub fn adapters(&self) -> &[PathBuf] {
        &self.adapters
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

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    pub fn generating(&self) -> bool {
        self.gate.available_permits() == 0
    }

    pub fn low_memory_signal(&self) -> bool {
        self.low_memory.as_ref().is_some_and(|s| s.is_low())
    }

    fn touch(&self) {
        *self.last_activity.lock().unwrap_or_else(|p| p.into_inner()) = Instant::now();
    }

    pub fn idle_secs(&self) -> f64 {
        self.last_activity.lock().unwrap_or_else(|p| p.into_inner()).elapsed().as_secs_f64()
    }

    /// Wait for the generation slot (bounded queue), then make sure the model is loaded.
    pub async fn acquire(self: &Arc<Self>) -> Result<Slot, Error> {
        let t0 = Instant::now();
        let permit = match self.gate.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                let before = self.waiting.fetch_add(1, Ordering::SeqCst);
                let _waiting = Waiting(&self.waiting);
                if before >= self.cfg.server.max_queue {
                    self.stats.refused_queue();
                    return Err(Error::queue_full(format!(
                        "one generation is running and {before} requests are waiting (server.max_queue = {})",
                        self.cfg.server.max_queue
                    )));
                }
                self.gate.clone().acquire_owned().await.map_err(|_| Error::engine("runtime is shutting down"))?
            }
        };
        let queue_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.touch();
        let (port, load_ms) = self.ensure_loaded().await?;
        self.pacer.before_generation().await;
        Ok(Slot { _permit: permit, sup: self.clone(), port, queue_ms, load_ms })
    }

    /// Returns the child's port and how long loading took now (0 if it was already loaded).
    async fn ensure_loaded(&self) -> Result<(u16, f64), Error> {
        let mut st = self.state.lock().await;
        if let Some(run) = st.as_mut() {
            if matches!(run.child.try_wait(), Ok(None)) {
                return Ok((run.port, 0.0));
            }
            warn!("llama-server had exited by itself; starting it again");
            let run = st.take().expect("checked above");
            self.finish_stop(run, UnloadReason::Crashed).await;
        }
        if let Err(why) = budget::check_before_load(&self.cfg) {
            self.stats.refused_memory();
            warn!("{why}");
            return Err(Error::insufficient_memory(why));
        }
        self.set_phase(Phase::Loading);
        let t0 = Instant::now();
        match self.start_child().await {
            Ok(run) => {
                let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let port = run.port;
                *self.info.lock().unwrap_or_else(|p| p.into_inner()) = Some(ChildInfo {
                    pid: run.pid,
                    port,
                    loaded_at: Instant::now(),
                    load_ms,
                    job: run.job.clone(),
                });
                *st = Some(run);
                self.set_phase(Phase::Loaded);
                self.stats.loaded(load_ms);
                info!("model loaded in {:.0} ms (llama-server pid {}, port {port})", load_ms, st.as_ref().map_or(0, |r| r.pid));
                Ok((port, load_ms))
            }
            Err(e) => {
                self.set_phase(Phase::Unloaded);
                self.stats.load_failed();
                warn!("model load failed: {}", e.message);
                Err(e)
            }
        }
    }

    async fn start_child(&self) -> Result<Running, Error> {
        let m = &self.cfg.model;
        let port = if m.port != 0 { m.port } else { free_port().map_err(|e| Error::load_failed(format!("no free port: {e}")))? };
        let args = llama_args(m, port, self.threads, &self.adapters);
        info!("starting llama-server: {} {}", m.llama_server.display(), args.join(" "));

        let mut cmd = tokio::process::Command::new(&m.llama_server);
        cmd.args(&args).stdin(Stdio::null()).kill_on_drop(true);
        let (out, err) = log_stdio(&m.log_file);
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
        let cap = (self.cfg.budget.memory_cap_mb > 0).then(|| self.cfg.budget.memory_cap_mb * 1024 * 1024);
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
                let mut msg = format!("llama-server exited while loading ({status})");
                let tail = log_tail(&m.log_file, 8);
                if let Some(cap) = cap {
                    // One big allocation can fail while the peak is still well under the cap,
                    // so an allocation error in the log counts too.
                    let near_cap = run.job.peak_process_commit().is_some_and(|p| p as f64 >= cap as f64 * 0.97);
                    let lower = tail.to_lowercase();
                    let alloc_failed = lower.contains("alloc") || lower.contains("mem_buffer");
                    if near_cap || alloc_failed {
                        msg.push_str(&format!(
                            "; memory allocation failed under the memory cap of {} MB (budget.memory_cap_mb): raise the cap or lower model.ctx_size",
                            self.cfg.budget.memory_cap_mb
                        ));
                    }
                }
                if !tail.is_empty() {
                    msg.push_str(&format!(". Last lines of {}:\n{tail}", m.log_file.display()));
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
                    "the model was not ready after {} s (model.load_timeout_secs)",
                    m.load_timeout_secs
                )));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    /// Kill the child now, even mid-generation. Returns false if it was not loaded.
    pub async fn unload(&self, reason: UnloadReason) -> bool {
        let mut st = self.state.lock().await;
        match st.take() {
            Some(run) => {
                self.finish_stop(run, reason).await;
                true
            }
            None => false,
        }
    }

    /// Wait for the running generation (and the queue ahead) to finish, then unload.
    pub async fn unload_when_free(&self, reason: UnloadReason) -> bool {
        let _permit = self.gate.clone().acquire_owned().await;
        self.unload(reason).await
    }

    async fn finish_stop(&self, mut run: Running, reason: UnloadReason) {
        let mem = win::process_memory_by_pid(run.pid).unwrap_or_default();
        run.job.terminate();
        let _ = run.child.start_kill();
        let _ = run.child.wait().await;
        self.record_peaks(&mem, &run.job);
        *self.info.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self.set_phase(Phase::Unloaded);
        self.stats.unloaded(reason);
        info!(
            "model unloaded ({}); child peak working set {:.0} MB, peak private {:.0} MB",
            reason.as_str(),
            mb(mem.peak_working_set),
            mb(mem.peak_private)
        );
    }

    fn record_peaks(&self, mem: &ProcessMemory, job: &Job) {
        let job_peak = job.peak_process_commit().unwrap_or(0);
        self.stats.child_peaks(mem.peak_working_set, mem.peak_private, job_peak);
    }

    /// Live memory of the child, if it is loaded.
    pub fn child_snapshot(&self) -> Option<ChildSnapshot> {
        let info = self.info.lock().unwrap_or_else(|p| p.into_inner()).clone()?;
        let mem = win::process_memory_by_pid(info.pid).unwrap_or_default();
        self.record_peaks(&mem, &info.job);
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

    pub fn spawn_watchdog(self: &Arc<Self>) {
        let sup = self.clone();
        let every = Duration::from_millis(self.cfg.budget.watch_interval_ms);
        tokio::spawn(async move {
            let mut low_count = 0u32;
            let mut tick = tokio::time::interval(every);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                sup.watch_once(&mut low_count).await;
            }
        });
    }

    async fn watch_once(&self, low_count: &mut u32) {
        // 1. The child died by itself (crash, or it hit the memory cap).
        if let Ok(mut st) = self.state.try_lock() {
            let exited = st.as_mut().and_then(|r| r.child.try_wait().ok().flatten());
            if let Some(status) = exited {
                let run = st.take().expect("child present");
                let near_cap = self.cfg.budget.memory_cap_mb > 0
                    && run.job.peak_process_commit().is_some_and(|p| p as f64 >= (self.cfg.budget.memory_cap_mb * 1024 * 1024) as f64 * 0.97);
                warn!(
                    "llama-server exited by itself ({status}){}; the model is unloaded",
                    if near_cap { ", after reaching the memory cap" } else { "" }
                );
                self.finish_stop(run, UnloadReason::Crashed).await;
            }
        }
        if self.phase() != Phase::Loaded {
            *low_count = 0;
            return;
        }
        // 2. Memory pressure: two checks in a row, so a short dip does not unload.
        match budget::pressure(&self.cfg, self.low_memory.as_ref()) {
            Some(why) => {
                *low_count += 1;
                if *low_count >= 2 {
                    warn!("memory pressure ({why}): unloading the model now");
                    self.unload(UnloadReason::MemoryPressure).await;
                    *low_count = 0;
                    return;
                }
            }
            None => *low_count = 0,
        }
        // 3. Idle.
        let idle = self.cfg.budget.idle_unload_secs;
        if idle > 0 && self.idle_secs() > idle as f64 {
            if let Ok(_permit) = self.gate.clone().try_acquire_owned() {
                if self.idle_secs() > idle as f64 {
                    info!("no requests for {idle} s: unloading the model");
                    self.unload(UnloadReason::Idle).await;
                }
            }
        }
    }

    pub async fn shutdown(&self) {
        self.unload(UnloadReason::Shutdown).await;
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
    for path in adapters {
        a.push("--lora".into());
        a.push(path.display().to_string());
    }
    if !adapters.is_empty() {
        // Adapters are loaded once with scale 0; each chain step sets its own scale per request.
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
        assert!(s.contains("--lora a.gguf --lora b.gguf --lora-init-without-apply"));
        assert!(s.ends_with("--cache-reuse 256"));
        let plain = llama_args(&ModelConfig { no_repack: false, ..Default::default() }, 1, 1, &[]).join(" ");
        assert!(!plain.contains("--lora") && !plain.contains("-lm none") && !plain.contains("--no-repack"));
    }
}
