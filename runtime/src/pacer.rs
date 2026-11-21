//! Pacer v1, "don't freeze the PC": while the user is busy, hard-cap the llama-server child's
//! CPU through its Job Object; when the user is idle again, remove the cap.
//!
//! Why this lever (see docs/results/03-pacer-v1.md):
//! - llama-server has no per-request thread count, and restarting it with fewer threads costs
//!   a model load.
//! - llama-server has no per-request throttle. Reading its token stream more slowly does not
//!   slow its compute: the slot keeps generating into a queue.
//! - A Job Object CPU rate hard cap (`JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP`) can be set and
//!   removed at any time on the running child, with no restart.
//!
//! Signals, sampled every `pacer.poll_ms`: time since the last keyboard/mouse input
//! (GetLastInputInfo), CPU used by other processes (system busy time minus the child's job
//! CPU time), and free RAM. Any one over its threshold = busy.

use crate::config::PacerConfig;
use crate::stats::{round2, RequestTiming};
use crate::win::{self, Job, SystemTimes};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// One sample of the "is the user busy" signals.
#[derive(Debug, Clone, Default)]
pub struct Signals {
    pub input_idle_ms: Option<u64>,
    pub others_cpu_pct: Option<f64>,
    pub free_mb: Option<u64>,
}

/// Why the user counts as busy now, or None.
pub fn busy_reason(cfg: &PacerConfig, s: &Signals) -> Option<String> {
    if let Some(ms) = s.input_idle_ms {
        if cfg.input_active_ms > 0 && ms < cfg.input_active_ms {
            return Some(format!("user input {ms} ms ago"));
        }
    }
    if let Some(pct) = s.others_cpu_pct {
        if cfg.others_cpu_busy_pct > 0.0 && pct > cfg.others_cpu_busy_pct {
            return Some(format!("other processes use {pct:.0}% CPU"));
        }
    }
    if let Some(free) = s.free_mb {
        if cfg.low_free_mb > 0 && free < cfg.low_free_mb {
            return Some(format!("{free} MB RAM free"));
        }
    }
    None
}

/// Should the cap be on? `since_busy` = time since the last busy sample (None = never busy).
pub fn want_cap(mode: &str, since_busy: Option<Duration>, release_after: Duration) -> bool {
    match mode {
        "always" => true,
        "off" => false,
        _ => since_busy.is_some_and(|d| d <= release_after),
    }
}

/// The Job Object CpuRate (cycles per 10000 of all logical processors) that gives the child
/// `busy_pct` % of the CPU it uses at full speed with `threads` threads.
pub fn cpu_rate(busy_pct: u32, threads: u32, logical: u32) -> u32 {
    let full = threads.min(logical) as f64 / logical.max(1) as f64;
    ((busy_pct as f64 / 100.0) * full * 10_000.0).round().clamp(1.0, 10_000.0) as u32
}

/// CPU used by processes other than the child, in % of all logical processors, between two samples.
pub fn others_cpu_pct(prev: SystemTimes, now: SystemTimes, child_prev: u64, child_now: u64) -> Option<f64> {
    let total = now.total.checked_sub(prev.total)?;
    let idle = now.idle.checked_sub(prev.idle)?;
    if total == 0 {
        return None;
    }
    let busy = total.saturating_sub(idle);
    let child = child_now.saturating_sub(child_prev);
    Some(busy.saturating_sub(child) as f64 * 100.0 / total as f64)
}

#[derive(Debug, Clone, Serialize)]
pub struct PacerSnapshot {
    pub mode: String,
    /// `no_model`, `idle` (no cap) or `busy` (capped).
    pub state: &'static str,
    pub capped: bool,
    /// Why the cap is on (the last busy reason).
    pub reason: Option<String>,
    /// The hard cap while capped, in % of all logical processors.
    pub cap_pct_of_system: f64,
    pub busy_cpu_pct: u32,
    pub busy_priority: String,
    pub input_idle_ms: Option<u64>,
    pub others_cpu_pct: Option<f64>,
    pub free_mb: Option<u64>,
    /// Time with the cap on, over the runtime's life.
    pub capped_secs_total: f64,
    /// How many times the cap was switched on.
    pub cap_on_count: u64,
    pub last_error: Option<String>,
    /// Generations that ended while capped / uncapped, and their last tokens/s.
    pub generations_capped: u64,
    pub generations_uncapped: u64,
    pub last_tokens_per_sec_capped: Option<f64>,
    pub last_tokens_per_sec_uncapped: Option<f64>,
}

#[derive(Default)]
struct Inner {
    prev_sys: Option<SystemTimes>,
    prev_child: Option<(u32, u64)>,
    last_busy_at: Option<Instant>,
    reason: Option<String>,
    signals: Signals,
    applied_pid: Option<u32>,
    /// A CPU rate cap is set on the current child's job (removing one that was never set fails).
    rate_set: bool,
    capped: bool,
    capped_since: Option<Instant>,
    capped_total: Duration,
    cap_on_count: u64,
    last_error: Option<String>,
    gens_capped: u64,
    gens_uncapped: u64,
    tps_capped: Option<f64>,
    tps_uncapped: Option<f64>,
}

pub struct Pacer {
    cfg: PacerConfig,
    rate: u32,
    base_priority: u32,
    inner: Mutex<Inner>,
}

impl Pacer {
    pub fn new(cfg: PacerConfig, threads: u32, logical: u32, low_priority: bool) -> Pacer {
        let rate = cpu_rate(cfg.busy_cpu_pct, threads, logical);
        let base_priority = if low_priority { win::BELOW_NORMAL_PRIORITY_CLASS } else { win::NORMAL_PRIORITY_CLASS };
        Pacer { cfg, rate, base_priority, inner: Mutex::new(Inner::default()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.cfg.poll_ms)
    }

    /// Called after the slot is taken and the model is loaded. Nothing to wait for in v1:
    /// the cap acts on the running child.
    pub async fn before_generation(&self) {}

    pub fn after_generation(&self, t: &RequestTiming) {
        let mut i = self.lock();
        if i.capped {
            i.gens_capped += 1;
            i.tps_capped = t.tokens_per_sec.or(i.tps_capped);
        } else {
            i.gens_uncapped += 1;
            i.tps_uncapped = t.tokens_per_sec.or(i.tps_uncapped);
        }
    }

    /// One sample and decision. `child` = the running llama-server (pid, job), if loaded.
    pub fn tick(&self, child: Option<(u32, Arc<Job>)>) {
        let now = Instant::now();
        let sys = win::system_times();
        let child_cpu = child.as_ref().and_then(|(pid, job)| job.cpu_time().map(|t| (*pid, t)));
        let mut i = self.lock();

        let others = match (i.prev_sys, sys, i.prev_child, child_cpu) {
            (Some(p), Some(n), Some((pp, pc)), Some((np, nc))) if pp == np => others_cpu_pct(p, n, pc, nc),
            (Some(p), Some(n), None, None) => others_cpu_pct(p, n, 0, 0),
            _ => None,
        };
        i.prev_sys = sys;
        i.prev_child = child_cpu;
        i.signals = Signals {
            input_idle_ms: win::input_idle_ms(),
            others_cpu_pct: others.map(|p| (p * 10.0).round() / 10.0),
            free_mb: win::memory_status().map(|m| m.avail / (1024 * 1024)),
        };
        if let Some(why) = busy_reason(&self.cfg, &i.signals) {
            i.last_busy_at = Some(now);
            i.reason = Some(why);
        }
        let want = want_cap(&self.cfg.mode, i.last_busy_at.map(|t| now - t), Duration::from_millis(self.cfg.release_after_ms))
            && child.is_some();

        let Some((pid, job)) = child else {
            // No child: nothing to cap. A new child starts uncapped (fresh job).
            self.set_capped(&mut i, false, now);
            i.applied_pid = None;
            i.rate_set = false;
            return;
        };
        if i.applied_pid != Some(pid) {
            i.rate_set = false; // a new child has a new job, uncapped
        }
        if i.applied_pid == Some(pid) && i.capped == want {
            return;
        }
        let mut err = None;
        if self.cfg.busy_cpu_pct < 100 && want != i.rate_set {
            match job.set_cpu_rate(want.then_some(self.rate)) {
                Ok(()) => i.rate_set = want,
                Err(e) => err = Some(format!("cannot {} the CPU cap: {e}", if want { "set" } else { "remove" })),
            }
        }
        if self.cfg.busy_priority == "idle" {
            let class = if want { win::IDLE_PRIORITY_CLASS } else { self.base_priority };
            if !win::set_priority_class(pid, class) {
                err = Some("cannot change the priority class".into());
            }
        }
        if let Some(e) = err {
            // Not applied: report it once and try again on the next sample.
            if i.last_error.as_ref() != Some(&e) {
                warn!("pacer: {e}");
            }
            i.last_error = Some(e);
            return;
        }
        if want != i.capped {
            if want {
                info!(
                    "pacer: capping llama-server to {:.1}% of the CPU ({})",
                    self.rate as f64 / 100.0,
                    i.reason.as_deref().unwrap_or(&self.cfg.mode)
                );
            } else {
                info!("pacer: user idle, cap removed");
            }
        }
        i.last_error = None;
        i.applied_pid = Some(pid);
        self.set_capped(&mut i, want, now);
    }

    fn set_capped(&self, i: &mut Inner, on: bool, now: Instant) {
        if on && !i.capped {
            i.cap_on_count += 1;
            i.capped_since = Some(now);
        }
        if !on {
            if let Some(t) = i.capped_since.take() {
                i.capped_total += now - t;
            }
        }
        i.capped = on;
    }

    pub fn capped(&self) -> bool {
        self.lock().capped
    }

    pub fn snapshot(&self, loaded: bool) -> PacerSnapshot {
        let i = self.lock();
        let total = i.capped_total + i.capped_since.map_or(Duration::ZERO, |t| t.elapsed());
        PacerSnapshot {
            mode: self.cfg.mode.clone(),
            state: if !loaded {
                "no_model"
            } else if i.capped {
                "busy"
            } else {
                "idle"
            },
            capped: i.capped,
            reason: if i.capped { i.reason.clone() } else { None },
            cap_pct_of_system: if self.cfg.busy_cpu_pct < 100 { self.rate as f64 / 100.0 } else { 100.0 },
            busy_cpu_pct: self.cfg.busy_cpu_pct,
            busy_priority: self.cfg.busy_priority.clone(),
            input_idle_ms: i.signals.input_idle_ms,
            others_cpu_pct: i.signals.others_cpu_pct,
            free_mb: i.signals.free_mb,
            capped_secs_total: round2(total.as_secs_f64()),
            cap_on_count: i.cap_on_count,
            last_error: i.last_error.clone(),
            generations_capped: i.gens_capped,
            generations_uncapped: i.gens_uncapped,
            last_tokens_per_sec_capped: i.tps_capped,
            last_tokens_per_sec_uncapped: i.tps_uncapped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_is_a_share_of_the_childs_own_cpu() {
        assert_eq!(cpu_rate(50, 3, 8), 1875, "3 of 8 threads at 50% = 18.75% of the system");
        assert_eq!(cpu_rate(100, 3, 8), 3750);
        assert_eq!(cpu_rate(50, 16, 8), 5000, "threads above the logical count are clamped");
        assert_eq!(cpu_rate(1, 1, 64), 2, "never 0");
    }

    #[test]
    fn busy_comes_from_input_cpu_or_ram() {
        let cfg = PacerConfig::default(); // input 5000 ms, cpu 10%, ram 2048 MB
        let idle = Signals { input_idle_ms: Some(60_000), others_cpu_pct: Some(3.0), free_mb: Some(6000) };
        assert_eq!(busy_reason(&cfg, &idle), None);
        let typing = Signals { input_idle_ms: Some(200), ..idle.clone() };
        assert!(busy_reason(&cfg, &typing).unwrap().contains("input"));
        let cpu = Signals { others_cpu_pct: Some(14.0), ..idle.clone() };
        assert!(busy_reason(&cfg, &cpu).unwrap().contains("CPU"));
        let ram = Signals { free_mb: Some(1500), ..idle.clone() };
        assert!(busy_reason(&cfg, &ram).unwrap().contains("RAM"));
        let unknown = Signals::default();
        assert_eq!(busy_reason(&cfg, &unknown), None, "missing signals never count as busy");
        let off = PacerConfig { input_active_ms: 0, others_cpu_busy_pct: 0.0, low_free_mb: 0, ..cfg };
        assert_eq!(busy_reason(&off, &Signals { input_idle_ms: Some(0), others_cpu_pct: Some(99.0), free_mb: Some(1) }), None);
    }

    #[test]
    fn the_cap_holds_for_release_after_then_lifts() {
        let r = Duration::from_millis(2000);
        assert!(!want_cap("auto", None, r));
        assert!(want_cap("auto", Some(Duration::ZERO), r));
        assert!(want_cap("auto", Some(Duration::from_millis(1999)), r));
        assert!(!want_cap("auto", Some(Duration::from_millis(2500)), r));
        assert!(want_cap("always", None, r));
        assert!(!want_cap("off", Some(Duration::ZERO), r));
    }

    #[test]
    fn others_cpu_excludes_the_child() {
        let p = SystemTimes { idle: 0, total: 0 };
        // 8 CPUs x 1 s = 80_000_000 units; 50% busy, of which the child used 30%.
        let n = SystemTimes { idle: 40_000_000, total: 80_000_000 };
        let pct = others_cpu_pct(p, n, 0, 24_000_000).unwrap();
        assert!((pct - 20.0).abs() < 1e-9);
        assert_eq!(others_cpu_pct(n, n, 0, 0), None);
    }

    #[test]
    fn tick_caps_a_real_job_and_lifts_the_cap() {
        let job = Arc::new(Job::new(None).unwrap());
        let always = Pacer::new(PacerConfig { mode: "always".into(), ..Default::default() }, 3, 8, true);
        always.tick(Some((1, job.clone())));
        assert!(always.capped());
        assert_eq!(always.snapshot(true).state, "busy");
        assert_eq!(always.snapshot(true).cap_on_count, 1);
        always.tick(None);
        assert!(!always.capped());
        assert_eq!(always.snapshot(false).state, "no_model");
        let off = Pacer::new(PacerConfig { mode: "off".into(), ..Default::default() }, 3, 8, true);
        off.tick(Some((1, job)));
        assert!(!off.capped());
        assert!(off.snapshot(true).last_error.is_none());
    }
}
