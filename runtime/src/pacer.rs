//! Pacer v2, "don't freeze the PC": when other programs need the CPU or RAM is short, hard-cap
//! the llama-server children's CPU through their Job Objects, at a level that fits how busy
//! the PC is; lift the cap when the PC is quiet again.
//!
//! Why this lever (see docs/results/03-pacer-v1.md):
//! - llama-server has no per-request thread count, and restarting it with fewer threads costs
//!   a model load.
//! - llama-server has no per-request throttle. Reading its token stream more slowly does not
//!   slow its compute: the slot keeps generating into a queue.
//! - A Job Object CPU rate hard cap (`JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP`) can be set,
//!   changed and removed at any time on a running child, with no restart.
//!
//! v2 (owner decision, 22 Nov 2026): no keyboard/mouse signal. Signals, sampled every
//! `poll_ms`:
//! - CPU used by other processes (system busy time minus all children's job CPU time),
//!   averaged over the last `smooth_ms` so one short spike does not count;
//! - free RAM.
//!
//! Levels (all thresholds configurable):
//!
//! | Level   | When                                                  | Cap (share of the child's own full-speed CPU) |
//! |---------|-------------------------------------------------------|------|
//! | `none`  | others < `light_cpu_pct` (15%)                        | none |
//! | `light` | others 15 to 40%                                      | `light_cap_pct` (75%) |
//! | `heavy` | others > `heavy_cpu_pct` (40%), or free RAM < `low_free_mb` | `heavy_cap_pct` (50%) |
//!
//! Hysteresis: a level goes up at once; it goes down only when the signals are below the
//! thresholds by `hysteresis_pct` (RAM: 256 MB above `low_free_mb`) and the current level has
//! held for `release_after_ms`.

use crate::config::PacerConfig;
use crate::stats::{round2, RequestTiming};
use crate::win::{self, Job, SystemTimes};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Free RAM must rise this far above `low_free_mb` before the RAM signal lets go.
const RAM_HYSTERESIS_MB: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    #[default]
    None = 0,
    Light = 1,
    Heavy = 2,
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Level::None => "none",
            Level::Light => "light",
            Level::Heavy => "heavy",
        }
    }
}

/// One sample of the signals. To add a signal (planned: a tiny RAM-latency probe that notices
/// memory-bandwidth contention, which CPU % does not show), add a field here, fill it in
/// `Pacer::tick`, and add its rule to `proposals`.
#[derive(Debug, Clone, Default)]
pub struct Signals {
    /// CPU used by processes other than the children, % of all logical processors, averaged over `smooth_ms`.
    pub others_cpu_pct: Option<f64>,
    pub free_mb: Option<u64>,
}

/// The pacer settings with `low_free_mb` resolved.
#[derive(Debug, Clone)]
pub struct Thresholds {
    pub light_cpu_pct: f64,
    pub heavy_cpu_pct: f64,
    pub low_free_mb: u64,
    pub hysteresis_pct: f64,
}

/// What each signal asks for. `margin` lowers the thresholds (used to decide whether a level may
/// go down), so a signal sitting just under a threshold does not make the level flap.
fn proposals(t: &Thresholds, s: &Signals, lowering: bool) -> Vec<(Level, String)> {
    let (cpu_margin, ram_margin) = if lowering { (t.hysteresis_pct, RAM_HYSTERESIS_MB) } else { (0.0, 0) };
    let mut out = Vec::new();
    if let Some(pct) = s.others_cpu_pct {
        if t.heavy_cpu_pct > 0.0 && pct > t.heavy_cpu_pct - cpu_margin {
            out.push((Level::Heavy, format!("other processes use {pct:.0}% CPU (> {:.0}%)", t.heavy_cpu_pct)));
        } else if t.light_cpu_pct > 0.0 && pct > t.light_cpu_pct - cpu_margin {
            out.push((Level::Light, format!("other processes use {pct:.0}% CPU (> {:.0}%)", t.light_cpu_pct)));
        }
    }
    if let Some(free) = s.free_mb {
        if t.low_free_mb > 0 && free < t.low_free_mb + ram_margin {
            out.push((Level::Heavy, format!("{free} MB RAM free (< {} MB)", t.low_free_mb)));
        }
    }
    out
}

fn highest(p: Vec<(Level, String)>) -> (Level, Option<String>) {
    p.into_iter().max_by_key(|(l, _)| *l).map_or((Level::None, None), |(l, r)| (l, Some(r)))
}

/// The next level: up at once to what the signals ask for; down only when the signals ask
/// for less even with the hysteresis margin, and the current level has held `release_after`.
pub fn next_level(t: &Thresholds, current: Level, s: &Signals, held: Duration, release_after: Duration) -> (Level, Option<String>) {
    let (up, why_up) = highest(proposals(t, s, false));
    if up >= current {
        return (up, why_up);
    }
    let (down, why_down) = highest(proposals(t, s, true));
    if down < current && held >= release_after {
        return (down, why_down);
    }
    (current, why_down.or(why_up))
}

/// The Job Object CpuRate (cycles per 10000 of all logical processors) that gives a child
/// `cap_pct` % of the CPU it uses at full speed with `threads` threads.
pub fn cpu_rate(cap_pct: u32, threads: u32, logical: u32) -> u32 {
    let full = threads.min(logical) as f64 / logical.max(1) as f64;
    ((cap_pct as f64 / 100.0) * full * 10_000.0).round().clamp(1.0, 10_000.0) as u32
}

/// CPU used by processes other than the children, in % of all logical processors, between two samples.
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
    /// `none`, `light` or `heavy`.
    pub level: Level,
    /// Why this level (the signal that set it).
    pub reason: Option<String>,
    pub capped: bool,
    /// The cap as a share of the child's own full-speed CPU (100 = no cap).
    pub cap_pct: u32,
    /// The same cap in % of all logical processors, for the default model's thread count.
    pub cap_pct_of_system: f64,
    pub others_cpu_pct: Option<f64>,
    pub free_mb: Option<u64>,
    /// Planned signal (memory-bandwidth probe); always null for now.
    pub mem_latency_ns: Option<f64>,
    pub thresholds: Value,
    pub busy_priority: String,
    /// Seconds spent at each level, over the runtime's life (while a model was loaded).
    pub secs_at_level: Value,
    pub level_changes: u64,
    /// Time with any cap on, and how often a cap was switched on (from none).
    pub capped_secs_total: f64,
    pub cap_on_count: u64,
    pub last_error: Option<String>,
    /// Generations that ended while capped / uncapped, and their last tokens/s.
    pub generations_capped: u64,
    pub generations_uncapped: u64,
    pub last_tokens_per_sec_capped: Option<f64>,
    pub last_tokens_per_sec_uncapped: Option<f64>,
}

use serde_json::{json, Value};

/// One loaded llama-server child, as the pacer sees it.
#[derive(Clone)]
pub struct Child {
    pub pid: u32,
    pub job: Arc<Job>,
    pub threads: u32,
}

struct Sample {
    at: Instant,
    sys: SystemTimes,
    children: HashMap<u32, u64>,
}

#[derive(Default)]
struct Inner {
    window: VecDeque<Sample>,
    signals: Signals,
    level: Level,
    reason: Option<String>,
    level_since: Option<Instant>,
    at_level: [Duration; 3],
    last_tick: Option<Instant>,
    level_changes: u64,
    /// Per child: the CPU rate set on its job now (None = no cap).
    applied: HashMap<u32, Option<u32>>,
    capped_since: Option<Instant>,
    capped_total: Duration,
    cap_on_count: u64,
    last_error: Option<String>,
    gens_capped: u64,
    gens_uncapped: u64,
    tps_capped: Option<f64>,
    tps_uncapped: Option<f64>,
}

/// One pacer for the whole pool: the signals are sampled once, "other processes' CPU" leaves
/// out every child, and the level's cap is set on every loaded child.
pub struct Pacer {
    cfg: PacerConfig,
    t: Thresholds,
    logical: u32,
    base_priority: u32,
    inner: Mutex<Inner>,
}

impl Pacer {
    /// `pressure_free_mb` is used when `pacer.low_free_mb` is 0.
    pub fn new(cfg: PacerConfig, logical: u32, low_priority: bool, pressure_free_mb: u64) -> Pacer {
        for (key, set) in [
            ("input_active_ms", cfg.input_active_ms.is_some()),
            ("others_cpu_busy_pct", cfg.others_cpu_busy_pct.is_some()),
            ("busy_cpu_pct", cfg.busy_cpu_pct.is_some()),
        ] {
            if set {
                warn!("pacer.{key} is from pacer v1 and is ignored (see docs/06-runtime-api.md, Pacer)");
            }
        }
        let base_priority = if low_priority { win::BELOW_NORMAL_PRIORITY_CLASS } else { win::NORMAL_PRIORITY_CLASS };
        let t = Thresholds {
            light_cpu_pct: cfg.light_cpu_pct,
            heavy_cpu_pct: cfg.heavy_cpu_pct,
            low_free_mb: if cfg.low_free_mb > 0 { cfg.low_free_mb } else { pressure_free_mb },
            hysteresis_pct: cfg.hysteresis_pct,
        };
        Pacer { cfg, t, logical, base_priority, inner: Mutex::new(Inner::default()) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.cfg.poll_ms)
    }

    /// The cap of a level, as a share of the child's own CPU (100 = none).
    pub fn cap_pct(&self, level: Level) -> u32 {
        match level {
            Level::None => 100,
            Level::Light => self.cfg.light_cap_pct,
            Level::Heavy => self.cfg.heavy_cap_pct,
        }
    }

    /// The hard cap for a child with `threads` threads at `level`, in CpuRate units; None = no cap.
    pub fn rate_for(&self, level: Level, threads: u32) -> Option<u32> {
        let pct = self.cap_pct(level);
        (pct < 100).then(|| cpu_rate(pct, threads, self.logical))
    }

    pub fn after_generation(&self, t: &RequestTiming) {
        let mut i = self.lock();
        if i.level != Level::None {
            i.gens_capped += 1;
            i.tps_capped = t.tokens_per_sec.or(i.tps_capped);
        } else {
            i.gens_uncapped += 1;
            i.tps_uncapped = t.tokens_per_sec.or(i.tps_uncapped);
        }
    }

    /// Average others' CPU over the window; resets the window when the set of children changes
    /// (a new child's CPU before its first sample is not known per interval).
    fn smoothed_others(&self, i: &mut Inner, now: Sample) -> Option<f64> {
        let same = i.window.back().is_some_and(|b| {
            b.children.len() == now.children.len() && now.children.keys().all(|k| b.children.contains_key(k))
        });
        if !same {
            i.window.clear();
        }
        i.window.push_back(now);
        let newest = i.window.back().expect("just pushed").at;
        let smooth = Duration::from_millis(self.cfg.smooth_ms);
        // Keep the newest sample that is at least `smooth` old as the window start.
        while i.window.len() > 2 && newest.duration_since(i.window[1].at) >= smooth {
            i.window.pop_front();
        }
        let (old, new) = (i.window.front()?, i.window.back()?);
        if i.window.len() < 2 {
            return None;
        }
        let child_old: u64 = old.children.values().sum();
        let child_new: u64 = new.children.values().sum();
        others_cpu_pct(old.sys, new.sys, child_old, child_new)
    }

    /// One sample and decision for all loaded children.
    pub fn tick(&self, children: &[Child]) {
        let now = Instant::now();
        let sys = win::system_times();
        let cpu: HashMap<u32, u64> = children.iter().filter_map(|c| c.job.cpu_time().map(|t| (c.pid, t))).collect();
        let mut i = self.lock();
        if let Some(last) = i.last_tick.replace(now) {
            if !children.is_empty() {
                let k = i.level as usize;
                i.at_level[k] += now - last;
            }
        }
        let others = match sys {
            Some(sys) if cpu.len() == children.len() => self.smoothed_others(&mut i, Sample { at: now, sys, children: cpu }),
            _ => {
                i.window.clear();
                None
            }
        };
        i.signals = Signals {
            others_cpu_pct: others.map(|p| (p * 10.0).round() / 10.0),
            free_mb: win::memory_status().map(|m| m.avail / (1024 * 1024)),
        };

        let held = i.level_since.map_or(Duration::MAX, |t| now - t);
        let release = Duration::from_millis(self.cfg.release_after_ms);
        let (want, reason) = match self.cfg.mode.as_str() {
            "off" => (Level::None, None),
            "always" => (Level::Heavy, Some("pacer.mode = always".to_string())),
            _ if i.signals.others_cpu_pct.is_none() => (i.level, i.reason.clone()), // no CPU reading this tick: keep
            _ => next_level(&self.t, i.level, &i.signals, held, release),
        };
        let want = if children.is_empty() { Level::None } else { want };

        // Forget children that are gone (a new child has a new, uncapped job).
        i.applied.retain(|pid, _| children.iter().any(|c| c.pid == *pid));
        let mut err = None;
        for c in children {
            let applied = i.applied.get(&c.pid).copied().flatten();
            let rate = self.rate_for(want, c.threads);
            if applied != rate {
                // Removing a cap that was never set fails, so only remove one that is set.
                match c.job.set_cpu_rate(rate) {
                    Ok(()) => {
                        i.applied.insert(c.pid, rate);
                    }
                    Err(e) => err = Some(format!("cannot set the CPU cap to {rate:?}: {e}")),
                }
            } else {
                i.applied.entry(c.pid).or_insert(None);
            }
            if self.cfg.busy_priority == "idle" && want != i.level {
                let class = if want != Level::None { win::IDLE_PRIORITY_CLASS } else { self.base_priority };
                if !win::set_priority_class(c.pid, class) {
                    err = Some("cannot change the priority class".into());
                }
            }
        }
        if let Some(e) = err {
            // Not applied everywhere: report it once and try again on the next sample.
            if i.last_error.as_ref() != Some(&e) {
                warn!("pacer: {e}");
            }
            i.last_error = Some(e);
            return;
        }
        i.last_error = None;
        if want != i.level {
            if !children.is_empty() {
                info!(
                    "pacer: level {} -> {} (cap {}% of the model's CPU){}",
                    i.level.as_str(),
                    want.as_str(),
                    self.cap_pct(want),
                    reason.as_deref().map(|r| format!(": {r}")).unwrap_or_default()
                );
            }
            if i.level == Level::None && want != Level::None {
                i.cap_on_count += 1;
                i.capped_since = Some(now);
            }
            if want == Level::None {
                if let Some(t) = i.capped_since.take() {
                    i.capped_total += now - t;
                }
            }
            if !children.is_empty() {
                i.level_changes += 1;
            }
            i.level = want;
            i.level_since = Some(now);
        }
        i.reason = if want == Level::None { None } else { reason };
    }

    pub fn capped(&self) -> bool {
        self.lock().level != Level::None
    }

    #[cfg(test)]
    pub fn level(&self) -> Level {
        self.lock().level
    }

    /// `threads` = the threads of the model whose cap `cap_pct_of_system` shows.
    pub fn snapshot(&self, loaded: bool, threads: u32) -> PacerSnapshot {
        let i = self.lock();
        let total = i.capped_total + i.capped_since.map_or(Duration::ZERO, |t| t.elapsed());
        let cap = self.cap_pct(i.level);
        PacerSnapshot {
            mode: self.cfg.mode.clone(),
            state: if !loaded {
                "no_model"
            } else if i.level != Level::None {
                "busy"
            } else {
                "idle"
            },
            level: i.level,
            reason: i.reason.clone(),
            capped: i.level != Level::None,
            cap_pct: cap,
            cap_pct_of_system: self.rate_for(i.level, threads).map_or(100.0, |r| r as f64 / 100.0),
            others_cpu_pct: i.signals.others_cpu_pct,
            free_mb: i.signals.free_mb,
            mem_latency_ns: None,
            thresholds: json!({
                "light_cpu_pct": self.t.light_cpu_pct, "heavy_cpu_pct": self.t.heavy_cpu_pct, "low_free_mb": self.t.low_free_mb,
                "light_cap_pct": self.cfg.light_cap_pct, "heavy_cap_pct": self.cfg.heavy_cap_pct,
                "hysteresis_pct": self.t.hysteresis_pct, "smooth_ms": self.cfg.smooth_ms, "release_after_ms": self.cfg.release_after_ms,
            }),
            busy_priority: self.cfg.busy_priority.clone(),
            secs_at_level: json!({
                "none": round2(i.at_level[0].as_secs_f64()), "light": round2(i.at_level[1].as_secs_f64()), "heavy": round2(i.at_level[2].as_secs_f64()),
            }),
            level_changes: i.level_changes,
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

    fn t() -> Thresholds {
        Thresholds { light_cpu_pct: 15.0, heavy_cpu_pct: 40.0, low_free_mb: 1024, hysteresis_pct: 5.0 }
    }
    fn s(cpu: f64, free: u64) -> Signals {
        Signals { others_cpu_pct: Some(cpu), free_mb: Some(free) }
    }
    const HELD: Duration = Duration::from_secs(10);
    const RELEASE: Duration = Duration::from_millis(2000);

    #[test]
    fn the_cap_is_a_share_of_the_childs_own_cpu() {
        assert_eq!(cpu_rate(50, 3, 8), 1875, "3 of 8 threads at 50% = 18.75% of the system");
        assert_eq!(cpu_rate(75, 3, 8), 2813);
        assert_eq!(cpu_rate(100, 3, 8), 3750);
        assert_eq!(cpu_rate(50, 16, 8), 5000, "threads above the logical count are clamped");
        assert_eq!(cpu_rate(1, 1, 64), 2, "never 0");
    }

    #[test]
    fn levels_follow_cpu_and_ram() {
        let t = t();
        let n = |c, f| next_level(&t, Level::None, &s(c, f), HELD, RELEASE).0;
        assert_eq!(n(5.0, 6000), Level::None);
        assert_eq!(n(15.0, 6000), Level::None, "15% is not above 15%");
        assert_eq!(n(20.0, 6000), Level::Light);
        assert_eq!(n(40.0, 6000), Level::Light);
        assert_eq!(n(41.0, 6000), Level::Heavy);
        assert_eq!(n(5.0, 900), Level::Heavy, "low RAM alone is heavy");
        let (_, why) = next_level(&t, Level::None, &s(25.0, 900), HELD, RELEASE);
        assert!(why.unwrap().contains("RAM"), "the highest level's reason wins");
        assert_eq!(next_level(&t, Level::None, &Signals::default(), HELD, RELEASE).0, Level::None, "no signals = none");
    }

    #[test]
    fn hysteresis_and_hold_stop_flapping() {
        let t = t();
        // Up at once, even right after a change.
        assert_eq!(next_level(&t, Level::None, &s(50.0, 6000), Duration::ZERO, RELEASE).0, Level::Heavy);
        // Down only below threshold - 5 ...
        assert_eq!(next_level(&t, Level::Heavy, &s(37.0, 6000), HELD, RELEASE).0, Level::Heavy);
        assert_eq!(next_level(&t, Level::Heavy, &s(34.0, 6000), HELD, RELEASE).0, Level::Light);
        assert_eq!(next_level(&t, Level::Light, &s(12.0, 6000), HELD, RELEASE).0, Level::Light);
        assert_eq!(next_level(&t, Level::Light, &s(9.0, 6000), HELD, RELEASE).0, Level::None);
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 6000), HELD, RELEASE).0, Level::None, "may drop two levels");
        // ... and only after the level held for release_after.
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 6000), Duration::from_millis(500), RELEASE).0, Level::Heavy);
        // RAM lets go 256 MB above the floor.
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 1100), HELD, RELEASE).0, Level::Heavy);
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 1400), HELD, RELEASE).0, Level::None);
    }

    #[test]
    fn others_cpu_excludes_the_children() {
        let p = SystemTimes { idle: 0, total: 0 };
        // 8 CPUs x 1 s = 80_000_000 units; 50% busy, of which the children used 30%.
        let n = SystemTimes { idle: 40_000_000, total: 80_000_000 };
        let pct = others_cpu_pct(p, n, 0, 24_000_000).unwrap();
        assert!((pct - 20.0).abs() < 1e-9);
        assert_eq!(others_cpu_pct(n, n, 0, 0), None);
    }

    #[test]
    fn tick_caps_real_jobs_by_mode() {
        let a = Child { pid: 1, job: Arc::new(Job::new(None).unwrap()), threads: 3 };
        let b = Child { pid: 2, job: Arc::new(Job::new(None).unwrap()), threads: 1 };
        let always = Pacer::new(PacerConfig { mode: "always".into(), ..Default::default() }, 8, true, 1024);
        assert_eq!((always.rate_for(Level::Heavy, 3), always.rate_for(Level::Light, 3), always.rate_for(Level::None, 3)), (Some(1875), Some(2813), None));
        always.tick(&[a.clone()]);
        assert_eq!(always.level(), Level::Heavy);
        let snap = always.snapshot(true, 3);
        assert_eq!((snap.state, snap.cap_pct, snap.cap_on_count), ("busy", 50, 1));
        always.tick(&[a.clone(), b.clone()]); // a second child joins and is capped too
        assert!(always.capped() && always.snapshot(true, 3).last_error.is_none());
        always.tick(&[]);
        assert!(!always.capped());
        assert_eq!(always.snapshot(false, 3).state, "no_model");
        let off = Pacer::new(PacerConfig { mode: "off".into(), ..Default::default() }, 8, true, 1024);
        off.tick(&[a.clone(), b.clone()]);
        assert!(!off.capped());
        assert!(off.snapshot(true, 3).last_error.is_none());
        // auto: the first tick has no CPU average yet, so the level stays none.
        let auto = Pacer::new(PacerConfig::default(), 8, true, 1024);
        auto.tick(&[a, b]);
        assert!(auto.snapshot(true, 3).last_error.is_none());
    }
}
