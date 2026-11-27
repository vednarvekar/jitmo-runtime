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
//! - CPU used by other processes (system busy time minus all children's job CPU time, and
//!   minus the client's own processes: `--parent-pid` and its other descendants, because the
//!   assistant's own router/tagger work is not "another program needs the PC"),
//!   averaged over the last `smooth_ms` so one short spike does not count;
//! - free RAM.
//!
//! Levels (all thresholds configurable):
//!
//! | Level   | When                                                  | Cap (share of the child's own full-speed CPU) |
//! |---------|-------------------------------------------------------|------|
//! | `none`  | others < `light_cpu_pct` (20%)                        | none |
//! | `light` | others 20 to 40%                                      | `light_cap_pct` (75%) |
//! | `heavy` | others > `heavy_cpu_pct` (40%), or free RAM < `low_free_mb` | `heavy_cap_pct` (50%) |
//!
//! Hysteresis: a level goes up at once; it goes down only when the signals are below the
//! thresholds by `hysteresis_pct` (RAM: 256 MB above `low_free_mb`) and the current level has
//! held for `release_after_ms`.
//!
//! v0.3: heavy lets go later, because light <-> heavy flapped 3 to 9 times per 40 s around the
//! 40% line (results/06-low-ram.md). To leave heavy, others' CPU must be `heavy_hysteresis_pct`
//! (10 points) under `heavy_cpu_pct`, RAM must be fine, and both must have stayed so for
//! `heavy_release_after_ms` (10 s) in a row. While a level holds, the reason says so.

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
    /// Hysteresis for leaving heavy (CPU points under `heavy_cpu_pct`).
    pub heavy_hysteresis_pct: f64,
    /// A level holds at least this long after a change before it goes down.
    pub release_after: Duration,
    /// Heavy goes down only after the signals stayed clear of it this long in a row.
    pub heavy_release_after: Duration,
}

impl Thresholds {
    pub fn from_config(cfg: &PacerConfig, pressure_free_mb: u64) -> Thresholds {
        Thresholds {
            light_cpu_pct: cfg.light_cpu_pct,
            heavy_cpu_pct: cfg.heavy_cpu_pct,
            low_free_mb: if cfg.low_free_mb > 0 { cfg.low_free_mb } else { pressure_free_mb },
            hysteresis_pct: cfg.hysteresis_pct,
            heavy_hysteresis_pct: cfg.heavy_hysteresis_pct,
            release_after: Duration::from_millis(cfg.release_after_ms),
            heavy_release_after: Duration::from_millis(cfg.heavy_release_after_ms),
        }
    }
}

/// What each signal asks for. `lowering` lowers the thresholds by the hysteresis (used to
/// decide whether a level may go down), so a signal sitting just under a threshold does not
/// make the level flap. A signal that is only inside that margin says it is holding.
fn proposals(t: &Thresholds, s: &Signals, lowering: bool) -> Vec<(Level, String)> {
    let (light_margin, heavy_margin, ram_margin) =
        if lowering { (t.hysteresis_pct, t.heavy_hysteresis_pct, RAM_HYSTERESIS_MB) } else { (0.0, 0.0, 0) };
    let cpu = |pct: f64, limit: f64| {
        if pct > limit {
            format!("other processes use {pct:.0}% CPU (> {limit:.0}%)")
        } else {
            format!("holding: other processes used over {limit:.0}% CPU, now {pct:.0}%")
        }
    };
    let mut out = Vec::new();
    if let Some(pct) = s.others_cpu_pct {
        if t.heavy_cpu_pct > 0.0 && pct > t.heavy_cpu_pct - heavy_margin {
            out.push((Level::Heavy, cpu(pct, t.heavy_cpu_pct)));
        } else if t.light_cpu_pct > 0.0 && pct > t.light_cpu_pct - light_margin {
            out.push((Level::Light, cpu(pct, t.light_cpu_pct)));
        }
    }
    if let Some(free) = s.free_mb {
        if t.low_free_mb > 0 && free < t.low_free_mb + ram_margin {
            let why = if free < t.low_free_mb {
                format!("{free} MB RAM free (< {} MB)", t.low_free_mb)
            } else {
                format!("holding: RAM was below {} MB, now {free} MB", t.low_free_mb)
            };
            out.push((Level::Heavy, why));
        }
    }
    out
}

/// True if the signals keep heavy from going down (heavy, with the hysteresis margin). The
/// pacer notes the last sample where this was true: heavy lets go only `heavy_release_after` later.
pub fn holds_heavy(t: &Thresholds, s: &Signals) -> bool {
    highest(proposals(t, s, true)).0 == Level::Heavy
}

fn highest(p: Vec<(Level, String)>) -> (Level, Option<String>) {
    p.into_iter().max_by_key(|(l, _)| *l).map_or((Level::None, None), |(l, r)| (l, Some(r)))
}

/// The next level: up at once to what the signals ask for; down only when the signals ask
/// for less even with the hysteresis margin, and the current level has held long enough:
/// `held` (since the last change) >= `release_after`, and for heavy also `quiet` (since the
/// last sample where `holds_heavy` was true) >= `heavy_release_after`.
pub fn next_level(t: &Thresholds, current: Level, s: &Signals, held: Duration, quiet: Duration) -> (Level, Option<String>) {
    let (up, why_up) = highest(proposals(t, s, false));
    if up >= current {
        return (up, why_up);
    }
    let (down, why_down) = highest(proposals(t, s, true));
    if down >= current {
        return (current, why_down); // inside the hysteresis margin: the reason says "holding"
    }
    let wait = if current == Level::Heavy {
        t.release_after.saturating_sub(held).max(t.heavy_release_after.saturating_sub(quiet))
    } else {
        t.release_after.saturating_sub(held)
    };
    if wait.is_zero() {
        return (down, why_down);
    }
    let now = why_down.or(why_up).unwrap_or_else(|| "the signals are under the thresholds".into());
    (current, Some(format!("holding {} for {:.0} s more; now {now}", current.as_str(), wait.as_secs_f64().ceil())))
}

/// The Job Object CpuRate (cycles per 10000 of all logical processors) that gives a child
/// `cap_pct` % of the CPU it uses at full speed with `threads` threads.
pub fn cpu_rate(cap_pct: u32, threads: u32, logical: u32) -> u32 {
    let full = threads.min(logical) as f64 / logical.max(1) as f64;
    ((cap_pct as f64 / 100.0) * full * 10_000.0).round().clamp(1.0, 10_000.0) as u32
}

/// CPU used by other processes, in % of all logical processors, between two samples.
/// `excluded` = CPU time of the children (and the client's processes) in that interval.
pub fn others_cpu_pct(prev: SystemTimes, now: SystemTimes, excluded: u64) -> Option<f64> {
    let total = now.total.checked_sub(prev.total)?;
    let idle = now.idle.checked_sub(prev.idle)?;
    if total == 0 {
        return None;
    }
    let busy = total.saturating_sub(idle);
    Some(busy.saturating_sub(excluded) as f64 * 100.0 / total as f64)
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
    /// The client process left out of `others_cpu_pct` (`--parent-pid`), and how many of its processes.
    pub client_pid: Option<u32>,
    pub client_processes: usize,
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
    /// An inproc engine worker's port: it is paced by `POST /pace` instead of a CPU cap.
    pub pace_port: Option<u16>,
}

/// What an inproc worker is told at a level: compute share (%) and threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerPace {
    pub duty_pct: u32,
    pub threads: u32,
}

struct Sample {
    at: Instant,
    sys: SystemTimes,
    children: HashMap<u32, u64>,
    /// CPU time of the client's processes (see `Pacer::set_client`).
    clients: HashMap<u32, u64>,
}

/// CPU time used between two samples by the pids present in both.
fn delta(old: &HashMap<u32, u64>, new: &HashMap<u32, u64>) -> u64 {
    new.iter().filter_map(|(pid, t)| old.get(pid).map(|o| t.saturating_sub(*o))).sum()
}

/// How often the client's process tree is listed again.
const CLIENT_TREE_REFRESH: Duration = Duration::from_secs(5);

#[derive(Default)]
struct Inner {
    window: VecDeque<Sample>,
    signals: Signals,
    level: Level,
    reason: Option<String>,
    level_since: Option<Instant>,
    /// Last sample where the signals held heavy (see `holds_heavy`).
    heavy_seen: Option<Instant>,
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
    /// The client's process tree and when it was listed.
    client_tree: Vec<u32>,
    client_tree_at: Option<Instant>,
}

/// One pacer for the whole pool: the signals are sampled once, "other processes' CPU" leaves
/// out every child, and the level's cap is set on every loaded child.
pub struct Pacer {
    cfg: PacerConfig,
    t: Thresholds,
    logical: u32,
    base_priority: u32,
    /// The process that started the runtime (`--parent-pid`): its CPU, and its other
    /// children's, is the assistant's own work, not "other programs".
    client: Mutex<Option<u32>>,
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
        let t = Thresholds::from_config(&cfg, pressure_free_mb);
        Pacer { cfg, t, logical, base_priority, client: Mutex::new(None), inner: Mutex::new(Inner::default()) }
    }

    /// Leave this process and its descendants (other than the runtime) out of "other
    /// programs' CPU", if `pacer.exclude_client_cpu` is on.
    pub fn set_client(&self, pid: u32) {
        if self.cfg.exclude_client_cpu {
            *self.client.lock().unwrap_or_else(|p| p.into_inner()) = Some(pid);
        }
    }

    /// CPU times of the client's processes; the tree is listed again every few seconds.
    fn client_times(&self, i: &mut Inner, now: Instant) -> HashMap<u32, u64> {
        let Some(root) = *self.client.lock().unwrap_or_else(|p| p.into_inner()) else {
            return HashMap::new();
        };
        if i.client_tree_at.is_none_or(|t| now - t >= CLIENT_TREE_REFRESH) {
            i.client_tree = win::process_tree(&win::process_parents(), root, std::process::id());
            i.client_tree_at = Some(now);
        }
        i.client_tree.iter().filter_map(|&pid| win::process_cpu_time(pid).map(|t| (pid, t))).collect()
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

    pub fn pause_lever(&self) -> bool {
        self.cfg.inproc_lever != "cap"
    }

    /// The pace for an inproc worker with `threads` threads at the current level.
    pub fn worker_pace(&self, threads: u32) -> WorkerPace {
        let level = self.lock().level;
        let duty_pct = self.cap_pct(level);
        let threads = if level == Level::Heavy { threads.saturating_sub(1).max(1) } else { threads };
        WorkerPace { duty_pct, threads }
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
        let excluded = child_new.saturating_sub(child_old) + delta(&old.clients, &new.clients);
        others_cpu_pct(old.sys, new.sys, excluded)
    }

    /// One sample and decision for all loaded children.
    pub fn tick(&self, children: &[Child]) {
        let now = Instant::now();
        let sys = win::system_times();
        let cpu: HashMap<u32, u64> = children.iter().filter_map(|c| c.job.cpu_time().map(|t| (c.pid, t))).collect();
        let mut i = self.lock();
        let clients = self.client_times(&mut i, now);
        if let Some(last) = i.last_tick.replace(now) {
            if !children.is_empty() {
                let k = i.level as usize;
                i.at_level[k] += now - last;
            }
        }
        let others = match sys {
            Some(sys) if cpu.len() == children.len() => self.smoothed_others(&mut i, Sample { at: now, sys, children: cpu, clients }),
            _ => {
                i.window.clear();
                None
            }
        };
        i.signals = Signals {
            others_cpu_pct: others.map(|p| (p * 10.0).round() / 10.0),
            free_mb: win::memory_status().map(|m| m.avail / (1024 * 1024)),
        };

        if i.signals.others_cpu_pct.is_some() && holds_heavy(&self.t, &i.signals) {
            i.heavy_seen = Some(now);
        }
        let held = i.level_since.map_or(Duration::MAX, |t| now - t);
        let quiet = i.heavy_seen.map_or(Duration::MAX, |t| now - t);
        let (want, reason) = match self.cfg.mode.as_str() {
            "off" => (Level::None, None),
            "always" => (Level::Heavy, Some("pacer.mode = always".to_string())),
            _ if i.signals.others_cpu_pct.is_none() => (i.level, i.reason.clone()), // no CPU reading this tick: keep
            _ => next_level(&self.t, i.level, &i.signals, held, quiet),
        };
        let want = if children.is_empty() { Level::None } else { want };

        // Forget children that are gone (a new child has a new, uncapped job).
        i.applied.retain(|pid, _| children.iter().any(|c| c.pid == *pid));
        let mut err = None;
        for c in children {
            let applied = i.applied.get(&c.pid).copied().flatten();
            let rate = if c.pace_port.is_some() && self.pause_lever() { None } else { self.rate_for(want, c.threads) };
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
            client_pid: *self.client.lock().unwrap_or_else(|p| p.into_inner()),
            client_processes: i.client_tree.len(),
            mem_latency_ns: None,
            thresholds: json!({
                "light_cpu_pct": self.t.light_cpu_pct, "heavy_cpu_pct": self.t.heavy_cpu_pct, "low_free_mb": self.t.low_free_mb,
                "light_cap_pct": self.cfg.light_cap_pct, "heavy_cap_pct": self.cfg.heavy_cap_pct,
                "hysteresis_pct": self.t.hysteresis_pct, "smooth_ms": self.cfg.smooth_ms, "release_after_ms": self.cfg.release_after_ms,
                "heavy_hysteresis_pct": self.t.heavy_hysteresis_pct, "heavy_release_after_ms": self.cfg.heavy_release_after_ms,
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
        Thresholds {
            light_cpu_pct: 20.0,
            heavy_cpu_pct: 40.0,
            low_free_mb: 1024,
            hysteresis_pct: 5.0,
            heavy_hysteresis_pct: 10.0,
            release_after: RELEASE,
            heavy_release_after: HEAVY_RELEASE,
        }
    }
    fn s(cpu: f64, free: u64) -> Signals {
        Signals { others_cpu_pct: Some(cpu), free_mb: Some(free) }
    }
    const HELD: Duration = Duration::from_secs(20);
    const RELEASE: Duration = Duration::from_millis(2000);
    const HEAVY_RELEASE: Duration = Duration::from_millis(10_000);

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
        let n = |c, f| next_level(&t, Level::None, &s(c, f), HELD, HELD).0;
        assert_eq!(n(5.0, 6000), Level::None);
        assert_eq!(n(20.0, 6000), Level::None, "20% is not above 20%");
        assert_eq!(n(25.0, 6000), Level::Light);
        assert_eq!(n(40.0, 6000), Level::Light);
        assert_eq!(n(41.0, 6000), Level::Heavy);
        assert_eq!(n(5.0, 900), Level::Heavy, "low RAM alone is heavy");
        let (_, why) = next_level(&t, Level::None, &s(25.0, 900), HELD, HELD);
        assert!(why.unwrap().contains("RAM"), "the highest level's reason wins");
        assert_eq!(next_level(&t, Level::None, &Signals::default(), HELD, HELD).0, Level::None, "no signals = none");
        assert_eq!(PacerConfig::default().light_cpu_pct, 20.0, "v0.3 default");
    }

    #[test]
    fn hysteresis_and_hold_stop_flapping() {
        let t = t();
        // Up at once, even right after a change.
        assert_eq!(next_level(&t, Level::None, &s(50.0, 6000), Duration::ZERO, Duration::ZERO).0, Level::Heavy);
        // Heavy goes down only below 40 - 10 ...
        assert_eq!(next_level(&t, Level::Heavy, &s(31.0, 6000), HELD, HELD).0, Level::Heavy);
        assert_eq!(next_level(&t, Level::Heavy, &s(29.0, 6000), HELD, HELD).0, Level::Light);
        // ... light below 20 - 5.
        assert_eq!(next_level(&t, Level::Light, &s(16.0, 6000), HELD, HELD).0, Level::Light);
        assert_eq!(next_level(&t, Level::Light, &s(14.0, 6000), HELD, HELD).0, Level::None);
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 6000), HELD, HELD).0, Level::None, "may drop two levels");
        // Light needs release_after since the change; heavy also heavy_release_after of quiet.
        assert_eq!(next_level(&t, Level::Light, &s(2.0, 6000), Duration::from_millis(500), HELD).0, Level::Light);
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 6000), Duration::from_millis(500), HELD).0, Level::Heavy);
        let (l, why) = next_level(&t, Level::Heavy, &s(2.0, 6000), HELD, Duration::from_secs(4));
        assert_eq!(l, Level::Heavy);
        assert!(why.as_deref().unwrap().starts_with("holding heavy for 6 s more"), "{why:?}");
        // RAM lets go 256 MB above the floor.
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 1100), HELD, HELD).0, Level::Heavy);
        assert_eq!(next_level(&t, Level::Heavy, &s(2.0, 1400), HELD, HELD).0, Level::None);
    }

    #[test]
    fn a_held_level_says_it_is_holding() {
        let t = t();
        // RAM inside the 256 MB margin (results/06-low-ram.md: "1225 MB RAM free (< 1024 MB)").
        let (l, why) = next_level(&t, Level::Heavy, &s(2.0, 1225), HELD, HELD);
        assert_eq!(l, Level::Heavy);
        assert_eq!(why.as_deref(), Some("holding: RAM was below 1024 MB, now 1225 MB"));
        // Really below the floor: the plain reason.
        let (_, why) = next_level(&t, Level::Heavy, &s(2.0, 900), HELD, HELD);
        assert_eq!(why.as_deref(), Some("900 MB RAM free (< 1024 MB)"));
        // CPU inside the margin.
        let (_, why) = next_level(&t, Level::Heavy, &s(35.0, 6000), HELD, HELD);
        assert_eq!(why.as_deref(), Some("holding: other processes used over 40% CPU, now 35%"));
        let (_, why) = next_level(&t, Level::Light, &s(17.0, 6000), HELD, HELD);
        assert_eq!(why.as_deref(), Some("holding: other processes used over 20% CPU, now 17%"));
        let (_, why) = next_level(&t, Level::Light, &s(45.0, 6000), HELD, HELD);
        assert_eq!(why.as_deref(), Some("other processes use 45% CPU (> 40%)"));
        // Waiting for the hold time.
        let (_, why) = next_level(&t, Level::Light, &s(2.0, 6000), Duration::from_millis(500), HELD);
        assert!(why.as_deref().unwrap().starts_with("holding light for 2 s more"), "{why:?}");
    }

    /// Feeds a CPU series to the decision the way `Pacer::tick` does (one sample per `poll`),
    /// and returns the number of level changes.
    fn simulate(t: &Thresholds, series: &[f64], poll: Duration) -> u32 {
        let start = Instant::now();
        let (mut level, mut since, mut heavy_seen, mut changes) = (Level::None, None::<Instant>, None::<Instant>, 0);
        for (k, &cpu) in series.iter().enumerate() {
            let now = start + poll * k as u32;
            let sig = s(cpu, 6000);
            if holds_heavy(t, &sig) {
                heavy_seen = Some(now);
            }
            let held = since.map_or(Duration::MAX, |x| now - x);
            let quiet = heavy_seen.map_or(Duration::MAX, |x| now - x);
            let (want, _) = next_level(t, level, &sig, held, quiet);
            if want != level {
                changes += 1;
                level = want;
                since = Some(now);
            }
        }
        changes
    }

    /// A noisy series: `center` +/- `spread`, from a fixed pseudo-random sequence.
    fn noisy(center: f64, spread: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (x >> 11) as f64 / (1u64 << 53) as f64; // 0..1
                center + spread * (2.0 * u - 1.0)
            })
            .collect()
    }

    #[test]
    fn noisy_cpu_around_the_heavy_line_changes_level_rarely() {
        let poll = Duration::from_millis(300);
        let n = 134; // 40 s
        // v0.2 behaviour: 5 points, 5 s hold since the change, no quiet time for heavy.
        let old = Thresholds {
            light_cpu_pct: 15.0,
            hysteresis_pct: 5.0,
            heavy_hysteresis_pct: 5.0,
            release_after: Duration::from_secs(5),
            heavy_release_after: Duration::ZERO,
            ..t()
        };
        let new = Thresholds::from_config(&PacerConfig::default(), 1024);
        assert_eq!((new.heavy_hysteresis_pct, new.heavy_release_after), (10.0, Duration::from_secs(10)));
        let mut old_total = 0;
        for seed in 1..=5 {
            for (center, spread) in [(40.0, 8.0), (38.0, 10.0), (40.0, 14.0)] {
                let series = noisy(center, spread, n, seed);
                let (o, c) = (simulate(&old, &series, poll), simulate(&new, &series, poll));
                old_total += o;
                eprintln!("seed {seed}, {center} +/- {spread}: v0.2 rules {o} changes, v0.3 rules {c}");
                assert!(c <= 2, "seed {seed}, {center} +/- {spread}: {c} changes in 40 s (v0.2 rules: {o})");
            }
        }
        assert!(old_total >= 15 * 2, "the v0.2 rules flap on these series ({old_total} changes in 15 runs)");
    }

    #[test]
    fn heavy_lets_go_after_a_quiet_spell() {
        let poll = Duration::from_millis(300);
        let t = Thresholds::from_config(&PacerConfig::default(), 1024);
        // 10 s busy, then 20 s at about 10%: up once, down once (to none) after about 10 s.
        let mut series = noisy(50.0, 5.0, 34, 7);
        series.extend(noisy(10.0, 3.0, 67, 8));
        assert_eq!(simulate(&t, &series, poll), 2);
        // Quiet for only 6 s between two busy spells: heavy holds the whole time.
        let mut series = noisy(50.0, 5.0, 34, 9);
        series.extend(noisy(10.0, 3.0, 20, 10));
        series.extend(noisy(50.0, 5.0, 34, 11));
        assert_eq!(simulate(&t, &series, poll), 1);
    }

    #[test]
    fn others_cpu_excludes_the_children() {
        let p = SystemTimes { idle: 0, total: 0 };
        // 8 CPUs x 1 s = 80_000_000 units; 50% busy, of which the children used 30%.
        let n = SystemTimes { idle: 40_000_000, total: 80_000_000 };
        let pct = others_cpu_pct(p, n, 24_000_000).unwrap();
        assert!((pct - 20.0).abs() < 1e-9);
        assert_eq!(others_cpu_pct(n, n, 0), None);
        // Only pids seen in both samples count (a new or gone process has no interval).
        let old = HashMap::from([(10, 100), (11, 50)]);
        let new = HashMap::from([(10, 400), (12, 999)]);
        assert_eq!(delta(&old, &new), 300);
    }

    #[test]
    fn client_tree_follows_children_and_skips_the_runtime() {
        // 1 = jitmo; 2 = its model-worker; 3 = the runtime; 4 = a llama-server; 5 = worker's child; 9 = unrelated.
        let parents = [(1, 0), (2, 1), (3, 1), (4, 3), (5, 2), (9, 7)];
        let mut t = win::process_tree(&parents, 1, 3);
        t.sort();
        assert_eq!(t, vec![1, 2, 5]);
    }

    #[test]
    fn tick_caps_real_jobs_by_mode() {
        let a = Child { pid: 1, job: Arc::new(Job::new(None).unwrap()), threads: 3, pace_port: None };
        let b = Child { pid: 2, job: Arc::new(Job::new(None).unwrap()), threads: 1, pace_port: None };
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
        // An inproc worker under the pause lever gets no job cap; it is told a pace instead.
        let w = Child { pid: 3, job: Arc::new(Job::new(None).unwrap()), threads: 3, pace_port: Some(1) };
        always.tick(&[w]);
        assert_eq!(always.level(), Level::Heavy);
        assert_eq!(always.worker_pace(3), WorkerPace { duty_pct: 50, threads: 2 });
        assert_eq!(off.worker_pace(3), WorkerPace { duty_pct: 100, threads: 3 });
    }
}
