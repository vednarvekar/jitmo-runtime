//! RAM budget: may the model start now, is the system short of memory, and how many threads.

use crate::config::Config;
use crate::win::{self, LowMemorySignal};

const MB: u64 = 1024 * 1024;
/// Private memory assumed when there is no cap (KV cache, compute buffers, overhead).
const UNCAPPED_PRIVATE: u64 = 1024 * MB;

/// RAM the loaded model is expected to take: the mmap'd weights (about the file size) plus
/// private memory, which the commit cap bounds.
pub fn load_estimate_bytes(cfg: &Config) -> u64 {
    if cfg.budget.load_estimate_mb > 0 {
        return cfg.budget.load_estimate_mb * MB;
    }
    let private = if cfg.budget.memory_cap_mb > 0 { cfg.budget.memory_cap_mb * MB } else { UNCAPPED_PRIVATE };
    std::fs::metadata(&cfg.model.path).map_or(0, |m| m.len()) + private
}

/// Ok if starting the model would leave at least `reserve_free_mb` free; otherwise the reason.
pub fn check_before_load(cfg: &Config) -> Result<(), String> {
    let Some(mem) = win::memory_status() else {
        tracing::warn!("cannot read free RAM; starting the model without the budget check");
        return Ok(());
    };
    let need = load_estimate_bytes(cfg);
    let reserve = cfg.budget.reserve_free_mb * MB;
    if mem.avail < need + reserve {
        return Err(format!(
            "not enough free RAM to start the model: {} MB free, the model needs about {} MB and {} MB must stay free \
             (budget.load_estimate_mb / budget.reserve_free_mb). Close some apps and try again.",
            mem.avail / MB,
            need / MB,
            reserve / MB
        ));
    }
    Ok(())
}

/// Some(reason) while the system is short of memory.
pub fn pressure(cfg: &Config, signal: Option<&LowMemorySignal>) -> Option<String> {
    if signal.is_some_and(|s| s.is_low()) {
        return Some("Windows low-memory signal".into());
    }
    let mem = win::memory_status()?;
    let floor = cfg.budget.pressure_free_mb * MB;
    (mem.avail < floor).then(|| format!("{} MB free, below budget.pressure_free_mb = {}", mem.avail / MB, floor / MB))
}

/// Physical cores minus one, between 1 and 3 (3 on a 4-core laptop): the model must never
/// take the whole CPU, and on a laptop more threads mostly wait on RAM anyway.
pub fn default_threads() -> u32 {
    let logical = std::thread::available_parallelism().map_or(2, |n| n.get() as u32);
    let physical = win::physical_cores().unwrap_or((logical / 2).max(1));
    physical.saturating_sub(1).clamp(1, 3)
}

pub fn threads(cfg: &Config) -> u32 {
    if cfg.model.threads > 0 {
        cfg.model.threads
    } else {
        default_threads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_never_take_the_whole_cpu() {
        assert!((1..=3).contains(&default_threads()));
        let mut cfg = Config::default();
        cfg.model.threads = 6;
        assert_eq!(threads(&cfg), 6);
    }

    #[test]
    fn an_impossible_reserve_refuses_the_load() {
        let mut cfg = Config::default();
        cfg.budget.load_estimate_mb = 1;
        cfg.budget.reserve_free_mb = 1 << 30; // 1 PB
        let err = check_before_load(&cfg).unwrap_err();
        assert!(err.contains("not enough free RAM"));
        cfg.budget.reserve_free_mb = 0;
        assert!(check_before_load(&cfg).is_ok());
    }
}
