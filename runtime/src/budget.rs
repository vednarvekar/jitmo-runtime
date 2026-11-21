//! RAM budget: the child's memory cap, may the model start now, is the system short of
//! memory, and how many threads.

use crate::config::{Config, MemoryCap, ModelConfig};
use crate::gguf::Gguf;
use crate::win::{self, LowMemorySignal};
use serde::Serialize;

const MB: u64 = 1024 * 1024;
/// Private memory assumed when there is no cap (KV cache, compute buffers, overhead).
const UNCAPPED_PRIVATE: u64 = 1024 * MB;
/// Cap used when `memory_cap_mb = "auto"` but the GGUF metadata cannot be read.
const FALLBACK_CAP: u64 = 1536 * MB;
/// Fixed part of the automatic cap: tokenizer, server state, the graph, heap slack. Measured
/// (private bytes after load minus KV, recurrent state and compute buffers, 21 Nov 2026):
/// Qwen3-4B 128 MiB, Gemma 3n E2B 180 MiB, Qwen3.5-0.8B 517 MiB. 640 MiB covers all three
/// with room for the growth seen during generation.
pub const CAP_OVERHEAD: u64 = 640 * MB;
/// llama-server's default physical batch (`-ub`); the compute buffers scale with it.
const N_UBATCH: u64 = 512;

/// Bytes per element of a KV cache type (block types: bytes per block / 32 elements).
fn kv_type_bytes(ty: &str) -> f64 {
    match ty {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q4_0" | "iq4_nl" => 18.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "q5_0" => 22.0 / 32.0,
        "q5_1" => 24.0 / 32.0,
        _ => 2.0,
    }
}

/// Private memory the child needs, from the GGUF metadata and the model settings.
/// Every part is in bytes. The estimate errs on the large side when metadata is missing.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct MemoryEstimate {
    pub arch: String,
    pub n_vocab: u64,
    /// Layers that keep a KV cache (full-context and sliding-window).
    pub kv_layers: u32,
    pub swa_layers: u32,
    pub kv_bytes: u64,
    /// Part of `kv_bytes` in sliding-window layers (sized to window + batch, not ctx).
    pub swa_kv_bytes: u64,
    /// Recurrent state of SSM / linear-attention layers (Qwen3.5, Mamba).
    pub recurrent_bytes: u64,
    /// Compute buffers: 16 x ubatch x (max FFN width + embedding width x altup inputs).
    pub compute_bytes: u64,
    /// `model.cache_ram_mb`.
    pub cache_ram_bytes: u64,
    /// `model.ctx_checkpoints` x (SWA KV + recurrent state), only for SWA/hybrid models.
    pub checkpoint_bytes: u64,
    pub overhead_bytes: u64,
    /// Sum of all parts, rounded up to 64 MiB: the automatic memory cap.
    pub total_bytes: u64,
}

impl MemoryEstimate {
    pub fn from_gguf(g: &Gguf, m: &ModelConfig) -> Option<MemoryEstimate> {
        let n_layer = g.num("block_count")? as usize;
        let n_embd = g.num("embedding_length")?;
        let n_head = g.per_layer("attention.head_count", 0).unwrap_or(1.0).max(1.0);
        let ctx = m.ctx_size as u64;
        let bk = kv_type_bytes(&m.kv_cache_type);
        let bv = bk;

        // Which layers have attention at all (hybrid models), and which keep their own KV.
        let interval = g.num("full_attention_interval").map(|v| v as usize).filter(|&v| v > 0);
        let shared = g.num("attention.shared_kv_layers").unwrap_or(0.0) as usize;
        let n_swa = g.num("attention.sliding_window").unwrap_or(0.0) as u64;
        let pattern_arr = g.array("attention.sliding_window_pattern");
        let pattern_n = g.num("attention.sliding_window_pattern").map(|v| v as usize).filter(|&v| v > 0);
        let swa_cells = ctx.min((n_swa + N_UBATCH).div_ceil(256) * 256);

        let mut est = MemoryEstimate {
            arch: g.arch().unwrap_or("").to_string(),
            n_vocab: g.len_of("tokenizer.ggml.tokens").unwrap_or(0),
            ..Default::default()
        };
        let mut kv = 0.0f64;
        let mut swa_kv = 0.0f64;
        let mut recurrent_layers = 0u64;
        for il in 0..n_layer {
            let attention = interval.is_none_or(|n| (il + 1) % n == 0);
            if !attention {
                recurrent_layers += 1;
                continue;
            }
            if shared > 0 && il >= n_layer.saturating_sub(shared) {
                continue; // reuses an earlier layer's KV (Gemma 3n)
            }
            let swa = n_swa > 0
                && match (pattern_arr, pattern_n) {
                    (Some(p), _) => p.get(il).is_some_and(|&v| v != 0.0),
                    (None, Some(n)) => il % n < n - 1,
                    // No pattern in the file: count the layer at full ctx (safe side).
                    (None, None) => false,
                };
            let heads_kv = g.per_layer("attention.head_count_kv", il).unwrap_or(n_head);
            let k_len = g.per_layer("attention.key_length", il).unwrap_or(n_embd / n_head);
            let v_len = g.per_layer("attention.value_length", il).unwrap_or(n_embd / n_head);
            let cells = if swa { swa_cells } else { ctx } as f64;
            let bytes = cells * heads_kv * (k_len * bk + v_len * bv);
            kv += bytes;
            est.kv_layers += 1;
            if swa {
                est.swa_layers += 1;
                swa_kv += bytes;
            }
        }
        est.kv_bytes = kv.ceil() as u64;
        est.swa_kv_bytes = swa_kv.ceil() as u64;

        if let (Some(inner), Some(state)) = (g.num("ssm.inner_size"), g.num("ssm.state_size")) {
            let conv = g.num("ssm.conv_kernel").unwrap_or(4.0);
            let groups = g.num("ssm.group_count").unwrap_or(1.0);
            let per_layer = (conv - 1.0) * (inner + 2.0 * groups * state) * 4.0 + inner * state * 4.0;
            let layers = if interval.is_some() { recurrent_layers } else { n_layer as u64 };
            est.recurrent_bytes = (per_layer * layers as f64).ceil() as u64;
        }

        let n_ff = (0..n_layer).filter_map(|il| g.per_layer("feed_forward_length", il)).fold(0.0, f64::max);
        let altup = g.num("altup.num_inputs").unwrap_or(1.0).max(1.0);
        est.compute_bytes = (16.0 * N_UBATCH as f64 * (n_ff + n_embd * altup)) as u64;
        est.cache_ram_bytes = m.cache_ram_mb * MB;
        if est.swa_kv_bytes > 0 || est.recurrent_bytes > 0 {
            est.checkpoint_bytes = m.ctx_checkpoints as u64 * (est.swa_kv_bytes + est.recurrent_bytes);
        }
        est.overhead_bytes = CAP_OVERHEAD;
        let sum = est.kv_bytes + est.recurrent_bytes + est.compute_bytes + est.cache_ram_bytes + est.checkpoint_bytes + est.overhead_bytes;
        est.total_bytes = sum.div_ceil(64 * MB) * 64 * MB;
        Some(est)
    }
}

/// The memory cap in force for this runtime, decided once at startup.
#[derive(Debug, Clone, Serialize)]
pub struct MemPlan {
    /// Commit cap on the child in bytes; None = no cap.
    pub cap_bytes: Option<u64>,
    /// `auto`, `config` (explicit number), `none` (0), or `fallback` (auto, but the GGUF could not be read).
    pub source: &'static str,
    pub estimate: Option<MemoryEstimate>,
}

impl MemPlan {
    pub fn new(cfg: &Config) -> MemPlan {
        let estimate = match Gguf::read(&cfg.model.path) {
            Ok(g) => {
                let e = MemoryEstimate::from_gguf(&g, &cfg.model);
                if e.is_none() {
                    tracing::warn!("GGUF metadata lacks block_count/embedding_length; cannot estimate the model's memory");
                }
                e
            }
            Err(e) => {
                tracing::warn!("{e:#}");
                None
            }
        };
        MemPlan::decide(cfg.budget.memory_cap_mb, estimate)
    }

    pub fn decide(setting: MemoryCap, estimate: Option<MemoryEstimate>) -> MemPlan {
        match (setting, &estimate) {
            (MemoryCap::Mb(0), _) => MemPlan { cap_bytes: None, source: "none", estimate },
            (MemoryCap::Mb(n), _) => MemPlan { cap_bytes: Some(n * MB), source: "config", estimate },
            (MemoryCap::Auto, Some(e)) => MemPlan { cap_bytes: Some(e.total_bytes), source: "auto", estimate },
            (MemoryCap::Auto, None) => MemPlan { cap_bytes: Some(FALLBACK_CAP), source: "fallback", estimate },
        }
    }

    pub fn cap_mb(&self) -> u64 {
        self.cap_bytes.unwrap_or(0) / MB
    }
}

/// RAM the loaded model is expected to take: the mmap'd weights (about the file size) plus
/// private memory, which the commit cap bounds.
pub fn load_estimate_bytes(cfg: &Config, plan: &MemPlan) -> u64 {
    if cfg.budget.load_estimate_mb > 0 {
        return cfg.budget.load_estimate_mb * MB;
    }
    std::fs::metadata(&cfg.model.path).map_or(0, |m| m.len()) + plan.cap_bytes.unwrap_or(UNCAPPED_PRIVATE)
}

/// Ok if starting the model would leave at least `reserve_free_mb` free; otherwise the reason.
pub fn check_before_load(cfg: &Config, plan: &MemPlan) -> Result<(), String> {
    let Some(mem) = win::memory_status() else {
        tracing::warn!("cannot read free RAM; starting the model without the budget check");
        return Ok(());
    };
    let need = load_estimate_bytes(cfg, plan);
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

/// What the watchdog should do about memory on this check.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAction {
    None,
    /// Short of memory, but not critically: finish the running generation, then unload.
    UnloadWhenFree(String),
    /// Unload at once, even mid-generation.
    UnloadNow(String),
}

/// `free` = available RAM in bytes, `low_signal` = Windows' low-memory notification,
/// `low_count` = how many checks in a row (including this one) were below the pressure floor.
pub fn memory_action(cfg: &Config, free: u64, low_signal: bool, low_count: u32) -> MemoryAction {
    let critical = cfg.budget.critical_free_mb * MB;
    if critical > 0 && free < critical {
        return MemoryAction::UnloadNow(format!(
            "{} MB free, below budget.critical_free_mb = {}",
            free / MB,
            cfg.budget.critical_free_mb
        ));
    }
    if low_count >= 2 {
        let why = if low_signal {
            "Windows low-memory signal".to_string()
        } else {
            format!("{} MB free, below budget.pressure_free_mb = {}", free / MB, cfg.budget.pressure_free_mb)
        };
        return MemoryAction::UnloadWhenFree(why);
    }
    MemoryAction::None
}

/// True if this check counts toward memory pressure.
pub fn is_low(cfg: &Config, free: u64, signal: Option<&LowMemorySignal>) -> (bool, bool) {
    let sig = signal.is_some_and(|s| s.is_low());
    (sig || free < cfg.budget.pressure_free_mb * MB, sig)
}

/// Physical cores minus one, but never more than half of the logical processors (min 1):
/// the model must never take the whole CPU, and on a laptop more threads mostly wait on RAM.
/// On the i5-1135G7 (4 cores, 8 threads): min(3, 4) = 3.
pub fn default_threads_for(physical: u32, logical: u32) -> u32 {
    physical.saturating_sub(1).min(logical / 2).max(1)
}

pub fn logical_processors() -> u32 {
    std::thread::available_parallelism().map_or(2, |n| n.get() as u32)
}

pub fn default_threads() -> u32 {
    let logical = logical_processors();
    let physical = win::physical_cores().unwrap_or((logical / 2).max(1));
    default_threads_for(physical, logical)
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
    use crate::gguf::testfile::{build, V};

    fn parse(kvs: &[(&str, V)]) -> Gguf {
        Gguf::parse(&mut std::io::Cursor::new(build(kvs))).unwrap()
    }

    fn mib(b: u64) -> f64 {
        b as f64 / MB as f64
    }

    #[test]
    fn threads_follow_the_owner_rule() {
        assert_eq!(default_threads_for(4, 8), 3, "i5-1135G7");
        assert_eq!(default_threads_for(4, 4), 2, "no SMT: half of 4");
        assert_eq!(default_threads_for(8, 16), 7);
        assert_eq!(default_threads_for(6, 6), 3);
        assert_eq!(default_threads_for(2, 4), 1);
        assert_eq!(default_threads_for(1, 1), 1);
        assert_eq!(default_threads_for(1, 2), 1);
        let d = default_threads();
        assert!(d >= 1 && d <= logical_processors() / 2 + 1);
        let mut cfg = Config::default();
        cfg.model.threads = 6;
        assert_eq!(threads(&cfg), 6);
    }

    #[test]
    fn an_impossible_reserve_refuses_the_load() {
        let mut cfg = Config::default();
        let plan = MemPlan::decide(MemoryCap::Mb(0), None);
        cfg.budget.load_estimate_mb = 1;
        cfg.budget.reserve_free_mb = 1 << 30; // 1 PB
        let err = check_before_load(&cfg, &plan).unwrap_err();
        assert!(err.contains("not enough free RAM"));
        cfg.budget.reserve_free_mb = 0;
        assert!(check_before_load(&cfg, &plan).is_ok());
    }

    /// The KV and recurrent sizes must match what llama.cpp build 12248 printed for the real
    /// files at ctx 4096, f16 KV (llama-server -lv 4, 21 Nov 2026).
    #[test]
    fn kv_estimates_match_llama_cpp_for_the_three_test_models() {
        let m = ModelConfig::default(); // ctx 4096, f16, cache_ram 256, 4 checkpoints

        let qwen3_4b = parse(&[
            ("general.architecture", V::Str("qwen3")),
            ("qwen3.block_count", V::U32(36)),
            ("qwen3.embedding_length", V::U32(2560)),
            ("qwen3.feed_forward_length", V::U32(9728)),
            ("qwen3.attention.head_count", V::U32(32)),
            ("qwen3.attention.head_count_kv", V::U32(8)),
            ("qwen3.attention.key_length", V::U32(128)),
            ("qwen3.attention.value_length", V::U32(128)),
        ]);
        let e = MemoryEstimate::from_gguf(&qwen3_4b, &m).unwrap();
        assert_eq!(e.kv_bytes, 603_979_776, "llama.cpp: KV 576.00 MiB, 36 layers");
        assert_eq!((e.kv_layers, e.swa_layers, e.recurrent_bytes, e.checkpoint_bytes), (36, 0, 0, 0));
        assert_eq!(mib(e.total_bytes), 1600.0);

        let qwen35_08b = parse(&[
            ("general.architecture", V::Str("qwen35")),
            ("qwen35.block_count", V::U32(24)),
            ("qwen35.embedding_length", V::U32(1024)),
            ("qwen35.feed_forward_length", V::U32(3584)),
            ("qwen35.attention.head_count", V::U32(8)),
            ("qwen35.attention.head_count_kv", V::U32(2)),
            ("qwen35.attention.key_length", V::U32(256)),
            ("qwen35.attention.value_length", V::U32(256)),
            ("qwen35.ssm.conv_kernel", V::U32(4)),
            ("qwen35.ssm.state_size", V::U32(128)),
            ("qwen35.ssm.group_count", V::U32(16)),
            ("qwen35.ssm.inner_size", V::U32(2048)),
            ("qwen35.full_attention_interval", V::U32(4)),
        ]);
        let e = MemoryEstimate::from_gguf(&qwen35_08b, &m).unwrap();
        assert_eq!(mib(e.kv_bytes), 48.0, "llama.cpp: KV 48.00 MiB, 6 layers");
        assert_eq!(e.kv_layers, 6);
        assert!((mib(e.recurrent_bytes) - 19.27).abs() < 0.01, "llama.cpp: RS 19.27 MiB, got {}", mib(e.recurrent_bytes));
        assert_eq!(e.checkpoint_bytes, 4 * e.recurrent_bytes);

        let mut pattern = [true, true, true, true, false].repeat(6);
        pattern.truncate(30);
        let ff = [8192u32; 30];
        let gemma3n = parse(&[
            ("general.architecture", V::Str("gemma3n")),
            ("gemma3n.block_count", V::U32(30)),
            ("gemma3n.embedding_length", V::U32(2048)),
            ("gemma3n.feed_forward_length", V::ArrU32(&ff)),
            ("gemma3n.attention.head_count", V::U32(8)),
            ("gemma3n.attention.head_count_kv", V::U32(2)),
            ("gemma3n.attention.key_length", V::U32(256)),
            ("gemma3n.attention.value_length", V::U32(256)),
            ("gemma3n.attention.sliding_window", V::U32(512)),
            ("gemma3n.attention.sliding_window_pattern", V::ArrBool(&pattern)),
            ("gemma3n.attention.shared_kv_layers", V::U32(10)),
            ("gemma3n.altup.num_inputs", V::U32(4)),
        ]);
        let e = MemoryEstimate::from_gguf(&gemma3n, &m).unwrap();
        assert_eq!((e.kv_layers, e.swa_layers), (20, 16), "llama.cpp: 4 full layers + 16 SWA layers");
        assert_eq!(mib(e.kv_bytes), 64.0, "llama.cpp: 32 MiB (4096 cells) + 32 MiB (1024 cells)");
        assert_eq!(mib(e.swa_kv_bytes), 32.0);
        assert_eq!(mib(e.compute_bytes), 128.0);

        let mut q8 = m.clone();
        q8.kv_cache_type = "q8_0".into();
        let e8 = MemoryEstimate::from_gguf(&qwen3_4b, &q8).unwrap();
        assert_eq!(mib(e8.kv_bytes), 306.0, "q8_0 is 34 bytes per 32 values");
    }

    /// The same numbers from the real files, when they are on this PC (skipped otherwise).
    #[test]
    fn the_real_model_files_give_the_same_estimates() {
        let m = ModelConfig::default();
        let dir = std::path::Path::new(r"V:\jitmo-sandbox\models");
        let cases = [
            ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 576.0, 0.0, 1600.0),
            ("Qwen3.5-0.8B-Q4_K_M.gguf", 48.0, 19.27, 1088.0),
            ("gemma-3n-E2B-it-Q4_K_M.gguf", 64.0, 0.0, 1216.0),
        ];
        for (file, kv, rs, total) in cases {
            let path = dir.join(file);
            if !path.is_file() {
                eprintln!("skipped: {} not found", path.display());
                continue;
            }
            let e = MemoryEstimate::from_gguf(&Gguf::read(&path).unwrap(), &m).unwrap();
            assert_eq!(mib(e.kv_bytes), kv, "{file}");
            assert!((mib(e.recurrent_bytes) - rs).abs() < 0.01, "{file}");
            assert_eq!(mib(e.total_bytes), total, "{file}");
            assert!(e.n_vocab > 100_000, "{file}");
        }
    }

    #[test]
    fn the_cap_setting_picks_the_source() {
        let e = MemoryEstimate { total_bytes: 1600 * MB, ..Default::default() };
        let p = MemPlan::decide(MemoryCap::Auto, Some(e.clone()));
        assert_eq!((p.cap_mb(), p.source), (1600, "auto"));
        assert_eq!(MemPlan::decide(MemoryCap::Mb(900), Some(e.clone())).cap_mb(), 900);
        assert_eq!(MemPlan::decide(MemoryCap::Mb(0), Some(e)).cap_bytes, None);
        assert_eq!(MemPlan::decide(MemoryCap::Auto, None).source, "fallback");
    }

    #[test]
    fn memory_pressure_waits_for_the_generation_unless_critical() {
        let cfg = Config::default(); // pressure 1024, critical 500
        assert_eq!(memory_action(&cfg, 4000 * MB, false, 0), MemoryAction::None);
        assert_eq!(memory_action(&cfg, 800 * MB, false, 1), MemoryAction::None, "one low check is not enough");
        assert!(matches!(memory_action(&cfg, 800 * MB, false, 2), MemoryAction::UnloadWhenFree(_)));
        assert!(matches!(memory_action(&cfg, 499 * MB, false, 1), MemoryAction::UnloadNow(_)), "critical acts on one check");
        let mut off = Config::default();
        off.budget.critical_free_mb = 0;
        assert!(matches!(memory_action(&off, 10 * MB, false, 2), MemoryAction::UnloadWhenFree(_)));
    }
}
