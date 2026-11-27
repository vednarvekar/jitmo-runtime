//! RAM budget: each child's memory cap, what a loaded model is expected to take, is the
//! system short of memory, and how many threads.

use crate::config::{BudgetConfig, Config, MemoryCap, ModelConfig};
use crate::gguf::Gguf;
use crate::win::{self, LowMemorySignal};
use serde::Serialize;

const MB: u64 = 1024 * 1024;
/// Private memory assumed when there is no cap and no estimate (KV cache, buffers, overhead).
const UNCAPPED_PRIVATE: u64 = 1024 * MB;
/// Cap used when `memory_cap_mb = "auto"` but the GGUF metadata cannot be read.
const FALLBACK_CAP: u64 = 1536 * MB;
/// Fixed part of the automatic cap: tokenizer, server state, the graph, heap slack. Measured
/// (private bytes after load minus KV, recurrent state and compute buffers, 21 Nov 2026):
/// Qwen3-4B 128 MiB, Gemma 3n E2B 180 MiB, Qwen3.5-0.8B 517 MiB. 640 MiB covers all three
/// with room for the growth seen during generation.
pub const CAP_OVERHEAD: u64 = 640 * MB;
/// Fixed part of the load estimate's overhead (tokenizer, server state, heap). The load
/// estimate is what the free-RAM check uses; unlike the cap it is not a hard limit, so it uses
/// a smaller overhead. Measured private bytes after a 128-token answer, minus KV, state and
/// compute buffers (22 Nov 2026): router 270M about 96 MiB, Gemma 3n E2B about 180 MiB,
/// Qwen3-4B about 128 MiB (21 Nov 2026).
pub const LOAD_OVERHEAD_BASE: u64 = 96 * MB;
/// Size-scaled part of the load overhead: this share of the weights (tensor bytes in the file).
pub const LOAD_OVERHEAD_PCT: u64 = 5;
/// Architectures whose weights end up in private memory even with mmap and `--no-repack`.
/// Measured on Qwen3.5-0.8B (22 Nov 2026): working set 714 MB, private 656 MB, so only about
/// 58 MB of the 507 MB file stayed mapped. Their weights count as private, not mapped.
pub const PRIVATE_WEIGHTS_ARCHS: &[&str] = &["qwen35"];
/// llama-server's default physical batch (`-ub`); sliding-window KV is sized to window + this.
const N_UBATCH: u64 = 512;
/// Tensors that stay on disk: llama.cpp reads only the rows of the tokens it sees. Gemma 3n's
/// Per-Layer Embeddings are about half of its file (results/02-gemma3n.md).
pub const ON_DISK_TENSORS: &[&str] = &["per_layer_token_embd"];

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
    /// `cache_ram_mb`.
    pub cache_ram_bytes: u64,
    /// `ctx_checkpoints` x (SWA KV + recurrent state), only for SWA/hybrid models.
    pub checkpoint_bytes: u64,
    pub overhead_bytes: u64,
    /// Sum of all parts, rounded up to 64 MiB: the automatic memory cap.
    pub total_bytes: u64,
    /// Tensor bytes in the file (the weights).
    pub weights_bytes: u64,
    /// Overhead used by the load estimate: `LOAD_OVERHEAD_BASE` + `LOAD_OVERHEAD_PCT` % of the weights.
    pub load_overhead_bytes: u64,
    /// Private part of the load estimate: KV + state + compute + prompt cache + checkpoints +
    /// load overhead, rounded up to 16 MiB. The prompt cache is counted in full: it fills with
    /// use (Qwen3.5-0.8B: private 656 MB after one answer, 849 MB peak after the smoke test).
    pub load_private_bytes: u64,
    /// The weights end up in private memory (`PRIVATE_WEIGHTS_ARCHS`).
    pub weights_private: bool,
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
        est.compute_bytes = (16.0 * m.ubatch() as f64 * (n_ff + n_embd * altup)) as u64;
        est.cache_ram_bytes = m.cache_ram_mb * MB;
        if est.swa_kv_bytes > 0 || est.recurrent_bytes > 0 {
            est.checkpoint_bytes = m.ctx_checkpoints as u64 * (est.swa_kv_bytes + est.recurrent_bytes);
        }
        est.overhead_bytes = CAP_OVERHEAD;
        let sum = est.kv_bytes + est.recurrent_bytes + est.compute_bytes + est.cache_ram_bytes + est.checkpoint_bytes + est.overhead_bytes;
        est.total_bytes = sum.div_ceil(64 * MB) * 64 * MB;

        est.weights_bytes = g.tensors.iter().map(|t| t.1).sum();
        est.load_overhead_bytes = LOAD_OVERHEAD_BASE + est.weights_bytes * LOAD_OVERHEAD_PCT / 100;
        let load = est.kv_bytes + est.recurrent_bytes + est.compute_bytes + est.cache_ram_bytes + est.checkpoint_bytes + est.load_overhead_bytes;
        est.load_private_bytes = load.div_ceil(16 * MB) * 16 * MB;
        est.weights_private = PRIVATE_WEIGHTS_ARCHS.contains(&est.arch.as_str());
        Some(est)
    }
}

/// The memory plan of one model, decided once at startup: its commit cap and its load estimate.
#[derive(Debug, Clone, Serialize)]
pub struct MemPlan {
    /// Commit cap on the child in bytes; None = no cap.
    pub cap_bytes: Option<u64>,
    /// `auto`, `config` (explicit number), `none` (0), or `fallback` (auto, but the GGUF could not be read).
    pub source: &'static str,
    pub estimate: Option<MemoryEstimate>,
    pub load: LoadEstimate,
}

/// RAM the model is expected to take once loaded: what the free-RAM check and the pool budget use.
///
/// `total = private + mapped (+ mapped again without --no-repack)`, where
/// - `private` = KV cache + recurrent state + compute buffers + prompt cache + checkpoints + a
///   small, size-scaled overhead (`MemoryEstimate::load_private_bytes`). A configured cap smaller
///   than that is used instead (the child cannot commit more). Without GGUF metadata: the
///   configured or fallback cap, or 1024 MiB if there is no cap.
/// - `mapped` = the part of the memory-mapped file expected to be resident: the whole file,
///   minus tensors llama.cpp leaves on disk (`ON_DISK_TENSORS`). For `PRIVATE_WEIGHTS_ARCHS`
///   it is 0 and the same bytes count as private instead.
///
/// v0.3: the private part no longer uses the cap's 640 MiB overhead, which made the estimate
/// 1.7 to 3.8 times the real use (results/06-low-ram.md).
///
/// With repacking on, the CPU backend copies the weights into private memory too, so they
/// count twice. A number in `load_estimate_mb` replaces the whole estimate.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct LoadEstimate {
    pub file_bytes: u64,
    pub private_bytes: u64,
    pub mapped_bytes: u64,
    /// (tensor prefix, bytes) left out of `mapped_bytes`.
    pub on_disk: Vec<(String, u64)>,
    pub repack_bytes: u64,
    pub total_bytes: u64,
    /// `estimate` or `config` (load_estimate_mb).
    pub source: &'static str,
}

impl MemPlan {
    pub fn new(m: &ModelConfig, b: &BudgetConfig) -> MemPlan {
        let file_bytes = std::fs::metadata(&m.path).map_or(0, |f| f.len());
        let gguf = match Gguf::read(&m.path) {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::warn!("{e:#}");
                None
            }
        };
        let estimate = gguf.as_ref().and_then(|g| {
            let e = MemoryEstimate::from_gguf(g, m);
            if e.is_none() {
                tracing::warn!("model '{}': GGUF metadata lacks block_count/embedding_length; cannot estimate its memory", m.name);
            }
            e
        });
        let on_disk: Vec<(String, u64)> = gguf
            .as_ref()
            .map(|g| ON_DISK_TENSORS.iter().map(|p| (p.to_string(), g.tensor_bytes(p))).filter(|(_, n)| *n > 0).collect())
            .unwrap_or_default();
        let override_mb = m.load_estimate_mb.unwrap_or(b.load_estimate_mb);
        let mut plan = MemPlan::decide(m.memory_cap_mb.unwrap_or(b.memory_cap_mb), estimate, file_bytes, on_disk, m.no_repack, override_mb);
        if m.gpu_layers.as_deref().is_some_and(|n| n != "0") {
            plan.add_gpu_offload();
        }
        plan
    }

    /// GPU offload (`gpu_layers`): on an integrated GPU the device buffers are system RAM that
    /// counts as the process's private memory (measured 28 Nov: Gemma 3 270M on Vulkan, private
    /// 150 -> 766 MB; Gemma 3n under the plain auto cap failed to load). Count the weights once
    /// more as private, in the auto cap and in the load estimate. Mapped pages stay counted too:
    /// the file is still mapped while the weights are uploaded.
    pub fn add_gpu_offload(&mut self) {
        let w = self.load.mapped_bytes.max(self.estimate.as_ref().map_or(0, |e| e.weights_bytes));
        if self.source == "auto" {
            self.cap_bytes = self.cap_bytes.map(|c| c + w);
        }
        if self.load.source == "estimate" {
            self.load.private_bytes += w;
            self.load.total_bytes += w;
        }
    }

    pub fn decide(
        setting: MemoryCap,
        estimate: Option<MemoryEstimate>,
        file_bytes: u64,
        on_disk: Vec<(String, u64)>,
        no_repack: bool,
        load_estimate_mb: u64,
    ) -> MemPlan {
        let (cap_bytes, source) = match (setting, &estimate) {
            (MemoryCap::Mb(0), _) => (None, "none"),
            (MemoryCap::Mb(n), _) => (Some(n * MB), "config"),
            (MemoryCap::Auto, Some(e)) => (Some(e.total_bytes), "auto"),
            (MemoryCap::Auto, None) => (Some(FALLBACK_CAP), "fallback"),
        };
        let excluded: u64 = on_disk.iter().map(|(_, n)| n).sum();
        let file_mapped = file_bytes.saturating_sub(excluded);
        let (private_bytes, mapped_bytes) = match &estimate {
            Some(e) => {
                let (p, mapped) = if e.weights_private { (e.load_private_bytes + file_mapped, 0) } else { (e.load_private_bytes, file_mapped) };
                let p = match (source, cap_bytes) {
                    ("config", Some(c)) => p.min(c),
                    _ => p,
                };
                (p, mapped)
            }
            None if source == "none" => (UNCAPPED_PRIVATE, file_mapped),
            None => (cap_bytes.unwrap_or(UNCAPPED_PRIVATE), file_mapped),
        };
        let repack_bytes = if no_repack { 0 } else { file_mapped };
        let mut load = LoadEstimate {
            file_bytes,
            private_bytes,
            mapped_bytes,
            on_disk,
            repack_bytes,
            total_bytes: private_bytes + mapped_bytes + repack_bytes,
            source: "estimate",
        };
        if load_estimate_mb > 0 {
            load.total_bytes = load_estimate_mb * MB;
            load.source = "config";
        }
        MemPlan { cap_bytes, source, estimate, load }
    }

    pub fn cap_mb(&self) -> u64 {
        self.cap_bytes.unwrap_or(0) / MB
    }

    pub fn load_bytes(&self) -> u64 {
        self.load.total_bytes
    }
}

/// Ok if loading `need` bytes would leave at least `reserve_free_mb` free; otherwise the reason.
pub fn check_free_ram(need: u64, reserve_free_mb: u64) -> Result<(), String> {
    let Some(mem) = win::memory_status() else {
        tracing::warn!("cannot read free RAM; starting the model without the budget check");
        return Ok(());
    };
    let reserve = reserve_free_mb * MB;
    if mem.avail < need + reserve {
        return Err(format!(
            "{} MB free, the model needs about {} MB and {} MB must stay free (budget.reserve_free_mb)",
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
    /// Short of memory, but not critically: finish running generations, then unload.
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

/// Under memory pressure, has a running generation had its time? True once it waited
/// `pressure_finish_max_ms` (0 = no limit) and free RAM is still short.
pub fn pressure_wait_over(waited: Option<std::time::Duration>, pressure_finish_max_ms: u64, still_low: bool) -> bool {
    pressure_finish_max_ms > 0 && still_low && waited.is_some_and(|w| w.as_millis() >= pressure_finish_max_ms as u128)
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

pub fn threads(m: &ModelConfig) -> u32 {
    if m.threads > 0 {
        m.threads
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
        let m = ModelConfig { threads: 6, ..Default::default() };
        assert_eq!(threads(&m), 6);
    }

    #[test]
    fn an_impossible_reserve_refuses_the_load() {
        let err = check_free_ram(MB, 1 << 30).unwrap_err(); // 1 PB reserve
        assert!(err.contains("must stay free"), "{err}");
        assert!(check_free_ram(MB, 0).is_ok());
    }

    #[test]
    fn the_load_estimate_leaves_on_disk_tensors_out() {
        let e = MemoryEstimate { total_bytes: 1216 * MB, load_private_bytes: 560 * MB, ..Default::default() };
        // Gemma 3n E2B: 2887 MiB file, 1440 MiB of Per-Layer Embeddings.
        let ple = vec![("per_layer_token_embd".to_string(), 1440 * MB)];
        let p = MemPlan::decide(MemoryCap::Auto, Some(e.clone()), 2887 * MB, ple.clone(), true, 0);
        assert_eq!(p.load.mapped_bytes, 1447 * MB);
        assert_eq!(p.load_bytes(), (560 + 1447) * MB, "v0.2 counted the 1216 MiB cap as private");
        assert_eq!(p.cap_mb(), 1216, "the cap itself is unchanged");
        // GPU offload: the weights count once more, in the cap and the estimate.
        let mut g = p.clone();
        g.add_gpu_offload();
        assert_eq!((g.cap_mb(), g.load_bytes()), (1216 + 1447, (560 + 2 * 1447) * MB));
        // Repacking copies the mapped weights into private memory.
        let r = MemPlan::decide(MemoryCap::Auto, Some(e.clone()), 2887 * MB, ple, false, 0);
        assert_eq!(r.load_bytes(), (560 + 2 * 1447) * MB);
        // A configured cap below the estimate bounds the private part; above it, it does not.
        let c = MemPlan::decide(MemoryCap::Mb(400), Some(e.clone()), 100 * MB, vec![], true, 0);
        assert_eq!(c.load.private_bytes, 400 * MB);
        let c = MemPlan::decide(MemoryCap::Mb(900), Some(e.clone()), 100 * MB, vec![], true, 0);
        assert_eq!(c.load.private_bytes, 560 * MB);
        // No estimate: the configured cap, or 1024 MiB with no cap.
        let c = MemPlan::decide(MemoryCap::Mb(900), None, 100 * MB, vec![], true, 0);
        assert_eq!(c.load.private_bytes, 900 * MB);
        let n = MemPlan::decide(MemoryCap::Mb(0), None, 100 * MB, vec![], true, 0);
        assert_eq!(n.load_bytes(), 1124 * MB);
        // Weights that end up private are counted once, as private.
        let w = MemoryEstimate { weights_private: true, ..e.clone() };
        let q = MemPlan::decide(MemoryCap::Auto, Some(w), 507 * MB, vec![], true, 0);
        assert_eq!((q.load.private_bytes, q.load.mapped_bytes, q.load_bytes()), ((560 + 507) * MB, 0, (560 + 507) * MB));
        let o = MemPlan::decide(MemoryCap::Auto, Some(e), 100 * MB, vec![], true, 777);
        assert_eq!((o.load_bytes(), o.load.source), (777 * MB, "config"));
    }

    /// Real use after a 128-token answer (results/06-low-ram.md, 22 Nov 2026), in MB:
    /// (file, private, working set). The load estimate must never be below either, and should
    /// be well under the v0.2 estimate. The estimates are from the GGUF metadata of the real
    /// files (the test is skipped for a file that is not on this PC).
    #[test]
    fn the_load_estimate_covers_the_measured_use() {
        let m = ModelConfig::default();
        let dir = std::path::Path::new(r"V:\jitmo-sandbox\models");
        // (file, real private MB, real working set MB, v0.2 estimate MB, max estimate / working set)
        // The 0.8B is there twice: after one answer, and its peak after the whole smoke test
        // (several prompts in the prompt cache, 22 Nov 2026).
        let cases = [
            ("gemma-3-270m-it-router-Q4_K_M.gguf", 146.0, 334.0, 1265.0, 2.2),
            ("Qwen3.5-0.8B-Q4_K_M.gguf", 656.0, 714.0, 1596.0, 1.6),
            ("Qwen3.5-0.8B-Q4_K_M.gguf", 849.0, 909.0, 1596.0, 1.3),
            ("gemma-3n-E2B-it-Q4_K_M.gguf", 371.0, 1579.0, 2663.0, 1.5),
        ];
        let b = BudgetConfig::default();
        for (file, private, ws, old, max_ratio) in cases {
            let path = dir.join(file);
            if !path.is_file() {
                eprintln!("skipped: {} not found", path.display());
                continue;
            }
            let p = MemPlan::new(&ModelConfig { path, ..m.clone() }, &b);
            let total = mib(p.load_bytes());
            let priv_est = mib(p.load.private_bytes);
            eprintln!("{file}: load estimate {total:.0} MiB (private {priv_est:.0}), real {ws} / {private} MB, v0.2 {old}");
            // MiB >= MB, so comparing MiB to MB is on the safe side.
            assert!(priv_est >= private, "{file}: private estimate {priv_est} < real {private}");
            assert!(total >= ws, "{file}: estimate {total} < real working set {ws}");
            assert!(total <= ws * max_ratio, "{file}: estimate {total} is over {max_ratio}x the real {ws}");
            assert!(total < old * 0.9, "{file}: estimate {total} is not below v0.2's {old}");
        }
        // Qwen3-4B still needs about its real amount: peak working set 3077 MB, peak private
        // 841 MB through the runtime, repack off (results/02-gemma3n.md).
        let path = dir.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf");
        if path.is_file() {
            let p = MemPlan::new(&ModelConfig { path, ..m }, &b);
            let total = mib(p.load_bytes());
            eprintln!("Qwen3-4B: load estimate {total:.0} MiB (private {:.0})", mib(p.load.private_bytes));
            assert!(total >= 3077.0 && total <= 3077.0 * 1.2, "Qwen3-4B: {total}");
            assert!(total < 3981.0 * 0.9, "below v0.2's 3981");
            assert!(mib(p.load.private_bytes) >= 841.0, "Qwen3-4B private: {}", mib(p.load.private_bytes));
        }
    }

    /// The load overhead grows with the weights and the formula adds up its parts.
    #[test]
    fn the_load_private_part_is_kv_compute_and_a_scaled_overhead() {
        let m = ModelConfig::default();
        let kvs = [
            ("general.architecture", V::Str("qwen3")),
            ("qwen3.block_count", V::U32(36)),
            ("qwen3.embedding_length", V::U32(2560)),
            ("qwen3.feed_forward_length", V::U32(9728)),
            ("qwen3.attention.head_count", V::U32(32)),
            ("qwen3.attention.head_count_kv", V::U32(8)),
        ];
        let small = Gguf::parse(&mut std::io::Cursor::new(crate::gguf::testfile::build_with_tensors(&kvs, &[("w", MB)]))).unwrap();
        let big = Gguf::parse(&mut std::io::Cursor::new(crate::gguf::testfile::build_with_tensors(&kvs, &[("w", 21 * MB)]))).unwrap();
        let (s, b) = (MemoryEstimate::from_gguf(&small, &m).unwrap(), MemoryEstimate::from_gguf(&big, &m).unwrap());
        assert_eq!((s.weights_bytes, b.weights_bytes), (MB, 21 * MB));
        assert_eq!(s.load_overhead_bytes, LOAD_OVERHEAD_BASE + MB * LOAD_OVERHEAD_PCT / 100);
        assert!(b.load_overhead_bytes > s.load_overhead_bytes + MB / 2, "grows with the weights");
        let parts = s.kv_bytes + s.recurrent_bytes + s.compute_bytes + s.cache_ram_bytes + s.checkpoint_bytes + s.load_overhead_bytes;
        assert!(s.load_private_bytes >= parts && s.load_private_bytes < parts + 16 * MB);
        assert!(s.load_private_bytes < s.total_bytes, "no 640 MiB overhead");
        assert!(!s.weights_private);
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
            ("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 576.0, 0.0, 1600.0, 0.0),
            ("Qwen3.5-0.8B-Q4_K_M.gguf", 48.0, 19.27, 1088.0, 0.0),
            ("gemma-3n-E2B-it-Q4_K_M.gguf", 64.0, 0.0, 1216.0, 1440.0),
        ];
        for (file, kv, rs, total, ple) in cases {
            let path = dir.join(file);
            if !path.is_file() {
                eprintln!("skipped: {} not found", path.display());
                continue;
            }
            let g = Gguf::read(&path).unwrap();
            let e = MemoryEstimate::from_gguf(&g, &m).unwrap();
            assert_eq!(mib(e.kv_bytes), kv, "{file}");
            assert!((mib(e.recurrent_bytes) - rs).abs() < 0.01, "{file}");
            assert_eq!(mib(e.total_bytes), total, "{file}");
            assert!(e.n_vocab > 100_000, "{file}");
            // The tensor sizes add up to the file, and Gemma 3n's PLE table is 1440 MiB.
            let data: u64 = g.tensors.iter().map(|t| t.1).sum();
            let len = std::fs::metadata(&path).unwrap().len();
            assert!(data < len && len - data < 16 * MB, "{file}: tensors {data} of {len} bytes");
            assert!((mib(g.tensor_bytes("per_layer_token_embd")) - ple).abs() < 1.0, "{file}");
        }
    }

    #[test]
    fn the_cap_setting_picks_the_source() {
        let e = MemoryEstimate { total_bytes: 1600 * MB, ..Default::default() };
        let d = |c, e| MemPlan::decide(c, e, 0, vec![], true, 0);
        let p = d(MemoryCap::Auto, Some(e.clone()));
        assert_eq!((p.cap_mb(), p.source), (1600, "auto"));
        assert_eq!(d(MemoryCap::Mb(900), Some(e.clone())).cap_mb(), 900);
        assert_eq!(d(MemoryCap::Mb(0), Some(e)).cap_bytes, None);
        assert_eq!(d(MemoryCap::Auto, None).source, "fallback");
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

    #[test]
    fn finish_then_unload_has_a_time_limit() {
        use std::time::Duration;
        let max = Config::default().budget.pressure_finish_max_ms;
        assert_eq!(max, 15_000);
        let s = |n| Some(Duration::from_secs(n));
        assert!(!pressure_wait_over(s(14), max, true), "still within the limit");
        assert!(pressure_wait_over(s(15), max, true));
        assert!(pressure_wait_over(s(43), max, true), "results/06-low-ram.md: it waited 43 s");
        assert!(!pressure_wait_over(s(43), max, false), "RAM came back: let the generation finish");
        assert!(!pressure_wait_over(s(600), 0, true), "0 = no limit (v0.2)");
        assert!(!pressure_wait_over(None, max, true), "no pressure wait in progress");
    }
}
