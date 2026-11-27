//! The one TOML config file: API server, models (each a llama-server child), RAM budget,
//! pacer, chains. Every field has a default; relative paths are resolved against the config
//! file's folder.
//!
//! Models: either the v0 form, one `[model]` table, or several `[[models]]` entries. After
//! loading, `Config::models` always holds the list (the `[model]` table becomes its only entry).

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const DEFAULT_CONFIG_FILE: &str = "jitmo-runtime.toml";

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    /// v0 form: one model. Moved into `models` by `normalize`.
    pub model: Option<ModelConfig>,
    pub models: Vec<ModelConfig>,
    pub budget: BudgetConfig,
    pub pacer: PacerConfig,
    pub chains: Vec<ChainConfig>,
}

/// The runtime's own HTTP API.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    /// 0 = pick a free port (printed as `JITMO_RUNTIME_LISTENING port=<n>`).
    pub port: u16,
    /// Requests allowed to wait for a generation slot. More are refused with 503.
    pub max_queue: usize,
    /// Generations that may run at the same time across all models. The models share the
    /// memory bandwidth, so 1 is the default.
    pub max_concurrent: usize,
    /// Web page origins (exact `Origin` header values) allowed to call the API. Empty = none.
    pub allowed_origins: Vec<String>,
    /// A waiting interactive request stops a running background generation (which gets
    /// `503 preempted` and may retry).
    pub preempt_background: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "127.0.0.1".into(),
            port: 8090,
            max_queue: 8,
            max_concurrent: 1,
            allowed_origins: Vec::new(),
            preempt_background: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    #[default]
    Chat,
    Embedding,
}

impl ModelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelKind::Chat => "chat",
            ModelKind::Embedding => "embedding",
        }
    }
}

/// One model and how its llama-server child is started.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Model id clients send in `"model"`.
    pub name: String,
    /// `chat` (default) or `embedding` (llama-server runs with `--embeddings`).
    pub kind: ModelKind,
    pub path: PathBuf,
    pub llama_server: PathBuf,
    /// `llama-server` (default: a llama-server child) or `inproc` (our own worker process with
    /// llama.dll inside: docs/08-inproc-engine.md). Chat models only, no LoRA yet.
    pub engine: String,
    /// `inproc`: the llama.cpp build folder with llama.dll (llama.cpp b12606, CPU or Vulkan).
    /// Unset = the folder of `llama_server`.
    pub llama_dir: Option<PathBuf>,
    /// Never evicted to make room for another model, and never unloaded for idleness.
    /// Memory pressure still unloads it.
    pub pinned: bool,
    /// Private port of the child. 0 = pick a free port at each start.
    pub port: u16,
    pub ctx_size: u32,
    /// 0 = auto: physical cores - 1, but at most half of the logical processors (min 1).
    pub threads: u32,
    /// KV cache type for K and V (`f16`, `q8_0`, ...). Quantized V needs flash attention on.
    pub kv_cache_type: String,
    /// `on`, `off` or `auto`.
    pub flash_attn: String,
    /// false = pass `-lm none` (weights copied into private memory, counted by the cap).
    pub mmap: bool,
    /// true = pass `--no-repack`: no repacked private copy of the weights. Measured on Qwen3-4B:
    /// private bytes 2211 -> 524 MB, working set 4201 -> 2518 MB, about 20% slower generation.
    pub no_repack: bool,
    /// Run the child at below-normal CPU priority.
    pub low_priority: bool,
    /// llama-server prompt cache in RAM (`--cache-ram`), in MiB. 0 = off. It grows the child's
    /// private memory, so the automatic memory cap includes it. llama-server's own default is 8192.
    pub cache_ram_mb: u64,
    /// Context checkpoints per slot (`--ctx-checkpoints`) for SWA and hybrid models. Each one
    /// holds a copy of the SWA/recurrent state, so the automatic memory cap includes them.
    /// llama-server's own default is 32.
    pub ctx_checkpoints: u32,
    /// Embedding models only: `none`, `mean`, `cls`, `last` or `rank`. None = the model's default.
    pub pooling: Option<String>,
    /// LoRA adapters loaded with this model (scale 0) for requests to pick by id
    /// (`"lora": [{"id": 0, "scale": 1.0}]`). Adapters used by chain steps on this model are
    /// added after these.
    pub adapters: Vec<PathBuf>,
    /// Sent as `chat_template_kwargs` on every request that does not set its own.
    pub chat_template_kwargs: Option<Value>,
    /// GPU offload (`-ngl`): `"0"` = CPU only, a layer count, `"auto"` or `"all"`. Needs a GPU
    /// build of llama-server in `llama_server` (for example Vulkan for Iris Xe). Unset = the
    /// llama-server default (a GPU build may then offload everything).
    pub gpu_layers: Option<String>,
    /// Device(s) for the offload (`--device`, for example `"Vulkan0"`). Unset = llama-server's choice.
    pub device: Option<String>,
    /// Extra llama-server arguments, appended as-is.
    pub extra_args: Vec<String>,
    /// File for llama-server's own log (truncated at each start). Empty = discard.
    /// Unset = `llama-server.log` with one model, `llama-server-<name>.log` with several.
    pub log_file: Option<PathBuf>,
    /// Give up if the model is not ready after this long.
    pub load_timeout_secs: u64,
    /// Per-model overrides of the `[budget]` values of the same name.
    pub memory_cap_mb: Option<MemoryCap>,
    pub load_estimate_mb: Option<u64>,
    pub idle_unload_secs: Option<u64>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            name: "jitmo-base".into(),
            kind: ModelKind::Chat,
            path: r"V:\jitmo-sandbox\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf".into(),
            llama_server: r"V:\jitmo-sandbox\llama\llama-server.exe".into(),
            engine: "llama-server".into(),
            llama_dir: None,
            pinned: false,
            port: 0,
            ctx_size: 4096,
            threads: 0,
            kv_cache_type: "f16".into(),
            flash_attn: "on".into(),
            mmap: true,
            no_repack: true,
            low_priority: true,
            cache_ram_mb: 256,
            ctx_checkpoints: 4,
            pooling: None,
            adapters: Vec::new(),
            chat_template_kwargs: None,
            gpu_layers: None,
            device: None,
            extra_args: Vec::new(),
            log_file: None,
            load_timeout_secs: 180,
            memory_cap_mb: None,
            load_estimate_mb: None,
            idle_unload_secs: None,
        }
    }
}

impl ModelConfig {
    /// Physical batch size. Embedding models read the whole input in one batch, so it is the ctx.
    pub fn ubatch(&self) -> u32 {
        match self.kind {
            ModelKind::Embedding => self.ctx_size,
            ModelKind::Chat => 512,
        }
    }

    pub fn inproc(&self) -> bool {
        self.engine == "inproc"
    }

    /// The llama.cpp build folder for `engine = "inproc"`.
    pub fn lib_dir(&self) -> PathBuf {
        self.llama_dir.clone().unwrap_or_else(|| self.llama_server.parent().map(PathBuf::from).unwrap_or_default())
    }

    pub fn log_path(&self) -> PathBuf {
        self.log_file.clone().unwrap_or_default()
    }
}

/// `budget.memory_cap_mb`: `"auto"` (computed from the model and ctx), a number of MiB, or 0 (no cap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCap {
    Auto,
    Mb(u64),
}

impl<'de> Deserialize<'de> for MemoryCap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Num(u64),
            Text(String),
        }
        match Raw::deserialize(d)? {
            Raw::Num(n) => Ok(MemoryCap::Mb(n)),
            Raw::Text(t) if t == "auto" => Ok(MemoryCap::Auto),
            Raw::Text(t) => Err(serde::de::Error::custom(format!("memory_cap_mb must be \"auto\" or a number of MiB, not {t:?}"))),
        }
    }
}

/// RAM budget and unload rules.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetConfig {
    /// Job Object per-process commit (private bytes) limit for each child.
    /// `auto` = KV cache + recurrent state + compute buffers + prompt cache + checkpoints + a fixed
    /// overhead, from the GGUF metadata (see `budget::MemoryEstimate`). 0 = no cap.
    /// It does not count mmap'd weight pages.
    pub memory_cap_mb: MemoryCap,
    /// Refuse to start a model if free RAM after the load would be below this.
    pub reserve_free_mb: u64,
    /// RAM a model is expected to take when loaded. 0 = estimate it (private estimate + the
    /// part of the mapped file that will be resident; see `budget::LoadEstimate`).
    pub load_estimate_mb: u64,
    /// Total load estimate of all loaded models may not exceed this. 0 = no pool limit (only
    /// the free-RAM check). Least recently used idle, unpinned models are evicted to make room.
    pub pool_budget_mb: u64,
    /// Memory pressure: free RAM below this on two checks in a row (or Windows signals low
    /// memory). Running generations may finish first; then their models are unloaded.
    pub pressure_free_mb: u64,
    /// Free RAM below this on one check: unload every model at once, even mid-generation. 0 = never.
    pub critical_free_mb: u64,
    /// Under memory pressure, a running generation may finish for at most this long. If free RAM
    /// is still under `pressure_free_mb` then, the generation is cancelled (the client gets a
    /// `memory_pressure` error) and the model is unloaded. 0 = wait for the generation to end.
    pub pressure_finish_max_ms: u64,
    /// Unload a model after this many seconds without requests. 0 = never.
    pub idle_unload_secs: u64,
    /// How often the watchdog checks idle time, memory pressure and the children.
    pub watch_interval_ms: u64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        BudgetConfig {
            memory_cap_mb: MemoryCap::Auto,
            reserve_free_mb: 1536,
            load_estimate_mb: 0,
            pool_budget_mb: 0,
            pressure_free_mb: 1024,
            critical_free_mb: 500,
            pressure_finish_max_ms: 15_000,
            idle_unload_secs: 90,
            watch_interval_ms: 1000,
        }
    }
}

/// "Don't freeze the PC": cap the children's CPU, at a level that fits how busy the PC is.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PacerConfig {
    /// `auto` (level from the signals), `always` (always the heavy level) or `off`.
    pub mode: String,
    /// How often the pacer samples CPU and RAM.
    pub poll_ms: u64,
    /// Other processes' CPU is averaged over this long, so one short spike does not count.
    pub smooth_ms: u64,
    /// Other processes above this share of all logical processors = light level. 0 = never.
    pub light_cpu_pct: f64,
    /// Above this = heavy level. 0 = never.
    pub heavy_cpu_pct: f64,
    /// Free RAM below this = heavy level. 0 = use `budget.pressure_free_mb`.
    pub low_free_mb: u64,
    /// Caps, as a share of the CPU a child uses at full speed (its threads / logical processors).
    pub light_cap_pct: u32,
    pub heavy_cap_pct: u32,
    /// To go down a level, other processes' CPU must be this many points below the threshold.
    pub hysteresis_pct: f64,
    /// A level holds at least this long before it may go down.
    pub release_after_ms: u64,
    /// To leave heavy, other processes' CPU must be this many points below `heavy_cpu_pct`.
    pub heavy_hysteresis_pct: f64,
    /// Heavy goes down only after the signals stayed clear of it (CPU under
    /// `heavy_cpu_pct - heavy_hysteresis_pct`, RAM fine) this long in a row.
    pub heavy_release_after_ms: u64,
    /// While capped, also set the children to this priority class: `unchanged` or `idle`.
    pub busy_priority: String,
    /// Leave the `--parent-pid` process and its other descendants out of "other programs' CPU".
    pub exclude_client_cpu: bool,
    /// Pacer v1 keys, accepted and ignored (a warning is logged).
    pub input_active_ms: Option<u64>,
    pub others_cpu_busy_pct: Option<f64>,
    pub busy_cpu_pct: Option<u32>,
}

impl Default for PacerConfig {
    fn default() -> Self {
        PacerConfig {
            mode: "auto".into(),
            poll_ms: 300,
            smooth_ms: 1500,
            light_cpu_pct: 20.0,
            heavy_cpu_pct: 40.0,
            low_free_mb: 0,
            light_cap_pct: 75,
            heavy_cap_pct: 50,
            hysteresis_pct: 5.0,
            release_after_ms: 5000,
            heavy_hysteresis_pct: 10.0,
            heavy_release_after_ms: 10_000,
            busy_priority: "unchanged".into(),
            exclude_client_cpu: true,
            input_active_ms: None,
            others_cpu_busy_pct: None,
            busy_cpu_pct: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainConfig {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub steps: Vec<StepConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepConfig {
    pub name: String,
    /// Model for this step (a chat model's name). Default: the first chat model.
    pub model: Option<String>,
    /// System prompt template.
    #[serde(default)]
    pub system: String,
    /// User message template. Default: `{input}` for the first step, `{prev}` after that.
    pub input: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// JSON schema for the output: a TOML table, or a string holding JSON.
    pub json_schema: Option<Value>,
    /// GBNF grammar for the output.
    pub grammar: Option<String>,
    pub lora: Option<LoraConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraConfig {
    pub path: PathBuf,
    #[serde(default = "default_lora_scale")]
    pub scale: f64,
}

fn default_max_tokens() -> u32 {
    256
}
fn default_temperature() -> f64 {
    0.2
}
fn default_lora_scale() -> f64 {
    1.0
}

const KV_TYPES: &[&str] = &["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"];
const POOLING: &[&str] = &["none", "mean", "cls", "last", "rank"];
/// Step names that templates use for other things.
pub const RESERVED_NAMES: &[&str] = &["input", "prev"];

impl Config {
    /// Parse TOML text and check it. Paths are left as written.
    pub fn from_toml(text: &str) -> Result<Config> {
        let mut cfg: Config = toml::from_str(text).context("config is not valid")?;
        cfg.normalize()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read a config file; relative paths become relative to the file's folder.
    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read config {}", path.display()))?;
        let mut cfg = Config::from_toml(&text).with_context(|| format!("in {}", path.display()))?;
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        cfg.resolve_paths(&base);
        Ok(cfg)
    }

    /// Built-in defaults, with relative paths resolved against `base`.
    pub fn defaults(base: &Path) -> Config {
        let mut cfg = Config::from_toml("").expect("the defaults are valid");
        cfg.resolve_paths(base);
        cfg
    }

    pub fn resolve_paths(&mut self, base: &Path) {
        let fix = |p: &mut PathBuf| {
            if !p.as_os_str().is_empty() && p.is_relative() {
                *p = base.join(&*p);
            }
        };
        for m in &mut self.models {
            fix(&mut m.path);
            fix(&mut m.llama_server);
            if let Some(l) = m.log_file.as_mut() {
                fix(l);
            }
            for a in &mut m.adapters {
                fix(a);
            }
        }
        for chain in &mut self.chains {
            for step in &mut chain.steps {
                if let Some(l) = step.lora.as_mut() {
                    fix(&mut l.path);
                }
            }
        }
    }

    /// `[model]` becomes the only `[[models]]` entry (none at all = the default model); log
    /// files get their default names; a JSON schema given as a string is parsed into JSON.
    fn normalize(&mut self) -> Result<()> {
        match (self.model.take(), self.models.is_empty()) {
            (Some(_), false) => bail!("use either [model] (one model) or [[models]] (several), not both"),
            (Some(m), true) => self.models.push(m),
            (None, true) => self.models.push(ModelConfig::default()),
            (None, false) => {}
        }
        let several = self.models.len() > 1;
        for m in &mut self.models {
            if m.log_file.is_none() {
                m.log_file =
                    Some(if several { format!("llama-server-{}.log", m.name).into() } else { "llama-server.log".into() });
            }
        }
        for chain in &mut self.chains {
            for step in &mut chain.steps {
                if let Some(Value::String(s)) = &step.json_schema {
                    let parsed: Value = serde_json::from_str(s)
                        .with_context(|| format!("chain '{}' step '{}': json_schema is not valid JSON", chain.name, step.name))?;
                    step.json_schema = Some(parsed);
                }
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        let mut names = HashSet::new();
        for m in &self.models {
            let at = || format!("model '{}'", m.name);
            if m.name.is_empty() || m.name.starts_with("chain:") {
                bail!("{}: name must be set and must not start with 'chain:'", at());
            }
            if !names.insert(m.name.as_str()) {
                bail!("{}: model name used twice", at());
            }
            if !KV_TYPES.contains(&m.kv_cache_type.as_str()) {
                bail!("{}: kv_cache_type '{}' is not one of {:?}", at(), m.kv_cache_type, KV_TYPES);
            }
            if !["on", "off", "auto"].contains(&m.flash_attn.as_str()) {
                bail!("{}: flash_attn must be on, off or auto", at());
            }
            if let Some(n) = &m.gpu_layers {
                if !(n == "auto" || n == "all" || (!n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))) {
                    bail!("{}: gpu_layers must be a layer count, \"auto\" or \"all\"", at());
                }
            }
            if m.ctx_size < 256 {
                bail!("{}: ctx_size must be at least 256", at());
            }
            if let Some(p) = &m.pooling {
                if m.kind != ModelKind::Embedding {
                    bail!("{}: pooling is only for kind = \"embedding\"", at());
                }
                if !POOLING.contains(&p.as_str()) {
                    bail!("{}: pooling must be one of {:?}", at(), POOLING);
                }
            }
            if !["llama-server", "inproc"].contains(&m.engine.as_str()) {
                bail!("{}: engine must be \"llama-server\" or \"inproc\"", at());
            }
            if m.inproc() {
                if m.kind != ModelKind::Chat {
                    bail!("{}: engine = \"inproc\" is for chat models only (embeddings: use llama-server)", at());
                }
                if !m.adapters.is_empty() {
                    bail!("{}: engine = \"inproc\" has no LoRA yet", at());
                }
                if m.device.is_some() {
                    bail!("{}: engine = \"inproc\" does not take `device` yet", at());
                }
                if crate::engine::kv_type_id(&m.kv_cache_type).is_none() {
                    bail!("{}: kv_cache_type '{}' is not known to the inproc engine", at(), m.kv_cache_type);
                }
            }
            if m.kind == ModelKind::Embedding && !m.adapters.is_empty() {
                bail!("{}: LoRA adapters are for chat models", at());
            }
        }
        if self.server.max_concurrent == 0 {
            bail!("server.max_concurrent must be at least 1");
        }
        if self.budget.watch_interval_ms < 100 {
            bail!("budget.watch_interval_ms must be at least 100");
        }
        if self.budget.critical_free_mb > 0 && self.budget.critical_free_mb >= self.budget.pressure_free_mb {
            bail!("budget.critical_free_mb must be below budget.pressure_free_mb (or 0)");
        }
        let p = &self.pacer;
        if !["auto", "always", "off"].contains(&p.mode.as_str()) {
            bail!("pacer.mode must be auto, always or off");
        }
        if !["unchanged", "idle"].contains(&p.busy_priority.as_str()) {
            bail!("pacer.busy_priority must be unchanged or idle");
        }
        if !(1..=100).contains(&p.light_cap_pct) || !(1..=100).contains(&p.heavy_cap_pct) {
            bail!("pacer.light_cap_pct and pacer.heavy_cap_pct must be between 1 and 100");
        }
        if p.heavy_cap_pct > p.light_cap_pct {
            bail!("pacer.heavy_cap_pct must not be above pacer.light_cap_pct (heavy caps harder)");
        }
        if p.heavy_cpu_pct > 0.0 && p.light_cpu_pct > p.heavy_cpu_pct {
            bail!("pacer.light_cpu_pct must not be above pacer.heavy_cpu_pct");
        }
        if p.hysteresis_pct < 0.0 || p.heavy_hysteresis_pct < 0.0 {
            bail!("pacer.hysteresis_pct and pacer.heavy_hysteresis_pct must not be negative");
        }
        if p.poll_ms < 50 {
            bail!("pacer.poll_ms must be at least 50");
        }
        if p.smooth_ms < p.poll_ms {
            bail!("pacer.smooth_ms must be at least pacer.poll_ms");
        }
        let mut chain_names = HashSet::new();
        for chain in &self.chains {
            check_name(&chain.name).with_context(|| "chain name".to_string())?;
            if !chain_names.insert(chain.name.as_str()) {
                bail!("chain '{}' is defined twice", chain.name);
            }
            if chain.steps.is_empty() {
                bail!("chain '{}' has no steps", chain.name);
            }
            let all_steps: Vec<&str> = chain.steps.iter().map(|s| s.name.as_str()).collect();
            let mut earlier: Vec<&str> = Vec::new();
            for (i, step) in chain.steps.iter().enumerate() {
                let at = || format!("chain '{}' step '{}'", chain.name, step.name);
                check_name(&step.name).with_context(at)?;
                if RESERVED_NAMES.contains(&step.name.as_str()) {
                    bail!("{}: 'input' and 'prev' are reserved step names", at());
                }
                if earlier.contains(&step.name.as_str()) {
                    bail!("{}: step name used twice", at());
                }
                if step.json_schema.is_some() && step.grammar.is_some() {
                    bail!("{}: use json_schema or grammar, not both", at());
                }
                if let Some(s) = &step.json_schema {
                    if !s.is_object() {
                        bail!("{}: json_schema must be a table/object", at());
                    }
                }
                if let Some(name) = &step.model {
                    match self.models.iter().find(|m| &m.name == name) {
                        None => bail!("{}: no model named '{name}'", at()),
                        Some(m) if m.kind != ModelKind::Chat => bail!("{}: model '{name}' is not a chat model", at()),
                        _ => {}
                    }
                } else if self.default_chat().is_none() {
                    bail!("{}: there is no chat model", at());
                }
                for (field, text) in [("system", step.system.as_str()), ("input", step.input_template(i))] {
                    check_template(text, i, &earlier, &all_steps).with_context(|| format!("{}: {field}", at()))?;
                }
                earlier.push(&step.name);
            }
        }
        Ok(())
    }

    /// Check that the files the runtime needs exist (done at startup, not in unit tests).
    pub fn check_files(&self) -> Result<()> {
        for m in &self.models {
            if !m.path.is_file() {
                bail!("model '{}': file not found: {}", m.name, m.path.display());
            }
            if !m.llama_server.is_file() {
                bail!("model '{}': llama-server not found: {}", m.name, m.llama_server.display());
            }
            for path in self.adapters_for(&m.name) {
                if !path.is_file() {
                    bail!("model '{}': LoRA adapter not found: {}", m.name, path.display());
                }
            }
        }
        Ok(())
    }

    /// The first chat model: what a request without `model` (and a chain step without one) uses.
    pub fn default_chat(&self) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.kind == ModelKind::Chat)
    }

    /// The model name a chain step runs on.
    pub fn step_model<'a>(&'a self, step: &'a StepConfig) -> &'a str {
        step.model.as_deref().or(self.default_chat().map(|m| m.name.as_str())).unwrap_or("")
    }

    /// LoRA adapter files of one model: its own `adapters`, then those of chain steps that run on
    /// it, without repeats. The index in this list is the adapter id llama-server uses.
    pub fn adapters_for(&self, model: &str) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        let Some(m) = self.models.iter().find(|m| m.name == model) else { return out };
        for a in &m.adapters {
            if !out.contains(a) {
                out.push(a.clone());
            }
        }
        for step in self.chains.iter().flat_map(|c| &c.steps) {
            if let Some(l) = &step.lora {
                if self.step_model(step) == model && !out.contains(&l.path) {
                    out.push(l.path.clone());
                }
            }
        }
        out
    }

    #[cfg(test)]
    pub fn chain(&self, name: &str) -> Option<&ChainConfig> {
        self.chains.iter().find(|c| c.name == name)
    }
}

/// A template may use `{input}`, `{prev}` (not in the first step), earlier steps' outputs and
/// request variables (any other name). Naming this step or a later one is an error.
pub fn check_template(text: &str, index: usize, earlier: &[&str], all_steps: &[&str]) -> Result<()> {
    let t = crate::chain::Template::parse(text);
    for var in t.var_names() {
        if var == "prev" && index == 0 {
            bail!("uses {{prev}}, but the first step has no previous step");
        }
        if all_steps.contains(&var) && !earlier.contains(&var) {
            bail!("uses {{{var}}}, but step '{var}' has not run yet here (only earlier steps can be used)");
        }
    }
    Ok(())
}

impl StepConfig {
    pub fn input_template(&self, index: usize) -> &str {
        match &self.input {
            Some(s) => s,
            None if index == 0 => "{input}",
            None => "{prev}",
        }
    }
}

fn check_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if !ok {
        bail!("'{name}' is not a valid name (letters, digits, _ and -, starting with a letter or _)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_gives_the_defaults() {
        let cfg = Config::from_toml("").unwrap();
        assert_eq!(cfg.server.port, 8090);
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!((cfg.server.max_queue, cfg.server.max_concurrent), (8, 1));
        assert_eq!(cfg.models.len(), 1);
        let m = &cfg.models[0];
        assert_eq!(m.name, "jitmo-base");
        assert_eq!(m.kind, ModelKind::Chat);
        assert_eq!(m.ctx_size, 4096);
        assert_eq!(m.kv_cache_type, "f16");
        assert!(m.no_repack && m.mmap && !m.pinned);
        assert_eq!(m.log_file, Some(PathBuf::from("llama-server.log")));
        assert_eq!(cfg.budget.memory_cap_mb, MemoryCap::Auto);
        assert_eq!((cfg.budget.reserve_free_mb, cfg.budget.pool_budget_mb), (1536, 0));
        assert_eq!((cfg.budget.pressure_free_mb, cfg.budget.critical_free_mb), (1024, 500));
        assert_eq!((m.cache_ram_mb, m.ctx_checkpoints), (256, 4));
        assert_eq!(cfg.pacer.mode, "auto");
        assert_eq!((cfg.pacer.light_cpu_pct, cfg.pacer.heavy_cpu_pct), (20.0, 40.0));
        assert_eq!((cfg.pacer.heavy_hysteresis_pct, cfg.pacer.heavy_release_after_ms), (10.0, 10_000));
        assert_eq!(cfg.budget.pressure_finish_max_ms, 15_000);
        assert_eq!((cfg.pacer.light_cap_pct, cfg.pacer.heavy_cap_pct, cfg.pacer.smooth_ms), (75, 50, 1500));
        // v1 keys still parse (ignored with a warning).
        assert!(Config::from_toml("[pacer]\ninput_active_ms = 5000\nbusy_cpu_pct = 50\nothers_cpu_busy_pct = 10.0").is_ok());
        assert_eq!(cfg.budget.idle_unload_secs, 90);
        assert!(cfg.chains.is_empty());
        assert!(cfg.model.is_none(), "moved into models");
    }

    #[test]
    fn the_shipped_config_parses() {
        let text = include_str!("../jitmo-runtime.toml");
        let cfg = Config::from_toml(text).unwrap();
        assert!(cfg.chain("classify-answer").is_some());
        assert!(cfg.adapters_for(&cfg.models[0].name).is_empty(), "the example must work without adapters");
    }

    #[test]
    fn full_config_parses_with_overrides() {
        let text = r#"
            [server]
            port = 9000
            [model]
            name = "base"
            path = "models/m.gguf"
            threads = 2
            kv_cache_type = "q8_0"
            chat_template_kwargs = { enable_thinking = false }
            extra_args = ["--cache-reuse", "256"]
            [budget]
            memory_cap_mb = 0
            idle_unload_secs = 0

            [[chains]]
            name = "plan-write"
            [[chains.steps]]
            name = "plan"
            system = "Plan."
            max_tokens = 64
            [[chains.steps]]
            name = "write"
            system = "Follow the plan."
            input = "Plan:\n{plan}\n\nRequest: {input}"
            temperature = 0.7
            lora = { path = "adapters/writer.gguf", scale = 0.5 }
        "#;
        let mut cfg = Config::from_toml(text).unwrap();
        assert_eq!(cfg.budget.memory_cap_mb, MemoryCap::Mb(0));
        assert_eq!(Config::from_toml("[budget]\nmemory_cap_mb = 2048").unwrap().budget.memory_cap_mb, MemoryCap::Mb(2048));
        assert_eq!(Config::from_toml("[budget]\nmemory_cap_mb = \"auto\"").unwrap().budget.memory_cap_mb, MemoryCap::Auto);
        assert_eq!(cfg.server.port, 9000);
        assert_eq!(cfg.server.max_queue, 8, "unset fields keep defaults");
        assert_eq!(cfg.models[0].threads, 2);
        assert_eq!(cfg.models[0].chat_template_kwargs.as_ref().unwrap()["enable_thinking"], Value::Bool(false));
        let chain = cfg.chain("plan-write").unwrap();
        assert_eq!(chain.steps[0].input_template(0), "{input}");
        assert_eq!(chain.steps[0].temperature, 0.2);
        assert_eq!(chain.steps[1].max_tokens, 256);
        assert_eq!(chain.steps[1].lora.as_ref().unwrap().scale, 0.5);

        cfg.resolve_paths(Path::new(r"C:\cfg"));
        assert_eq!(cfg.models[0].path, PathBuf::from(r"C:\cfg\models/m.gguf"));
        assert_eq!(cfg.models[0].log_file, Some(PathBuf::from(r"C:\cfg\llama-server.log")));
        assert_eq!(cfg.adapters_for("base"), vec![PathBuf::from(r"C:\cfg\adapters/writer.gguf")]);
    }

    #[test]
    fn several_models_with_kinds_steps_and_adapters() {
        let text = r#"
            [budget]
            pool_budget_mb = 3000
            [server]
            max_concurrent = 2
            [[models]]
            name = "router"
            path = "r.gguf"
            ctx_size = 2048
            pinned = true
            adapters = ["a0.gguf"]
            [[models]]
            name = "writer"
            path = "w.gguf"
            memory_cap_mb = 900
            idle_unload_secs = 30
            [[models]]
            name = "embed"
            kind = "embedding"
            path = "e.gguf"
            ctx_size = 512
            pooling = "mean"
            [[chains]]
            name = "c"
            [[chains.steps]]
            name = "pick"
            lora = { path = "a1.gguf" }
            [[chains.steps]]
            name = "write"
            model = "writer"
            lora = { path = "a2.gguf" }
        "#;
        let cfg = Config::from_toml(text).unwrap();
        assert_eq!(cfg.models.len(), 3);
        assert_eq!((cfg.budget.pool_budget_mb, cfg.server.max_concurrent), (3000, 2));
        assert_eq!(cfg.models[2].kind, ModelKind::Embedding);
        assert_eq!(cfg.models[2].ubatch(), 512);
        assert_eq!(cfg.models[0].ubatch(), 512);
        assert_eq!(cfg.models[1].memory_cap_mb, Some(MemoryCap::Mb(900)));
        assert_eq!(cfg.models[1].log_file, Some(PathBuf::from("llama-server-writer.log")));
        assert_eq!(cfg.default_chat().unwrap().name, "router");
        let c = cfg.chain("c").unwrap();
        assert_eq!(cfg.step_model(&c.steps[0]), "router");
        assert_eq!(cfg.adapters_for("router"), vec![PathBuf::from("a0.gguf"), PathBuf::from("a1.gguf")]);
        assert_eq!(cfg.adapters_for("writer"), vec![PathBuf::from("a2.gguf")]);
        assert!(cfg.adapters_for("embed").is_empty());
    }

    #[test]
    fn json_schema_may_be_a_table_or_a_json_string() {
        let text = r#"
            [[chains]]
            name = "c"
            [[chains.steps]]
            name = "a"
            json_schema = { type = "object", properties = { kind = { type = "string", enum = ["x", "y"] } } }
            [[chains.steps]]
            name = "b"
            json_schema = '{"type":"object","properties":{"n":{"type":"integer"}}}'
        "#;
        let cfg = Config::from_toml(text).unwrap();
        let steps = &cfg.chains[0].steps;
        assert_eq!(steps[0].json_schema.as_ref().unwrap()["properties"]["kind"]["enum"][1], "y");
        assert_eq!(steps[1].json_schema.as_ref().unwrap()["properties"]["n"]["type"], "integer");
    }

    #[test]
    fn bad_configs_are_rejected_with_a_reason() {
        let cases = [
            ("[model]\nkv_cache_type = \"q3\"", "kv_cache_type"),
            ("[model]\nnmae = \"x\"", "unknown field"),
            ("[model]\nname = \"a\"\n[[models]]\nname = \"b\"", "not both"),
            ("[[models]]\nname = \"a\"\n[[models]]\nname = \"a\"", "used twice"),
            ("[[models]]\nname = \"a\"\nkind = \"encoder\"", "unknown variant"),
            ("[[models]]\nname = \"a\"\npooling = \"mean\"", "only for kind"),
            ("[[models]]\nname = \"a\"\nkind = \"embedding\"\npooling = \"avg\"", "pooling must be"),
            ("[server]\nmax_concurrent = 0", "max_concurrent"),
            ("[budget]\nmemory_cap_mb = \"big\"", "auto"),
            ("[budget]\ncritical_free_mb = 2000", "critical_free_mb"),
            ("[pacer]\nmode = \"sometimes\"", "pacer.mode"),
            ("[pacer]\nheavy_cap_pct = 0", "heavy_cap_pct"),
            ("[pacer]\nheavy_cap_pct = 90", "heavy caps harder"),
            ("[pacer]\nlight_cpu_pct = 50.0", "light_cpu_pct"),
            ("[pacer]\nsmooth_ms = 100", "smooth_ms"),
            ("[pacer]\nbusy_priority = \"high\"", "busy_priority"),
            ("[[chains]]\nname = \"c\"\nsteps = []", "no steps"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\ninput = \"{prev}\"", "{prev}"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\nsystem = \"{later}\"\n[[chains.steps]]\nname = \"later\"", "{later}"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\nsystem = \"{a}\"", "{a}"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\n[[chains.steps]]\nname = \"a\"", "twice"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"input\"", "reserved"),
            ("[[chains]]\nname = \"bad name\"\n[[chains.steps]]\nname = \"a\"", "not a valid name"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\nmodel = \"nope\"", "no model named"),
            (
                "[[models]]\nname = \"e\"\nkind = \"embedding\"\n[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\nmodel = \"e\"",
                "not a chat model",
            ),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\ngrammar = \"root ::= [0-9]\"\njson_schema = { type = \"object\" }", "not both"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\njson_schema = '{nope'", "not valid JSON"),
        ];
        for (text, want) in cases {
            let err = format!("{:#}", Config::from_toml(text).unwrap_err());
            assert!(err.contains(want), "config {text:?}: error {err:?} should mention {want:?}");
        }
    }

    #[test]
    fn named_outputs_fields_and_request_vars_are_allowed() {
        let text = r#"
            [[chains]]
            name = "c"
            [[chains.steps]]
            name = "classify"
            input = "Q: {question}"
            [[chains.steps]]
            name = "answer"
            system = "Kind: {classify.kind}. Raw: {classify}. Previous: {prev}. Passage: {passage}"
            input = "{input}"
        "#;
        Config::from_toml(text).unwrap();
    }
}
