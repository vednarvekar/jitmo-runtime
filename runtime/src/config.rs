//! The one TOML config file: API server, model + llama-server child, RAM budget, chains.
//! Every field has a default; relative paths are resolved against the config file's folder.

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
    pub model: ModelConfig,
    pub budget: BudgetConfig,
    pub chains: Vec<ChainConfig>,
}

/// The runtime's own HTTP API.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Requests allowed to wait while one generation runs. More are refused with 503.
    pub max_queue: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig { host: "127.0.0.1".into(), port: 8090, max_queue: 8 }
    }
}

/// The base model and how the llama-server child is started.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Model id clients send in `"model"`.
    pub name: String,
    pub path: PathBuf,
    pub llama_server: PathBuf,
    /// Private port of the child. 0 = pick a free port at each start.
    pub port: u16,
    pub ctx_size: u32,
    /// 0 = auto: physical cores - 1, between 1 and 3.
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
    /// Sent as `chat_template_kwargs` on every request that does not set its own.
    pub chat_template_kwargs: Option<Value>,
    /// Extra llama-server arguments, appended as-is.
    pub extra_args: Vec<String>,
    /// File for llama-server's own log (truncated at each start). Empty = discard.
    pub log_file: PathBuf,
    /// Give up if the model is not ready after this long.
    pub load_timeout_secs: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            name: "jitmo-base".into(),
            path: r"V:\jitmo-sandbox\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf".into(),
            llama_server: r"V:\jitmo-sandbox\llama\llama-server.exe".into(),
            port: 0,
            ctx_size: 4096,
            threads: 0,
            kv_cache_type: "f16".into(),
            flash_attn: "on".into(),
            mmap: true,
            no_repack: true,
            low_priority: true,
            chat_template_kwargs: None,
            extra_args: Vec::new(),
            log_file: "llama-server.log".into(),
            load_timeout_secs: 180,
        }
    }
}

/// RAM budget and unload rules.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetConfig {
    /// Job Object per-process commit (private bytes) limit for the child. 0 = no cap.
    /// It does not count mmap'd weight pages: it bounds the KV cache, buffers and any copied weights.
    pub memory_cap_mb: u64,
    /// Refuse to start the model if free RAM after the load would be below this.
    pub reserve_free_mb: u64,
    /// RAM the model is expected to take when loaded. 0 = model file size + memory_cap_mb
    /// (or + 1024 MB when there is no cap).
    pub load_estimate_mb: u64,
    /// Unload the model when free RAM stays below this (or Windows signals low memory).
    pub pressure_free_mb: u64,
    /// Unload after this many seconds without requests. 0 = never.
    pub idle_unload_secs: u64,
    /// How often the watchdog checks idle time, memory pressure and the child.
    pub watch_interval_ms: u64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        BudgetConfig {
            memory_cap_mb: 1536,
            reserve_free_mb: 1536,
            load_estimate_mb: 0,
            pressure_free_mb: 512,
            idle_unload_secs: 90,
            watch_interval_ms: 1000,
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
const RESERVED_NAMES: &[&str] = &["input", "prev"];

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
        let mut cfg = Config::default();
        cfg.resolve_paths(base);
        cfg
    }

    pub fn resolve_paths(&mut self, base: &Path) {
        let fix = |p: &mut PathBuf| {
            if !p.as_os_str().is_empty() && p.is_relative() {
                *p = base.join(&*p);
            }
        };
        fix(&mut self.model.path);
        fix(&mut self.model.llama_server);
        fix(&mut self.model.log_file);
        for chain in &mut self.chains {
            for step in &mut chain.steps {
                if let Some(l) = step.lora.as_mut() {
                    fix(&mut l.path);
                }
            }
        }
    }

    /// A JSON schema given as a string is parsed into JSON here.
    fn normalize(&mut self) -> Result<()> {
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
        if self.model.name.is_empty() || self.model.name.starts_with("chain:") {
            bail!("model.name must be set and must not start with 'chain:'");
        }
        if !KV_TYPES.contains(&self.model.kv_cache_type.as_str()) {
            bail!("model.kv_cache_type '{}' is not one of {:?}", self.model.kv_cache_type, KV_TYPES);
        }
        if !["on", "off", "auto"].contains(&self.model.flash_attn.as_str()) {
            bail!("model.flash_attn must be on, off or auto");
        }
        if self.model.ctx_size < 256 {
            bail!("model.ctx_size must be at least 256");
        }
        if self.budget.watch_interval_ms < 100 {
            bail!("budget.watch_interval_ms must be at least 100");
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
            let mut known: Vec<&str> = vec!["input"];
            for (i, step) in chain.steps.iter().enumerate() {
                let at = || format!("chain '{}' step '{}'", chain.name, step.name);
                check_name(&step.name).with_context(at)?;
                if RESERVED_NAMES.contains(&step.name.as_str()) {
                    bail!("{}: 'input' and 'prev' are reserved step names", at());
                }
                if known.contains(&step.name.as_str()) {
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
                if i > 0 {
                    known.push("prev");
                }
                for (field, text) in [("system", step.system.as_str()), ("input", step.input_template(i))] {
                    let t = crate::chain::Template::parse(text);
                    for var in t.var_names() {
                        if !known.contains(&var) {
                            bail!(
                                "{}: {} uses {{{}}}, but only {} are available here",
                                at(),
                                field,
                                var,
                                known.iter().map(|k| format!("{{{k}}}")).collect::<Vec<_>>().join(", ")
                            );
                        }
                    }
                }
                if i > 0 {
                    known.pop(); // "prev"
                }
                known.push(&step.name);
            }
        }
        Ok(())
    }

    /// Check that the files the runtime needs exist (done at startup, not in unit tests).
    pub fn check_files(&self) -> Result<()> {
        if !self.model.path.is_file() {
            bail!("model file not found: {}", self.model.path.display());
        }
        if !self.model.llama_server.is_file() {
            bail!("llama-server not found: {}", self.model.llama_server.display());
        }
        for path in self.adapters() {
            if !path.is_file() {
                bail!("LoRA adapter not found: {}", path.display());
            }
        }
        Ok(())
    }

    /// Unique LoRA adapter files used by any chain step, in first-use order. The index in
    /// this list is the adapter id llama-server uses.
    pub fn adapters(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        for step in self.chains.iter().flat_map(|c| &c.steps) {
            if let Some(l) = &step.lora {
                if !out.contains(&l.path) {
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
        assert_eq!(cfg.model.name, "jitmo-base");
        assert_eq!(cfg.model.ctx_size, 4096);
        assert_eq!(cfg.model.kv_cache_type, "f16");
        assert!(cfg.model.no_repack && cfg.model.mmap);
        assert_eq!(cfg.budget.memory_cap_mb, 1536);
        assert_eq!(cfg.budget.reserve_free_mb, 1536);
        assert_eq!(cfg.budget.idle_unload_secs, 90);
        assert!(cfg.chains.is_empty());
    }

    #[test]
    fn the_shipped_config_parses() {
        let text = include_str!("../jitmo-runtime.toml");
        let cfg = Config::from_toml(text).unwrap();
        assert!(cfg.chain("classify-answer").is_some());
        assert!(cfg.adapters().is_empty(), "the example must work without adapters");
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
        assert_eq!(cfg.server.port, 9000);
        assert_eq!(cfg.server.max_queue, 8, "unset fields keep defaults");
        assert_eq!(cfg.model.threads, 2);
        assert_eq!(cfg.model.chat_template_kwargs.as_ref().unwrap()["enable_thinking"], Value::Bool(false));
        let chain = cfg.chain("plan-write").unwrap();
        assert_eq!(chain.steps[0].input_template(0), "{input}");
        assert_eq!(chain.steps[0].temperature, 0.2);
        assert_eq!(chain.steps[1].max_tokens, 256);
        assert_eq!(chain.steps[1].lora.as_ref().unwrap().scale, 0.5);

        cfg.resolve_paths(Path::new(r"C:\cfg"));
        assert_eq!(cfg.model.path, PathBuf::from(r"C:\cfg\models/m.gguf"));
        assert_eq!(cfg.adapters(), vec![PathBuf::from(r"C:\cfg\adapters/writer.gguf")]);
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
            ("[[chains]]\nname = \"c\"\nsteps = []", "no steps"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\ninput = \"{prev}\"", "{prev}"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\nsystem = \"{later}\"\n[[chains.steps]]\nname = \"later\"", "{later}"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\n[[chains.steps]]\nname = \"a\"", "twice"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"input\"", "reserved"),
            ("[[chains]]\nname = \"bad name\"\n[[chains.steps]]\nname = \"a\"", "not a valid name"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\ngrammar = \"root ::= [0-9]\"\njson_schema = { type = \"object\" }", "not both"),
            ("[[chains]]\nname = \"c\"\n[[chains.steps]]\nname = \"a\"\njson_schema = '{nope'", "not valid JSON"),
        ];
        for (text, want) in cases {
            let err = format!("{:#}", Config::from_toml(text).unwrap_err());
            assert!(err.contains(want), "config {text:?}: error {err:?} should mention {want:?}");
        }
    }

    #[test]
    fn named_outputs_and_fields_of_earlier_steps_are_allowed() {
        let text = r#"
            [[chains]]
            name = "c"
            [[chains.steps]]
            name = "classify"
            [[chains.steps]]
            name = "answer"
            system = "Kind: {classify.kind}. Raw: {classify}. Previous: {prev}."
            input = "{input}"
        "#;
        Config::from_toml(text).unwrap();
    }
}
