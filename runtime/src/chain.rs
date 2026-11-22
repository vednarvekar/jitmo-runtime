//! Chains: ordered steps. Each step fills its prompt templates from the chain input, the
//! request's variables and earlier outputs, then runs one generation on its model (default:
//! the first chat model). Steps on the same model only change per-request settings (system
//! prompt, LoRA scale, schema), never reload the model.
//!
//! Template placeholders: `{input}` (the chain input), `{prev}` (previous step's output),
//! `{step}` (that step's output), `{step.field}` / `{step.a.b}` (a field of a step's JSON
//! output), and `{name}` / `{name.field}` for any request variable. Any other text in braces,
//! like a JSON example, is left as it is.

use crate::config::{check_template, Config, StepConfig, RESERVED_NAMES};
use crate::error::Error;
use crate::llama::{self, EngineTimings};
use crate::pool::Pool;
use crate::sched::RequestGuard;
use crate::stats::{round2, RequestTiming};
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
enum Piece {
    Text(String),
    Var { name: String, path: Vec<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Template {
    pieces: Vec<Piece>,
}

fn is_placeholder(s: &str) -> bool {
    let mut parts = s.split('.');
    let first = parts.next().unwrap_or("");
    let name_ok = first.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && first.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    name_ok && parts.all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
}

impl Template {
    pub fn parse(s: &str) -> Template {
        let mut pieces = Vec::new();
        let mut text = String::new();
        let mut rest = s;
        while let Some(open) = rest.find('{') {
            text.push_str(&rest[..open]);
            let after = &rest[open + 1..];
            match after.find('}') {
                Some(close) if is_placeholder(&after[..close]) => {
                    if !text.is_empty() {
                        pieces.push(Piece::Text(std::mem::take(&mut text)));
                    }
                    let mut parts = after[..close].split('.').map(str::to_string);
                    let name = parts.next().unwrap_or_default();
                    pieces.push(Piece::Var { name, path: parts.collect() });
                    rest = &after[close + 1..];
                }
                _ => {
                    text.push('{');
                    rest = after;
                }
            }
        }
        text.push_str(rest);
        if !text.is_empty() {
            pieces.push(Piece::Text(text));
        }
        Template { pieces }
    }

    /// Names the template refers to (`input`, `prev`, step names or request variables).
    pub fn var_names(&self) -> impl Iterator<Item = &str> {
        self.pieces.iter().filter_map(|p| match p {
            Piece::Var { name, .. } => Some(name.as_str()),
            Piece::Text(_) => None,
        })
    }

    pub fn render(&self, vars: &Vars) -> Result<String, String> {
        let mut out = String::new();
        for p in &self.pieces {
            match p {
                Piece::Text(t) => out.push_str(t),
                Piece::Var { name, path } => out.push_str(&vars.lookup(name, path)?),
            }
        }
        Ok(out)
    }
}

/// Output of a finished step.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub name: String,
    pub text: String,
    pub json: Option<Value>,
}

pub struct Vars<'a> {
    pub input: &'a str,
    pub outputs: &'a [StepOutput],
    /// Request variables (`"vars"` in the chain request).
    pub vars: &'a Map<String, Value>,
}

impl Vars<'_> {
    fn lookup(&self, name: &str, path: &[String]) -> Result<String, String> {
        if let Some(v) = self.vars.get(name).filter(|_| !RESERVED_NAMES.contains(&name)) {
            if !self.outputs.iter().any(|o| o.name == name) {
                return var_value(name, path, v);
            }
        }
        let (text, json) = match name {
            "input" => (self.input, None),
            "prev" => match self.outputs.last() {
                Some(o) => (o.text.as_str(), o.json.as_ref()),
                None => return Err("{prev} is empty in the first step".into()),
            },
            _ => match self.outputs.iter().find(|o| o.name == name) {
                Some(o) => (o.text.as_str(), o.json.as_ref()),
                None => return Err(format!("no step output or request variable named '{name}'")),
            },
        };
        if path.is_empty() {
            return Ok(text.to_string());
        }
        let shown = format!("{{{name}.{}}}", path.join("."));
        let parsed;
        let mut v = match json {
            Some(v) => v,
            None => {
                parsed = serde_json::from_str::<Value>(text)
                    .map_err(|_| format!("{shown}: the output of '{name}' is not JSON: {text:?}"))?;
                &parsed
            }
        };
        for key in path {
            let next = match v {
                Value::Object(m) => m.get(key),
                Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
                _ => None,
            };
            v = next.ok_or_else(|| format!("{shown}: the output of '{name}' has no field '{key}': {text}"))?;
        }
        Ok(match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
    }
}

/// A request variable as text: a string as it is, anything else as JSON; `{name.a.b}` reads a
/// field of an object (or an item of an array).
fn var_value(name: &str, path: &[String], v: &Value) -> Result<String, String> {
    let mut v = v;
    for key in path {
        let next = match v {
            Value::Object(m) => m.get(key),
            Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
            _ => None,
        };
        v = next.ok_or_else(|| format!("{{{name}.{}}}: variable '{name}' has no field '{key}'", path.join(".")))?;
    }
    Ok(match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

/// A chain with its templates parsed, models and LoRA adapters resolved.
pub struct Chain {
    pub name: String,
    pub description: String,
    pub steps: Vec<Step>,
    /// Placeholder names that are not `input`, `prev` or a step: the variables a run must send.
    pub var_names: BTreeSet<String>,
}

pub struct Step {
    pub name: String,
    /// Index into the pool's models.
    pub model: usize,
    pub model_name: String,
    system: Template,
    input: Template,
    max_tokens: u32,
    temperature: f64,
    pub json_schema: Option<Value>,
    grammar: Option<String>,
    /// (adapter id, scale)
    pub lora: Option<(usize, f64)>,
    pub lora_path: Option<String>,
}

impl Chain {
    pub fn compile_all(cfg: &Config) -> Vec<Chain> {
        cfg.chains
            .iter()
            .map(|c| {
                let steps: Vec<Step> = c.steps.iter().enumerate().map(|(i, s)| Step::compile(cfg, s, i)).collect();
                let step_names: Vec<&str> = steps.iter().map(|s| s.name.as_str()).collect();
                let var_names = steps
                    .iter()
                    .flat_map(|s| s.system.var_names().chain(s.input.var_names()).map(str::to_string).collect::<Vec<_>>())
                    .filter(|n| !RESERVED_NAMES.contains(&n.as_str()) && !step_names.contains(&n.as_str()))
                    .collect();
                Chain { name: c.name.clone(), description: c.description.clone(), steps, var_names }
            })
            .collect()
    }

    /// Check a run's variables and system overrides before any generation; returns the
    /// overrides parsed, by step index.
    pub fn check_run(&self, vars: &Map<String, Value>, system: &Map<String, Value>) -> Result<HashMap<usize, Template>, Error> {
        let step_names: Vec<&str> = self.steps.iter().map(|s| s.name.as_str()).collect();
        for k in vars.keys() {
            if RESERVED_NAMES.contains(&k.as_str()) || step_names.contains(&k.as_str()) {
                return Err(Error::bad_request(format!(
                    "variable '{k}' has the name of {}; pick another name",
                    if RESERVED_NAMES.contains(&k.as_str()) { "a built-in placeholder" } else { "a step" }
                )));
            }
        }
        let mut overrides = HashMap::new();
        for (step, text) in system {
            let Some(i) = self.steps.iter().position(|s| &s.name == step) else {
                return Err(Error::bad_request(format!("'system' names step '{step}', but chain '{}' has steps {step_names:?}", self.name)));
            };
            let Some(text) = text.as_str() else {
                return Err(Error::bad_request(format!("'system.{step}' must be a string")));
            };
            check_template(text, i, &step_names[..i], &step_names).map_err(|e| Error::bad_request(format!("'system.{step}': {e:#}")))?;
            overrides.insert(i, Template::parse(text));
        }
        let mut needed: BTreeSet<String> = self.var_names.clone();
        for t in overrides.values() {
            needed.extend(t.var_names().filter(|n| !RESERVED_NAMES.contains(n) && !step_names.contains(n)).map(str::to_string));
        }
        let missing: Vec<&String> = needed.iter().filter(|n| !vars.contains_key(*n)).collect();
        if !missing.is_empty() {
            return Err(Error::bad_request(format!(
                "chain '{}' needs the variable(s) {missing:?} in \"vars\"",
                self.name
            )));
        }
        Ok(overrides)
    }
}

impl Step {
    fn compile(cfg: &Config, s: &StepConfig, index: usize) -> Step {
        let model_name = cfg.step_model(s).to_string();
        let model = cfg.models.iter().position(|m| m.name == model_name).unwrap_or(0);
        let adapters = cfg.adapters_for(&model_name);
        let lora = s.lora.as_ref().and_then(|l| adapters.iter().position(|p| *p == l.path).map(|id| (id, l.scale)));
        Step {
            name: s.name.clone(),
            model,
            model_name,
            system: Template::parse(&s.system),
            input: Template::parse(s.input_template(index)),
            max_tokens: s.max_tokens,
            temperature: s.temperature,
            json_schema: s.json_schema.clone(),
            grammar: s.grammar.clone(),
            lora,
            lora_path: s.lora.as_ref().map(|l| l.path.display().to_string()),
        }
    }

    /// The llama-server request for this step.
    pub fn request(&self, system: &str, user: &str, template_kwargs: Option<&Value>) -> Value {
        let mut messages = Vec::new();
        if !system.is_empty() {
            messages.push(json!({ "role": "system", "content": system }));
        }
        messages.push(json!({ "role": "user", "content": user }));
        let mut body = json!({
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": false,
            "cache_prompt": true,
        });
        if let Some(schema) = &self.json_schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": { "name": self.name, "strict": true, "schema": schema }
            });
        }
        if let Some(g) = &self.grammar {
            body["grammar"] = json!(g);
        }
        if let Some((id, scale)) = self.lora {
            body["lora"] = json!([{ "id": id, "scale": scale }]);
        }
        if let Some(k) = template_kwargs {
            body["chat_template_kwargs"] = k.clone();
        }
        body
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StepReport {
    pub name: String,
    pub model: String,
    pub output: String,
    /// The output parsed as JSON, for steps with a json_schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<Value>,
    pub finish_reason: Option<String>,
    pub timing: RequestTiming,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChainReport {
    pub chain: String,
    pub request_id: String,
    /// The last step's output.
    pub output: String,
    pub steps: Vec<StepReport>,
    pub queue_ms: f64,
    pub load_ms: f64,
    pub total_ms: f64,
}

/// What one chain run gets from the request.
pub struct RunInput<'a> {
    pub input: &'a str,
    pub vars: &'a Map<String, Value>,
    pub system: &'a Map<String, Value>,
}

/// Run a whole chain while holding one generation slot (other requests wait). Each step
/// holds its own model while it runs, so a step on another model may load it (and evict an
/// idle one).
pub async fn run(pool: &Pool, chain: &Chain, run: RunInput<'_>, req: &RequestGuard) -> Result<ChainReport, Error> {
    let started = Instant::now();
    let overrides = chain.check_run(run.vars, run.system)?;
    let permit = pool.sched.acquire(req.priority).await.inspect_err(|e| {
        if e.code == "queue_full" {
            pool.stats.refused_queue();
        }
    })?;
    let queue_ms = started.elapsed().as_secs_f64() * 1000.0;
    req.set_running();
    let mut outputs: Vec<StepOutput> = Vec::new();
    let mut reports = Vec::new();
    let mut load_total = 0.0;
    for (i, step) in chain.steps.iter().enumerate() {
        let at = format!("chain '{}' step '{}'", chain.name, step.name);
        let fail = |e: String| Error::chain_step(format!("{at}: {e}"));
        let vars = Vars { input: run.input, outputs: &outputs, vars: run.vars };
        let system = overrides.get(&i).unwrap_or(&step.system).render(&vars).map_err(fail)?;
        let user = step.input.render(&vars).map_err(fail)?;
        let model = &pool.models[step.model];
        let mut body = step.request(&system, &user, model.cfg.chat_template_kwargs.as_ref());
        llama::full_lora(&mut body, model.adapters.len()).map_err(|e| e.with_prefix(&at))?;
        let t0 = Instant::now();
        let (hold, port, load_ms) = pool.use_model(model).await.map_err(|e| e.with_prefix(&at))?;
        let v = llama::chat_once(&pool.http, port, &body).await.map_err(|e| e.with_prefix(&at))?;
        drop(hold);
        model.count_generation();
        load_total += load_ms;
        let text = llama::content(&v);
        let json = step.json_schema.as_ref().and_then(|_| serde_json::from_str::<Value>(&text).ok());
        let q = if i == 0 { queue_ms } else { 0.0 };
        let engine = llama::engine_timings(&v).unwrap_or_else(EngineTimings::default);
        let timing = RequestTiming::new(
            "chain_step",
            format!("{}/{}", chain.name, step.name),
            q,
            load_ms,
            &engine,
            None,
            t0.elapsed().as_secs_f64() * 1000.0 + q,
        )
        .with_ids(&req.id, &step.model_name);
        pool.pacer.after_generation(&timing);
        pool.stats.record(timing.clone());
        reports.push(StepReport {
            name: step.name.clone(),
            model: step.model_name.clone(),
            output: text.clone(),
            json: json.clone(),
            finish_reason: v["choices"][0]["finish_reason"].as_str().map(str::to_string),
            timing,
        });
        outputs.push(StepOutput { name: step.name.clone(), text, json });
    }
    drop(permit);
    Ok(ChainReport {
        chain: chain.name.clone(),
        request_id: req.id.clone(),
        output: outputs.last().map(|o| o.text.clone()).unwrap_or_default(),
        steps: reports,
        queue_ms: round2(queue_ms),
        load_ms: round2(load_total),
        total_ms: round2(started.elapsed().as_secs_f64() * 1000.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(name: &str, text: &str) -> StepOutput {
        StepOutput { name: name.into(), text: text.into(), json: None }
    }

    #[test]
    fn placeholders_are_filled() {
        let outputs = vec![out("classify", r#"{"kind":"question","score":0.9,"tags":["a","b"]}"#), out("plan", "1. do it")];
        let none = Map::new();
        let vars = Vars { input: "what is RAM?", outputs: &outputs, vars: &none };
        let t = Template::parse("Kind={classify.kind} score={classify.score} tag={classify.tags.1} prev=[{prev}] in={input}");
        assert_eq!(t.render(&vars).unwrap(), "Kind=question score=0.9 tag=b prev=[1. do it] in=what is RAM?");
        let whole = Template::parse("{classify}");
        assert_eq!(whole.render(&vars).unwrap(), outputs[0].text);
    }

    #[test]
    fn json_and_other_braces_stay_literal() {
        let t = Template::parse(r#"Answer like {"kind": "x"} or { spaced } or {} or {1abc}; user: {input}"#);
        assert_eq!(t.var_names().collect::<Vec<_>>(), vec!["input"]);
        let none = Map::new();
        let vars = Vars { input: "hi", outputs: &[], vars: &none };
        assert_eq!(t.render(&vars).unwrap(), r#"Answer like {"kind": "x"} or { spaced } or {} or {1abc}; user: hi"#);
        assert_eq!(Template::parse("{{input}}").render(&vars).unwrap(), "{hi}");
        assert_eq!(Template::parse("no braces").render(&vars).unwrap(), "no braces");
        assert_eq!(Template::parse("").render(&vars).unwrap(), "");
    }

    #[test]
    fn missing_fields_and_non_json_outputs_are_clear_errors() {
        let outputs = vec![out("classify", r#"{"kind":"task"}"#), out("plain", "not json")];
        let none = Map::new();
        let vars = Vars { input: "x", outputs: &outputs, vars: &none };
        let e = Template::parse("{classify.mood}").render(&vars).unwrap_err();
        assert!(e.contains("no field 'mood'"), "{e}");
        let e = Template::parse("{plain.kind}").render(&vars).unwrap_err();
        assert!(e.contains("not JSON"), "{e}");
        let e = Template::parse("{prev}").render(&Vars { input: "x", outputs: &[], vars: &none }).unwrap_err();
        assert!(e.contains("first step"), "{e}");
    }

    #[test]
    fn a_step_request_has_schema_lora_and_template_kwargs() {
        let text = r#"
            [model]
            chat_template_kwargs = { enable_thinking = false }
            [[chains]]
            name = "c"
            [[chains.steps]]
            name = "classify"
            system = "Classify."
            max_tokens = 10
            temperature = 0.0
            json_schema = { type = "object" }
            lora = { path = "x.gguf", scale = 0.75 }
            [[chains.steps]]
            name = "answer"
        "#;
        let cfg = Config::from_toml(text).unwrap();
        let chains = Chain::compile_all(&cfg);
        let step = &chains[0].steps[0];
        let kwargs = cfg.models[0].chat_template_kwargs.clone();
        let body = step.request("Classify.", "hello", kwargs.as_ref());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "hello");
        assert_eq!(body["max_tokens"], 10);
        assert_eq!(body["response_format"]["json_schema"]["schema"]["type"], "object");
        assert_eq!(body["lora"][0]["id"], 0);
        assert_eq!(body["lora"][0]["scale"], 0.75);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        // Second step: no system prompt, input defaults to {prev}, no lora.
        let second = &chains[0].steps[1];
        let body = second.request("", "prev text", kwargs.as_ref());
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert!(body.get("lora").is_none());
        assert_eq!(second.input, Template::parse("{prev}"));
    }

    #[test]
    fn request_vars_fill_templates_and_runs_are_checked() {
        let text = r#"
            [[models]]
            name = "small"
            [[models]]
            name = "big"
            [[chains]]
            name = "qa"
            [[chains.steps]]
            name = "pick"
            input = "Q: {question}"
            [[chains.steps]]
            name = "write"
            model = "big"
            system = "Use this passage: {passage}"
            input = "{question} ({pick}) {meta.lang} {meta.tags.1}"
        "#;
        let cfg = Config::from_toml(text).unwrap();
        let chains = Chain::compile_all(&cfg);
        let c = &chains[0];
        assert_eq!(c.var_names.iter().cloned().collect::<Vec<_>>(), vec!["meta", "passage", "question"]);
        assert_eq!((c.steps[0].model, c.steps[1].model), (0, 1));
        assert_eq!(c.steps[1].model_name, "big");

        let vars: Map<String, Value> = serde_json::from_value(json!({
            "question": "Who wrote it?", "passage": "Ann wrote it.", "meta": { "lang": "en", "tags": ["a", "b"] }
        }))
        .unwrap();
        let no_sys = Map::new();
        assert!(c.check_run(&vars, &no_sys).unwrap().is_empty());
        let outputs = vec![out("pick", "3")];
        let v = Vars { input: "", outputs: &outputs, vars: &vars };
        assert_eq!(c.steps[1].input.render(&v).unwrap(), "Who wrote it? (3) en b");
        assert_eq!(c.steps[1].system.render(&v).unwrap(), "Use this passage: Ann wrote it.");

        let mut short = vars.clone();
        short.remove("passage");
        let e = c.check_run(&short, &no_sys).unwrap_err();
        assert!(e.status == 400 && e.message.contains("passage"), "{}", e.message);
        let clash: Map<String, Value> = serde_json::from_value(json!({ "pick": "x", "question": "", "passage": "", "meta": {} })).unwrap();
        assert!(c.check_run(&clash, &no_sys).unwrap_err().message.contains("a step"));

        // Per-request system override for a step, with its own variable.
        let sys: Map<String, Value> = serde_json::from_value(json!({ "write": "Be {tone}. Passage: {passage}. Pick: {pick}" })).unwrap();
        assert!(c.check_run(&vars, &sys).unwrap_err().message.contains("tone"));
        let mut toned = vars.clone();
        toned.insert("tone".into(), json!("brief"));
        let o = c.check_run(&toned, &sys).unwrap();
        let v = Vars { input: "", outputs: &outputs, vars: &toned };
        assert_eq!(o[&1].render(&v).unwrap(), "Be brief. Passage: Ann wrote it.. Pick: 3");
        let bad: Map<String, Value> = serde_json::from_value(json!({ "pick": "{write}" })).unwrap();
        assert!(c.check_run(&vars, &bad).unwrap_err().message.contains("not run yet"));
        let unknown: Map<String, Value> = serde_json::from_value(json!({ "nope": "x" })).unwrap();
        assert!(c.check_run(&vars, &unknown).unwrap_err().message.contains("names step 'nope'"));
    }
}
