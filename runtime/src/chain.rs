//! Chains: ordered steps on the one base model. Each step fills its prompt templates from
//! the chain input and earlier outputs, then runs one generation. Switching steps only
//! changes per-request settings (system prompt, LoRA scale, schema), never the model.
//!
//! Template placeholders: `{input}` (the chain input), `{prev}` (previous step's output),
//! `{step}` (that step's output), `{step.field}` / `{step.a.b}` (a field of a step's JSON
//! output). Any other text in braces, like a JSON example, is left as it is.

use crate::config::{Config, StepConfig};
use crate::error::Error;
use crate::llama::{self, EngineTimings};
use crate::stats::{round2, RequestTiming};
use crate::supervisor::Supervisor;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
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

    /// Names the template refers to (`input`, `prev` or step names).
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
}

impl Vars<'_> {
    fn lookup(&self, name: &str, path: &[String]) -> Result<String, String> {
        let (text, json) = match name {
            "input" => (self.input, None),
            "prev" => match self.outputs.last() {
                Some(o) => (o.text.as_str(), o.json.as_ref()),
                None => return Err("{prev} is empty in the first step".into()),
            },
            _ => match self.outputs.iter().find(|o| o.name == name) {
                Some(o) => (o.text.as_str(), o.json.as_ref()),
                None => return Err(format!("no output named '{name}' yet")),
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

/// A chain with its templates parsed and LoRA adapters resolved to llama-server ids.
pub struct Chain {
    pub name: String,
    pub description: String,
    pub steps: Vec<Step>,
}

pub struct Step {
    pub name: String,
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
        let adapters = cfg.adapters();
        cfg.chains
            .iter()
            .map(|c| Chain {
                name: c.name.clone(),
                description: c.description.clone(),
                steps: c.steps.iter().enumerate().map(|(i, s)| Step::compile(s, i, &adapters)).collect(),
            })
            .collect()
    }
}

impl Step {
    fn compile(s: &StepConfig, index: usize, adapters: &[std::path::PathBuf]) -> Step {
        let lora = s.lora.as_ref().and_then(|l| adapters.iter().position(|p| *p == l.path).map(|id| (id, l.scale)));
        Step {
            name: s.name.clone(),
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
    pub fn request(&self, system: &str, user: &str, cfg: &Config) -> Value {
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
        if let Some(k) = &cfg.model.chat_template_kwargs {
            body["chat_template_kwargs"] = k.clone();
        }
        body
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StepReport {
    pub name: String,
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
    /// The last step's output.
    pub output: String,
    pub steps: Vec<StepReport>,
    pub queue_ms: f64,
    pub load_ms: f64,
    pub total_ms: f64,
}

/// Run a whole chain while holding the one generation slot (other requests wait).
pub async fn run(sup: &Arc<Supervisor>, cfg: &Config, chain: &Chain, input: &str) -> Result<ChainReport, Error> {
    let started = Instant::now();
    let slot = sup.acquire().await?;
    let mut outputs: Vec<StepOutput> = Vec::new();
    let mut reports = Vec::new();
    for (i, step) in chain.steps.iter().enumerate() {
        let fail = |e: String| Error::chain_step(format!("chain '{}' step '{}': {e}", chain.name, step.name));
        let vars = Vars { input, outputs: &outputs };
        let system = step.system.render(&vars).map_err(fail)?;
        let user = step.input.render(&vars).map_err(fail)?;
        let body = step.request(&system, &user, cfg);
        let t0 = Instant::now();
        let v = llama::chat_once(&sup.http, slot.port, &body)
            .await
            .map_err(|e| e.with_prefix(&format!("chain '{}' step '{}'", chain.name, step.name)))?;
        let text = llama::content(&v);
        let json = step.json_schema.as_ref().and_then(|_| serde_json::from_str::<Value>(&text).ok());
        let (queue_ms, load_ms) = if i == 0 { (slot.queue_ms, slot.load_ms) } else { (0.0, 0.0) };
        let engine = llama::engine_timings(&v).unwrap_or_else(EngineTimings::default);
        let timing = RequestTiming::new(
            "chain_step",
            format!("{}/{}", chain.name, step.name),
            queue_ms,
            load_ms,
            &engine,
            None,
            t0.elapsed().as_secs_f64() * 1000.0 + queue_ms + load_ms,
        );
        sup.pacer.after_generation(&timing);
        sup.stats.record(timing.clone());
        reports.push(StepReport {
            name: step.name.clone(),
            output: text.clone(),
            json: json.clone(),
            finish_reason: v["choices"][0]["finish_reason"].as_str().map(str::to_string),
            timing,
        });
        outputs.push(StepOutput { name: step.name.clone(), text, json });
    }
    let (queue_ms, load_ms) = (slot.queue_ms, slot.load_ms);
    drop(slot);
    Ok(ChainReport {
        chain: chain.name.clone(),
        output: outputs.last().map(|o| o.text.clone()).unwrap_or_default(),
        steps: reports,
        queue_ms: round2(queue_ms),
        load_ms: round2(load_ms),
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
        let vars = Vars { input: "what is RAM?", outputs: &outputs };
        let t = Template::parse("Kind={classify.kind} score={classify.score} tag={classify.tags.1} prev=[{prev}] in={input}");
        assert_eq!(t.render(&vars).unwrap(), "Kind=question score=0.9 tag=b prev=[1. do it] in=what is RAM?");
        let whole = Template::parse("{classify}");
        assert_eq!(whole.render(&vars).unwrap(), outputs[0].text);
    }

    #[test]
    fn json_and_other_braces_stay_literal() {
        let t = Template::parse(r#"Answer like {"kind": "x"} or { spaced } or {} or {1abc}; user: {input}"#);
        assert_eq!(t.var_names().collect::<Vec<_>>(), vec!["input"]);
        let vars = Vars { input: "hi", outputs: &[] };
        assert_eq!(t.render(&vars).unwrap(), r#"Answer like {"kind": "x"} or { spaced } or {} or {1abc}; user: hi"#);
        assert_eq!(Template::parse("{{input}}").render(&vars).unwrap(), "{hi}");
        assert_eq!(Template::parse("no braces").render(&vars).unwrap(), "no braces");
        assert_eq!(Template::parse("").render(&vars).unwrap(), "");
    }

    #[test]
    fn missing_fields_and_non_json_outputs_are_clear_errors() {
        let outputs = vec![out("classify", r#"{"kind":"task"}"#), out("plain", "not json")];
        let vars = Vars { input: "x", outputs: &outputs };
        let e = Template::parse("{classify.mood}").render(&vars).unwrap_err();
        assert!(e.contains("no field 'mood'"), "{e}");
        let e = Template::parse("{plain.kind}").render(&vars).unwrap_err();
        assert!(e.contains("not JSON"), "{e}");
        let e = Template::parse("{prev}").render(&Vars { input: "x", outputs: &[] }).unwrap_err();
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
        let body = step.request("Classify.", "hello", &cfg);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["content"], "hello");
        assert_eq!(body["max_tokens"], 10);
        assert_eq!(body["response_format"]["json_schema"]["schema"]["type"], "object");
        assert_eq!(body["lora"][0]["id"], 0);
        assert_eq!(body["lora"][0]["scale"], 0.75);
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        // Second step: no system prompt, input defaults to {prev}, no lora.
        let second = &chains[0].steps[1];
        let body = second.request("", "prev text", &cfg);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert!(body.get("lora").is_none());
        assert_eq!(second.input, Template::parse("{prev}"));
    }
}
