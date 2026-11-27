//! JSON schema -> GBNF grammar, for `response_format: json_schema` on the in-process engine
//! (llama-server does this in C++; libllama doesn't).
//!
//! Supported: object (properties in the listed order; required ones first, optional ones after
//! them), string (with `enum`/`const`), number, integer, boolean, null, array (`items`), `enum`,
//! `const`, `anyOf`/`oneOf`, a list of types, and `{}` / unknown = any JSON value.
//! Not supported (refused with an error): `$ref`, `pattern`, `format`, string length limits.

use serde_json::Value;
use std::collections::BTreeMap;

const BASE: &str = r#"ws ::= ( " " | "\n" [ \t]{0,20} )?
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] )
integer ::= "-"? ( "0" | [1-9] [0-9]{0,15} )
number ::= integer ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
boolean ::= "true" | "false"
null ::= "null"
value ::= object-any | array-any | string | number | boolean | null
object-any ::= "{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* )? ws "}"
array-any ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
"#;

struct Gen {
    rules: BTreeMap<String, String>,
    n: usize,
}

/// A GBNF string literal for a JSON text.
fn lit(json_text: &str) -> String {
    let mut s = String::from("\"");
    for c in json_text.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c => s.push(c),
        }
    }
    s.push('"');
    s
}

impl Gen {
    fn rule(&mut self, hint: &str, body: String) -> String {
        self.n += 1;
        let name = format!("{}-{}", hint.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').take(24).collect::<String>().to_lowercase(), self.n);
        let name = if name.starts_with(|c: char| c.is_ascii_digit()) || name.starts_with('-') { format!("r{name}") } else { name };
        self.rules.insert(name.clone(), body);
        name
    }

    fn literal_alts(vals: &[Value]) -> String {
        vals.iter().map(|v| lit(&v.to_string())).collect::<Vec<_>>().join(" | ")
    }

    fn schema(&mut self, s: &Value, hint: &str) -> Result<String, String> {
        let Some(o) = s.as_object() else {
            return if s == &Value::Bool(true) { Ok("value".into()) } else { Err("a schema must be an object".into()) };
        };
        for k in ["$ref", "pattern", "format", "minLength", "maxLength", "patternProperties"] {
            if o.contains_key(k) {
                return Err(format!("json_schema '{k}' is not supported by the in-process engine"));
            }
        }
        if let Some(c) = o.get("const") {
            return Ok(format!("( {} )", lit(&c.to_string())));
        }
        if let Some(Value::Array(vals)) = o.get("enum") {
            if vals.is_empty() {
                return Err("json_schema 'enum' is empty".into());
            }
            return Ok(format!("( {} )", Self::literal_alts(vals)));
        }
        for k in ["anyOf", "oneOf"] {
            if let Some(Value::Array(alts)) = o.get(k) {
                let parts: Vec<String> = alts.iter().map(|a| self.schema(a, hint)).collect::<Result<_, _>>()?;
                return Ok(format!("( {} )", parts.join(" | ")));
            }
        }
        match o.get("type") {
            None => {
                if o.contains_key("properties") {
                    self.object(o, hint)
                } else {
                    Ok("value".into())
                }
            }
            Some(Value::Array(types)) => {
                let parts: Vec<String> = types
                    .iter()
                    .map(|t| {
                        let mut one = o.clone();
                        one.insert("type".into(), t.clone());
                        self.schema(&Value::Object(one), hint)
                    })
                    .collect::<Result<_, _>>()?;
                Ok(format!("( {} )", parts.join(" | ")))
            }
            Some(Value::String(t)) => match t.as_str() {
                "string" => Ok("string".into()),
                "number" => Ok("number".into()),
                "integer" => Ok("integer".into()),
                "boolean" => Ok("boolean".into()),
                "null" => Ok("null".into()),
                "object" => self.object(o, hint),
                "array" => {
                    let item = match o.get("items") {
                        Some(i) => self.schema(i, &format!("{hint}-item"))?,
                        None => "value".into(),
                    };
                    let body = format!("\"[\" ws ( {item} ( ws \",\" ws {item} )* )? ws \"]\"");
                    Ok(self.rule(&format!("{hint}-array"), body))
                }
                other => Err(format!("json_schema type '{other}' is not supported")),
            },
            Some(_) => Err("json_schema 'type' must be a string or a list".into()),
        }
    }

    fn object(&mut self, o: &serde_json::Map<String, Value>, hint: &str) -> Result<String, String> {
        let props = match o.get("properties") {
            Some(Value::Object(p)) => p,
            None => return Ok("object-any".into()),
            Some(_) => return Err("json_schema 'properties' must be an object".into()),
        };
        let required: Vec<&str> = o.get("required").and_then(Value::as_array).map(|r| r.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        let kv = |name: &str, sub: &Value, g: &mut Gen| -> Result<String, String> {
            let v = g.schema(sub, name)?;
            Ok(format!("{} ws \":\" ws {v}", lit(&Value::String(name.into()).to_string())))
        };
        let mut req = Vec::new();
        let mut opt = Vec::new();
        for (name, sub) in props {
            let item = kv(name, sub, self)?;
            if required.contains(&name.as_str()) {
                req.push(item);
            } else {
                opt.push(item);
            }
        }
        let body = if !req.is_empty() {
            let mut b = req.join(" ws \",\" ws ");
            for it in &opt {
                b.push_str(&format!(" ( ws \",\" ws {it} )?"));
            }
            format!("\"{{\" ws {b} ws \"}}\"")
        } else if !opt.is_empty() {
            // Each optional property may appear only after the ones before it.
            let mut b = String::new();
            for it in opt.iter().rev() {
                b = if b.is_empty() { it.clone() } else { format!("{it} ( ws \",\" ws {b} )?") };
            }
            format!("\"{{\" ws ( {b} )? ws \"}}\"")
        } else {
            "\"{\" ws \"}\"".into()
        };
        Ok(self.rule(&format!("{hint}-object"), body))
    }
}

/// The GBNF grammar (root rule `root`) for a JSON schema.
pub fn from_schema(schema: &Value) -> Result<String, String> {
    let mut g = Gen { rules: BTreeMap::new(), n: 0 };
    let root = g.schema(schema, "root")?;
    let mut out = format!("root ::= {root}\n{BASE}");
    for (name, body) in &g.rules {
        out.push_str(&format!("{name} ::= {body}\n"));
    }
    Ok(out)
}

/// The grammar for an OpenAI `response_format`, or None for plain text.
/// Accepts `{"type":"json_schema","json_schema":{"schema":...}}`, llama-server's
/// `{"type":"json_object","schema":...}` and plain `{"type":"json_object"}`.
pub fn from_response_format(rf: &Value) -> Result<Option<String>, String> {
    match rf["type"].as_str() {
        None | Some("text") => Ok(None),
        Some("json_schema") => {
            let s = rf["json_schema"].get("schema").or_else(|| rf.get("schema")).ok_or("response_format json_schema needs json_schema.schema")?;
            from_schema(s).map(Some)
        }
        Some("json_object") => match rf.get("schema") {
            Some(s) => from_schema(s).map(Some),
            None => from_schema(&serde_json::json!({ "type": "object" })).map(Some),
        },
        Some(t) => Err(format!("response_format type '{t}' is not supported")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn jitmo_pick_and_fill_schemas() {
        let pick = json!({"type":"object","properties":{"tool":{"type":"string","enum":["email.send","none"]}},"required":["tool"],"additionalProperties":false});
        let g = from_schema(&pick).unwrap();
        assert!(g.starts_with("root ::= root-object-"));
        assert!(g.contains(r#""\"tool\"" ws ":" ws ( "\"email.send\"" | "\"none\"" )"#), "{g}");
        let fill = json!({"type":"object","properties":{"to":{"type":"string"},"body":{"type":"string"}},"required":["to","body"]});
        let g = from_schema(&fill).unwrap();
        assert!(g.contains(r#""\"to\"" ws ":" ws string ws "," ws "\"body\"" ws ":" ws string"#), "{g}");
    }

    #[test]
    fn optional_nested_arrays_and_unions() {
        let s = json!({"type":"object","properties":{
            "a":{"type":"integer"},
            "tags":{"type":"array","items":{"type":"string"}},
            "x":{"type":["string","null"]},
            "inner":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}
        },"required":["a"]});
        let g = from_schema(&s).unwrap();
        assert!(g.contains("( ws \",\" ws \"\\\"tags\\\"\""), "{g}");
        assert!(g.contains("( string | null )"));
        assert!(g.contains("inner-object-"));
        assert!(from_schema(&json!({"type":"string","pattern":"x"})).is_err());
        assert_eq!(from_response_format(&json!({"type":"text"})).unwrap(), None);
        assert!(from_response_format(&json!({"type":"json_object"})).unwrap().unwrap().contains("root ::= object-any"));
    }
}
