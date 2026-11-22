//! Talking to the llama-server child over plain HTTP on 127.0.0.1, and reading its timings.

use crate::error::Error;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use hyper_util::client::legacy::{connect::HttpConnector, Client};
use hyper_util::rt::TokioExecutor;
use serde_json::Value;
use std::time::{Duration, Instant};

pub type HttpClient = Client<HttpConnector, Full<Bytes>>;

pub fn client() -> HttpClient {
    Client::builder(TokioExecutor::new()).build_http()
}

fn url(port: u16, path: &str) -> String {
    format!("http://127.0.0.1:{port}{path}")
}

/// Status of a GET, or None if there was no answer within `timeout`.
pub async fn get_status(client: &HttpClient, port: u16, path: &str, timeout: Duration) -> Option<StatusCode> {
    let req = Request::get(url(port, path)).body(Full::new(Bytes::new())).ok()?;
    let fut = async {
        let resp = client.request(req).await.ok()?;
        let status = resp.status();
        let _ = resp.into_body().collect().await;
        Some(status)
    };
    tokio::time::timeout(timeout, fut).await.ok().flatten()
}

pub async fn post_json(client: &HttpClient, port: u16, path: &str, body: &Value) -> Result<Response<Incoming>, Error> {
    let bytes = serde_json::to_vec(body).map_err(|e| Error::bad_request(e.to_string()))?;
    let req = Request::post(url(port, path))
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(bytes)))
        .map_err(|e| Error::engine(e.to_string()))?;
    client.request(req).await.map_err(|e| {
        Error::engine(format!(
            "no answer from the model process ({e}); it may have crashed or been unloaded for memory pressure"
        ))
    })
}

pub async fn collect(resp: Response<Incoming>) -> Result<(StatusCode, Bytes), Error> {
    let status = resp.status();
    let body = resp
        .into_body()
        .collect()
        .await
        .map_err(|e| Error::engine(format!("the model process stopped while answering ({e})")))?
        .to_bytes();
    Ok((status, body))
}

/// One non-streamed chat completion; an engine error status becomes an `Error`.
pub async fn chat_once(client: &HttpClient, port: u16, body: &Value) -> Result<Value, Error> {
    let resp = post_json(client, port, "/v1/chat/completions", body).await?;
    let (status, bytes) = collect(resp).await?;
    let v: Value = serde_json::from_slice(&bytes)
        .map_err(|_| Error::engine(format!("engine returned {status} with a non-JSON body")))?;
    if !status.is_success() {
        let msg = v["error"]["message"].as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
        return Err(Error::engine(format!("engine returned {status}: {msg}")));
    }
    Ok(v)
}

/// Make the request's LoRA list complete: one entry per loaded adapter, scale 0 for every
/// adapter the request does not name. llama.cpp (build 12248) applies an adapter that a request
/// leaves out at its load-time scale (1.0), even with `--lora-init-without-apply`, so a request
/// without `lora` would otherwise run with every adapter on (measured 22 Nov 2026).
pub fn full_lora(req: &mut Value, n_adapters: usize) -> Result<(), Error> {
    if n_adapters == 0 {
        return Ok(());
    }
    let mut scales = vec![0.0f64; n_adapters];
    match req.get("lora") {
        None | Some(Value::Null) => {}
        Some(Value::Array(items)) => {
            for it in items {
                let id = it["id"].as_u64().ok_or_else(|| Error::bad_request("each 'lora' entry needs a numeric 'id'"))? as usize;
                let scale = it["scale"].as_f64().ok_or_else(|| Error::bad_request("each 'lora' entry needs a numeric 'scale'"))?;
                if id >= n_adapters {
                    return Err(Error::bad_request(format!("'lora' id {id} does not exist; this model has {n_adapters} adapter(s), ids 0 to {}", n_adapters - 1)));
                }
                scales[id] = scale;
            }
        }
        Some(_) => return Err(Error::bad_request("'lora' must be an array of {\"id\", \"scale\"}")),
    }
    req["lora"] = Value::Array(scales.into_iter().enumerate().map(|(id, scale)| serde_json::json!({ "id": id, "scale": scale })).collect());
    Ok(())
}

/// The assistant text of a chat completion.
pub fn content(v: &Value) -> String {
    v["choices"][0]["message"]["content"].as_str().unwrap_or("").trim().to_string()
}

/// llama-server's own timings for one generation (the `timings` object in its responses).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineTimings {
    pub prompt_n: u64,
    pub prompt_ms: f64,
    pub predicted_n: u64,
    pub predicted_ms: f64,
}

/// Timings from a response (or final stream chunk). Falls back to `usage` token counts.
pub fn engine_timings(v: &Value) -> Option<EngineTimings> {
    if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
        return Some(EngineTimings {
            prompt_n: t["prompt_n"].as_u64().unwrap_or(0),
            prompt_ms: t["prompt_ms"].as_f64().unwrap_or(0.0),
            predicted_n: t["predicted_n"].as_u64().unwrap_or(0),
            predicted_ms: t["predicted_ms"].as_f64().unwrap_or(0.0),
        });
    }
    let u = v.get("usage").filter(|u| u.is_object())?;
    Some(EngineTimings {
        prompt_n: u["prompt_tokens"].as_u64().unwrap_or(0),
        predicted_n: u["completion_tokens"].as_u64().unwrap_or(0),
        ..Default::default()
    })
}

/// Watches an SSE stream as it passes through: when the first token arrives, and the
/// timings in the final chunk.
#[derive(Default)]
pub struct SseTracker {
    line: Vec<u8>,
    pub first_token_at: Option<Instant>,
    pub timings: Option<EngineTimings>,
    pub done: bool,
}

impl SseTracker {
    pub fn feed(&mut self, chunk: &[u8]) {
        for &b in chunk {
            if b == b'\n' {
                let line = std::mem::take(&mut self.line);
                self.on_line(&line);
            } else if self.line.len() < 4 << 20 {
                self.line.push(b);
            }
        }
    }

    fn on_line(&mut self, line: &[u8]) {
        let Ok(text) = std::str::from_utf8(line) else { return };
        let Some(data) = text.trim_end_matches('\r').strip_prefix("data:") else { return };
        let data = data.trim();
        if data == "[DONE]" {
            self.done = true;
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { return };
        if self.first_token_at.is_none() {
            let delta = &v["choices"][0]["delta"];
            let has_text = |k: &str| delta[k].as_str().is_some_and(|s| !s.is_empty());
            if has_text("content") || has_text("reasoning_content") {
                self.first_token_at = Some(Instant::now());
            }
        }
        if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
            self.timings = engine_timings(&serde_json::json!({ "timings": t }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_lora_list_is_always_complete() {
        let mut none = json!({ "messages": [] });
        full_lora(&mut none, 2).unwrap();
        assert_eq!(none["lora"], json!([{ "id": 0, "scale": 0.0 }, { "id": 1, "scale": 0.0 }]));
        let mut one = json!({ "lora": [{ "id": 1, "scale": 0.5 }] });
        full_lora(&mut one, 2).unwrap();
        assert_eq!(one["lora"], json!([{ "id": 0, "scale": 0.0 }, { "id": 1, "scale": 0.5 }]));
        let mut bad = json!({ "lora": [{ "id": 2, "scale": 1.0 }] });
        assert_eq!(full_lora(&mut bad, 2).unwrap_err().status, 400);
        let mut untouched = json!({ "x": 1 });
        full_lora(&mut untouched, 0).unwrap();
        assert!(untouched.get("lora").is_none());
    }

    #[test]
    fn timings_are_read_from_timings_or_usage() {
        let v = json!({"timings": {"prompt_n": 12, "prompt_ms": 80.5, "predicted_n": 30, "predicted_ms": 1500.0}});
        assert_eq!(
            engine_timings(&v),
            Some(EngineTimings { prompt_n: 12, prompt_ms: 80.5, predicted_n: 30, predicted_ms: 1500.0 })
        );
        let u = json!({"usage": {"prompt_tokens": 5, "completion_tokens": 7}});
        assert_eq!(engine_timings(&u).unwrap().predicted_n, 7);
        assert_eq!(engine_timings(&json!({})), None);
    }

    #[test]
    fn the_sse_tracker_finds_first_token_and_final_timings_across_chunk_splits() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":3,\"prompt_ms\":10.0,\"predicted_n\":2,\"predicted_ms\":50.0}}\n\n",
            "data: [DONE]\n\n"
        );
        let mut t = SseTracker::default();
        let (a, b) = stream.as_bytes().split_at(70);
        t.feed(a);
        assert!(t.first_token_at.is_none());
        t.feed(b);
        assert!(t.first_token_at.is_some());
        assert_eq!(t.timings.as_ref().unwrap().predicted_n, 2);
        assert!(t.done);
    }
}
