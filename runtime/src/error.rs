//! One error type for the whole runtime. The API turns it into an OpenAI-style error body:
//! `{"error": {"message": ..., "type": <code>, "code": <code>}}`.

#[derive(Debug, Clone)]
pub struct Error {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl Error {
    fn new(status: u16, code: &'static str, message: impl Into<String>) -> Error {
        Error { status, code, message: message.into() }
    }
    pub fn bad_request(m: impl Into<String>) -> Error {
        Error::new(400, "invalid_request_error", m)
    }
    pub fn model_not_found(m: impl Into<String>) -> Error {
        Error::new(404, "model_not_found", m)
    }
    pub fn chain_not_found(m: impl Into<String>) -> Error {
        Error::new(404, "chain_not_found", m)
    }
    /// A chain template could not be filled (for example a JSON field is missing).
    pub fn chain_step(m: impl Into<String>) -> Error {
        Error::new(422, "chain_step_failed", m)
    }
    pub fn queue_full(m: impl Into<String>) -> Error {
        Error::new(503, "queue_full", m)
    }
    pub fn insufficient_memory(m: impl Into<String>) -> Error {
        Error::new(503, "insufficient_memory", m)
    }
    pub fn load_failed(m: impl Into<String>) -> Error {
        Error::new(503, "model_load_failed", m)
    }
    /// llama-server failed or went away during a request.
    pub fn engine(m: impl Into<String>) -> Error {
        Error::new(502, "engine_error", m)
    }

    /// The request was cancelled with `DELETE /v1/requests/{id}` (status 499, as nginx uses
    /// for "client closed request").
    pub fn cancelled(m: impl Into<String>) -> Error {
        Error::new(499, "request_cancelled", m)
    }
    /// The runtime stopped the generation to unload its model: free RAM stayed short for
    /// longer than `budget.pressure_finish_max_ms`.
    pub fn memory_pressure(m: impl Into<String>) -> Error {
        Error::new(503, "memory_pressure", m)
    }
    /// Refused by the local-only guard (a web page, a foreign Host, a non-JSON body).
    pub fn forbidden(m: impl Into<String>) -> Error {
        Error::new(403, "forbidden", m)
    }
    /// A background generation stopped so a waiting interactive request can run.
    pub fn preempted(m: impl Into<String>) -> Error {
        Error::new(503, "preempted", m)
    }
    pub fn request_not_found(m: impl Into<String>) -> Error {
        Error::new(404, "request_not_found", m)
    }
    /// A client-chosen request id that is already in use.
    pub fn conflict(m: impl Into<String>) -> Error {
        Error::new(409, "request_id_in_use", m)
    }
    /// The model exists but cannot do this (for example embeddings on a chat model).
    pub fn wrong_kind(m: impl Into<String>) -> Error {
        Error::new(400, "wrong_model_kind", m)
    }

    pub fn with_prefix(mut self, prefix: &str) -> Error {
        self.message = format!("{prefix}: {}", self.message);
        self
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}
