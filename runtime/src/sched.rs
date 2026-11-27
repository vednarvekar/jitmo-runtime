//! Who may generate now: a priority queue of generation slots shared by all models, cancel
//! tokens, and the registry of in-flight requests (for `DELETE /v1/requests/{id}`).
//!
//! - `server.max_concurrent` slots (default 1: the models share the memory bandwidth).
//! - Waiting requests are ordered interactive first, then by arrival.
//! - Preemption (`server.preempt_background`, on by default): when an interactive request has
//!   to wait and no slot runs interactive work, the newest running background request is
//!   stopped with `503 preempted` (the client may retry it), so the interactive one starts at
//!   once instead of after the whole background answer.
//! - A waiting request that is cancelled (or whose client goes away) leaves the queue, because
//!   its future is dropped; `QueueGuard` removes it.

use crate::error::Error;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{oneshot, Notify};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Interactive = 0,
    Background = 1,
}

impl Priority {
    pub fn parse(s: &str) -> Option<Priority> {
        match s.trim().to_ascii_lowercase().as_str() {
            "interactive" => Some(Priority::Interactive),
            "background" => Some(Priority::Background),
            _ => None,
        }
    }
}

// ---------- cancel token ----------

#[derive(Default)]
struct CancelInner {
    flag: AtomicBool,
    notify: Notify,
    /// Set when the runtime (not the client) cancels: the error the client gets instead of
    /// `request_cancelled`.
    reason: std::sync::Mutex<Option<Error>>,
}

#[derive(Clone, Default)]
pub struct CancelToken(Arc<CancelInner>);

impl CancelToken {
    pub fn cancel(&self) {
        self.0.flag.store(true, Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }
    /// Cancel with the error the client should get (the runtime stops the request).
    pub fn cancel_with(&self, e: Error) {
        let mut r = self.0.reason.lock().unwrap_or_else(|p| p.into_inner());
        r.get_or_insert(e);
        drop(r);
        self.cancel();
    }
    /// The error set by `cancel_with`, if any.
    pub fn reason(&self) -> Option<Error> {
        self.0.reason.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.flag.load(Ordering::SeqCst)
    }
    /// Resolves once `cancel` has been called.
    pub async fn cancelled(&self) {
        loop {
            let n = self.0.notify.notified();
            tokio::pin!(n);
            n.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            n.await;
        }
    }
}

// ---------- scheduler ----------

struct Waiter {
    seq: u64,
    priority: Priority,
    token: Option<CancelToken>,
    tx: oneshot::Sender<Permit>,
}

/// A slot in use.
struct Running {
    id: u64,
    priority: Priority,
    token: Option<CancelToken>,
}

#[derive(Default)]
struct SchedInner {
    running: Vec<Running>,
    seq: u64,
    queue: Vec<Waiter>,
    preempted: u64,
}

pub struct Scheduler {
    max: usize,
    max_queue: usize,
    preempt: bool,
    inner: Mutex<SchedInner>,
}

/// One generation slot. Dropping it hands the slot to the best waiting request.
pub struct Permit {
    sched: Option<Arc<Scheduler>>,
    id: u64,
}

impl Drop for Permit {
    fn drop(&mut self) {
        if let Some(s) = self.sched.take() {
            s.release(self.id);
        }
    }
}

/// Removes a waiter from the queue if its request is dropped (cancelled, client gone).
struct QueueGuard<'a> {
    sched: &'a Scheduler,
    seq: u64,
}

impl Drop for QueueGuard<'_> {
    fn drop(&mut self) {
        self.sched.lock().queue.retain(|w| w.seq != self.seq);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SchedSnapshot {
    pub max_concurrent: usize,
    pub running: usize,
    pub waiting: usize,
    pub waiting_interactive: usize,
    pub waiting_background: usize,
    pub max_queue: usize,
    pub preempt_background: bool,
    /// Background generations stopped for an interactive request, over the runtime's life.
    pub preempted_total: u64,
}

impl Scheduler {
    pub fn new(max_concurrent: usize, max_queue: usize, preempt: bool) -> Arc<Scheduler> {
        Arc::new(Scheduler { max: max_concurrent.max(1), max_queue, preempt, inner: Mutex::new(SchedInner::default()) })
    }

    fn start(&self, i: &mut SchedInner, priority: Priority, token: Option<CancelToken>) -> u64 {
        i.seq += 1;
        let id = i.seq;
        i.running.push(Running { id, priority, token });
        id
    }

    /// An interactive request must wait: if no slot runs interactive work, stop the newest
    /// running background request that is not stopped yet.
    fn preempt_for_interactive(&self, i: &mut SchedInner) {
        if !self.preempt || i.running.iter().any(|r| r.priority == Priority::Interactive) {
            return;
        }
        let victim = i.running.iter().rev().find_map(|r| r.token.clone().filter(|t| !t.is_cancelled()));
        if let Some(t) = victim {
            t.cancel_with(Error::preempted("stopped so an interactive request could run; retry later"));
            i.preempted += 1;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SchedInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `token` = the request's cancel token, so a running background request can be preempted.
    pub async fn acquire(self: &Arc<Self>, priority: Priority, token: Option<CancelToken>) -> Result<Permit, Error> {
        let (rx, seq) = {
            let mut i = self.lock();
            // Waiters exist only while every slot is taken, so a free slot means nobody is ahead.
            if i.running.len() < self.max && i.queue.is_empty() {
                let id = self.start(&mut i, priority, token);
                return Ok(Permit { sched: Some(self.clone()), id });
            }
            if i.queue.len() >= self.max_queue {
                return Err(Error::queue_full(format!(
                    "{} generation(s) running and {} requests waiting (server.max_queue = {})",
                    i.running.len(),
                    i.queue.len(),
                    self.max_queue
                )));
            }
            i.seq += 1;
            let seq = i.seq;
            let (tx, rx) = oneshot::channel();
            i.queue.push(Waiter { seq, priority, token, tx });
            if priority == Priority::Interactive {
                self.preempt_for_interactive(&mut i);
            }
            (rx, seq)
        };
        let _guard = QueueGuard { sched: self, seq };
        rx.await.map_err(|_| Error::engine("runtime is shutting down"))
    }

    fn release(self: &Arc<Self>, id: u64) {
        let mut i = self.lock();
        i.running.retain(|r| r.id != id);
        // Hand the free slot to the best waiter. If its request is gone, try the next one.
        while i.running.len() < self.max {
            let best = i.queue.iter().enumerate().min_by_key(|(_, w)| (w.priority, w.seq)).map(|(k, _)| k);
            let Some(k) = best else { return };
            let w = i.queue.remove(k);
            let id = self.start(&mut i, w.priority, w.token);
            if let Err(mut p) = w.tx.send(Permit { sched: Some(self.clone()), id }) {
                p.sched = None; // not handed over: free the slot here, without re-entering release
                i.running.retain(|r| r.id != id);
            }
        }
    }

    pub fn snapshot(&self) -> SchedSnapshot {
        let i = self.lock();
        let bg = i.queue.iter().filter(|w| w.priority == Priority::Background).count();
        SchedSnapshot {
            max_concurrent: self.max,
            running: i.running.len(),
            waiting: i.queue.len(),
            waiting_interactive: i.queue.len() - bg,
            waiting_background: bg,
            max_queue: self.max_queue,
            preempt_background: self.preempt,
            preempted_total: i.preempted,
        }
    }
}

// ---------- in-flight requests ----------

const QUEUED: u8 = 0;
const RUNNING: u8 = 1;

struct Entry {
    token: CancelToken,
    kind: &'static str,
    model: String,
    priority: Priority,
    started: Instant,
    state: Arc<AtomicU8>,
}

#[derive(Default)]
pub struct Requests {
    next: AtomicU64,
    prefix: String,
    map: Mutex<HashMap<String, Entry>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestInfo {
    pub id: String,
    pub kind: &'static str,
    pub model: String,
    pub priority: Priority,
    /// `queued` (waiting for a slot) or `running` (loading its model or generating).
    pub state: &'static str,
    pub age_ms: f64,
}

/// A registered request. Dropping it removes it from the registry.
pub struct RequestGuard {
    reqs: Arc<Requests>,
    pub id: String,
    pub token: CancelToken,
    pub priority: Priority,
    state: Arc<AtomicU8>,
}

impl RequestGuard {
    pub fn set_running(&self) {
        self.state.store(RUNNING, Ordering::SeqCst);
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.reqs.lock().remove(&self.id);
    }
}

fn valid_id(id: &str) -> bool {
    (1..=128).contains(&id.len()) && id.chars().all(|c| c.is_ascii_alphanumeric() || "._:-".contains(c))
}

impl Requests {
    pub fn new() -> Arc<Requests> {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        Arc::new(Requests { prefix: format!("req-{t:x}"), ..Default::default() })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Entry>> {
        self.map.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Register a request. `wanted` = an id chosen by the client (`X-Jitmo-Request-Id`), so it
    /// can cancel a request before any response header arrives.
    pub fn begin(
        self: &Arc<Self>,
        wanted: Option<&str>,
        kind: &'static str,
        model: &str,
        priority: Priority,
    ) -> Result<RequestGuard, Error> {
        let id = match wanted {
            Some(w) if !valid_id(w) => {
                return Err(Error::bad_request("X-Jitmo-Request-Id must be 1 to 128 characters: letters, digits, . _ : -"))
            }
            Some(w) => w.to_string(),
            None => format!("{}-{}", self.prefix, self.next.fetch_add(1, Ordering::SeqCst) + 1),
        };
        let token = CancelToken::default();
        let state = Arc::new(AtomicU8::new(QUEUED));
        let mut map = self.lock();
        if map.contains_key(&id) {
            return Err(Error::conflict(format!("request id '{id}' is already in use by a running request")));
        }
        map.insert(
            id.clone(),
            Entry { token: token.clone(), kind, model: model.to_string(), priority, started: Instant::now(), state: state.clone() },
        );
        Ok(RequestGuard { reqs: self.clone(), id, token, priority, state })
    }

    /// Cancel a request: returns the state it was in, or None if no such request is in flight.
    pub fn cancel(&self, id: &str) -> Option<&'static str> {
        let map = self.lock();
        let e = map.get(id)?;
        e.token.cancel();
        Some(if e.state.load(Ordering::SeqCst) == RUNNING { "running" } else { "queued" })
    }

    pub fn list(&self) -> Vec<RequestInfo> {
        let map = self.lock();
        let mut v: Vec<RequestInfo> = map
            .iter()
            .map(|(id, e)| RequestInfo {
                id: id.clone(),
                kind: e.kind,
                model: e.model.clone(),
                priority: e.priority,
                state: if e.state.load(Ordering::SeqCst) == RUNNING { "running" } else { "queued" },
                age_ms: (e.started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0,
            })
            .collect();
        v.sort_by(|a, b| b.age_ms.total_cmp(&a.age_ms));
        v
    }
}

/// Run `work` unless `token` is cancelled first; on cancel, `work` is dropped (which closes the
/// connection to llama-server, and llama-server stops that generation).
pub async fn cancellable<T>(token: &CancelToken, id: &str, work: impl std::future::Future<Output = Result<T, Error>>) -> Result<T, Error> {
    tokio::select! {
        biased;
        _ = token.cancelled() => Err(token.reason().unwrap_or_else(|| Error::cancelled(format!("request {id} was cancelled")))),
        r = work => r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn take(s: Arc<Scheduler>, p: Priority, log: Arc<Mutex<Vec<&'static str>>>, name: &'static str) {
        let _permit = s.acquire(p, None).await.unwrap();
        log.lock().unwrap().push(name);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    #[tokio::test]
    async fn interactive_requests_go_before_queued_background_ones() {
        let s = Scheduler::new(1, 8, false);
        let log = Arc::new(Mutex::new(Vec::new()));
        let first = s.acquire(Priority::Background, None).await.unwrap(); // running, not interrupted
        let mut tasks = Vec::new();
        for (p, name) in [(Priority::Background, "bg1"), (Priority::Background, "bg2"), (Priority::Interactive, "ia1"), (Priority::Interactive, "ia2")] {
            tasks.push(tokio::spawn(take(s.clone(), p, log.clone(), name)));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let snap = s.snapshot();
        assert_eq!((snap.running, snap.waiting, snap.waiting_background), (1, 4, 2));
        drop(first);
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(*log.lock().unwrap(), vec!["ia1", "ia2", "bg1", "bg2"]);
        assert_eq!(s.snapshot().running, 0);
    }

    #[tokio::test]
    async fn an_interactive_request_preempts_a_running_background_one() {
        let s = Scheduler::new(1, 8, true);
        let bg = CancelToken::default();
        let held = s.acquire(Priority::Background, Some(bg.clone())).await.unwrap();
        // A background waiter does not preempt.
        let s2 = s.clone();
        let bg2 = tokio::spawn(async move { s2.acquire(Priority::Background, Some(CancelToken::default())).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(!bg.is_cancelled());
        let s3 = s.clone();
        let ia = tokio::spawn(async move { s3.acquire(Priority::Interactive, None).await.map(|_| "ia") });
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(bg.is_cancelled(), "the running background request is stopped");
        assert_eq!(bg.reason().unwrap().code, "preempted");
        assert_eq!(s.snapshot().preempted_total, 1);
        drop(held); // the stopped work ends and frees its slot
        assert_eq!(ia.await.unwrap().unwrap(), "ia");
        bg2.await.unwrap().unwrap();
        assert_eq!(s.snapshot().running, 0);
    }

    #[tokio::test]
    async fn no_preemption_when_off_or_interactive_is_running() {
        let off = Scheduler::new(1, 8, false);
        let bg = CancelToken::default();
        let held = off.acquire(Priority::Background, Some(bg.clone())).await.unwrap();
        let o2 = off.clone();
        let w = tokio::spawn(async move { o2.acquire(Priority::Interactive, None).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(!bg.is_cancelled());
        drop(held);
        w.await.unwrap().unwrap();

        let on = Scheduler::new(2, 8, true);
        let bg = CancelToken::default();
        let _ia = on.acquire(Priority::Interactive, None).await.unwrap();
        let _b = on.acquire(Priority::Background, Some(bg.clone())).await.unwrap();
        let o3 = on.clone();
        let _w = tokio::spawn(async move { o3.acquire(Priority::Interactive, None).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(!bg.is_cancelled(), "interactive work already runs, so background keeps going");
    }

    #[tokio::test]
    async fn a_dropped_waiter_leaves_the_queue_and_the_queue_is_bounded() {
        let s = Scheduler::new(1, 1, true);
        let held = s.acquire(Priority::Interactive, None).await.unwrap();
        let s2 = s.clone();
        let waiter = tokio::spawn(async move { s2.acquire(Priority::Interactive, None).await.map(|_| ()) });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(s.snapshot().waiting, 1);
        assert!(s.acquire(Priority::Interactive, None).await.is_err(), "queue full");
        waiter.abort();
        let _ = waiter.await;
        assert_eq!(s.snapshot().waiting, 0, "an aborted waiter is removed");
        drop(held);
        assert_eq!(s.snapshot().running, 0);
        let again = s.acquire(Priority::Background, None).await.unwrap();
        assert_eq!(s.snapshot().running, 1);
        drop(again);
    }

    #[tokio::test]
    async fn max_concurrent_two_runs_two() {
        let s = Scheduler::new(2, 4, true);
        let a = s.acquire(Priority::Interactive, None).await.unwrap();
        let _b = s.acquire(Priority::Interactive, None).await.unwrap();
        assert_eq!(s.snapshot().running, 2);
        let s2 = s.clone();
        let c = tokio::spawn(async move { s2.acquire(Priority::Interactive, None).await.is_ok() });
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(s.snapshot().waiting, 1);
        drop(a);
        assert!(c.await.unwrap());
    }

    #[tokio::test]
    async fn cancel_ends_the_work_and_the_registry_tracks_state() {
        let reqs = Requests::new();
        let g = reqs.begin(None, "chat", "m", Priority::Interactive).unwrap();
        assert!(g.id.starts_with("req-"));
        assert!(reqs.begin(Some(&g.id), "chat", "m", Priority::Interactive).is_err(), "ids are unique");
        assert!(reqs.begin(Some("bad id!"), "chat", "m", Priority::Interactive).is_err());
        let mine = reqs.begin(Some("bar-42"), "chat", "m", Priority::Background).unwrap();
        assert_eq!(reqs.list().len(), 2);
        assert_eq!(reqs.cancel("bar-42"), Some("queued"));
        let r: Result<(), Error> = cancellable(&mine.token, &mine.id, std::future::pending()).await;
        assert_eq!(r.unwrap_err().status, 499);
        g.set_running();
        assert_eq!(reqs.cancel(&g.id), Some("running"));
        drop(mine);
        drop(g);
        assert!(reqs.list().is_empty());
        assert_eq!(reqs.cancel("bar-42"), None);
        assert_eq!(Priority::parse(" Background "), Some(Priority::Background));
        assert_eq!(Priority::parse("urgent"), None);
    }

    #[tokio::test]
    async fn a_runtime_cancel_gives_its_own_error() {
        let t = CancelToken::default();
        t.cancel_with(Error::memory_pressure("free RAM stayed low"));
        t.cancel_with(Error::cancelled("later cancels do not replace the first reason"));
        let r: Result<(), Error> = cancellable(&t, "x", std::future::pending()).await;
        let e = r.unwrap_err();
        assert_eq!((e.status, e.code, e.message.as_str()), (503, "memory_pressure", "free RAM stayed low"));
        let plain = CancelToken::default();
        plain.cancel();
        assert!(plain.reason().is_none());
        let r: Result<(), Error> = cancellable(&plain, "y", std::future::pending()).await;
        assert_eq!(r.unwrap_err().code, "request_cancelled");
    }
}
