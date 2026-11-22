//! jitmo-runtime: a local model runtime. See docs/06-runtime-api.md.

mod api;
mod budget;
mod chain;
mod config;
mod error;
mod gguf;
mod llama;
mod pacer;
mod pool;
mod sched;
mod stats;
mod win;

use anyhow::{bail, Context, Result};
use config::Config;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

const USAGE: &str = "\
jitmo-runtime v0.2 - local model runtime (a pool of llama-server children, chains, OpenAI-compatible API)

Usage: jitmo-runtime [options]
  --config <file>        TOML config (default: ./jitmo-runtime.toml if it exists, else built-in defaults)
  --model <file.gguf>    override the path of the first model
  --llama-server <exe>   override llama_server of every model
  --port <n>             override server.port (0 = any free port)
  --parent-pid <pid>     exit (and stop all models) when that process exits
  --port-file <file>     also write the JITMO_RUNTIME_LISTENING line to this file (removed at exit)
  --log-file <file>      write the log to this file instead of stderr
  --log-level <level>    error | warn | info | debug | trace (default: info)
  -h, --help             show this help

stdout carries exactly one line, `JITMO_RUNTIME_LISTENING port=<n>`, once the API is listening.
The log goes to stderr (or --log-file).";

struct Args {
    config: Option<PathBuf>,
    model: Option<PathBuf>,
    llama_server: Option<PathBuf>,
    port: Option<u16>,
    parent_pid: Option<u32>,
    port_file: Option<PathBuf>,
    log_file: Option<PathBuf>,
    log_level: tracing::Level,
}

fn parse_args() -> Result<Option<Args>> {
    let mut args = Args {
        config: None,
        model: None,
        llama_server: None,
        port: None,
        parent_pid: None,
        port_file: None,
        log_file: None,
        log_level: tracing::Level::INFO,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--config" => args.config = Some(value()?.into()),
            "--model" => args.model = Some(value()?.into()),
            "--llama-server" => args.llama_server = Some(value()?.into()),
            "--port" => args.port = Some(value()?.parse().context("--port must be a number")?),
            "--parent-pid" => args.parent_pid = Some(value()?.parse().context("--parent-pid must be a number")?),
            "--port-file" => args.port_file = Some(value()?.into()),
            "--log-file" => args.log_file = Some(value()?.into()),
            "--log-level" => args.log_level = value()?.parse().map_err(|_| anyhow::anyhow!("bad --log-level"))?,
            other => bail!("unknown argument '{other}'\n\n{USAGE}"),
        }
    }
    Ok(Some(args))
}

fn load_config(args: &Args) -> Result<Config> {
    let mut cfg = match &args.config {
        Some(p) => Config::load(p)?,
        None => {
            let p = PathBuf::from(config::DEFAULT_CONFIG_FILE);
            if p.is_file() {
                Config::load(&p)?
            } else {
                warn!("no config file given and no ./{} found: using built-in defaults (no chains)", config::DEFAULT_CONFIG_FILE);
                Config::defaults(&std::env::current_dir().unwrap_or_default())
            }
        }
    };
    if let Some(m) = &args.model {
        cfg.models[0].path = std::path::absolute(m).unwrap_or(m.clone());
    }
    if let Some(s) = &args.llama_server {
        for m in &mut cfg.models {
            m.llama_server = std::path::absolute(s).unwrap_or(s.clone());
        }
    }
    if let Some(p) = args.port {
        cfg.server.port = p;
    }
    cfg.check_files()?;
    Ok(cfg)
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(Some(a)) => a,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            return ExitCode::from(2);
        }
    };
    let sub = tracing_subscriber::fmt().with_max_level(args.log_level).with_target(false);
    match &args.log_file {
        Some(path) => match std::fs::File::create(path) {
            Ok(f) => sub.with_ansi(false).with_writer(std::sync::Mutex::new(f)).init(),
            Err(e) => {
                eprintln!("error: cannot write log file {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        None => sub.with_writer(std::io::stderr).init(),
    }
    match run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e:#}");
            ExitCode::FAILURE
        }
    }
}

/// Resolves when the runtime should stop: Ctrl+C, or the parent process exited.
async fn stop_signal(parent: Option<(u32, win::ProcessWaiter)>) {
    let parent_gone = async move {
        match parent {
            Some((pid, w)) => {
                let _ = tokio::task::spawn_blocking(move || w.wait()).await;
                info!("parent process {pid} exited");
            }
            None => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => info!("Ctrl+C"),
        _ = parent_gone => {}
    }
}

fn write_port_file(path: &std::path::Path, line: &str) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, format!("{line}\n")).with_context(|| format!("cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

async fn run(args: Args) -> Result<()> {
    // Open the parent first: if it is already gone there is nothing to serve.
    let parent = match args.parent_pid {
        Some(pid) => {
            let w = win::ProcessWaiter::open(pid).with_context(|| format!("--parent-pid {pid}: no such process"))?;
            if w.exited() {
                bail!("--parent-pid {pid}: that process has already exited");
            }
            Some((pid, w))
        }
        None => None,
    };
    let cfg = Arc::new(load_config(&args)?);
    let stats = Arc::new(stats::Stats::new());
    let pool = pool::Pool::new(cfg.clone(), stats);
    let chains = chain::Chain::compile_all(&cfg);
    for m in &pool.models {
        info!(
            "model '{}' ({}{}) = {} | ctx {} | threads {} | kv {} | cap {} MB ({}) | load estimate {} MB",
            m.name(),
            m.cfg.kind.as_str(),
            if m.cfg.pinned { ", pinned" } else { "" },
            m.cfg.path.display(),
            m.cfg.ctx_size,
            m.threads,
            m.cfg.kv_cache_type,
            m.plan.cap_mb(),
            m.plan.source,
            m.plan.load_bytes() / (1024 * 1024),
        );
    }
    info!(
        "pool budget {} | max concurrent {} | reserve {} MB | idle unload {} s | pacer {} | chains: {}",
        if cfg.budget.pool_budget_mb > 0 { format!("{} MB", cfg.budget.pool_budget_mb) } else { "none".into() },
        cfg.server.max_concurrent,
        cfg.budget.reserve_free_mb,
        cfg.budget.idle_unload_secs,
        cfg.pacer.mode,
        if chains.is_empty() { "none".to_string() } else { chains.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ") }
    );
    pool.spawn_watchdog();
    pool.spawn_pacer();

    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("cannot listen on {addr}"))?;
    let port = listener.local_addr().context("no local address")?.port();
    let line = format!("JITMO_RUNTIME_LISTENING port={port}");
    if let Some(p) = &args.port_file {
        write_port_file(p, &line)?;
    }
    {
        // A closed stdout (parent stopped reading) must not stop the runtime.
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
    info!("listening on http://{}:{port} (models load on demand)", cfg.server.host);
    let app = Arc::new(api::App { cfg: cfg.clone(), pool: pool.clone(), chains });

    let shutdown_pool = pool.clone();
    let shutdown = async move {
        stop_signal(parent).await;
        info!("stopping the models and shutting down");
        // Stop the children first so running generations end now instead of holding the server open.
        shutdown_pool.shutdown().await;
        tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(5)).await;
            warn!("open connections did not close in 5 s; exiting anyway");
            std::process::exit(0);
        });
    };
    axum::serve(listener, api::router(app)).with_graceful_shutdown(shutdown).await.context("server error")?;
    pool.shutdown().await;
    if let Some(p) = &args.port_file {
        let _ = std::fs::remove_file(p);
    }
    info!("stopped");
    Ok(())
}
