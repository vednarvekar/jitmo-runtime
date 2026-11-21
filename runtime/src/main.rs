//! jitmo-runtime v0: a local model runtime. See docs/06-runtime-api.md.

mod api;
mod budget;
mod chain;
mod config;
mod error;
mod gguf;
mod llama;
mod pacer;
mod stats;
mod supervisor;
mod win;

use anyhow::{bail, Context, Result};
use config::Config;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

const USAGE: &str = "\
jitmo-runtime v0 - local model runtime (one llama-server child, chains, OpenAI-compatible API)

Usage: jitmo-runtime [options]
  --config <file>        TOML config (default: ./jitmo-runtime.toml if it exists, else built-in defaults)
  --model <file.gguf>    override model.path
  --llama-server <exe>   override model.llama_server
  --port <n>             override server.port
  --log-level <level>    error | warn | info | debug | trace (default: info)
  -h, --help             show this help";

struct Args {
    config: Option<PathBuf>,
    model: Option<PathBuf>,
    llama_server: Option<PathBuf>,
    port: Option<u16>,
    log_level: tracing::Level,
}

fn parse_args() -> Result<Option<Args>> {
    let mut args = Args { config: None, model: None, llama_server: None, port: None, log_level: tracing::Level::INFO };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--config" => args.config = Some(value()?.into()),
            "--model" => args.model = Some(value()?.into()),
            "--llama-server" => args.llama_server = Some(value()?.into()),
            "--port" => args.port = Some(value()?.parse().context("--port must be a number")?),
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
        cfg.model.path = std::path::absolute(m).unwrap_or(m.clone());
    }
    if let Some(s) = &args.llama_server {
        cfg.model.llama_server = std::path::absolute(s).unwrap_or(s.clone());
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
    tracing_subscriber::fmt().with_max_level(args.log_level).with_target(false).init();
    match run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<()> {
    let cfg = Arc::new(load_config(&args)?);
    let stats = Arc::new(stats::Stats::new());
    let sup = supervisor::Supervisor::new(cfg.clone(), stats);
    let chains = chain::Chain::compile_all(&cfg);
    info!(
        "model '{}' = {} | ctx {} | threads {} | kv {} | cap {} MB ({}) | reserve {} MB | idle unload {} s | pacer {} | chains: {}",
        cfg.model.name,
        cfg.model.path.display(),
        cfg.model.ctx_size,
        sup.threads(),
        cfg.model.kv_cache_type,
        sup.plan.cap_mb(),
        sup.plan.source,
        cfg.budget.reserve_free_mb,
        cfg.budget.idle_unload_secs,
        cfg.pacer.mode,
        if chains.is_empty() { "none".to_string() } else { chains.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ") }
    );
    sup.spawn_watchdog();
    sup.spawn_pacer();

    let addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("cannot listen on {addr}"))?;
    info!("listening on http://{addr} (model loads on the first request)");
    let app = Arc::new(api::App { cfg: cfg.clone(), sup: sup.clone(), chains });

    let shutdown_sup = sup.clone();
    let shutdown = async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("Ctrl+C: stopping the model and shutting down");
        // Stop the child first so running generations end now instead of holding the server open.
        shutdown_sup.shutdown().await;
        tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(5)).await;
            warn!("open connections did not close in 5 s; exiting anyway");
            std::process::exit(0);
        });
    };
    axum::serve(listener, api::router(app)).with_graceful_shutdown(shutdown).await.context("server error")?;
    sup.shutdown().await;
    info!("stopped");
    Ok(())
}
