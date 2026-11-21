//! Foreground "user" probe. Simulates an app while something else (the model) runs.
//! Usage: probe [--secs 20] [--mb 200] [--tag name]
//! Phase 1 "frame" (secs): loop { sleep 16 ms; then do a fixed CPU task }.
//!   lateness_ms = how much later than 16 ms the sleep returned (frame jitter).
//!   work_ms     = time to finish the fixed CPU task (about 2 ms when the PC is idle).
//! Phase 2 "membuf" (secs): repeatedly touch (read+write) one byte per 64-byte line of a 200 MB buffer.
//!   pass_ms = time per full pass (a memory-bound app).
//! Prints epoch-ms timestamps of each phase so other logs (model tokens) can be matched to the window.
use jitmo_bench::stats::*;
use std::time::{Duration, Instant};

#[inline(never)]
fn cpu_task(iters: u64) -> u64 {
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    for _ in 0..iters {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    x
}

fn summary(name: &str, v: &[f64]) {
    println!(
        "{}: n={} p50={:.3} p95={:.3} p99={:.3} max={:.3} mean={:.3}",
        name,
        v.len(),
        pct(v, 50.0),
        pct(v, 95.0),
        pct(v, 99.0),
        max(v),
        mean(v)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| -> String {
        args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap_or(d.to_string())
    };
    let secs: f64 = get("--secs", "20").parse().unwrap();
    let mb: usize = get("--mb", "200").parse().unwrap();
    let tag = get("--tag", "run");
    let iters: u64 = get("--iters", "3000000").parse().unwrap();
    let dur = Duration::from_secs_f64(secs);

    println!("tag={}", tag);
    // Phase 1: frame loop
    let start = now_epoch_ms();
    println!("frame_start_epoch_ms={:.1}", start);
    let t0 = Instant::now();
    let mut late = Vec::new();
    let mut work = Vec::new();
    let mut acc = 0u64;
    while t0.elapsed() < dur {
        let a = Instant::now();
        std::thread::sleep(Duration::from_millis(16));
        let b = Instant::now();
        late.push(((b - a).as_secs_f64() * 1000.0) - 16.0);
        acc = acc.wrapping_add(cpu_task(iters));
        work.push(b.elapsed().as_secs_f64() * 1000.0);
    }
    std::hint::black_box(acc);
    println!("frame_end_epoch_ms={:.1}", now_epoch_ms());
    summary("frame_lateness_ms", &late);
    summary("frame_work_ms", &work);

    // Phase 2: memory-bound pass loop
    let mut buf = vec![1u8; mb * 1024 * 1024];
    for i in (0..buf.len()).step_by(4096) {
        buf[i] = 2; // pre-fault pages
    }
    println!("membuf_start_epoch_ms={:.1}", now_epoch_ms());
    let t1 = Instant::now();
    let mut pass = Vec::new();
    while t1.elapsed() < dur {
        let a = Instant::now();
        let mut i = 0;
        while i < buf.len() {
            buf[i] = buf[i].wrapping_add(1);
            i += 64;
        }
        std::hint::black_box(&buf);
        pass.push(a.elapsed().as_secs_f64() * 1000.0);
    }
    println!("membuf_end_epoch_ms={:.1}", now_epoch_ms());
    summary("membuf_pass_ms", &pass);
}
