//! RAM bandwidth: sequential read, write (fill) and copy over a buffer far larger than cache.
//! Usage: membw [--size-mb 2048] [--reps 5] [--threads 1,2,4,8] [--passes 4]
//! GB/s = 1e9 bytes/s. Copy counts bytes read + bytes written (STREAM convention).
use jitmo_bench::stats::*;
use std::sync::Barrier;
use std::time::Instant;

#[inline(never)]
fn read_sum(s: &[u64]) -> u64 {
    let mut a = [0u64; 16];
    for c in s.chunks_exact(16) {
        for i in 0..16 {
            a[i] = a[i].wrapping_add(c[i]);
        }
    }
    a.iter().fold(0u64, |x, y| x.wrapping_add(*y))
}

/// Second read kernel (plain fold) as a cross-check on the first.
#[inline(never)]
fn read_fold(s: &[u64]) -> u64 {
    s.iter().fold(0u64, |a, &b| a.wrapping_add(b))
}

#[inline(never)]
fn fill(s: &mut [u64], v: u64) {
    for x in s.iter_mut() {
        *x = v;
    }
}

fn split_mut(buf: &mut [u64], n: usize) -> Vec<&mut [u64]> {
    let per = buf.len() / n;
    let mut out = Vec::new();
    let mut rest = buf;
    for _ in 0..n {
        let (a, b) = rest.split_at_mut(per);
        out.push(a);
        rest = b;
    }
    out
}

/// Runs `f` on n threads, each on its own chunk, `passes` times; returns seconds from barrier release to last finish.
fn timed<F>(chunks: Vec<&mut [u64]>, extra: Option<Vec<&mut [u64]>>, passes: usize, f: F) -> f64
where
    F: Fn(&mut [u64], Option<&mut [u64]>) -> u64 + Sync,
{
    let n = chunks.len();
    let bar = Barrier::new(n + 1);
    let mut t = 0.0;
    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        let mut extra = extra.map(|v| v.into_iter());
        for c in chunks {
            let e = extra.as_mut().map(|it| it.next().unwrap());
            let bar = &bar;
            let f = &f;
            hs.push(sc.spawn(move || {
                let mut e = e;
                bar.wait();
                let mut acc = 0u64;
                for _ in 0..passes {
                    // black_box stops LLVM from merging repeated passes over the same (unchanged) memory:
                    // without it, read_sum() was computed once for all passes (measured 4x too high).
                    let c2 = std::hint::black_box(&mut *c);
                    let e2 = std::hint::black_box(e.as_deref_mut());
                    acc = acc.wrapping_add(f(c2, e2));
                }
                std::hint::black_box(acc)
            }));
        }
        bar.wait();
        let t0 = Instant::now();
        for h in hs {
            h.join().unwrap();
        }
        t = t0.elapsed().as_secs_f64();
    });
    t
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| -> String {
        args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap_or(d.to_string())
    };
    let size_mb: usize = get("--size-mb", "2048").parse().unwrap();
    let reps: usize = get("--reps", "5").parse().unwrap();
    let passes: usize = get("--passes", "4").parse().unwrap();
    let threads: Vec<usize> = get("--threads", "1,2,4,8").split(',').map(|s| s.parse().unwrap()).collect();

    let words = size_mb * 1024 * 1024 / 8;
    let mut buf: Vec<u64> = vec![0u64; words];
    // Touch every page (first-touch commit) with non-trivial data.
    for (i, x) in buf.iter_mut().enumerate() {
        *x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    let bytes = (words * 8) as f64;
    println!("# membw buffer={} MiB reps={} passes={} threads={:?}", size_mb, reps, passes, threads);
    println!("test,threads,rep,gbps");
    let mut summary = Vec::new();
    for &n in &threads {
        // "read" (16-lane array kernel) is also run as a cross-check; it compiles to slow code, use "readfold".
        for test in ["read", "readfold", "write", "copy"] {
            let mut v = Vec::new();
            for r in 0..reps {
                let gbps = match test {
                    "read" => {
                        let t = timed(split_mut(&mut buf, n), None, passes, |c, _| read_sum(c));
                        bytes * passes as f64 / t / 1e9
                    }
                    "readfold" => {
                        let t = timed(split_mut(&mut buf, n), None, passes, |c, _| read_fold(c));
                        bytes * passes as f64 / t / 1e9
                    }
                    "write" => {
                        let t = timed(split_mut(&mut buf, n), None, passes, |c, _| {
                            fill(c, 7);
                            0
                        });
                        bytes * passes as f64 / t / 1e9
                    }
                    _ => {
                        let half = buf.len() / 2;
                        let (src, dst) = buf.split_at_mut(half);
                        let t = timed(split_mut(src, n), Some(split_mut(dst, n)), passes, |s, d| {
                            d.unwrap().copy_from_slice(s);
                            0
                        });
                        // half the buffer read + half written per pass
                        bytes * passes as f64 / t / 1e9
                    }
                };
                println!("{},{},{},{:.2}", test, n, r, gbps);
                v.push(gbps);
            }
            summary.push((test, n, median(&v), min(&v), max(&v)));
        }
    }
    println!("# summary: test,threads,median_gbps,min,max");
    for (t, n, m, lo, hi) in summary {
        println!("# {},{},{:.2},{:.2},{:.2}", t, n, m, lo, hi);
    }
}
