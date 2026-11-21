//! Unbuffered (FILE_FLAG_NO_BUFFERING) SSD benchmark with an I/O completion port for queue depth.
//! Usage:
//!   diskbench create <file> <size_gib>        write the test file (unbuffered + write-through, 4 MiB blocks)
//!   diskbench run <file> [--dur 5] [--reps 3] [--tests seq:1M:1,rand:4K:32,...]
//! Buffers come from VirtualAlloc (page aligned, satisfies sector alignment). The Windows file cache is
//! bypassed, so every read goes to the SSD. MB/s = 1e6 bytes/s.
use jitmo_bench::stats::*;
use jitmo_bench::win::*;
use std::ptr::null_mut;
use std::time::{Duration, Instant};

#[repr(C)]
struct Slot {
    ov: OVERLAPPED, // must be first: completion returns *mut OVERLAPPED which we cast back to *mut Slot
    buf: *mut u8,
    issued: Instant,
}

fn parse_size(s: &str) -> usize {
    let s = s.to_uppercase();
    if let Some(k) = s.strip_suffix('K') {
        k.parse::<usize>().unwrap() * 1024
    } else if let Some(m) = s.strip_suffix('M') {
        m.parse::<usize>().unwrap() * 1024 * 1024
    } else {
        s.parse().unwrap()
    }
}

fn create(path: &str, gib: u64) {
    let block: usize = 4 * 1024 * 1024;
    let total = gib * 1024 * 1024 * 1024;
    unsafe {
        let h = CreateFileW(
            wide(path).as_ptr(),
            GENERIC_WRITE,
            0,
            null_mut(),
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH,
            null_mut(),
        );
        assert!(h != INVALID_HANDLE_VALUE, "CreateFileW failed: {}", GetLastError());
        let buf = VirtualAlloc(null_mut(), block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as *mut u64;
        let words = block / 8;
        let mut x: u64 = 0x1234_5678_9ABC_DEF1;
        let t0 = Instant::now();
        let mut done: u64 = 0;
        while done < total {
            // fresh pseudo-random data per block (not compressible / not zero)
            for i in 0..words {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *buf.add(i) = x;
            }
            let mut w = 0u32;
            let ok = WriteFile(h, buf as *const _, block as u32, &mut w, null_mut());
            assert!(ok != 0 && w as usize == block, "WriteFile failed: {}", GetLastError());
            done += block as u64;
        }
        FlushFileBuffers(h);
        let t = t0.elapsed().as_secs_f64();
        CloseHandle(h);
        VirtualFree(buf as *mut _, 0, MEM_RELEASE);
        println!(
            "created {} ({} GiB) in {:.1} s = {:.0} MB/s (includes generating random data; write-through, QD1, 4 MiB)",
            path,
            gib,
            t,
            total as f64 / t / 1e6
        );
    }
}

struct Res {
    mbps: f64,
    iops: f64,
    lat_avg_us: f64,
    lat_p50_us: f64,
    lat_p99_us: f64,
}

fn run_one(path: &str, random: bool, bs: usize, qd: usize, dur: Duration, seed: u64) -> Res {
    unsafe {
        let h = CreateFileW(
            wide(path).as_ptr(),
            GENERIC_READ,
            FILE_SHARE_READ,
            null_mut(),
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            null_mut(),
        );
        assert!(h != INVALID_HANDLE_VALUE, "open failed: {}", GetLastError());
        let mut fsz: i64 = 0;
        GetFileSizeEx(h, &mut fsz);
        let nblocks = (fsz as u64) / bs as u64;
        let port = CreateIoCompletionPort(h, null_mut(), 1, 1);
        assert!(!port.is_null(), "IOCP failed: {}", GetLastError());

        let mut slots: Vec<Slot> = (0..qd)
            .map(|_| Slot {
                ov: OVERLAPPED::default(),
                buf: VirtualAlloc(null_mut(), bs, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) as *mut u8,
                issued: Instant::now(),
            })
            .collect();
        let mut rng = seed | 1;
        let mut seq_next: u64 = 0;
        let mut next_off = |rng: &mut u64| -> u64 {
            if random {
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                (*rng % nblocks) * bs as u64
            } else {
                let o = seq_next * bs as u64;
                seq_next = (seq_next + 1) % nblocks;
                o
            }
        };
        let issue = |s: *mut Slot, off: u64| {
            let s = &mut *s;
            s.ov = OVERLAPPED { Offset: off as u32, OffsetHigh: (off >> 32) as u32, ..Default::default() };
            s.issued = Instant::now();
            let ok = ReadFile(h, s.buf as *mut _, bs as u32, null_mut(), &mut s.ov);
            if ok == 0 {
                let e = GetLastError();
                assert!(e == ERROR_IO_PENDING, "ReadFile failed: {}", e);
            }
        };
        let t0 = Instant::now();
        for i in 0..qd {
            let off = next_off(&mut rng);
            issue(&mut slots[i] as *mut Slot, off);
        }
        let mut lats: Vec<f64> = Vec::with_capacity(1 << 20);
        let mut bytes: u64 = 0;
        let mut inflight = qd;
        let mut stopping = false;
        let mut t_end = t0;
        while inflight > 0 {
            let mut n = 0u32;
            let mut key = 0usize;
            let mut pov: *mut OVERLAPPED = null_mut();
            let ok = GetQueuedCompletionStatus(port, &mut n, &mut key, &mut pov, INFINITE);
            assert!(ok != 0, "GQCS failed: {}", GetLastError());
            let s = pov as *mut Slot;
            let now = Instant::now();
            inflight -= 1;
            if !stopping {
                lats.push((now - (*s).issued).as_secs_f64() * 1e6);
                bytes += n as u64;
                if now - t0 >= dur {
                    stopping = true;
                    t_end = now;
                } else {
                    let off = next_off(&mut rng);
                    issue(s, off);
                    inflight += 1;
                }
            }
        }
        let secs = (t_end - t0).as_secs_f64();
        for s in &slots {
            VirtualFree(s.buf as *mut _, 0, MEM_RELEASE);
        }
        CloseHandle(port);
        CloseHandle(h);
        Res {
            mbps: bytes as f64 / secs / 1e6,
            iops: lats.len() as f64 / secs,
            lat_avg_us: mean(&lats),
            lat_p50_us: pct(&lats, 50.0),
            lat_p99_us: pct(&lats, 99.0),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("create") => create(&args[2], args[3].parse().unwrap()),
        Some("run") => {
            let path = &args[2];
            let get = |k: &str, d: &str| -> String {
                args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap_or(d.to_string())
            };
            let dur = Duration::from_secs_f64(get("--dur", "5").parse().unwrap());
            let reps: usize = get("--reps", "3").parse().unwrap();
            let tests = get(
                "--tests",
                "seq:1M:1,seq:1M:32,seq:4M:1,seq:4M:16,rand:4K:1,rand:4K:32,rand:64K:1,rand:64K:32,rand:1M:1,rand:1M:16",
            );
            let tests: Vec<(bool, usize, usize, String)> = tests
                .split(',')
                .map(|t| {
                    let p: Vec<&str> = t.split(':').collect();
                    (p[0] == "rand", parse_size(p[1]), p[2].parse().unwrap(), t.to_string())
                })
                .collect();
            println!("# diskbench file={} dur={:?} reps={} (reps are interleaved: rep loop outside test loop)", path, dur, reps);
            println!("test,rep,MBps,IOPS,lat_avg_us,lat_p50_us,lat_p99_us");
            let mut all: Vec<Vec<Res>> = (0..tests.len()).map(|_| Vec::new()).collect();
            for r in 0..reps {
                for (i, (rnd, bs, qd, name)) in tests.iter().enumerate() {
                    let res = run_one(path, *rnd, *bs, *qd, dur, 0x9E37_79B9 + (r * 1000 + i) as u64);
                    println!(
                        "{},{},{:.1},{:.0},{:.1},{:.1},{:.1}",
                        name, r, res.mbps, res.iops, res.lat_avg_us, res.lat_p50_us, res.lat_p99_us
                    );
                    all[i].push(res);
                }
            }
            println!("# summary: test,median_MBps,min_MBps,max_MBps,median_IOPS,median_lat_p50_us,median_lat_p99_us");
            for (i, (_, _, _, name)) in tests.iter().enumerate() {
                let mb: Vec<f64> = all[i].iter().map(|r| r.mbps).collect();
                let io: Vec<f64> = all[i].iter().map(|r| r.iops).collect();
                let p50: Vec<f64> = all[i].iter().map(|r| r.lat_p50_us).collect();
                let p99: Vec<f64> = all[i].iter().map(|r| r.lat_p99_us).collect();
                println!(
                    "# {},{:.1},{:.1},{:.1},{:.0},{:.1},{:.1}",
                    name,
                    median(&mb),
                    min(&mb),
                    max(&mb),
                    median(&io),
                    median(&p50),
                    median(&p99)
                );
            }
        }
        _ => eprintln!("usage: diskbench create <file> <gib> | diskbench run <file> [--dur s] [--reps n] [--tests ...]"),
    }
}
