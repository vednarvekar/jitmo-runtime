//! Launch a process, optionally under memory limits, and sample its memory every N ms.
//! Usage: runmon [--csv out.csv] [--interval-ms 100] [--timeout-s 0]
//!               [--commit-limit-mb N]   Job Object JOB_OBJECT_LIMIT_PROCESS_MEMORY (private commit cap)
//!               [--ws-max-mb N]         hard max working set (SetProcessWorkingSetSizeEx, HARDWS_MAX_ENABLE)
//!               -- program args...
//! The child is created suspended, put in the job / limited, then resumed, so limits apply from the first instruction.
//! Reported per sample: working set, private bytes (commit charge), page faults, process read bytes, system available RAM.
use jitmo_bench::win::*;
use std::io::Write;
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::process::Command;
use std::ptr::null_mut;
use std::time::Instant;

const CREATE_SUSPENDED: u32 = 0x4;
const MB: f64 = 1024.0 * 1024.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sep = args.iter().position(|a| a == "--").expect("need -- before the program");
    let opts = &args[1..sep];
    let prog = &args[sep + 1..];
    let get = |k: &str| -> Option<String> { opts.iter().position(|a| a == k).map(|i| opts[i + 1].clone()) };
    let csv_path = get("--csv");
    let interval: u64 = get("--interval-ms").map(|s| s.parse().unwrap()).unwrap_or(100);
    let timeout_s: f64 = get("--timeout-s").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let commit_mb: Option<usize> = get("--commit-limit-mb").map(|s| s.parse().unwrap());
    let ws_mb: Option<usize> = get("--ws-max-mb").map(|s| s.parse().unwrap());

    let mut child = Command::new(&prog[0]).args(&prog[1..]).creation_flags(CREATE_SUSPENDED).spawn().expect("spawn failed");
    let h = child.as_raw_handle() as HANDLE;
    let job;
    unsafe {
        job = CreateJobObjectW(null_mut(), null_mut());
        assert!(!job.is_null());
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if let Some(c) = commit_mb {
            info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            info.ProcessMemoryLimit = c * 1024 * 1024;
        }
        let ok = SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            &mut info as *mut _ as *mut _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        assert!(ok != 0, "SetInformationJobObject failed: {}", GetLastError());
        let ok = AssignProcessToJobObject(job, h);
        assert!(ok != 0, "AssignProcessToJobObject failed: {}", GetLastError());
        if let Some(w) = ws_mb {
            let ok = SetProcessWorkingSetSizeEx(
                h,
                16 * 1024 * 1024,
                w * 1024 * 1024,
                QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE,
            );
            if ok == 0 {
                eprintln!("[runmon] SetProcessWorkingSetSizeEx FAILED err={}", GetLastError());
            } else {
                eprintln!("[runmon] hard working-set max = {} MiB", w);
            }
        }
        if let Some(c) = commit_mb {
            eprintln!("[runmon] job process commit limit = {} MiB", c);
        }
        NtResumeProcess(h);
    }
    let mut csv = csv_path.as_ref().map(|p| std::fs::File::create(p).unwrap());
    if let Some(f) = csv.as_mut() {
        writeln!(f, "t_ms,ws_mb,private_mb,page_faults,read_mb,sys_avail_mb,sys_commit_mb").unwrap();
    }
    let t0 = Instant::now();
    let mut max_priv = 0usize;
    let mut max_ws = 0usize;
    let mut min_avail = u64::MAX;
    let mut max_sys_commit = 0usize;
    let mut last = PROCESS_MEMORY_COUNTERS_EX::default();
    let mut last_io = IO_COUNTERS::default();
    let code;
    loop {
        let r = unsafe { WaitForSingleObject(h, interval as u32) };
        let pm = proc_mem(h);
        if r != WAIT_OBJECT_0 || pm.cb != 0 {
            if pm.WorkingSetSize > 0 {
                last = pm;
                last_io = io_counters(h);
            }
        }
        let ms = mem_status();
        let pi = perf_info();
        let sys_commit = pi.CommitTotal * pi.PageSize;
        if r != WAIT_OBJECT_0 {
            max_priv = max_priv.max(pm.PrivateUsage);
            max_ws = max_ws.max(pm.WorkingSetSize);
            min_avail = min_avail.min(ms.ullAvailPhys);
            max_sys_commit = max_sys_commit.max(sys_commit);
            if let Some(f) = csv.as_mut() {
                let io = io_counters(h);
                writeln!(
                    f,
                    "{},{:.1},{:.1},{},{:.1},{:.0},{:.0}",
                    t0.elapsed().as_millis(),
                    pm.WorkingSetSize as f64 / MB,
                    pm.PrivateUsage as f64 / MB,
                    pm.PageFaultCount,
                    io.ReadTransferCount as f64 / MB,
                    ms.ullAvailPhys as f64 / MB,
                    sys_commit as f64 / MB
                )
                .unwrap();
            }
        }
        if r == WAIT_OBJECT_0 {
            let mut c = 0u32;
            unsafe { GetExitCodeProcess(h, &mut c) };
            code = c as i32;
            break;
        }
        if timeout_s > 0.0 && t0.elapsed().as_secs_f64() > timeout_s {
            eprintln!("[runmon] timeout, killing child");
            let _ = child.kill();
        }
    }
    let _ = child.wait();
    let after = proc_mem(h); // peaks are kept in the process object after exit
    last.PeakWorkingSetSize = last.PeakWorkingSetSize.max(after.PeakWorkingSetSize);
    last.PeakPagefileUsage = last.PeakPagefileUsage.max(after.PeakPagefileUsage);
    last.PageFaultCount = last.PageFaultCount.max(after.PageFaultCount);
    let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    unsafe {
        QueryInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            &mut info as *mut _ as *mut _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            null_mut(),
        );
    }
    // Final API peaks (from the last good sample before exit).
    println!(
        "RUNMON exit_code={} secs={:.1} peak_ws_mb={:.1} peak_commit_mb(PeakPagefileUsage)={:.1} max_sampled_private_mb={:.1} max_sampled_ws_mb={:.1} job_peak_process_commit_mb={:.1} page_faults={} read_mb={:.1} min_sys_avail_mb={:.0} max_sys_commit_mb={:.0}",
        code,
        t0.elapsed().as_secs_f64(),
        last.PeakWorkingSetSize as f64 / MB,
        last.PeakPagefileUsage as f64 / MB,
        max_priv as f64 / MB,
        max_ws as f64 / MB,
        info.PeakProcessMemoryUsed as f64 / MB,
        last.PageFaultCount,
        last_io.ReadTransferCount as f64 / MB,
        min_avail as f64 / MB,
        max_sys_commit as f64 / MB
    );
    std::process::exit(if code == 0 { 0 } else { 3 });
}
