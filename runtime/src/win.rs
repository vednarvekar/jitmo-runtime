//! Hand-written Win32 FFI (kernel32 + ntdll), adapted from `bench/src/win.rs`.
//! Only what the runtime needs: memory status, Job Objects, process memory counters,
//! the low-memory notification, physical core count, and resuming a suspended child.
#![allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms)]

use std::ffi::c_void;
use std::io;
use std::ptr::null_mut;

pub type HANDLE = *mut c_void;
type BOOL = i32;

pub const CREATE_SUSPENDED: u32 = 0x0000_0004;
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;
pub const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
pub const NORMAL_PRIORITY_CLASS: u32 = 0x0000_0020;
pub const IDLE_PRIORITY_CLASS: u32 = 0x0000_0040;
const PROCESS_SET_INFORMATION: u32 = 0x0200;
const JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS: i32 = 1;
const JOB_OBJECT_CPU_RATE_CONTROL_INFORMATION_CLASS: i32 = 15;
const JOB_OBJECT_CPU_RATE_CONTROL_ENABLE: u32 = 0x1;
const JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP: u32 = 0x4;

const JOB_OBJECT_LIMIT_PROCESS_MEMORY: u32 = 0x100;
const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: i32 = 9;
const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
const PROCESS_VM_READ: u32 = 0x0010;
const LOW_MEMORY_RESOURCE_NOTIFICATION: i32 = 0;
const RELATION_PROCESSOR_CORE: u32 = 0;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct MEMORYSTATUSEX {
    dwLength: u32,
    dwMemoryLoad: u32,
    ullTotalPhys: u64,
    ullAvailPhys: u64,
    ullTotalPageFile: u64,
    ullAvailPageFile: u64,
    ullTotalVirtual: u64,
    ullAvailVirtual: u64,
    ullAvailExtendedVirtual: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct PROCESS_MEMORY_COUNTERS_EX {
    cb: u32,
    PageFaultCount: u32,
    PeakWorkingSetSize: usize,
    WorkingSetSize: usize,
    QuotaPeakPagedPoolUsage: usize,
    QuotaPagedPoolUsage: usize,
    QuotaPeakNonPagedPoolUsage: usize,
    QuotaNonPagedPoolUsage: usize,
    PagefileUsage: usize,
    PeakPagefileUsage: usize,
    PrivateUsage: usize,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IO_COUNTERS {
    ReadOperationCount: u64,
    WriteOperationCount: u64,
    OtherOperationCount: u64,
    ReadTransferCount: u64,
    WriteTransferCount: u64,
    OtherTransferCount: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
    PerProcessUserTimeLimit: i64,
    PerJobUserTimeLimit: i64,
    LimitFlags: u32,
    MinimumWorkingSetSize: usize,
    MaximumWorkingSetSize: usize,
    ActiveProcessLimit: u32,
    Affinity: usize,
    PriorityClass: u32,
    SchedulingClass: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION,
    IoInfo: IO_COUNTERS,
    ProcessMemoryLimit: usize,
    JobMemoryLimit: usize,
    PeakProcessMemoryUsed: usize,
    PeakJobMemoryUsed: usize,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct JOBOBJECT_BASIC_ACCOUNTING_INFORMATION {
    TotalUserTime: i64,
    TotalKernelTime: i64,
    ThisPeriodTotalUserTime: i64,
    ThisPeriodTotalKernelTime: i64,
    TotalPageFaultCount: u32,
    TotalProcesses: u32,
    ActiveProcesses: u32,
    TotalTerminatedProcesses: u32,
}

/// JOBOBJECT_CPU_RATE_CONTROL_INFORMATION; the union is used as CpuRate.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
    ControlFlags: u32,
    CpuRate: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct FILETIME {
    lo: u32,
    hi: u32,
}

impl FILETIME {
    fn get(self) -> u64 {
        (self.hi as u64) << 32 | self.lo as u64
    }
}

/// SYSTEM_LOGICAL_PROCESSOR_INFORMATION; the 16-byte union is kept opaque.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SYSTEM_LOGICAL_PROCESSOR_INFORMATION {
    ProcessorMask: usize,
    Relationship: u32,
    Union: [u64; 2],
}

#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(m: *mut MEMORYSTATUSEX) -> BOOL;
    fn CreateJobObjectW(sa: *mut c_void, name: *const u16) -> HANDLE;
    fn SetInformationJobObject(job: HANDLE, class: i32, info: *mut c_void, len: u32) -> BOOL;
    fn QueryInformationJobObject(job: HANDLE, class: i32, info: *mut c_void, len: u32, ret: *mut u32) -> BOOL;
    fn AssignProcessToJobObject(job: HANDLE, process: HANDLE) -> BOOL;
    fn TerminateJobObject(job: HANDLE, code: u32) -> BOOL;
    fn CloseHandle(h: HANDLE) -> BOOL;
    fn OpenProcess(access: u32, inherit: BOOL, pid: u32) -> HANDLE;
    fn K32GetProcessMemoryInfo(process: HANDLE, c: *mut PROCESS_MEMORY_COUNTERS_EX, cb: u32) -> BOOL;
    fn CreateMemoryResourceNotification(kind: i32) -> HANDLE;
    fn QueryMemoryResourceNotification(h: HANDLE, state: *mut BOOL) -> BOOL;
    fn GetLogicalProcessorInformation(buf: *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION, len: *mut u32) -> BOOL;
    fn GetSystemTimes(idle: *mut FILETIME, kernel: *mut FILETIME, user: *mut FILETIME) -> BOOL;
    fn SetPriorityClass(process: HANDLE, class: u32) -> BOOL;
    fn WaitForSingleObject(h: HANDLE, ms: u32) -> u32;
}

#[link(name = "ntdll")]
extern "system" {
    fn NtResumeProcess(process: HANDLE) -> i32;
}

/// System-wide physical memory.
#[derive(Debug, Clone, Copy)]
pub struct MemoryStatus {
    pub total: u64,
    pub avail: u64,
    pub load_pct: u32,
}

pub fn memory_status() -> Option<MemoryStatus> {
    let mut m = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    // SAFETY: `m` is a valid struct whose size is in dwLength.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(MemoryStatus { total: m.ullTotalPhys, avail: m.ullAvailPhys, load_pct: m.dwMemoryLoad })
}

/// Memory counters of one process, in bytes. "private" is the commit charge (private bytes).
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessMemory {
    pub working_set: u64,
    pub peak_working_set: u64,
    pub private: u64,
    pub peak_private: u64,
}

fn process_memory_raw(h: HANDLE) -> Option<ProcessMemory> {
    let mut c = PROCESS_MEMORY_COUNTERS_EX { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32, ..Default::default() };
    // SAFETY: valid out-struct with its size; `h` is a process handle owned by the caller.
    let ok = unsafe { K32GetProcessMemoryInfo(h, &mut c, c.cb) } != 0;
    ok.then_some(ProcessMemory {
        working_set: c.WorkingSetSize as u64,
        peak_working_set: c.PeakWorkingSetSize as u64,
        private: c.PrivateUsage as u64,
        peak_private: c.PeakPagefileUsage as u64,
    })
}

/// Memory of a process by PID (opens and closes its own handle, so it is safe to call
/// while another task may be ending that process).
pub fn process_memory_by_pid(pid: u32) -> Option<ProcessMemory> {
    // SAFETY: plain Win32 calls; the handle is closed before returning.
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid);
        if h.is_null() {
            return None;
        }
        let r = process_memory_raw(h);
        CloseHandle(h);
        r
    }
}

/// Resume a process that was created with CREATE_SUSPENDED.
pub fn resume_process(h: HANDLE) -> bool {
    // SAFETY: `h` is a live process handle owned by the caller.
    unsafe { NtResumeProcess(h) >= 0 }
}

/// A Job Object that ends its processes when closed (kill-on-close), with an optional
/// per-process commit (private bytes) limit.
pub struct Job(HANDLE);

// SAFETY: a job handle is a kernel handle; it may be used from any thread.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    pub fn new(process_commit_limit: Option<u64>) -> io::Result<Job> {
        // SAFETY: plain Win32 calls; the handle is owned by the returned Job (closed on drop).
        unsafe {
            let h = CreateJobObjectW(null_mut(), std::ptr::null());
            if h.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Job(h);
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if let Some(limit) = process_commit_limit {
                info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
                info.ProcessMemoryLimit = limit as usize;
            }
            let ok = SetInformationJobObject(
                job.0,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }
    }

    pub fn assign(&self, process: HANDLE) -> io::Result<()> {
        // SAFETY: both handles are live.
        let ok = unsafe { AssignProcessToJobObject(self.0, process) } != 0;
        if ok {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Kill every process in the job now.
    pub fn terminate(&self) {
        // SAFETY: live job handle.
        unsafe { TerminateJobObject(self.0, 1) };
    }

    /// Hard-cap the job's CPU use to `rate` / 10000 of all logical processors, or remove the
    /// cap (`None`). Takes effect at once on running processes.
    pub fn set_cpu_rate(&self, rate: Option<u32>) -> io::Result<()> {
        let mut info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION::default();
        if let Some(r) = rate {
            info.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
            info.CpuRate = r.clamp(1, 10_000);
        }
        // SAFETY: valid in-struct with its size; live job handle.
        let ok = unsafe {
            SetInformationJobObject(
                self.0,
                JOB_OBJECT_CPU_RATE_CONTROL_INFORMATION_CLASS,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>() as u32,
            )
        } != 0;
        if ok {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// CPU time (user + kernel) used by all processes of the job so far, in 100 ns units.
    pub fn cpu_time(&self) -> Option<u64> {
        let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        // SAFETY: valid out-struct with its size.
        let ok = unsafe {
            QueryInformationJobObject(
                self.0,
                JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                null_mut(),
            )
        } != 0;
        ok.then_some((info.TotalUserTime + info.TotalKernelTime) as u64)
    }

    /// Highest commit (private bytes) any process in the job has reached, in bytes.
    pub fn peak_process_commit(&self) -> Option<u64> {
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        // SAFETY: valid out-struct with its size.
        let ok = unsafe {
            QueryInformationJobObject(
                self.0,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                null_mut(),
            )
        } != 0;
        ok.then_some(info.PeakProcessMemoryUsed as u64)
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // SAFETY: we own the handle. Closing the last handle kills the job's processes.
        unsafe { CloseHandle(self.0) };
    }
}

/// Windows' own "physical memory is low" signal.
pub struct LowMemorySignal(HANDLE);

// SAFETY: kernel handle, usable from any thread.
unsafe impl Send for LowMemorySignal {}
unsafe impl Sync for LowMemorySignal {}

impl LowMemorySignal {
    pub fn new() -> Option<LowMemorySignal> {
        // SAFETY: plain Win32 call; null means failure.
        let h = unsafe { CreateMemoryResourceNotification(LOW_MEMORY_RESOURCE_NOTIFICATION) };
        (!h.is_null()).then_some(LowMemorySignal(h))
    }

    pub fn is_low(&self) -> bool {
        let mut state: BOOL = 0;
        // SAFETY: valid out-pointer, live handle.
        let ok = unsafe { QueryMemoryResourceNotification(self.0, &mut state) } != 0;
        ok && state != 0
    }
}

impl Drop for LowMemorySignal {
    fn drop(&mut self) {
        // SAFETY: we own the handle.
        unsafe { CloseHandle(self.0) };
    }
}

/// System CPU time summed over all logical processors, in 100 ns units.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemTimes {
    pub idle: u64,
    /// Kernel + user (kernel time includes idle time).
    pub total: u64,
}

pub fn system_times() -> Option<SystemTimes> {
    let (mut i, mut k, mut u) = (FILETIME::default(), FILETIME::default(), FILETIME::default());
    // SAFETY: three valid out-pointers.
    let ok = unsafe { GetSystemTimes(&mut i, &mut k, &mut u) } != 0;
    ok.then(|| SystemTimes { idle: i.get(), total: k.get() + u.get() })
}

/// Set the priority class of a process by PID.
pub fn set_priority_class(pid: u32, class: u32) -> bool {
    // SAFETY: plain Win32 calls; the handle is closed before returning.
    unsafe {
        let h = OpenProcess(PROCESS_SET_INFORMATION, 0, pid);
        if h.is_null() {
            return false;
        }
        let ok = SetPriorityClass(h, class) != 0;
        CloseHandle(h);
        ok
    }
}

const SYNCHRONIZE: u32 = 0x0010_0000;
const INFINITE: u32 = 0xFFFF_FFFF;

/// A handle to another process (the runtime's parent) that can be waited on. Holding the
/// handle means a reused PID cannot be mistaken for the parent.
pub struct ProcessWaiter(HANDLE);

// SAFETY: kernel handle, usable from any thread.
unsafe impl Send for ProcessWaiter {}
unsafe impl Sync for ProcessWaiter {}

impl ProcessWaiter {
    pub fn open(pid: u32) -> io::Result<ProcessWaiter> {
        // SAFETY: plain Win32 call; null means failure.
        let h = unsafe { OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if h.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok(ProcessWaiter(h))
        }
    }

    /// Block this thread until the process has exited.
    pub fn wait(&self) {
        // SAFETY: live handle opened with SYNCHRONIZE.
        unsafe { WaitForSingleObject(self.0, INFINITE) };
    }

    /// True if the process has exited (does not block).
    pub fn exited(&self) -> bool {
        // SAFETY: live handle; a 0 ms wait only polls. WAIT_OBJECT_0 = 0.
        unsafe { WaitForSingleObject(self.0, 0) == 0 }
    }
}

impl Drop for ProcessWaiter {
    fn drop(&mut self) {
        // SAFETY: we own the handle.
        unsafe { CloseHandle(self.0) };
    }
}

/// Number of physical CPU cores (not logical processors).
pub fn physical_cores() -> Option<u32> {
    let mut len: u32 = 0;
    // SAFETY: the first call only reports the needed buffer size; the second fills a buffer
    // of at least that size.
    unsafe {
        GetLogicalProcessorInformation(null_mut(), &mut len);
        let n = len as usize / std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
        if n == 0 {
            return None;
        }
        let mut buf = vec![SYSTEM_LOGICAL_PROCESSOR_INFORMATION::default(); n];
        if GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) == 0 {
            return None;
        }
        let cores = buf.iter().filter(|e| e.Relationship == RELATION_PROCESSOR_CORE).count() as u32;
        (cores > 0).then_some(cores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_the_win64_abi() {
        assert_eq!(std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>(), 144);
        assert_eq!(std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>(), 32);
        assert_eq!(std::mem::size_of::<MEMORYSTATUSEX>(), 64);
        assert_eq!(std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>(), 80);
        assert_eq!(std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>(), 48);
        assert_eq!(std::mem::size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>(), 8);
    }

    #[test]
    fn memory_and_cores_are_plausible() {
        let m = memory_status().unwrap();
        assert!(m.avail > 0 && m.avail <= m.total);
        let cores = physical_cores().unwrap();
        let logical = std::thread::available_parallelism().unwrap().get() as u32;
        assert!(cores >= 1 && cores <= logical);
        let t = system_times().unwrap();
        assert!(t.total >= t.idle && t.total > 0);
    }

    #[test]
    fn a_process_can_be_waited_on() {
        let me = ProcessWaiter::open(std::process::id()).unwrap();
        assert!(!me.exited());
        let mut child = std::process::Command::new("cmd").args(["/C", "exit 0"]).spawn().unwrap();
        let w = ProcessWaiter::open(child.id()).unwrap();
        w.wait();
        assert!(w.exited());
        child.wait().unwrap();
        assert!(ProcessWaiter::open(0xFFFF_FFF0).is_err());
    }

    #[test]
    fn a_job_cpu_cap_can_be_set_and_removed_at_runtime() {
        let job = Job::new(None).unwrap();
        job.set_cpu_rate(Some(1875)).unwrap();
        job.set_cpu_rate(None).unwrap();
        assert_eq!(job.cpu_time(), Some(0));
    }
}
