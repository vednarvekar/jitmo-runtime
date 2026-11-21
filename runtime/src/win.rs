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
    }

    #[test]
    fn memory_and_cores_are_plausible() {
        let m = memory_status().unwrap();
        assert!(m.avail > 0 && m.avail <= m.total);
        let cores = physical_cores().unwrap();
        let logical = std::thread::available_parallelism().unwrap().get() as u32;
        assert!(cores >= 1 && cores <= logical);
    }
}
