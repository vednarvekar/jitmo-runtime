//! Minimal hand-written Win32 FFI (kernel32 + ntdll). No external crates.
#![allow(non_snake_case, non_camel_case_types, clippy::upper_case_acronyms, dead_code)]
use std::ffi::c_void;

pub type HANDLE = *mut c_void;
pub type BOOL = i32;
pub const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;

pub const GENERIC_READ: u32 = 0x8000_0000;
pub const GENERIC_WRITE: u32 = 0x4000_0000;
pub const FILE_SHARE_READ: u32 = 1;
pub const CREATE_ALWAYS: u32 = 2;
pub const OPEN_EXISTING: u32 = 3;
pub const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
pub const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
pub const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
pub const FILE_FLAG_WRITE_THROUGH: u32 = 0x8000_0000;
pub const ERROR_IO_PENDING: u32 = 997;
pub const INFINITE: u32 = 0xFFFF_FFFF;
pub const MEM_COMMIT: u32 = 0x1000;
pub const MEM_RESERVE: u32 = 0x2000;
pub const MEM_RELEASE: u32 = 0x8000;
pub const PAGE_READWRITE: u32 = 0x04;
pub const WAIT_OBJECT_0: u32 = 0;
pub const WAIT_TIMEOUT: u32 = 258;

pub const JOB_OBJECT_LIMIT_WORKINGSET: u32 = 0x1;
pub const JOB_OBJECT_LIMIT_PROCESS_MEMORY: u32 = 0x100;
pub const JOB_OBJECT_LIMIT_JOB_MEMORY: u32 = 0x200;
pub const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
pub const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: i32 = 9;
pub const QUOTA_LIMITS_HARDWS_MIN_DISABLE: u32 = 0x2;
pub const QUOTA_LIMITS_HARDWS_MAX_ENABLE: u32 = 0x4;

#[repr(C)]
#[derive(Default)]
pub struct OVERLAPPED {
    pub Internal: usize,
    pub InternalHigh: usize,
    pub Offset: u32,
    pub OffsetHigh: u32,
    pub hEvent: usize,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct PROCESS_MEMORY_COUNTERS_EX {
    pub cb: u32,
    pub PageFaultCount: u32,
    pub PeakWorkingSetSize: usize,
    pub WorkingSetSize: usize,
    pub QuotaPeakPagedPoolUsage: usize,
    pub QuotaPagedPoolUsage: usize,
    pub QuotaPeakNonPagedPoolUsage: usize,
    pub QuotaNonPagedPoolUsage: usize,
    pub PagefileUsage: usize,
    pub PeakPagefileUsage: usize,
    pub PrivateUsage: usize,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct IO_COUNTERS {
    pub ReadOperationCount: u64,
    pub WriteOperationCount: u64,
    pub OtherOperationCount: u64,
    pub ReadTransferCount: u64,
    pub WriteTransferCount: u64,
    pub OtherTransferCount: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
    pub PerProcessUserTimeLimit: i64,
    pub PerJobUserTimeLimit: i64,
    pub LimitFlags: u32,
    pub MinimumWorkingSetSize: usize,
    pub MaximumWorkingSetSize: usize,
    pub ActiveProcessLimit: u32,
    pub Affinity: usize,
    pub PriorityClass: u32,
    pub SchedulingClass: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
    pub BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION,
    pub IoInfo: IO_COUNTERS,
    pub ProcessMemoryLimit: usize,
    pub JobMemoryLimit: usize,
    pub PeakProcessMemoryUsed: usize,
    pub PeakJobMemoryUsed: usize,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct MEMORYSTATUSEX {
    pub dwLength: u32,
    pub dwMemoryLoad: u32,
    pub ullTotalPhys: u64,
    pub ullAvailPhys: u64,
    pub ullTotalPageFile: u64,
    pub ullAvailPageFile: u64,
    pub ullTotalVirtual: u64,
    pub ullAvailVirtual: u64,
    pub ullAvailExtendedVirtual: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct PERFORMANCE_INFORMATION {
    pub cb: u32,
    pub CommitTotal: usize,
    pub CommitLimit: usize,
    pub CommitPeak: usize,
    pub PhysicalTotal: usize,
    pub PhysicalAvailable: usize,
    pub SystemCache: usize,
    pub KernelTotal: usize,
    pub KernelPaged: usize,
    pub KernelNonpaged: usize,
    pub PageSize: usize,
    pub HandleCount: u32,
    pub ProcessCount: u32,
    pub ThreadCount: u32,
}

#[link(name = "kernel32")]
extern "system" {
    pub fn CreateFileW(name: *const u16, access: u32, share: u32, sa: *mut c_void, disp: u32, flags: u32, tmpl: HANDLE) -> HANDLE;
    pub fn ReadFile(h: HANDLE, buf: *mut c_void, n: u32, read: *mut u32, ov: *mut OVERLAPPED) -> BOOL;
    pub fn WriteFile(h: HANDLE, buf: *const c_void, n: u32, written: *mut u32, ov: *mut OVERLAPPED) -> BOOL;
    pub fn FlushFileBuffers(h: HANDLE) -> BOOL;
    pub fn GetFileSizeEx(h: HANDLE, size: *mut i64) -> BOOL;
    pub fn CloseHandle(h: HANDLE) -> BOOL;
    pub fn GetLastError() -> u32;
    pub fn CreateIoCompletionPort(file: HANDLE, existing: HANDLE, key: usize, threads: u32) -> HANDLE;
    pub fn GetQueuedCompletionStatus(port: HANDLE, bytes: *mut u32, key: *mut usize, ov: *mut *mut OVERLAPPED, ms: u32) -> BOOL;
    pub fn VirtualAlloc(addr: *mut c_void, size: usize, ty: u32, prot: u32) -> *mut c_void;
    pub fn VirtualFree(addr: *mut c_void, size: usize, ty: u32) -> BOOL;
    pub fn WaitForSingleObject(h: HANDLE, ms: u32) -> u32;
    pub fn GetExitCodeProcess(h: HANDLE, code: *mut u32) -> BOOL;
    pub fn TerminateProcess(h: HANDLE, code: u32) -> BOOL;
    pub fn CreateJobObjectW(sa: *mut c_void, name: *const u16) -> HANDLE;
    pub fn SetInformationJobObject(job: HANDLE, class: i32, info: *mut c_void, len: u32) -> BOOL;
    pub fn QueryInformationJobObject(job: HANDLE, class: i32, info: *mut c_void, len: u32, ret: *mut u32) -> BOOL;
    pub fn AssignProcessToJobObject(job: HANDLE, process: HANDLE) -> BOOL;
    pub fn SetProcessWorkingSetSizeEx(process: HANDLE, min: usize, max: usize, flags: u32) -> BOOL;
    pub fn GlobalMemoryStatusEx(m: *mut MEMORYSTATUSEX) -> BOOL;
    pub fn K32GetProcessMemoryInfo(process: HANDLE, c: *mut PROCESS_MEMORY_COUNTERS_EX, cb: u32) -> BOOL;
    pub fn K32GetPerformanceInfo(p: *mut PERFORMANCE_INFORMATION, cb: u32) -> BOOL;
    pub fn GetProcessIoCounters(process: HANDLE, io: *mut IO_COUNTERS) -> BOOL;
}

#[link(name = "ntdll")]
extern "system" {
    pub fn NtResumeProcess(process: HANDLE) -> i32;
}

pub fn wide(s: &str) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    std::ffi::OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect()
}

pub fn mem_status() -> MEMORYSTATUSEX {
    let mut m = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    unsafe { GlobalMemoryStatusEx(&mut m) };
    m
}

pub fn perf_info() -> PERFORMANCE_INFORMATION {
    let mut p = PERFORMANCE_INFORMATION { cb: std::mem::size_of::<PERFORMANCE_INFORMATION>() as u32, ..Default::default() };
    unsafe { K32GetPerformanceInfo(&mut p, p.cb) };
    p
}

pub fn proc_mem(h: HANDLE) -> PROCESS_MEMORY_COUNTERS_EX {
    let mut c = PROCESS_MEMORY_COUNTERS_EX { cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32, ..Default::default() };
    unsafe { K32GetProcessMemoryInfo(h, &mut c, c.cb) };
    c
}

pub fn io_counters(h: HANDLE) -> IO_COUNTERS {
    let mut io = IO_COUNTERS::default();
    unsafe { GetProcessIoCounters(h, &mut io) };
    io
}
