//! Copy a file without putting either side in the Windows file cache (FILE_FLAG_NO_BUFFERING
//! on both handles). The copy is then "cold": a later read must come from the SSD. Used to
//! measure cold model loads without admin rights (emptying the standby list needs admin).
//!
//!   nocachecopy <src> <dst>
use jitmo_bench::win::*;
use std::ptr::null_mut;
use std::time::Instant;

extern "system" {
    fn SetFilePointerEx(h: HANDLE, dist: i64, new: *mut i64, method: u32) -> BOOL;
    fn SetEndOfFile(h: HANDLE) -> BOOL;
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 3, "usage: nocachecopy <src> <dst>");
    let block: usize = 8 * 1024 * 1024;
    unsafe {
        let src = CreateFileW(wide(&a[1]).as_ptr(), GENERIC_READ, FILE_SHARE_READ, null_mut(), OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, null_mut());
        assert!(src != INVALID_HANDLE_VALUE, "open src failed: {}", GetLastError());
        let mut len: i64 = 0;
        GetFileSizeEx(src, &mut len);
        let dst = CreateFileW(wide(&a[2]).as_ptr(), GENERIC_WRITE, 0, null_mut(), CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, null_mut());
        assert!(dst != INVALID_HANDLE_VALUE, "create dst failed: {}", GetLastError());
        let buf = VirtualAlloc(null_mut(), block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        let t0 = Instant::now();
        let mut done: i64 = 0;
        while done < len {
            let mut r = 0u32;
            let ok = ReadFile(src, buf, block as u32, &mut r, null_mut());
            assert!(ok != 0, "read failed: {}", GetLastError());
            if r == 0 {
                break;
            }
            // Unbuffered writes must be whole sectors: round the last block up, trim below.
            let n = (r as usize).div_ceil(4096) * 4096;
            let mut w = 0u32;
            let ok = WriteFile(dst, buf, n as u32, &mut w, null_mut());
            assert!(ok != 0 && w as usize == n, "write failed: {}", GetLastError());
            done += r as i64;
        }
        CloseHandle(dst);
        // Trim the sector padding: setting the end of file reads no data, so nothing is cached.
        let dst = CreateFileW(wide(&a[2]).as_ptr(), GENERIC_WRITE, 0, null_mut(), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, null_mut());
        assert!(dst != INVALID_HANDLE_VALUE, "reopen dst failed: {}", GetLastError());
        assert!(SetFilePointerEx(dst, len, null_mut(), 0) != 0 && SetEndOfFile(dst) != 0, "trim failed: {}", GetLastError());
        CloseHandle(dst);
        CloseHandle(src);
        VirtualFree(buf, 0, MEM_RELEASE);
        let s = t0.elapsed().as_secs_f64();
        println!("copied {} bytes in {:.2} s ({:.0} MB/s), uncached", len, s, len as f64 / 1048576.0 / s);
    }
}
