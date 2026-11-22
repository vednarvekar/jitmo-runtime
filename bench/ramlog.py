"""Free RAM once a second (GlobalMemoryStatusEx: ullAvailPhys, dwMemoryLoad), as JSON lines.

Usage: python -I ramlog.py <out.jsonl> [secs] [--kill-below MB --kill-pid WINPID]
--kill-below: if free RAM drops under MB, kill WINPID and its children (taskkill /T /F), log it, stop.
Used as a safety stop for the forced-model test. Stops by itself after `secs` (default 3600)."""
import ctypes
import json
import subprocess
import sys
import time


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def status():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


args = sys.argv[1:]
opt = lambda k: args[args.index(k) + 1] if k in args else None
out = args[0]
secs = float(args[1]) if len(args) > 1 and not args[1].startswith("--") else 3600
kill_below = float(opt("--kill-below") or 0)
kill_pid = opt("--kill-pid")
end = time.time() + secs
with open(out, "a") as f:
    while time.time() < end:
        m = status()
        free = m.ullAvailPhys / 2**20
        row = {"t": round(time.time() * 1000), "free_mb": round(free), "load_pct": m.dwMemoryLoad,
               "commit_free_mb": round(m.ullAvailPageFile / 2**20)}
        if kill_below and kill_pid and free < kill_below:
            subprocess.run(["taskkill", "/PID", kill_pid, "/T", "/F"], capture_output=True)
            row["killed"] = kill_pid
            print(f"SAFETY STOP: {free:.0f} MB free < {kill_below:.0f} MB, killed {kill_pid}", flush=True)
        f.write(json.dumps(row) + "\n")
        f.flush()
        if "killed" in row:
            break
        time.sleep(1)
