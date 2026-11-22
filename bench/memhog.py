"""Lower free RAM on purpose, in steps, to test the runtime's memory-pressure path.

Usage: python -I memhog.py <target_free_mb> [hold_secs] [floor_mb]
Allocates and touches 64 MB blocks until free RAM (GlobalMemoryStatusEx) is under target_free_mb,
holds for hold_secs (default 20) while topping up if free RAM rises again, then frees everything.
Safety: never allocates while free RAM is under floor_mb (default 300); frees at once if it gets there.
Prints one JSON line per second."""
import ctypes
import json
import sys
import time


class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]


def free_mb():
    m = MS()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2**20


target = float(sys.argv[1])
hold = float(sys.argv[2]) if len(sys.argv) > 2 else 20
floor = float(sys.argv[3]) if len(sys.argv) > 3 else 300
blocks = []
BLOCK = 64 * 2**20
t_hold = None
last = 0
while True:
    f = free_mb()
    if f < floor:
        blocks.clear()
        print(json.dumps({"t": round(time.time() * 1000), "event": "floor", "free_mb": round(f)}), flush=True)
        break
    if f > target and len(blocks) < 200:
        b = bytearray(BLOCK)
        for i in range(0, BLOCK, 4096):
            b[i] = 1
        blocks.append(b)
    elif t_hold is None:
        t_hold = time.time()
    if time.time() - last >= 1:
        last = time.time()
        print(json.dumps({"t": round(last * 1000), "free_mb": round(f), "held_mb": len(blocks) * 64,
                          "holding": t_hold is not None}), flush=True)
    if t_hold is not None and time.time() - t_hold > hold:
        break
    time.sleep(0.05 if f > target else 0.25)
blocks.clear()
print(json.dumps({"t": round(time.time() * 1000), "event": "released", "free_mb": round(free_mb())}), flush=True)
