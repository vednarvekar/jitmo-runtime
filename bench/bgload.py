"""Light background load: N processes, each busy `duty` of every 10 ms, for `secs` seconds
(each process stops by itself, so no process is left behind).
2 processes x 0.8 on 8 logical CPUs = about 20% of the CPU.

Usage: python -I bgload.py [procs] [duty] [secs]"""
import subprocess
import sys
import time


def spin(duty, secs):
    period = 0.010
    end = time.perf_counter() + secs
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < period * duty:
            pass
        time.sleep(max(0.0, period * (1 - duty)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        spin(float(sys.argv[2]), float(sys.argv[3]))
        sys.exit(0)
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    duty = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 36
    kids = [subprocess.Popen([sys.executable, "-I", __file__, "--child", str(duty), str(secs)]) for _ in range(procs)]
    try:
        for k in kids:
            k.wait()
    finally:
        for k in kids:
            k.kill()
