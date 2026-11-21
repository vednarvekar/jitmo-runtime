"""Model tokens/s inside the probe's time windows.

Usage: python -I window_tps.py <tokens.json> <probe_output.txt>
Counts streamed tokens whose wall-clock stamp falls inside each probe phase window.
"""
import json
import re
import sys

tok = json.load(open(sys.argv[1]))["t_ms"]
txt = open(sys.argv[2]).read()
g = lambda k: float(re.search(k + r"=([\d.]+)", txt).group(1))
wins = {
    "frame": (g("frame_start_epoch_ms"), g("frame_end_epoch_ms")),
    "membuf": (g("membuf_start_epoch_ms"), g("membuf_end_epoch_ms")),
}
wins["whole"] = (wins["frame"][0], wins["membuf"][1])
first, last = (tok[0], tok[-1]) if tok else (0, 0)
out = {}
for k, (a, b) in wins.items():
    n = sum(1 for t in tok if a <= t <= b)
    covered = first <= a and last >= b
    out[k] = {"tok_s": round(n / ((b - a) / 1000), 2), "tokens": n, "window_s": round((b - a) / 1000, 1), "fully_covered": covered}
print(json.dumps(out))
