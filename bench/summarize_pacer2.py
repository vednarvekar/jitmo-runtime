"""Pacer v2 levels during the probe windows (bench/out/pacer2/*_pacer.jsonl, sampled once a second).

Usage: python -I summarize_pacer2.py [dir]
Per condition and repeat: share of samples at each level, median smoothed other-process CPU,
in the probe's frame phase (CPU task) and membuf phase (memory-heavy), plus the reasons seen.
Run summarize_pacer.py on the same dir for the probe slowdown and model tok/s."""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out", "pacer2")
for f in sorted(glob.glob(os.path.join(D, "*_pacer.jsonl"))):
    c, r = re.match(r"(.+)_r(\d+)_pacer\.jsonl", os.path.basename(f)).groups()
    rows = [json.loads(l) for l in open(f) if l.strip()]
    txt = open(os.path.join(D, f"{c}_r{r}.txt")).read()
    g = lambda k: float(re.search(k + r"=([\d.]+)", txt).group(1))
    phases = {"frame": (g("frame_start_epoch_ms"), g("frame_end_epoch_ms")), "membuf": (g("membuf_start_epoch_ms"), g("membuf_end_epoch_ms"))}
    out = {}
    for ph, (a, b) in phases.items():
        rs = [x for x in rows if a <= x["t"] <= b]
        if not rs:
            continue
        lv = {k: round(sum(1 for x in rs if x["level"] == k) / len(rs), 2) for k in ("none", "light", "heavy")}
        cpu = [x["others_cpu_pct"] for x in rs if x["others_cpu_pct"] is not None]
        out[ph] = {"levels": lv, "others_cpu_median": round(st.median(cpu), 1) if cpu else None, "n": len(rs)}
    reasons = sorted({re.sub(r"\d+%", "N%", x["reason"]) for x in rows if x.get("reason")})
    changes = rows[-1]["level_changes"] - rows[0]["level_changes"] if rows else 0
    print(c, r, json.dumps(out), "level_changes_in_window:", changes, "reasons:", reasons)
