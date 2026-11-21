"""Summarize bench/out/pacer: foreground probe slowdown and model tok/s per condition.

Usage: python -I summarize_pacer.py [dir]
For each condition: medians over repeats of the probe's p50 numbers, slowdown vs "alone",
model tok/s inside the probe windows, and whether the user touched the PC during the window.
"""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out", "pacer")


def probe(path):
    t = open(path).read()
    g = lambda name, stat: float(re.search(name + r": n=\d+ .*?" + stat + r"=([\d.]+)", t).group(1))
    return {
        "late_p99": g("frame_lateness_ms", "p99"),
        "late_max": g("frame_lateness_ms", "max"),
        "work_p50": g("frame_work_ms", "p50"),
        "pass_p50": g("membuf_pass_ms", "p50"),
        "pass_p99": g("membuf_pass_ms", "p99"),
    }


conds = {}
for f in sorted(glob.glob(os.path.join(D, "*_r[0-9].txt"))):
    name = os.path.basename(f)
    if "_client" in name:
        continue
    c, r = re.match(r"(.+)_r(\d+)\.txt", name).groups()
    row = probe(f)
    tps = os.path.join(D, f"{c}_r{r}_tps.json")
    if os.path.exists(tps):
        w = json.load(open(tps))
        row.update(tps_frame=w["frame"]["tok_s"], tps_membuf=w["membuf"]["tok_s"], tps_whole=w["whole"]["tok_s"],
                   covered=w["frame"]["fully_covered"] and w["membuf"]["fully_covered"])
    pa = os.path.join(D, f"{c}_r{r}_pacer_after.json")
    pb = os.path.join(D, f"{c}_r{r}_pacer_before.json")
    if os.path.exists(pa):
        a = json.load(open(pa))
        b = json.load(open(pb)) if os.path.exists(pb) else {}
        row["input_idle_s_after"] = round((a.get("input_idle_ms") or 0) / 1000, 1)
        row["capped_s_in_window"] = round(a["capped_secs_total"] - b.get("capped_secs_total", 0), 1)
        row["others_cpu_pct_end"] = a.get("others_cpu_pct")
    conds.setdefault(c, []).append(row)

med = lambda rows, k: round(st.median([r[k] for r in rows if k in r]), 2) if any(k in r for r in rows) else None
base = conds.get("alone")
order = ["alone", "off", "cap50", "cap25", "idleprio", "auto"]
out = []
for c in sorted(conds, key=lambda x: order.index(x) if x in order else 99):
    rows = conds[c]
    s = {k: med(rows, k) for k in ["work_p50", "pass_p50", "pass_p99", "late_p99", "late_max", "tps_frame", "tps_membuf", "tps_whole"]}
    if base:
        s["work_slowdown_pct"] = round((s["work_p50"] / med(base, "work_p50") - 1) * 100, 1)
        s["pass_slowdown_pct"] = round((s["pass_p50"] / med(base, "pass_p50") - 1) * 100, 1)
    s["reps"] = len(rows)
    s["input_idle_s_after"] = [r.get("input_idle_s_after") for r in rows]
    s["capped_s_in_window"] = [r.get("capped_s_in_window") for r in rows]
    s["per_rep_pass_p50"] = [r["pass_p50"] for r in rows]
    s["per_rep_tps_whole"] = [r.get("tps_whole") for r in rows]
    out.append((c, s))
    print(c, json.dumps(s))
