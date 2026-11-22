"""Summarize run_pacer2_bg.sh: model tok/s, pacer levels and level changes per run.
Usage: python -I summarize_pacer2_bg.py [dir]"""
import collections, glob, json, os, re, statistics as st, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out", "pacer2")
for f in sorted(glob.glob(os.path.join(D, "bgonly_*_pacer.jsonl"))):
    tag = os.path.basename(f)[: -len("_pacer.jsonl")]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    t0, t1 = rows[0]["t"], rows[-1]["t"]
    tok = os.path.join(D, tag + "_tokens.json")
    tps = None
    if os.path.exists(tok):
        t = json.load(open(tok))["t_ms"]
        tps = round(sum(1 for x in t if t0 <= x <= t1) / ((t1 - t0) / 1000), 2)
    cpu = [x["others_cpu_pct"] for x in rows if x["others_cpu_pct"] is not None]
    s = {"tag": tag, "tok_s": tps, "levels": dict(collections.Counter(x["level"] for x in rows)),
         "others_cpu_median": round(st.median(cpu), 1) if cpu else None, "others_cpu_range": [min(cpu), max(cpu)] if cpu else None,
         "level_changes": sum(1 for a, b in zip(rows, rows[1:]) if a["level"] != b["level"]), "window_s": round((t1 - t0) / 1000, 1)}
    json.dump(s, open(os.path.join(D, tag + "_summary.json"), "w"))
    print(json.dumps(s))
