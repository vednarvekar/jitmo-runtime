"""Turn raw outputs in bench/out into markdown tables (printed to stdout).
Usage: python -I summarize.py [section ...]   sections: membw disk lb mem lag limit
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MODELS = r"V:\jitmo-sandbox\models"
secs = sys.argv[1:] or ["membw", "disk", "lb", "mem", "lag", "limit"]


def med_rng(v, fmt="{:.1f}"):
    return f"{fmt.format(st.median(v))} ({fmt.format(min(v))}–{fmt.format(max(v))})"


def membw():
    rows = {}
    for line in open(os.path.join(OUT, "membw.txt")):
        p = line.strip().split(",")
        if len(p) == 4 and p[0] in ("read", "readfold", "write", "copy"):
            rows.setdefault((p[0], int(p[1])), []).append(float(p[3]))
    print("| Test | Threads | GB/s median (min–max), 5 reps |\n|---|---|---|")
    for (t, n), v in rows.items():
        print(f"| {t} | {n} | {med_rng(v)} |")


def disk():
    rows = {}
    for line in open(os.path.join(OUT, "disk.txt")):
        p = line.strip().split(",")
        if len(p) == 7 and ":" in p[0]:
            rows.setdefault(p[0], []).append([float(x) for x in p[2:]])
    print("| Test (pattern:block:QD) | MB/s median (min–max) | IOPS median | p50 latency µs | p99 latency µs |\n|---|---|---|---|---|")
    for k, v in rows.items():
        mb = [r[0] for r in v]
        print(f"| {k} | {med_rng(mb)} | {st.median([r[1] for r in v]):.0f} | {st.median([r[3] for r in v]):.0f} | {st.median([r[4] for r in v]):.0f} |")


def lb(bw_gbps=None):
    print("| Model | File MB | Threads | pp512 t/s median (min–max) | tg128 t/s median (min–max) | tg effective GB/s (file bytes × t/s) |\n|---|---|---|---|---|---|")
    for name in ("q08", "g1b", "q4b", "g3n"):
        f = os.path.join(OUT, f"lb_{name}.json")
        if not os.path.exists(f):
            print(f"| {name} | – | – | missing | missing | – |")
            continue
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"| {name} | – | – | JSON error {e} | | |")
            continue
        res = {}
        for d in data:
            fn = os.path.basename(d["model_filename"])
            size = os.path.getsize(os.path.join(MODELS, fn))
            kind = "pp" if d["n_prompt"] > 0 else "tg"
            res.setdefault(d["n_threads"], {})[kind] = d["samples_ts"]
        for t in sorted(res):
            pp = res[t].get("pp", [])
            tg = res[t].get("tg", [])
            eff = f"{size * st.median(tg) / 1e9:.1f}" if tg else "–"
            print(f"| {fn} | {size / 1e6:.0f} | {t} | {med_rng(pp) if pp else '–'} | {med_rng(tg, '{:.2f}') if tg else '–'} | {eff} |")


def perf(txt):
    pe = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", txt)
    ev = re.search(r"\s eval time =\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) tokens per second", txt)
    ptok = int(pe.group(2)) if pe else None
    pps = float(pe.group(3)) if pe else None
    tgs = float(ev.group(3)) if ev else None
    return (ptok, pps), tgs


def runmon_line(txt):
    m = re.search(r"RUNMON (.*)", txt)
    if not m:
        return {}
    return {k: v for k, v in (kv.split("=") for kv in m.group(1).split())}


def mem_table(pattern, label):
    groups = {}
    for f in sorted(glob.glob(os.path.join(OUT, pattern))):
        tag = re.sub(r"_r\d+\.txt$", "", os.path.basename(f))
        txt = open(f, errors="replace").read()
        r = runmon_line(txt)
        (ptok, pps), tgs = perf(txt)
        groups.setdefault(tag, []).append((r, ptok, pps, tgs, txt))
    print(f"| {label} | runs ok | prompt tokens | peak private MB | peak working set MB | peak commit MB | prompt t/s | gen t/s |\n|---|---|---|---|---|---|---|---|")
    for tag, runs in groups.items():
        ok = [x for x in runs if x[0].get("exit_code") == "0"]
        def col(key):
            v = [float(x[0][key]) for x in ok if key in x[0]]
            return med_rng(v, "{:.0f}") if v else "–"
        pps = [x[2] for x in ok if x[2]]
        tgs = [x[3] for x in ok if x[3]]
        ptok = ok[0][1] if ok else "–"
        print(f"| {tag} | {len(ok)}/{len(runs)} | {ptok} | {col('max_sampled_private_mb')} | {col('peak_ws_mb')} | {col('peak_commit_mb(PeakPagefileUsage)')} | "
              f"{med_rng(pps) if pps else '–'} | {med_rng(tgs, '{:.2f}') if tgs else '–'} |")
        bad = [x for x in runs if x[0].get("exit_code") != "0"]
        for x in bad[:1]:
            errs = [l for l in x[4].splitlines() if re.search(r"error|fail|exception|alloc", l, re.I)][:3]
            print(f"|  ↳ failure ({tag}) exit={x[0].get('exit_code')} | {' / '.join(e.strip()[:160] for e in errs)} | | | | | | |")


def lag():
    def probe(f):
        t = open(f).read()
        g = lambda k: {m.group(1): float(m.group(2)) for m in re.finditer(k + r": n=\d+ (?:p50=([\d.]+))", t)}
        out = {}
        for name in ("frame_lateness_ms", "frame_work_ms", "membuf_pass_ms"):
            m = re.search(name + r": n=(\d+) p50=([\d.]+) p95=([\d.]+) p99=([\d.]+) max=([\d.]+)", t)
            out[name] = [float(x) for x in m.groups()[1:]] if m else None
        return out
    conds = {"alone": "lag_base_r*.txt", "model -t 4": "lag_t4_r?.txt", "model -t 2": "lag_t2_r?.txt"}
    base = {}
    print("| Condition | metric | p50 | p95 | p99 | max | slowdown of p50 vs alone | model t/s in window |\n|---|---|---|---|---|---|---|---|")
    for c, pat in conds.items():
        files = sorted(glob.glob(os.path.join(OUT, pat)))
        if not files:
            continue
        ps = [probe(f) for f in files]
        tps = []
        for f in files:
            j = f.replace(".txt", "_tps.json")
            if os.path.exists(j):
                d = json.loads(open(j).read().strip().splitlines()[-1])
                tps.append(d["whole"]["tok_s"])
        for metric in ("frame_lateness_ms", "frame_work_ms", "membuf_pass_ms"):
            v = [p[metric] for p in ps if p[metric]]
            meds = [st.median([x[i] for x in v]) for i in range(4)]
            if c == "alone":
                base[metric] = meds[0]
            slow = "–" if c == "alone" or metric == "frame_lateness_ms" else f"{(meds[0] / base[metric] - 1) * 100:+.0f}%"
            tp = med_rng(tps, "{:.2f}") if tps and metric == "frame_lateness_ms" else ""
            print(f"| {c} | {metric} | {meds[0]:.2f} | {meds[1]:.2f} | {meds[2]:.2f} | {meds[3]:.2f} | {slow} | {tp} |")


for s in secs:
    print(f"\n### {s}\n")
    {"membw": membw, "disk": disk, "lb": lb,
     "mem": lambda: mem_table("mem_*_r*.txt", "Config (ctx, load, KV)"),
     "lag": lag,
     "limit": lambda: mem_table("lim_*_r*.txt", "Limit config")}[s]()
