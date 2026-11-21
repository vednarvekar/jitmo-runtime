"""Table of llama-bench JSON files: median (min-max) tokens/s per model, repack, threads, test.

Usage: python -I summarize_lb2.py out/lb2_*.json
"""
import json
import statistics as st
import sys

for path in sys.argv[1:]:
    for e in json.load(open(path)):
        test = f"pp{e['n_prompt']}" if e["n_prompt"] else f"tg{e['n_gen']}"
        s = e["samples_ts"]
        print(f"{e['model_filename'].split('/')[-1]:40s} repack={int(e['repack'])} t={e['n_threads']} {test:6s} "
              f"{st.median(s):7.2f} ({min(s):.2f}-{max(s):.2f})  size={e['model_size']}")
