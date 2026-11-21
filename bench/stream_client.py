"""Stream tokens from a running llama-server and log the wall-clock time of every token.

Usage: python -I stream_client.py --port 8090 --n 400 --out tokens.json [--prompt-file f] [--prompt "text"]
Writes JSON: {"t_ms": [epoch ms of each streamed token], "timings": <server timings>, ...}
Uses only the Python standard library.
"""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="Write a long, detailed story about a lighthouse keeper who discovers a hidden map. Keep going for many paragraphs.")
ap.add_argument("--prompt-file")
ap.add_argument("--ignore-eos", action="store_true", default=True)
a = ap.parse_args()

prompt = open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else a.prompt
body = json.dumps({
    "prompt": prompt,
    "n_predict": a.n,
    "stream": True,
    "ignore_eos": a.ignore_eos,
    "temperature": 0.7,
    "seed": 42,
    "cache_prompt": False,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/completion", data=body,
                             headers={"Content-Type": "application/json"})
t_send = time.time() * 1000
stamps = []
timings = None
text = []
# Write partial results as we go so a killed client still leaves data.
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        now = time.time() * 1000
        if d.get("content"):
            stamps.append(now)
            text.append(d["content"])
        if d.get("stop"):
            timings = d.get("timings")
            break
        if len(stamps) % 5 == 0:
            with open(a.out, "w") as f:
                json.dump({"t_send_ms": t_send, "t_ms": stamps, "timings": None, "partial": True}, f)
with open(a.out, "w") as f:
    json.dump({"t_send_ms": t_send, "t_ms": stamps, "timings": timings, "partial": False,
               "text_head": "".join(text)[:300]}, f)
print(json.dumps({"tokens": len(stamps), "timings": timings}))
