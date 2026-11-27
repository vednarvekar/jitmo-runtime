"""llama-server vs our inproc engine, CPU vs Vulkan, through the runtime; and UI lag under the
pacer's two levers (Job Object CPU cap vs pause between tokens).

  python bench/engine_compare.py speed --model <gguf> [--reps 5]
  python bench/engine_compare.py lag --model <gguf> [--secs 15]

Writes JSON lines to bench/out/engine_compare/ and prints a table.
"""
import argparse, json, os, re, statistics as st, subprocess, sys, tempfile, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "..", "runtime", "target", "release", "jitmo-runtime.exe")
PROBE = os.path.join(HERE, "target", "release", "probe.exe")
CPU = r"V:\jitmo-sandbox\llama-b12606-bin-win-cpu-x64"
VK = r"V:\jitmo-sandbox\downloads\vulkan-b12606\bin"
OUT = os.path.join(HERE, "out", "engine_compare")

PICK_SYS = "Pick the tool number for the request, or 0 if no tool fits.\n" + "\n".join(
    f"{i}: {t}" for i, t in enumerate(["Send an email", "Set a reminder", "Play music", "Find files on this PC",
                                       "Search the web", "Open an app", "Set a timer", "Add a calendar event"], 1))
USERS = ["email priya the notes from today", "remind me to call mom at 6", "play some lofi beats", "find my resume pdf",
         "search for flights to goa", "open chrome", "set a timer for ten minutes", "add lunch with raj on friday"]


def post(base, path, body, timeout=600):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else b"",
                               headers={"Content-Type": "application/json"} if body is not None else {}, method="POST")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as resp:
        return json.load(resp)


def start(tmp, tag, engine, lib, model, ngl, pacer=None, lever="pause"):
    cfg = os.path.join(tmp, f"{tag}.toml")
    gpu = f'gpu_layers = "{ngl}"\n' if ngl else ""
    pac = f'[pacer]\nmode = "{pacer}"\ninproc_lever = "{lever}"\n' if pacer else '[pacer]\nmode = "off"\n'
    open(cfg, "w").write(f"""[server]
port = 0
{pac}[budget]
reserve_free_mb = 512
pressure_free_mb = 300
critical_free_mb = 150
idle_unload_secs = 0
[[models]]
name = "m"
engine = "{engine}"
path = '{model}'
llama_server = '{lib}\\llama-server.exe'
ctx_size = 2048
log_file = '{tmp}\\{tag}.log'
{gpu}""")
    pf = os.path.join(tmp, f"{tag}.port")
    p = subprocess.Popen([EXE, "--config", cfg, "--port", "0", "--parent-pid", str(os.getpid()), "--port-file", pf,
                          "--log-file", os.path.join(tmp, f"{tag}.runtime.log")], stdout=subprocess.DEVNULL)
    for _ in range(200):
        if os.path.exists(pf):
            break
        time.sleep(0.1)
    base = "http://127.0.0.1:" + open(pf).read().split("port=")[1].strip()
    t0 = time.perf_counter()
    post(base, "/admin/load", {"model": "m"})
    return p, base, (time.perf_counter() - t0) * 1000


def stop(p):
    p.terminate()
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()


def chat(user, system=None, **kw):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return {"model": "m", "messages": msgs, "temperature": 0, **kw}


CONFIGS = [("cpu-server", "llama-server", CPU, None), ("cpu-inproc", "inproc", CPU, None),
           ("vk-server", "llama-server", VK, "all"), ("vk-inproc", "inproc", VK, "all")]


def speed(a):
    tmp = tempfile.mkdtemp(prefix="engcmp-")
    rows = []
    for tag, engine, lib, ngl in CONFIGS:
        p, base, load_ms = start(tmp, tag, engine, lib, a.model, ngl)
        try:
            post(base, "/v1/chat/completions", chat("warm up", PICK_SYS, max_tokens=1, grammar="root ::= [0-8]"))
            post(base, "/v1/chat/completions", chat("Say hi.", max_tokens=4))
            pick_ms, pick_prompt = [], []
            for r in range(a.reps):
                for u in USERS:
                    t0 = time.perf_counter()
                    v = post(base, "/v1/chat/completions", chat(u, PICK_SYS, max_tokens=1, grammar="root ::= [0-8]"))
                    pick_ms.append((time.perf_counter() - t0) * 1000)
                    pick_prompt.append(v["timings"]["prompt_ms"])
            gen_tps, cold_prompt = [], []
            for r in range(a.reps):
                v = post(base, "/v1/chat/completions", chat(f"Write a short story about a {['cat','dog','fox','owl','bee'][r % 5]} (run {r}).", max_tokens=96))
                t = v["timings"]
                gen_tps.append(t["predicted_n"] * 1000 / t["predicted_ms"])
            s = get(base, "/stats")
            child = (s["models"][0].get("child") or {})
            row = {"config": tag, "model": os.path.basename(a.model), "load_ms": round(load_ms),
                   "pick_p50_ms": round(st.median(pick_ms), 1), "pick_prompt_p50_ms": round(st.median(pick_prompt), 1),
                   "gen_tps_p50": round(st.median(gen_tps), 1),
                   "working_set_mb": child.get("working_set_mb"), "private_mb": child.get("private_mb")}
        finally:
            stop(p)
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(3)
    save("speed", rows)
    print(f"\n{'config':11} {'load':>6} {'pick p50':>9} {'prompt':>7} {'gen t/s':>7} {'WS MB':>6} {'priv MB':>7}")
    for r in rows:
        print(f"{r['config']:11} {r['load_ms']:6} {r['pick_p50_ms']:9} {r['pick_prompt_p50_ms']:7} {r['gen_tps_p50']:7} {r['working_set_mb']!s:>6} {r['private_mb']!s:>7}")


def probe(secs, tag):
    out = subprocess.run([PROBE, "--secs", str(secs), "--tag", tag], capture_output=True, text=True).stdout
    res = {}
    for name in ("frame_lateness_ms", "frame_work_ms", "membuf_pass_ms"):
        m = re.search(name + r": n=\d+ p50=([\d.]+) p95=([\d.]+) p99=([\d.]+)", out)
        if m:
            res[name] = {"p50": float(m.group(1)), "p95": float(m.group(2)), "p99": float(m.group(3))}
    return res


def lag(a):
    tmp = tempfile.mkdtemp(prefix="englag-")
    rows = []
    base_probe = probe(a.secs, "idle")
    rows.append({"case": "no model", **{k: v["p50"] for k, v in base_probe.items()}, "frame_late_p99": base_probe["frame_lateness_ms"]["p99"], "req_per_s": None, "req_p50_ms": None})
    print(json.dumps(rows[-1]), flush=True)
    cases = [("inproc, no pacer", "inproc", None, "pause"), ("inproc, CPU cap 50%", "inproc", "always", "cap"),
             ("inproc, pause 50%", "inproc", "always", "pause"), ("llama-server, CPU cap 50%", "llama-server", "always", "cap")]
    for name, engine, pacer, lever in cases:
        p, base, _ = start(tmp, re.sub(r"\W+", "_", name), engine, CPU, a.model, None, pacer, lever)
        try:
            time.sleep(1.5)
            # jitmo-like load: short answers, fresh prompts (worst case: no prefix to reuse).
            stop_flag = threading.Event()
            lat = []
            def load():
                k = 0
                while not stop_flag.is_set():
                    k += 1
                    sys_ = f"Request {k}. " + PICK_SYS
                    t0 = time.perf_counter()
                    try:
                        if k % 2:
                            post(base, "/v1/chat/completions", chat(USERS[k % len(USERS)], sys_, max_tokens=1, grammar="root ::= [0-8]"))
                        else:
                            post(base, "/v1/chat/completions", chat(USERS[k % len(USERS)], f"Fill the values (call {k}). Values: to, body", max_tokens=24,
                                 response_format={"type": "json_schema", "json_schema": {"name": "v", "schema": {"type": "object",
                                 "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]}}}))
                        lat.append((time.perf_counter() - t0) * 1000)
                    except Exception:
                        pass
            th = threading.Thread(target=load)
            th.start()
            time.sleep(2)
            n0 = len(lat)
            t_start = time.perf_counter()
            pr = probe(a.secs, name)
            win = time.perf_counter() - t_start
            served = len(lat) - n0
            stop_flag.set()
            th.join(120)
            row = {"case": name, **{k: v["p50"] for k, v in pr.items()}, "frame_late_p99": pr.get("frame_lateness_ms", {}).get("p99"),
                   "req_per_s": round(served / win, 2), "req_p50_ms": round(st.median(lat[n0:n0 + served]), 0) if served else None}
        finally:
            stop(p)
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(3)
    save("lag", rows)
    print(f"\n{'case':28} {'late p50/p99':>14} {'work p50':>8} {'mem pass':>8} {'req/s':>6} {'req p50':>7}")
    for r in rows:
        print(f"{r['case']:28} {r.get('frame_lateness_ms', 0):6.2f}/{r.get('frame_late_p99') or 0:<7.2f} {r.get('frame_work_ms', 0):8.2f} {r.get('membuf_pass_ms', 0):8.2f} {r['req_per_s']!s:>6} {r['req_p50_ms']!s:>7}")


def save(kind, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["speed", "lag"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--secs", type=int, default=12)
    a = ap.parse_args()
    speed(a) if a.what == "speed" else lag(a)
