"""How much of a jitmo-style prompt does llama-server reuse from its cache?

Starts the runtime (as our child: --parent-pid, --port 0), sends jitmo-shaped requests
(fill values for a tool, pick a tool from candidates) in an order that mixes tools, and
prints prompt tokens, cached tokens and prompt time per request.

  python bench/prompt_cache.py --model V:/jitmo-sandbox/models/gemma-3-270m-it-router-Q4_K_M.gguf
"""
import argparse, json, os, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "..", "runtime", "target", "release", "jitmo-runtime.exe")

TOOLS = {
    "email.send": ("Send an email", [("to", "who the email is for"), ("subject", "what the email is about"), ("body", "the text of the email")]),
    "reminder.add": ("Set a reminder", [("text", "what to be reminded of"), ("when", "the date or time for the reminder")]),
    "music.play": ("Play music", [("query", "the song, artist, album or playlist to play")]),
    "files.find": ("Find files on this PC", [("query", "words in the file name"), ("kind", "type of file, for example pdf or photo")]),
    "web.search": ("Search the web", [("query", "what to search the web for")]),
}

def args_system(tid):
    title, args = TOOLS[tid]
    vals = "\n".join(f"- {n}: {d}" for n, d in args)
    return (f'Fill in the values for the tool "{title}" from the user\'s request. Copy the words the user said; '
            f"do not invent anything. Use an empty string for anything the user did not say.\n\nValues:\n{vals}")

def pick_system(cands):
    lines = "\n".join(f"{n + 1}: {TOOLS[t][0]}" for n, t in enumerate(cands))
    return f"Pick the tool number for the request, or 0 if no tool fits.\n{lines}"

def post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        v = json.load(r)
    return v, (time.perf_counter() - t0) * 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "..", "runtime", "jitmo-runtime.toml"))
    ap.add_argument("--llama-server", help="override llama-server.exe (for example a Vulkan build)")
    ap.add_argument("--extra", default="", help="ignored; for notes in the output")
    a = ap.parse_args()

    pf = os.path.join(tempfile.mkdtemp(), "port.txt")
    log = os.path.join(tempfile.gettempdir(), "prompt_cache_runtime.log")
    cmd = [EXE, "--config", a.config, "--model", a.model, "--port", "0", "--parent-pid", str(os.getpid()),
           "--port-file", pf, "--log-file", log]
    if a.llama_server:
        cmd += ["--llama-server", a.llama_server]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(pf):
            break
        time.sleep(0.1)
    port = open(pf).read().split("port=")[1].strip()
    base = f"http://127.0.0.1:{port}"
    urllib.request.urlopen(urllib.request.Request(base + "/admin/load", data=b"", method="POST"), timeout=120).read()

    seq = [
        ("fill", "email.send", "email priya that the meeting moved to 3pm"),
        ("fill", "email.send", "send an email to raj saying I'll be late"),
        ("fill", "reminder.add", "remind me to call mom tomorrow at 6"),
        ("fill", "email.send", "email the team about friday's demo"),
        ("fill", "music.play", "play some lofi beats"),
        ("fill", "reminder.add", "remind me to pay rent on the 1st"),
        ("pick", ["email.send", "reminder.add", "web.search"], "email priya the notes"),
        ("pick", ["music.play", "files.find", "web.search"], "play arijit singh"),
        ("pick", ["email.send", "reminder.add", "web.search"], "remind me to stretch"),
    ]
    print(f"model {os.path.basename(a.model)} {a.extra}")
    print(f"{'kind':5} {'prompt':>6} {'cached':>6} {'new':>4} {'prompt ms':>9} {'total ms':>8}  system")
    rows = []
    for kind, arg, user in seq:
        if kind == "fill":
            system, label = args_system(arg), arg
            schema = {"type": "object", "properties": {n: {"type": "string"} for n, _ in TOOLS[arg][1]},
                      "required": [n for n, _ in TOOLS[arg][1]], "additionalProperties": False}
            extra = {"response_format": {"type": "json_schema", "json_schema": {"name": "v", "schema": schema}}, "max_tokens": 60}
        else:
            system, label = pick_system(arg), "pick " + ",".join(t.split(".")[0] for t in arg)
            extra = {"grammar": 'root ::= [0-3]', "max_tokens": 1}
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0, "cache_prompt": True, **extra}
        v, ms = post(base, body)
        t = v.get("timings", {})
        n, c = t.get("prompt_n", 0), t.get("cache_n", 0)
        rows.append((kind, n + c, c, n, t.get("prompt_ms", 0), ms, label))
        print(f"{kind:5} {n + c:6} {c:6} {n:4} {t.get('prompt_ms', 0):9.1f} {ms:8.0f}  {label}")
    proc.terminate()

if __name__ == "__main__":
    main()
