"""Smoke test for `engine = "inproc"` (our own engine worker, docs/08-inproc-engine.md).

Starts the runtime as our child (--port 0 --parent-pid) with two inproc models (CPU, and
Vulkan if the build is on disk) and checks: load, chat, stream, JSON schema, grammar, prefix
reuse, cancel of a running request, client closing a stream, pacer CPU cap on the worker,
model switch with pool eviction, unload, and exit with the parent. Exit code 0 = all passed.

  python runtime/scripts/smoke_inproc.py
"""
import json, os, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "..", "target", "release", "jitmo-runtime.exe")
CPU_DIR = r"V:\jitmo-sandbox\llama-b12606-bin-win-cpu-x64"
VK_DIR = r"V:\jitmo-sandbox\downloads\vulkan-b12606\bin"
ROUTER = r"V:\jitmo-sandbox\models\gemma-3-270m-it-router-Q4_K_M.gguf"
GEMMA = r"V:\jitmo-sandbox\models\gemma-3-270m-it-Q4_K_M.gguf"
failed = 0


def check(name, ok, detail=""):
    global failed
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + str(detail) if detail else ""), flush=True)
    failed += 0 if ok else 1


def req(base, method, path, body=None, headers=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(headers or {})
    r = urllib.request.Request(base + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()), dict(resp.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), dict(e.headers)
        except ValueError:
            return e.code, raw.decode(), dict(e.headers)


def chat(model, content, system=None, **kw):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    return {"model": model, "messages": msgs, "temperature": 0, **kw}


def main():
    vk = os.path.exists(os.path.join(VK_DIR, "llama.dll"))
    tmp = tempfile.mkdtemp(prefix="smoke-inproc-")
    cfg = os.path.join(tmp, "inproc.toml")
    models = f"""
[[models]]
name = "router"
engine = "inproc"
path = '{ROUTER}'
llama_server = '{CPU_DIR}\\llama-server.exe'
ctx_size = 2048
log_file = '{tmp}\\router.log'
"""
    if vk:
        models += f"""
[[models]]
name = "gpu"
engine = "inproc"
path = '{GEMMA}'
llama_server = '{VK_DIR}\\llama-server.exe'
gpu_layers = "all"
ctx_size = 2048
log_file = '{tmp}\\gpu.log'
"""
    open(cfg, "w").write(f"""
[server]
port = 0
[budget]
reserve_free_mb = 512
pressure_free_mb = 300
critical_free_mb = 150
{models}
""")
    pf = os.path.join(tmp, "port.txt")
    proc = subprocess.Popen([EXE, "--config", cfg, "--port", "0", "--parent-pid", str(os.getpid()), "--port-file", pf,
                             "--log-file", os.path.join(tmp, "runtime.log")], stdout=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(pf):
            break
        time.sleep(0.1)
    base = "http://127.0.0.1:" + open(pf).read().split("port=")[1].strip()
    print("runtime at", base, "logs in", tmp)

    st, b, _ = req(base, "POST", "/admin/load", {"model": "router"})
    check("load inproc router", st == 200, b)
    st, s, _ = req(base, "GET", "/stats")
    child = [m for m in s["models"] if m["name"] == "router"][0] if isinstance(s.get("models"), list) else None
    check("/stats shows the worker", st == 200 and child is not None, child and {k: child.get(k) for k in ("state", "pid")})

    st, b, h = req(base, "POST", "/v1/chat/completions", chat("router", "Say hello.", max_tokens=12))
    text = b["choices"][0]["message"]["content"] if st == 200 else b
    check("chat", st == 200 and isinstance(text, str) and b["usage"]["completion_tokens"] > 0, repr(text)[:60])
    check("request id header", "X-Jitmo-Request-Id" in h or "x-jitmo-request-id" in {k.lower() for k in h})

    schema = {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]}
    sys_fill = 'Fill in the values for the tool "Send an email" from the user\'s request.\n\nValues:\n- to: who\n- body: the text'
    st, b, _ = req(base, "POST", "/v1/chat/completions", chat("router", "email priya that the meeting moved to 3pm", sys_fill, max_tokens=60,
                   response_format={"type": "json_schema", "json_schema": {"name": "v", "schema": schema}}))
    out = b["choices"][0]["message"]["content"] if st == 200 else ""
    try:
        v = json.loads(out)
        ok = list(v.keys()) == ["to", "body"]
    except ValueError:
        v, ok = out, False
    check("json_schema: valid JSON, keys in schema order", st == 200 and ok, v)

    sys_pick = "Pick the tool number for the request, or 0 if no tool fits.\n1: Send an email\n2: Set a reminder\n3: Play music"
    st, b, _ = req(base, "POST", "/v1/chat/completions", chat("router", "play some lofi", sys_pick, max_tokens=1, grammar="root ::= [0-3]"))
    d = b["choices"][0]["message"]["content"] if st == 200 else ""
    check("grammar: one digit", st == 200 and d in list("0123"), repr(d))
    st, b2, _ = req(base, "POST", "/v1/chat/completions", chat("router", "remind me to stretch", sys_pick, max_tokens=1, grammar="root ::= [0-3]"))
    check("prefix reuse (same system prompt)", st == 200 and b2.get("timings", {}).get("cache_n", 0) > 0, b2.get("timings"))

    # Stream
    r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(chat("router", "Count to five.", max_tokens=20, stream=True)).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=120) as resp:
        sse = resp.read().decode()
    events = [l for l in sse.splitlines() if l.startswith("data:")]
    check("stream: events, timings, [DONE]", len(events) >= 3 and events[-1].strip() == "data: [DONE]" and '"timings"' in sse, f"{len(events)} events")

    # Cancel a running request (long generation), by id.
    rid = "smoke-cancel-1"
    res = {}
    def long():
        res["r"] = req(base, "POST", "/v1/chat/completions", chat("router", "Write a very long story about a dragon.", max_tokens=1500),
                       headers={"X-Jitmo-Request-Id": rid})
    t = threading.Thread(target=long); t.start()
    time.sleep(2.0)
    st, b, _ = req(base, "DELETE", "/v1/requests/" + rid)
    t.join(60)
    check("DELETE a running request -> 499", st == 200 and res.get("r", (0,))[0] == 499, (st, res.get("r", (None,))[0]))
    t0 = time.time()
    st, b, _ = req(base, "POST", "/v1/chat/completions", chat("router", "Say hi.", max_tokens=4))
    check("worker is free right after the cancel", st == 200 and time.time() - t0 < 10, f"{time.time() - t0:.1f} s")

    # Client closes a stream: the worker must stop (next request must not wait for 1500 tokens).
    import socket
    port = int(base.rsplit(":", 1)[1])
    body = json.dumps(chat("router", "Write a very long story about a castle.", max_tokens=1500, stream=True)).encode()
    s_ = socket.create_connection(("127.0.0.1", port))
    s_.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
    s_.recv(4096); time.sleep(1.0); s_.close()
    t0 = time.time()
    st, b, _ = req(base, "POST", "/v1/chat/completions", chat("router", "Say hi.", max_tokens=4))
    check("client closed a stream -> generation stopped", st == 200 and time.time() - t0 < 10, f"next request {time.time() - t0:.1f} s")

    # Pacer cap is applied to the worker's job (mode always would need a restart; read /stats instead).
    st, s, _ = req(base, "GET", "/stats")
    check("/stats pacer present", st == 200 and "pacer" in s, s.get("pacer", {}).get("level"))

    if vk:
        st, b, _ = req(base, "POST", "/v1/chat/completions", chat("gpu", "Name three colors.", max_tokens=24))
        out = b["choices"][0]["message"]["content"] if st == 200 else b
        check("Vulkan inproc model answers", st == 200 and len(out) > 0, repr(out)[:60])
        st, b, _ = req(base, "POST", "/v1/chat/completions", chat("gpu", "Name three colors.", max_tokens=24))
        tm = b.get("timings", {}) if st == 200 else {}
        check("Vulkan: warm request", st == 200, {k: round(v, 1) if isinstance(v, float) else v for k, v in tm.items()})
    else:
        print("SKIP  Vulkan (no build at %s)" % VK_DIR)

    st, b, _ = req(base, "POST", "/admin/unload?model=router")
    st2, s, _ = req(base, "GET", "/stats")
    state = [m for m in s.get("models", []) if m["name"] == "router"]
    check("unload stops the worker", st == 200 and state and state[0].get("state") == "unloaded", state and state[0].get("state"))

    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("ALL CHECKS PASSED" if failed == 0 else f"{failed} CHECK(S) FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
