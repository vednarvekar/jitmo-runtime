"""Crash injection: kill model processes and check the runtime recovers.

- A request whose model process is killed mid-answer still succeeds (one retry after a restart).
- A pinned model killed while idle is started again by the watchdog.
- A crash loop (more than 3 crashes in 5 min) stops the automatic restarts; a request still loads it.
Both engines (llama-server and inproc). Exit code 0 = all passed.

  python runtime/scripts/crash_test.py
"""
import json, os, signal, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "..", "target", "release", "jitmo-runtime.exe")
CPU_DIR = r"V:\jitmo-sandbox\llama-b12606-bin-win-cpu-x64"
ROUTER = r"V:\jitmo-sandbox\models\gemma-3-270m-it-router-Q4_K_M.gguf"
failed = 0


def check(name, ok, detail=""):
    global failed
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + str(detail) if detail else ""), flush=True)
    failed += 0 if ok else 1


def req(base, method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    r = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def model_info(base, name):
    _, s = req(base, "GET", "/stats")
    m = [x for x in s.get("models", []) if x["name"] == name][0]
    return m.get("state"), (m.get("child") or {}).get("pid"), s


def kill(pid):
    try:
        os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
    except OSError:
        pass


def wait_state(base, name, want, secs):
    t0 = time.time()
    while time.time() - t0 < secs:
        st, pid, _ = model_info(base, name)
        if st == want and (want != "loaded" or pid):
            return st, pid
        time.sleep(0.3)
    return model_info(base, name)[:2]


def main():
    tmp = tempfile.mkdtemp(prefix="crash-")
    cfg = os.path.join(tmp, "crash.toml")
    entry = lambda name, engine: f"""
[[models]]
name = "{name}"
engine = "{engine}"
path = '{ROUTER}'
llama_server = '{CPU_DIR}\\llama-server.exe'
ctx_size = 2048
pinned = true
log_file = '{tmp}\\{name}.log'
"""
    open(cfg, "w").write("[server]\nport = 0\n[pacer]\nmode = \"off\"\n[budget]\nreserve_free_mb = 512\nidle_unload_secs = 0\n"
                         + entry("inproc", "inproc") + entry("server", "llama-server"))
    pf = os.path.join(tmp, "port")
    proc = subprocess.Popen([EXE, "--config", cfg, "--port", "0", "--parent-pid", str(os.getpid()), "--port-file", pf,
                             "--log-file", os.path.join(tmp, "runtime.log")], stdout=subprocess.DEVNULL)
    for _ in range(100):
        if os.path.exists(pf):
            break
        time.sleep(0.1)
    base = "http://127.0.0.1:" + open(pf).read().split("port=")[1].strip()
    print("runtime at", base, "logs in", tmp)

    for name in ("inproc", "server"):
        req(base, "POST", "/admin/load", {"model": name})
        st, pid, s0 = model_info(base, name)
        retried0 = s0.get("counters", s0).get("retried_after_crash", 0) if isinstance(s0.get("counters"), dict) else s0.get("retried_after_crash", 0)

        # 1. Kill mid-answer: the request is retried after a restart and succeeds.
        res = {}
        def long():
            res["r"] = req(base, "POST", "/v1/chat/completions", {"model": name, "temperature": 0, "max_tokens": 300,
                           "messages": [{"role": "user", "content": "Write a long story about a dragon."}]})
        t = threading.Thread(target=long)
        t.start()
        time.sleep(1.5)
        kill(pid)
        t.join(180)
        code, body = res.get("r", (None, {}))
        _, new_pid, s1 = model_info(base, name)
        check(f"{name}: killed mid-answer -> request still succeeds", code == 200 and body.get("usage", {}).get("completion_tokens", 0) > 0,
              f"status {code}, new pid {new_pid} (was {pid})")
        txt = json.dumps(s1)
        check(f"{name}: retry counted in /stats", '"retried_after_crash"' in txt and new_pid != pid)

        # 2. Pinned model killed while idle: the watchdog starts it again.
        _, pid = wait_state(base, name, "loaded", 5)
        kill(pid)
        st2, pid2 = wait_state(base, name, "loaded", 30)
        time.sleep(0.5)
        st2, pid2 = wait_state(base, name, "loaded", 30)
        check(f"{name}: pinned model killed while idle -> restarted", st2 == "loaded" and pid2 and pid2 != pid, f"{st2}, pid {pid} -> {pid2}")

    # 3. Crash loop (inproc): crashes 2..4 within 5 min -> after the 3rd counted crash, no restart.
    name = "inproc"
    states = []
    for _ in range(3):
        _, pid = wait_state(base, name, "loaded", 30)
        if not pid:
            break
        kill(pid)
        time.sleep(2.5)
        states.append(model_info(base, name)[0])
    time.sleep(3)
    st, pid, _ = model_info(base, name)
    check("crash loop -> automatic restarts stop", st != "loaded", f"states {states}, now {st}")
    code, body = req(base, "POST", "/v1/chat/completions", {"model": name, "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})
    check("crash loop -> a request still loads the model", code == 200, code)

    proc.terminate()
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("ALL CHECKS PASSED" if failed == 0 else f"{failed} CHECK(S) FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
