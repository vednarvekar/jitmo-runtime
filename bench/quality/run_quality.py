"""Quality run for picking the base model (roadmap step 4). Stdlib only.

Sends every item of tasks.jsonl to jitmo-runtime's POST /v1/chat/completions at temperature 0,
scores the answer with the item's checks, and records latency and tokens.

  python run_quality.py --model gemma3n                     # run all items on one model
  python run_quality.py --model qwen35-08b --no-grammar     # Qwen3.5: no digit grammar (see the doc)
  python run_quality.py --self-test                         # score each item's reference answer (no server)
  python run_quality.py --dry-run --model x                 # print what would be sent, send nothing
  python run_quality.py --compare a.jsonl b.jsonl [c.jsonl] # side-by-side table of finished runs

Output (default bench/out/quality/, git-ignored): <model>-<time>.jsonl (one line per item) and
<model>-<time>.md (summary table). The table is also printed.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TASKS = os.path.join(HERE, "tasks.jsonl")
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "out", "quality"))
CATEGORY_ORDER = ["tool_pick", "tool_pick_digit", "json_fill", "answer_passage", "instruction", "reasoning", "multilingual"]

# ----------------------------------------------------------------------------- text helpers


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def strip_think(s):
    """Removes a <think>...</think> block if a model wrote one anyway. Returns (text, had_think)."""
    if "<think>" in s or "</think>" in s:
        return re.sub(r"<think>.*?(</think>|$)", "", s, flags=re.S).strip(), True
    return s.strip(), False


def words(s):
    return [w for w in re.split(r"\s+", s.strip()) if re.search(r"\w", w)]


def sentences(s):
    parts = re.split(r"(?<=[.!?।])\s+", s.strip())
    return [p for p in parts if re.search(r"\w", p)]


def devanagari_share(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)


def final_answer(s):
    """The text after the last 'Answer:' in the output, or the last non-empty line."""
    m = re.findall(r"answer\s*[:：]\s*(.+)", s, flags=re.I)
    if m:
        return m[-1].strip()
    lines = [l for l in s.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def parse_number(s):
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None

# ----------------------------------------------------------------------------- a small JSON-schema check
# Covers what the task schemas use: object, properties, required, additionalProperties false,
# string/number/integer/boolean/array types, enum.


def schema_errors(value, schema, path="$"):
    errs = []
    t = schema.get("type")
    types = {"object": dict, "string": str, "array": list, "boolean": bool}
    if t in types and not isinstance(value, types[t]):
        return ["{}: expected {}".format(path, t)]
    if t in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return ["{}: expected {}".format(path, t)]
        if t == "integer" and not float(value).is_integer():
            return ["{}: expected integer".format(path)]
    if "enum" in schema and value not in schema["enum"]:
        errs.append("{}: {!r} not in enum".format(path, value))
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append("{}: missing {}".format(path, k))
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errs.append("{}: extra key {}".format(path, k))
        for k, sub in props.items():
            if k in value:
                errs += schema_errors(value[k], sub, path + "." + k)
    if t == "array" and "items" in schema:
        for i, v in enumerate(value):
            errs += schema_errors(v, schema["items"], "{}[{}]".format(path, i))
    return errs

# ----------------------------------------------------------------------------- scoring


def get_path(obj, path):
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            return None
    return obj


def run_check(c, text, item):
    """Returns (passed, note)."""
    t = c["t"]
    n = norm(text)
    if t == "equals":
        return norm(text).strip(" .!\"'") == norm(c["v"]).strip(" .!\"'"), ""
    if t == "not_equals":
        return norm(text).strip(" .!\"'") != norm(c["v"]).strip(" .!\"'"), ""
    if t == "regex":
        return re.search(c["v"], text.strip(), flags=re.I | re.M) is not None, ""
    if t == "not_regex":
        return re.search(c["v"], text.strip(), flags=re.I | re.M) is None, ""
    if t == "contains_all":
        missing = [v for v in c["v"] if norm(v) not in n]
        return not missing, ("missing " + ", ".join(missing)) if missing else ""
    if t == "contains_any":
        return any(norm(v) in n for v in c["v"]), ""
    if t == "not_contains":
        found = [v for v in c["v"] if norm(v) in n]
        return not found, ("has " + ", ".join(found)) if found else ""
    if t == "starts_with":
        return n.startswith(norm(c["v"])), ""
    if t == "not_starts_with":
        return not n.startswith(norm(c["v"])), ""
    if t == "max_words":
        k = len(words(text))
        return k <= c["v"], "{} words".format(k)
    if t == "max_sentences":
        k = len(sentences(text))
        return k <= c["v"], "{} sentences".format(k)
    if t == "lines":
        k = len([l for l in text.splitlines() if l.strip()])
        return k == c["v"], "{} lines".format(k)
    if t == "lowercase":
        return text == text.lower(), ""
    if t == "script":
        share = devanagari_share(text)
        return share >= 0.5, "devanagari {:.0%}".format(share)
    if t == "digit":
        m = re.search(r"\d", text)
        return bool(m) and int(m.group(0)) == c["v"], ""
    if t == "answer_num":
        got = parse_number(final_answer(text))
        if got is None:
            return False, "no number"
        return math.isclose(got, c["v"], abs_tol=c.get("tol", 1e-9)), "got {}".format(got)
    if t == "answer_text":
        got = final_answer(text)
        return norm(c["v"]) in norm(got), "got {!r}".format(got[:40])
    if t in ("json_schema", "field"):
        try:
            obj = json.loads(text)
        except ValueError:
            return False, "not JSON"
        if t == "json_schema":
            errs = schema_errors(obj, item["schema"])
            return not errs, "; ".join(errs[:3])
        val = get_path(obj, c["path"])
        sval = norm(val) if isinstance(val, str) else ""
        op = c["op"]
        if op == "equals":
            return val == c["v"], "got {!r}".format(val)
        if op == "contains_all":
            missing = [v for v in c["v"] if norm(v) not in sval]
            return isinstance(val, str) and not missing, "{}={!r}".format(c["path"], val)
        if op == "contains_any":
            return isinstance(val, str) and any(norm(v) in sval for v in c["v"]), "{}={!r}".format(c["path"], val)
        if op == "empty":
            return isinstance(val, str) and sval == "", "{}={!r}".format(c["path"], val)
        if op == "not_empty":
            return isinstance(val, str) and sval != "", "{}={!r}".format(c["path"], val)
    raise ValueError("unknown check {}".format(c))


def score(item, text):
    results = []
    for c in item["checks"]:
        ok, note = run_check(c, text, item)
        results.append({"check": c["t"] + ("/" + c["path"] if "path" in c else ""), "ok": ok, "note": note})
    return results

# ----------------------------------------------------------------------------- requests


def build_body(item, args):
    body = {
        "model": args.model,
        "messages": item["messages"],
        "temperature": 0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": item["max_tokens"],
        "cache_prompt": True,
    }
    if not args.no_template_kwargs:
        # What jitmo agent sends: Qwen3 / Qwen3.5 templates read it; Gemma's ignores it.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if "grammar" in item and not args.no_grammar:
        body["grammar"] = item["grammar"]
    if "response_format" in item and not args.no_schema:
        body["response_format"] = item["response_format"]
    return body


def post(url, body, timeout, req_id=None):
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "X-Jitmo-Priority": "interactive"}
    if req_id:
        headers["X-Jitmo-Request-Id"] = req_id
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        status = e.code
    ms = (time.perf_counter() - t0) * 1000.0
    try:
        return status, json.loads(raw), ms
    except ValueError:
        return status, {"raw": raw[:500]}, ms


def warm_up(base, model, timeout):
    status, resp, ms = post(base + "/admin/load", {"model": model}, timeout)
    if status != 200:
        sys.exit("warm-up failed ({}): {}".format(status, json.dumps(resp)[:400]))
    return resp.get("load_ms", ms)

# ----------------------------------------------------------------------------- summary


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def p50(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=0):
    return "-" if x is None else ("{:.%df}" % nd).format(x)


def summary_rows(records):
    cats = [c for c in CATEGORY_ORDER if any(r["category"] == c for r in records)]
    cats += sorted({r["category"] for r in records} - set(cats))
    rows = []
    for cat in cats + ["ALL"]:
        rs = [r for r in records if cat == "ALL" or r["category"] == cat]
        rows.append({
            "category": cat, "n": len(rs), "pass": sum(r["pass"] for r in rs),
            "checks": mean([r["check_share"] for r in rs]),
            "errors": sum(1 for r in rs if r.get("error")),
            "p50_ms": p50([r.get("latency_ms") for r in rs]),
            "prompt_tok": mean([r.get("prompt_tokens") for r in rs]),
            "out_tok": mean([r.get("completion_tokens") for r in rs]),
            "gen_tps": mean([r.get("gen_tps") for r in rs]),
        })
    return rows


def summary_table(records, title):
    out = ["## " + title, "",
           "| Category | Items | Pass | Pass % | Checks % | Errors | p50 latency ms | Prompt tok | Output tok | Gen tok/s |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary_rows(records):
        name = "**all**" if r["category"] == "ALL" else r["category"]
        out.append("| {} | {} | {} | {:.1f} | {:.1f} | {} | {} | {} | {} | {} |".format(
            name, r["n"], r["pass"], pct(r["pass"], r["n"]), 100 * (r["checks"] or 0), r["errors"],
            fmt(r["p50_ms"]), fmt(r["prompt_tok"]), fmt(r["out_tok"], 1), fmt(r["gen_tps"], 1)))
    fails = [r for r in records if not r["pass"]]
    if fails:
        out += ["", "Failed items:", ""]
        for r in fails:
            bad = "; ".join("{} {}".format(c["check"], c["note"]).strip() for c in r["checks"] if not c["ok"])
            ans = (r.get("answer") or r.get("error") or "").replace("\n", " \\n ")
            out.append("- `{}` ({}): {} -> `{}`".format(r["id"], r.get("kind", ""), bad, ans[:120]))
    return "\n".join(out)


def compare(paths):
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        name = recs[0].get("model", os.path.basename(p)) if recs else os.path.basename(p)
        runs.append((name, recs))
    head = "| Category | " + " | ".join("{} pass %".format(n) for n, _ in runs) + " | " + \
        " | ".join("{} p50 ms".format(n) for n, _ in runs) + " |"
    lines = ["## Comparison", "", head, "|" + "---|" * (1 + 2 * len(runs))]
    tables = [{r["category"]: r for r in summary_rows(recs)} for _, recs in runs]
    cats = []
    for t in tables:
        cats += [c for c in t if c not in cats]
    cats = [c for c in cats if c != "ALL"] + ["ALL"]
    for cat in cats:
        cells = []
        for t in tables:
            r = t.get(cat)
            cells.append("-" if not r else "{:.1f} ({}/{})".format(pct(r["pass"], r["n"]), r["pass"], r["n"]))
        lat = [fmt(t[cat]["p50_ms"]) if cat in t else "-" for t in tables]
        lines.append("| {} | {} | {} |".format("**all**" if cat == "ALL" else cat, " | ".join(cells), " | ".join(lat)))
    by_id = [{r["id"]: r for r in recs} for _, recs in runs]
    ids = [i for i in by_id[0] if all(i in b for b in by_id[1:])]
    diff = [i for i in ids if len({b[i]["pass"] for b in by_id}) > 1]
    lines += ["", "Items where the models differ ({}):".format(len(diff)), ""]
    for i in diff:
        lines.append("- `{}`: ".format(i) + ", ".join(
            "{} {}".format(n, "pass" if b[i]["pass"] else "FAIL") for (n, _), b in zip(runs, by_id)))
    return "\n".join(lines)

# ----------------------------------------------------------------------------- main


def load_tasks(path, cats, ids, limit):
    with open(path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    if cats:
        items = [i for i in items if i["category"] in cats]
    if ids:
        items = [i for i in items if i["id"] in ids]
    return items[:limit] if limit else items


def self_test(items):
    bad = 0
    for it in items:
        res = score(it, it["reference"])
        if not all(r["ok"] for r in res):
            bad += 1
            print("REFERENCE FAILS", it["id"], [r for r in res if not r["ok"]])
    counts = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    print("{} items, {} references fail their own checks".format(len(items), bad))
    for c in CATEGORY_ORDER:
        if c in counts:
            print("  {:16s} {}".format(c, counts[c]))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090", help="jitmo-runtime base URL")
    ap.add_argument("--model", help="model name as in the runtime config ([[models]] name)")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--tag", default="", help="extra word in the output file names")
    ap.add_argument("--category", action="append", help="only this category (repeatable)")
    ap.add_argument("--id", action="append", help="only this item id (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    ap.add_argument("--no-grammar", action="store_true", help="drop the digit grammar (Qwen3.5: grammar made llama-server use ~6 GB)")
    ap.add_argument("--no-schema", action="store_true", help="drop response_format json_schema (JSON from the prompt alone)")
    ap.add_argument("--no-template-kwargs", action="store_true", help="do not send chat_template_kwargs enable_thinking=false")
    ap.add_argument("--no-warmup", action="store_true", help="skip POST /admin/load before the run")
    ap.add_argument("--resume", metavar="RUN.jsonl",
                    help="keep the answered items of an earlier run of the same model; rerun only the ones that got a request error")
    ap.add_argument("--max-errors", type=int, default=3,
                    help="stop after this many request errors in a row (model unloaded for memory, runtime gone)")
    ap.add_argument("--dry-run", action="store_true", help="print the request bodies, send nothing")
    ap.add_argument("--self-test", action="store_true", help="score each item's reference answer; no server")
    ap.add_argument("--compare", nargs="+", metavar="RUN.jsonl", help="compare finished runs; no server")
    args = ap.parse_args()
    # Windows consoles default to cp1252, which cannot print Devanagari answers.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.compare:
        print(compare(args.compare))
        return 0
    items = load_tasks(args.tasks, args.category, args.id, args.limit)
    if args.self_test:
        return self_test(items)
    if not args.model:
        ap.error("--model is required for a run")
    if args.dry_run:
        for it in items:
            print(json.dumps({"id": it["id"], "body": build_body(it, args)}, ensure_ascii=False))
        print("{} requests".format(len(items)), file=sys.stderr)
        return 0

    base = args.url.rstrip("/")
    load_ms = None if args.no_warmup else warm_up(base, args.model, args.timeout)
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", args.model) + ("-" + args.tag if args.tag else "")
    jpath = os.path.join(args.out, "{}-{}.jsonl".format(safe, stamp))
    mpath = os.path.join(args.out, "{}-{}.md".format(safe, stamp))
    flags = {k: getattr(args, k) for k in ("no_grammar", "no_schema", "no_template_kwargs")}
    kept = {}
    if args.resume:
        with open(args.resume, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    old = json.loads(line)
                    if not old.get("error"):
                        kept[old["id"]] = old
        print("resume: keeping {} answered items from {}".format(len(kept), args.resume), flush=True)
    order = [it["id"] for it in items]
    items = [it for it in items if it["id"] not in kept]
    records = []
    errors_in_row = 0
    t_start = time.perf_counter()
    with open(jpath, "w", encoding="utf-8", newline="\n") as out:
        for n, it in enumerate(items, 1):
            body = build_body(it, args)
            rec = {"id": it["id"], "category": it["category"], "kind": it.get("kind"), "lang": it.get("lang"),
                   "model": args.model, "flags": flags}
            try:
                status, resp, ms = post(base + "/v1/chat/completions", body, args.timeout,
                                        req_id="q-{}-{}".format(stamp, it["id"]))
            except Exception as e:  # connection refused, timeout
                status, resp, ms = None, {}, None
                rec["error"] = "{}: {}".format(type(e).__name__, e)
            rec["status"], rec["latency_ms"] = status, ms
            text = ""
            if status == 200:
                try:
                    text = resp["choices"][0]["message"].get("content") or ""
                    rec["finish_reason"] = resp["choices"][0].get("finish_reason")
                except (KeyError, IndexError, TypeError):
                    rec["error"] = "no answer in response"
                usage, tm = resp.get("usage", {}), resp.get("timings", {})
                rec["prompt_tokens"] = usage.get("prompt_tokens")
                rec["completion_tokens"] = usage.get("completion_tokens")
                rec["prompt_ms"] = tm.get("prompt_ms")
                rec["gen_ms"] = tm.get("predicted_ms")
                rec["gen_tps"] = tm.get("predicted_per_second") if (tm.get("predicted_n") or 0) > 1 else None
                rec["cached_tokens"] = tm.get("cache_n")
            elif "error" not in rec:
                rec["error"] = "HTTP {}: {}".format(status, json.dumps(resp)[:300])
            text, had_think = strip_think(text)
            rec["answer"], rec["had_think"] = text, had_think
            checks = score(it, text) if "error" not in rec else [
                {"check": c["t"], "ok": False, "note": "request failed"} for c in it["checks"]]
            rec["checks"] = checks
            rec["check_share"] = sum(c["ok"] for c in checks) / len(checks)
            rec["pass"] = all(c["ok"] for c in checks)
            records.append(rec)
            print("[{:3d}/{}] {:14s} {:4s} {:>7s} ms  {}".format(
                n, len(items), it["id"], "ok" if rec["pass"] else "FAIL", fmt(ms),
                (text or rec.get("error", "")).replace("\n", " ")[:70]), flush=True)
            errors_in_row = errors_in_row + 1 if rec.get("error") else 0
            if errors_in_row >= args.max_errors:
                print("stopping: {} request errors in a row; rerun with --resume {}".format(errors_in_row, jpath), flush=True)
                break
        by_id = dict(kept)
        by_id.update({r["id"]: r for r in records})
        records = [by_id[i] for i in order if i in by_id]
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    wall = time.perf_counter() - t_start
    title = "{} ({} items, {:.0f} s, warm-up load {} ms, flags {})".format(
        args.model, len(records), wall, fmt(load_ms), ", ".join(k for k, v in flags.items() if v) or "none")
    table = summary_table(records, title)
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(table + "\n")
    print()
    print(table)
    print("\nwrote {}\n      {}".format(jpath, mpath))
    return 0


if __name__ == "__main__":
    sys.exit(main())
