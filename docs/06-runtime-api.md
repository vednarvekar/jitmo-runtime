# 06. Runtime v0.1: API for jitmo agent

This doc is for the jitmo agent team. It says what the runtime does, how to start it, and how to call it.
Code: `runtime/` (Rust). Config: `runtime/jitmo-runtime.toml`. Smoke test: `runtime/scripts/smoke.ps1`.

## What it does

- It runs **one base model** in a llama.cpp `llama-server` child process, on a private localhost port.
- It starts the model **on the first request** (lazy) and **unloads it** after 90 s without requests.
- It checks **free RAM before loading**. If too little would stay free, it refuses with a clear error.
- It **unloads the model when the PC is short of memory**: after the running answer is finished, or at once if free RAM is critically low.
- It runs **one generation at a time**. Other requests wait in a queue (8 by default).
- It gives the model **fewer threads** (physical cores − 1, at most half the logical processors) and **below-normal priority**.
- Its **pacer** caps the model's CPU while you use the PC (keyboard/mouse, other busy apps, low RAM) and lifts the cap when you stop. See [Pacer](#pacer).
- It **caps the model's private memory** with a limit computed from the model file and the context size.
- It speaks the **OpenAI API** (`/v1/chat/completions`, `/v1/models`), so any OpenAI client works.
- It runs **chains**: ordered steps on the same base model. Each step has its own prompt, settings and optional LoRA adapter. No model reload between steps.

## Start it

```powershell
cd V:\Code\jitmo-runtime\runtime
cargo build --release
.\target\release\jitmo-runtime.exe --config jitmo-runtime.toml
```

Options (all optional):

| Option | Meaning |
|---|---|
| `--config <file>` | Config file. Default: `./jitmo-runtime.toml` if it exists, else built-in defaults (no chains). |
| `--model <file.gguf>` | Override `model.path`. |
| `--llama-server <exe>` | Override `model.llama_server`. |
| `--port <n>` | Override `server.port`. |
| `--log-level <level>` | `error`, `warn`, `info` (default), `debug`, `trace`. |

Logs go to stdout. llama-server's own log goes to `model.log_file` (default `runtime\llama-server.log`).
**Ctrl+C** stops the model and exits. If the runtime is killed hard, Windows kills the child too (Job Object kill-on-close).

## Config reference

One TOML file. Every key is optional. Relative paths are relative to the config file.

### `[server]`

| Key | Default | Meaning |
|---|---|---|
| `host` | `"127.0.0.1"` | Address of the API. Keep it on localhost. |
| `port` | `8090` | Port of the API. |
| `max_queue` | `8` | Requests that may wait while one generation runs. More get `503 queue_full`. |

### `[model]`

| Key | Default | Meaning |
|---|---|---|
| `name` | `"jitmo-base"` | The model id clients send. |
| `path` | `V:\jitmo-sandbox\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | The GGUF file. |
| `llama_server` | `V:\jitmo-sandbox\llama\llama-server.exe` | The llama.cpp server. |
| `port` | `0` | Private port of the child. `0` = any free port. |
| `ctx_size` | `4096` | Context length in tokens. |
| `threads` | `0` | `0` = physical cores − 1, but at most half of the logical processors, at least 1. The i5-1135G7 (4 cores, 8 threads) gets min(3, 4) = 3. |
| `kv_cache_type` | `"f16"` | KV cache type for K and V. `q8_0` saves about 280 MB at ctx 4096 on Qwen3-4B, but read a long prompt 35% slower (measured below). |
| `flash_attn` | `"on"` | `on`, `off`, `auto`. A `q8_0` V cache needs `on`. |
| `mmap` | `true` | `false` passes `-lm none`: weights are copied into private memory and count against the cap. |
| `no_repack` | `true` | `true` passes `--no-repack`: no second, private copy of the weights. About half the RAM, about 20% slower (see below). |
| `low_priority` | `true` | Child runs at below-normal CPU priority. |
| `cache_ram_mb` | `256` | llama-server's prompt cache in RAM (`--cache-ram`). It keeps earlier prompts so repeated prefixes (chain system prompts) are not re-read. It grows the child's private memory, so the automatic cap includes it. llama-server's own default is 8192, which would outgrow any cap. `0` = off. |
| `ctx_checkpoints` | `4` | Context checkpoints per slot (`--ctx-checkpoints`) for sliding-window and hybrid models (Gemma 3n, Qwen3.5). Each holds a copy of the SWA KV or recurrent state; the automatic cap includes them. llama-server's own default is 32. |
| `chat_template_kwargs` | none | Added to every request that does not set its own. The shipped config sets `{ enable_thinking = false }` for Qwen models. |
| `extra_args` | `[]` | Extra llama-server arguments, passed as-is. |
| `log_file` | `"llama-server.log"` | llama-server's log, rewritten at each start. `""` = discard. |
| `load_timeout_secs` | `180` | Give up if the model is not ready after this long. |

### `[budget]`

| Key | Default | Meaning |
|---|---|---|
| `memory_cap_mb` | `"auto"` | Job Object commit (private bytes) cap on the child. `"auto"` = computed from the model file and ctx (below). A number = that many MiB. `0` = no cap. It does not count the mmap'd weights. |
| `reserve_free_mb` | `1536` | Refuse to load if free RAM after loading would be below this. |
| `load_estimate_mb` | `0` | RAM the loaded model is expected to take. `0` = model file size + the memory cap (+ 1024 MB if there is no cap). |
| `pressure_free_mb` | `1024` | Memory pressure: free RAM below this on two checks in a row, or Windows' low-memory signal. A running generation (or chain) may **finish first**; then the model is unloaded before the next request starts. |
| `critical_free_mb` | `500` | Free RAM below this on **one** check: unload at once, even mid-generation (that request fails with `engine_error`). `0` = never. Must be below `pressure_free_mb`. |
| `idle_unload_secs` | `90` | Unload after this many seconds without requests. `0` = never. |
| `watch_interval_ms` | `1000` | How often the watchdog checks idle time, memory and the child. |

After a pressure unload, queued requests behave as before: the next one tries to load the model again and gets `503 insufficient_memory` if the RAM reserve still cannot be met.

### `[pacer]`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `"auto"` | `auto` = cap while the user is busy; `always` = cap all the time; `off` = never cap (signals are still shown in `/stats`). |
| `poll_ms` | `300` | How often the pacer samples. Each sample is 4 cheap Win32 calls. |
| `input_active_ms` | `5000` | Keyboard or mouse input this recently = user active. `0` = ignore input. |
| `others_cpu_busy_pct` | `10.0` | Other processes (everything except the model) above this share of all logical processors = busy. `0` = ignore. |
| `low_free_mb` | `2048` | Free RAM below this = busy. `0` = ignore. |
| `busy_cpu_pct` | `50` | While busy, the model may use this share of the CPU it uses at full speed. With 3 of 8 threads that is a hard cap of 18.75% of the whole CPU. `100` = no CPU cap. |
| `busy_priority` | `"unchanged"` | `idle` = also drop the child to idle priority while busy. |
| `release_after_ms` | `2000` | Keep the cap this long after the last busy sample. |

### What the memory cap limits

The cap is a Windows Job Object `JOB_OBJECT_LIMIT_PROCESS_MEMORY`. It limits the child's **commit charge (private bytes)** only.

- It does **not** count the model weights when they are memory-mapped (`mmap = true`, the default). Those pages are file-backed; Windows can drop them and read them back.
- It **does** count the KV cache, recurrent state, compute buffers, the prompt cache, context checkpoints and any private copy of the weights.
- llama.cpp's CPU backend normally "repacks" the weights into a faster layout in private memory, while the mapped file also stays in RAM. The weights then sit in RAM twice.
  The baseline measured Qwen3-4B at 2211 MB private / 4201 MB working set with repacking, and 524 MB / 2518 MB without. So `no_repack = true` is the default.
  **With `no_repack = false` the automatic cap is too small** (it does not include a repacked copy): set `memory_cap_mb` by hand or to `0`.
- The real RAM use is the **working set** (about the touched part of the file + private bytes). `/stats` shows both.
- Do not use the cap as a whole-model limit with `mmap = false`: in the baseline a 2560 MB cap with `-lm none` failed at load (`failed to allocate buffer for kv cache`).

**The automatic cap** (`memory_cap_mb = "auto"`), computed at startup from the GGUF metadata:

```
cap = round_up_64MiB( KV + recurrent + compute + cache_ram + checkpoints + 640 MiB )

KV          = sum over layers that keep a KV cache of  cells x n_head_kv x (key_length + value_length) x bytes(kv_cache_type)
              cells = ctx_size, or min(ctx_size, sliding_window + 512) for sliding-window layers
              layers without KV: recurrent layers of hybrid models (full_attention_interval),
              and the last shared_kv_layers layers (Gemma 3n reuses earlier KV)
recurrent   = per recurrent layer: (conv_kernel - 1) x (inner_size + 2 x group_count x state_size) x 4 + inner_size x state_size x 4
compute     = 16 x 512 (ubatch) x (max feed_forward_length + embedding_length x altup.num_inputs)
cache_ram   = model.cache_ram_mb
checkpoints = model.ctx_checkpoints x (SWA part of KV + recurrent), only for SWA/hybrid models
640 MiB     = fixed overhead (tokenizer, server, heap); measured 128 to 517 MiB on the three test models
```

bytes(kv type): f16/bf16 2, f32 4, q8_0 34/32, q4_0 18/32, q5_0 22/32, q5_1 24/32.
If the metadata cannot be read, the cap falls back to 1536 MiB. `/stats` shows the cap, its source and every part.

Checked against the real files (ctx 4096, f16 KV, no repack, 21 Nov 2026). The KV and recurrent numbers equal what llama.cpp itself prints.

| Model | KV + recurrent: formula (llama.cpp) | Compute: formula (llama.cpp) | Private after load | Peak private in use | Auto cap | Headroom |
|---|---|---|---|---|---|---|
| Qwen3-4B | 576 (576) MiB | 96 (86) MiB | 794 MB | 841 MB | 1600 MB | 759 MB |
| Qwen3.5-0.8B | 48 + 19.3 (48 + 19.3) MiB | 36 (39) MiB | 624 MB | 849 MB | 1088 MB | 239 MB |
| Gemma 3n E2B | 64 (64) MiB | 128 (119) MiB | 370 MB | 388 MB | 1216 MB | 828 MB |

The fixed 640 MiB is the weak part: what is left after KV, state and buffers was 128 MiB (Qwen3-4B), 180 MiB (Gemma 3n) and 517 MiB (Qwen3.5-0.8B). The 517 MiB was not explained; it does not change with ctx (579 MB at ctx 512), and most of it is committed but not touched (working set 676 MB).

If the child goes over the cap, an allocation fails and llama-server exits. The runtime answers `503 model_load_failed` and says the memory cap was hit.

## Endpoints

All bodies are JSON. Errors look like OpenAI errors:

```json
{ "error": { "message": "not enough free RAM to start the model: ...", "type": "insufficient_memory", "code": "insufficient_memory" } }
```

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request_error` | Bad JSON or missing fields. |
| 404 | `model_not_found` / `chain_not_found` | Unknown model or chain name. |
| 422 | `chain_step_failed` | A chain template could not be filled (for example a JSON field is missing). |
| 502 | `engine_error` | llama-server failed or went away during the request. |
| 503 | `insufficient_memory` | Loading would leave less than `reserve_free_mb` free. Retry later. |
| 503 | `model_load_failed` | llama-server did not start (bad file, memory cap hit, timeout). |
| 503 | `queue_full` | Too many requests waiting. |

Errors from llama-server itself (for example a prompt longer than the context) are passed through with llama-server's status and body.

### `GET /v1/models`

```json
{ "object": "list", "data": [
  { "id": "jitmo-base", "object": "model", "created": 1795000000, "owned_by": "jitmo-runtime" },
  { "id": "chain:classify-answer", "object": "model", "created": 1795000000, "owned_by": "jitmo-runtime" }
] }
```

### `POST /v1/chat/completions`

OpenAI chat completions. The request goes to llama-server almost unchanged, so llama-server's extra fields work too
(`response_format` with a JSON schema, `grammar`, `lora`, `cache_prompt`, `top_k`, ...).
The runtime only adds `chat_template_kwargs` from the config when the request has none.

`model` must be the configured name (`jitmo-base`), or `chain:<name>` to run a chain. If `model` is missing, the base model is used.

```json
{ "model": "jitmo-base",
  "messages": [ { "role": "system", "content": "Be brief." }, { "role": "user", "content": "What is RAM?" } ],
  "max_tokens": 100, "temperature": 0.2 }
```

The response is llama-server's OpenAI response. It also has llama-server's `timings` object:

```json
{ "id": "chatcmpl-...", "object": "chat.completion", "model": "jitmo-base",
  "choices": [ { "index": 0, "message": { "role": "assistant", "content": "RAM is ..." }, "finish_reason": "stop" } ],
  "usage": { "prompt_tokens": 24, "completion_tokens": 31, "total_tokens": 55 },
  "timings": { "prompt_n": 24, "prompt_ms": 410.2, "predicted_n": 31, "predicted_ms": 3020.5, "predicted_per_second": 10.3 } }
```

**Streaming:** with `"stream": true` the answer comes as server-sent events (`data: {...}` lines, then `data: [DONE]`), passed through from llama-server as they are made.

**Constrained output:**

```json
{ "model": "jitmo-base", "messages": [ ... ],
  "response_format": { "type": "json_schema",
    "json_schema": { "name": "pick", "strict": true,
      "schema": { "type": "object", "properties": { "tool": { "type": "string", "enum": ["open_app", "none"] } }, "required": ["tool"] } } } }
```

**A chain through this endpoint:** `"model": "chain:classify-answer"`. The last user message is the chain input.
The answer is the last step's output. The response has an extra `chain` field with every step (same as `/v1/chains/{name}/run`).
With `"stream": true` the whole answer arrives in one chunk (chains are not streamed token by token in v0).

### `GET /v1/chains`

Lists the configured chains and their steps.

```json
{ "chains": [ { "name": "classify-answer", "description": "...",
  "steps": [ { "name": "classify", "json_schema": true, "lora": null }, { "name": "answer", "json_schema": false, "lora": null } ] } ] }
```

### `POST /v1/chains/{name}/run`

Request: `{ "input": "What is the capital of France?" }`

Response: each step's output and timings. `json` is there for steps with a `json_schema`.

```json
{ "chain": "classify-answer",
  "output": "The capital of France is Paris.",
  "steps": [
    { "name": "classify", "output": "{\"kind\":\"question\"}", "json": { "kind": "question" }, "finish_reason": "stop",
      "timing": { "kind": "chain_step", "label": "classify-answer/classify", "queue_ms": 0.0, "load_ms": 2100.0, "first_token_ms": 2600.5,
                  "prompt_tokens": 60, "prompt_ms": 500.5, "prompt_tokens_per_sec": 119.9, "completion_tokens": 6,
                  "generation_ms": 300.1, "tokens_per_sec": 20.0, "total_ms": 2905.0 } },
    { "name": "answer", "output": "The capital of France is Paris.", "finish_reason": "stop", "timing": { ... } } ],
  "queue_ms": 0.0, "load_ms": 2100.0, "total_ms": 4200.0 }
```

A chain holds the one generation slot from its first step to its last. Other requests wait until it is done.

### `GET /health`

Cheap status.

```json
{ "status": "ok", "model": "jitmo-base", "model_state": "loaded",
  "child_working_set_mb": 1013.2, "child_private_mb": 702.4, "system_free_mb": 8123.0,
  "requests_served": 5, "last_tokens_per_sec": 19.8, "pacer_capped": false, "uptime_secs": 120.5 }
```

`model_state` is `unloaded`, `loading` or `loaded`. The child fields are `null` when the model is unloaded.

### `GET /stats`

Everything: the config in use, the child's memory, system memory, queue, counters and the last 20 timings.

```json
{ "uptime_secs": 120.5, "model_state": "loaded",
  "model": { "name": "jitmo-base", "path": "...", "ctx_size": 4096, "threads": 3, "kv_cache_type": "f16", "flash_attn": "on", "mmap": true, "no_repack": true,
             "cache_ram_mb": 256, "ctx_checkpoints": 4, "adapters": [] },
  "child": { "pid": 1234, "port": 51234, "loaded_secs": 40.2, "load_ms": 2100.0, "working_set_mb": 1013.2, "private_mb": 702.4,
             "peak_working_set_mb": 1100.0, "peak_private_mb": 720.0, "job_peak_commit_mb": 720.0 },
  "system": { "total_mb": 16000.0, "free_mb": 8123.0, "memory_load_pct": 49, "low_memory_signal": false },
  "budget": { "memory_cap_mb": 1600, "memory_cap_source": "auto",
              "memory_estimate_mb": { "arch": "qwen3", "kv": 576.0, "recurrent": 0.0, "compute": 96.0, "cache_ram": 256.0, "checkpoints": 0.0,
                                      "overhead": 640.0, "total": 1600.0, "kv_layers": 36, "swa_layers": 0 },
              "reserve_free_mb": 1536, "load_estimate_mb": 3982, "pressure_free_mb": 1024, "critical_free_mb": 500, "unload_pending": false,
              "idle_unload_secs": 90, "idle_secs": 3.2 },
  "pacer": { "mode": "auto", "state": "busy", "capped": true, "reason": "user input 120 ms ago", "cap_pct_of_system": 18.75,
             "busy_cpu_pct": 50, "busy_priority": "unchanged", "input_idle_ms": 420, "others_cpu_pct": 6.9, "free_mb": 4070,
             "capped_secs_total": 35.2, "cap_on_count": 3, "last_error": null, "generations_capped": 2, "generations_uncapped": 1,
             "last_tokens_per_sec_capped": 4.1, "last_tokens_per_sec_uncapped": 7.6 },
  "queue": { "generating": false, "waiting": 0, "max_queue": 8 },
  "counters": { "requests_served": 5, "requests_failed": 0, "refused_insufficient_memory": 0, "refused_queue_full": 0,
                "loads": 1, "load_failures": 0, "last_load_ms": 2100.0, "unloads": { "idle": 1, "memory_pressure": 0, "memory_critical": 0 },
                "peak_child_working_set_mb": 1100.0, "peak_child_private_mb": 720.0, "peak_job_commit_mb": 720.0,
                "last_tokens_per_sec": 19.8, "last": { ... }, "recent": [ ... ] } }
```

New in v0.1:

| Field | Meaning |
|---|---|
| `budget.memory_cap_source` | `auto` (computed), `config` (a number in the config), `none` (0) or `fallback` (auto, but the GGUF could not be read: 1536 MiB). |
| `budget.memory_estimate_mb` | The parts of the automatic cap (see the formula above). Shown even when the cap is set by hand. |
| `budget.unload_pending` | Memory pressure was seen during a generation; the model is unloaded when it ends. |
| `pacer.state` | `no_model`, `idle` (no cap) or `busy` (capped). |
| `pacer.reason` | The busy signal that turned the cap on. |
| `pacer.input_idle_ms`, `others_cpu_pct`, `free_mb` | The last sample of the three signals. |
| `pacer.capped_secs_total`, `cap_on_count` | Time with the cap on, and how often it was switched on. |
| `pacer.generations_capped/uncapped`, `last_tokens_per_sec_capped/uncapped` | Generations by whether the cap was on when they ended. |
| `counters.unloads.memory_critical` | Unloads below `critical_free_mb`. |

Timing fields (per generation):

| Field | Meaning |
|---|---|
| `queue_ms` | Waiting for the generation slot. |
| `load_ms` | Starting the model for this request (0 if it was loaded). |
| `first_token_ms` | Request arrival to first token. Measured for streams. For non-stream requests: queue + load + prompt time. |
| `prompt_tokens`, `prompt_ms`, `prompt_tokens_per_sec` | Reading the prompt (llama-server's numbers). |
| `completion_tokens`, `generation_ms`, `tokens_per_sec` | Writing the answer (llama-server's numbers). |
| `total_ms` | Whole request inside the runtime. |

### `POST /admin/unload`

Stops the model now and frees its RAM. By default it waits for the running generation (and the queue ahead of it) to finish.
`POST /admin/unload?force=true` kills it at once; a running request then fails with `engine_error`.

```json
{ "unloaded": true, "model_state": "unloaded" }
```

The next request loads the model again.

## Pacer

"Don't freeze the PC", v1. Code: `runtime/src/pacer.rs`. Measured in [results/03-pacer-v1.md](results/03-pacer-v1.md).

Every `poll_ms` (300 ms) the pacer reads three signals:

| Signal | Source | Busy when |
|---|---|---|
| User input | `GetLastInputInfo` (keyboard and mouse, this session) | input in the last `input_active_ms` (5 s) |
| Other processes' CPU | `GetSystemTimes` minus the child's Job Object CPU time | above `others_cpu_busy_pct` (10%) of all logical processors |
| Free RAM | `GlobalMemoryStatusEx` | below `low_free_mb` (2048 MB) |

When any signal says busy, the pacer puts a **Job Object CPU rate hard cap** on the llama-server child: `busy_cpu_pct` (50%) of the CPU the child uses at full speed.
With 3 threads on 8 logical processors that is 18.75% of the whole CPU. The cap is set and removed on the running child, with no restart.
When no signal has said busy for `release_after_ms` (2 s), the cap is removed. Optionally (`busy_priority = "idle"`) the child also drops to idle priority while busy.

Why this lever: llama-server has no per-request thread count (a restart costs a model load), and it keeps generating at full speed even if the runtime reads its stream slowly. A CPU cap is the one lever that works on the running process.

The cost: a capped model is slower. A request that runs while the user types takes longer; `/stats` → `pacer` shows the tokens/s of capped and uncapped generations.

## Chains

A chain is a list of steps in the config. All steps use the same base model.

```toml
[[chains]]
name = "classify-answer"
description = "Decide what kind of message it is, then answer in a fitting style."

[[chains.steps]]
name = "classify"
system = "Classify the user's message as exactly one kind: question, task or chat."
input = "{input}"
max_tokens = 16
temperature = 0.0
json_schema = { type = "object", properties = { kind = { type = "string", enum = ["question", "task", "chat"] } }, required = ["kind"] }

[[chains.steps]]
name = "answer"
system = "The user's message is a {classify.kind}. Answer it."
input = "{input}"
max_tokens = 200
temperature = 0.3
# lora = { path = 'adapters/answer.gguf', scale = 1.0 }
```

Step keys:

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Letters, digits, `_`, `-`. Not `input` or `prev`. Unique in the chain. |
| `system` | `""` | System prompt template. Empty = no system message. |
| `input` | `{input}` for step 1, `{prev}` after | User message template. |
| `max_tokens` | `256` | |
| `temperature` | `0.2` | |
| `json_schema` | none | Output must match this JSON schema. A TOML table, or a string holding JSON. |
| `grammar` | none | GBNF grammar for the output. Not together with `json_schema`. |
| `lora` | none | `{ path = "...", scale = 1.0 }`. Optional LoRA adapter for this step. |

Template placeholders:

| Placeholder | Becomes |
|---|---|
| `{input}` | The chain input. |
| `{prev}` | The previous step's output. |
| `{stepname}` | That earlier step's output. |
| `{stepname.field}` | A field of that step's JSON output. Nested: `{stepname.a.b}`; arrays: `{stepname.items.0}`. |

Only `{name}` and `{name.path}` are placeholders. Other braces, like a JSON example in a prompt, stay as they are.
A placeholder that names an unknown or later step is a config error at startup.

**LoRA adapters.** Every adapter file used by any step is loaded once when the model starts (`--lora ... --lora-init-without-apply`, scale 0).
A step with `lora` sends `"lora": [{"id": N, "scale": S}]` with its request, so only that step uses it. Switching steps never reloads the model.
No adapters exist yet, so adapter steps are **untested**.

## How to use it from jitmo agent

- Point an OpenAI client at `http://127.0.0.1:8090/v1` with model `jitmo-base`. No API key is checked.
- Expect the first request after start or after an idle unload to take longer (the model loads; see `load_ms`).
- Handle `503 insufficient_memory`: the PC is short of RAM. Tell the user, or retry later.
- Handle `503 queue_full` and `502 engine_error` as "try again". `502` also happens when free RAM falls below `critical_free_mb` mid-answer.
- Expect generation to be slower while the user is typing or another app is busy (the pacer caps the model). `GET /health` → `pacer_capped` says whether it is capped now.
- Use `GET /health` before showing "model ready" in a UI. It never loads the model.
- Do not send many requests at once. They run one by one anyway.

## Limits of v0.1

- One base model per runtime. No model switching.
- One generation at a time. No batching.
- Chain output is not streamed token by token.
- No authentication. It listens on 127.0.0.1 only by default.
- The pacer has one lever: a CPU cap on the whole child. It slows the model while capped (see [Pacer](#pacer)). It cannot change threads per request (llama-server has no such setting) and does not pause between tokens.
- The pacer's "busy" signals are coarse: any keyboard/mouse input, any other process above 10% of the CPU, or free RAM below 2 GB. It does not know which app is in front or whether it is memory-heavy.
- The memory cap does not cover memory-mapped weights (see above), and the automatic cap does not cover a repacked copy (`no_repack = false`).
- The free-RAM check before a load counts the whole model file plus the cap. That is too strict for Gemma 3n, whose Per-Layer Embeddings stay on disk: it estimates 4.1 GB while the model uses 1.6 GB.
- Memory pressure waits for the running generation (or the whole chain). A long generation can keep the model loaded while RAM is short, until free RAM falls below `critical_free_mb`.
- Windows only (Job Objects, Win32 memory, CPU and input APIs).

## Measured in v0.1

- Gemma 3n E2B vs Qwen3-4B through the runtime (load, first token, tok/s, memory, PLE): [results/02-gemma3n.md](results/02-gemma3n.md).
- Pacer: probe slowdown and model tok/s with and without the cap: [results/03-pacer-v1.md](results/03-pacer-v1.md). In short: a 50% cap cut the memory-heavy app's slowdown from +29% to +2% and the model's speed from 5.3 to 3.2 tok/s while capped.
- Smoke test (`runtime\scripts\smoke.ps1`, Qwen3.5-0.8B) also checks: automatic cap above the measured peak, threads = min(cores − 1, logical/2), pressure during a generation lets it finish and then unloads, critical memory aborts it (`502`) and unloads, pacer `always` caps the child, `off` does not.
- A real critical-memory event happened once: Qwen3-4B with repacking drove free RAM to 311 MB during load and the runtime unloaded it at once.

## Measured on the owner's laptop (v0)

Default config then: cap 1536 MB fixed, `pressure_free_mb` 512, no pacer, llama-server's own `--cache-ram` (8192) and checkpoints (32).


Single runs on 21 Nov 2026. i5-1135G7, 16 GB, Windows 11, llama.cpp CPU build 12248. Default config: threads 3, ctx 4096, f16 KV, mmap, no repack, cap 1536 MB.
Other apps were open (about 4 to 6 GB free). Load times are with the model file already in the Windows file cache (warm); a cold load from the SSD is slower and was not measured.

| What | Qwen3.5-0.8B (smoke test) | Qwen3-4B |
|---|---|---|
| Load (start to ready, warm file cache) | 1.3 to 1.6 s | 1.9 s |
| First token, short prompt (model loaded) | 168 ms (stream, 20-token prompt) | 1.2 to 1.4 s (non-stream, 62 to 72-token prompt) |
| Prompt reading | 70 to 210 tok/s | 52 to 54 tok/s |
| Generation | 25 to 53 tok/s | 8.2 to 9.0 tok/s (short prompt), 6.9 tok/s after a 721-token prompt |
| Child peak private bytes (= what the cap counts) | 849 MB | 831 MB |
| Child peak working set (real RAM) | 908 MB | 3134 MB |
| Example chain `classify-answer` (2 steps), total | 1.3 s | 12.9 s (incl. 1.9 s load) |

KV cache type on Qwen3-4B, same requests:

| KV | Peak private | Peak working set | 721-token prompt | Generation after it |
|---|---|---|---|---|
| f16 (default) | 831 MB | 3134 MB | 54.2 tok/s | 6.85 tok/s |
| q8_0 | 549 MB | 2853 MB | 35.1 tok/s | 6.55 tok/s |

Checked to work (scripted, 0.8B model): lazy start, `/v1/models`, chat (normal and stream), unknown model (404), chain run, chain via `chain:<name>`, `/stats`, `/health`,
`/admin/unload` ends the child, reload after unload, idle unload (3 s setting), refusal when the RAM reserve cannot be met (503 `insufficient_memory`),
load failure under a too-small cap (503 `model_load_failed` with the reason), unload on memory pressure, queue (`max_queue = 1`: one runs, one waits, one gets 503),
Ctrl+C (exit code 0, child gone), and the child dying when the runtime is killed hard.

Not tested: LoRA adapters (none exist yet), a cold load from SSD, a real low-memory event from Windows (pressure was simulated with a high `pressure_free_mb`), and an 8 GB laptop.
