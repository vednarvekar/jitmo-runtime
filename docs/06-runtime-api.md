# 06. Runtime v0.3: API for jitmo agent

This doc is for the jitmo agent team. It says what the runtime does, how to start it (also as a child process of your app), how to configure it, and how to call it.
Code: `runtime/` (Rust). Config: `runtime/jitmo-runtime.toml`. Smoke test: `runtime/scripts/smoke.ps1`. Measurements: [results/04-v0.2.md](results/04-v0.2.md), [results/05-pacer-v2.md](results/05-pacer-v2.md).

## What it does

- It runs **one or more models**, each in its own llama.cpp `llama-server` child process on a private localhost port, each in its own Windows Job Object (memory cap, and killed if the runtime dies).
- A model **loads on the first request** that needs it, or when you call `POST /admin/load` (warm-up). It **unloads** after 90 s without requests.
- **Several models at once (pool):** loaded models must fit one RAM budget. To load another model, the runtime evicts the **least recently used** model that is idle and not pinned. It **never evicts a model that is generating**.
- It checks **free RAM before every load** and refuses with a clear error if too little would stay free.
- It **unloads models when the PC is short of memory**: after the running answer is finished, or at once if free RAM is critically low.
- It runs **one generation at a time across all models** by default (they share the memory bandwidth). Other requests wait in a queue. **Interactive requests go before background ones.**
- Every request gets a **request id**, and **you can cancel it** (`DELETE /v1/requests/{id}`). Cancelling, or closing the connection, makes llama-server stop computing.
- Its **pacer** caps the models' CPU while other programs need it, at a level that fits how busy the PC is.
- It speaks the **OpenAI API**: `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`.
- It runs **chains**: ordered steps, each with its own prompt template, settings, model and optional LoRA adapter. Templates take the input, **any number of request variables** and earlier steps' outputs.

## Start it

```powershell
cd V:\Code\jitmo-runtime\runtime
cargo build --release
.\target\release\jitmo-runtime.exe --config jitmo-runtime.toml
```

| Option | Meaning |
|---|---|
| `--config <file>` | Config file. Default: `./jitmo-runtime.toml` if it exists, else built-in defaults (one model, no chains). |
| `--model <file.gguf>` | Override the path of the first model. |
| `--llama-server <exe>` | Override `llama_server` of every model. |
| `--port <n>` | Override `server.port`. `0` = any free port. |
| `--parent-pid <pid>` | Exit cleanly (stopping all models) when that process exits. |
| `--port-file <file>` | Also write the `JITMO_RUNTIME_LISTENING` line to this file. It is removed at a clean exit. |
| `--log-file <file>` | Write the log to this file instead of stderr. |
| `--log-level <level>` | `error`, `warn`, `info` (default), `debug`, `trace`. |

**stdout** carries exactly one line, once the API is listening: `JITMO_RUNTIME_LISTENING port=<n>`. **The log goes to stderr** (or `--log-file`). Each model's llama-server log goes to its `log_file`.
**Ctrl+C** stops the models and exits with code 0. If the runtime is killed hard, Windows kills the children too (Job Object kill-on-close).

### As a child process of your app

```text
jitmo-runtime.exe --config <your.toml> --port 0 --parent-pid <your pid> --port-file <dir>\runtime.port --log-file <dir>\runtime.log
```

1. Start it. Read stdout until the line `JITMO_RUNTIME_LISTENING port=<n>` (or poll `--port-file` until it exists and is not empty; it is written by a rename, so it is never half-written).
2. Use `http://127.0.0.1:<n>`. `GET /health` answers at once; it never loads a model.
3. When your app exits (also if it crashes), the runtime notices within milliseconds, stops its models and exits with code 0 (measured: 73 ms, children gone, port file removed).
4. With `--log-file`, nothing but the one line is written to stdout or stderr, so a parent that does not read the pipes can never block the runtime.

If `--parent-pid` names no running process, the runtime exits at once with an error.

## Config reference

One TOML file. Every key is optional. Relative paths are relative to the config file.

### `[server]`

| Key | Default | Meaning |
|---|---|---|
| `host` | `"127.0.0.1"` | Address of the API. Keep it on localhost. |
| `port` | `8090` | Port of the API. `0` = any free port (see above). |
| `max_queue` | `8` | Requests that may wait for a generation slot. More get `503 queue_full`. |
| `max_concurrent` | `1` | Generations that may run at the same time, across all models. Keep 1 on a laptop: the models share the RAM bandwidth. More than 1 is meant for different models; two requests to the same model still run one after the other inside its llama-server. |
| `allowed_origins` | `[]` | 28 Nov. Web page origins (exact `Origin` values, for example `"http://localhost:5173"`) allowed to call the API. See [Local-only guard](#local-only-guard). |
| `preempt_background` | `true` | 28 Nov. When an interactive request has to wait and no interactive request is running, the newest running background request is stopped with `503 preempted`. `false` = v0.3 behaviour (a running request is never interrupted). |

### Models: `[model]` or `[[models]]`

One model: a `[model]` table (the v0.1 form, still works). Several models: `[[models]]` entries. Not both.
The **first chat model is the default**: requests without `model` and chain steps without `model` use it.

```toml
[budget]
pool_budget_mb = 4500

[[models]]
name = "router"
path = 'V:\jitmo-sandbox\models\gemma-3-270m-it-router-Q4_K_M.gguf'
ctx_size = 2048
pinned = true

[[models]]
name = "writer"
path = 'V:\jitmo-sandbox\models\gemma-3n-E2B-it-Q4_K_M.gguf'

[[models]]
name = "minilm"
kind = "embedding"
path = 'V:\jitmo-sandbox\models\all-MiniLM-L6-v2-Q8_0.gguf'
ctx_size = 512
```

| Key | Default | Meaning |
|---|---|---|
| `name` | `"jitmo-base"` | The model id clients send. Unique; must not start with `chain:`. |
| `kind` | `"chat"` | `chat`, or `embedding` (llama-server runs with `--embeddings`; use `POST /v1/embeddings`). |
| `path` | `V:\jitmo-sandbox\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | The GGUF file. |
| `llama_server` | `V:\jitmo-sandbox\llama\llama-server.exe` | The llama.cpp server. |
| `engine` | `"llama-server"` | 28 Nov. `"inproc"` = our own engine worker (`jitmo-runtime.exe --engine-worker`, llama.dll inside) instead of llama-server. Same API, same Job Object (memory cap, pacer, kill on close). Chat models only; no LoRA, no `device`, no `chat_template_kwargs` (built-in chat templates only, no Jinja). JSON schema: objects, strings, enums, numbers, booleans, arrays, unions (no `$ref`/`pattern`/`format`). Needs llama.cpp **b12606** DLLs. See [08-inproc-engine.md](08-inproc-engine.md). |
| `llama_dir` | unset | 28 Nov. `inproc`: the llama.cpp build folder with `llama.dll` (CPU or Vulkan). Unset = the folder of `llama_server`. |
| `pinned` | `false` | `true` = never evicted to make room for another model, and never unloaded for idleness. Memory pressure still unloads it. |
| `port` | `0` | Private port of the child. `0` = any free port. |
| `ctx_size` | `4096` | Context length in tokens. For embedding models also the batch size (one input is read in one batch), so set it to the model's own context (512 for MiniLM). |
| `threads` | `0` | `0` = physical cores − 1, but at most half of the logical processors, at least 1. The i5-1135G7 (4 cores, 8 threads) gets min(3, 4) = 3. |
| `kv_cache_type` | `"f16"` | KV cache type for K and V. `q8_0` saves about 280 MB at ctx 4096 on Qwen3-4B, but read a long prompt 35% slower. |
| `flash_attn` | `"on"` | `on`, `off`, `auto`. A `q8_0` V cache needs `on`. |
| `mmap` | `true` | `false` passes `-lm none`: weights are copied into private memory and count against the cap. |
| `no_repack` | `true` | `true` passes `--no-repack`: no second, private copy of the weights. About half the RAM, about 20% slower. |
| `low_priority` | `true` | Child runs at below-normal CPU priority. |
| `cache_ram_mb` | `256` | llama-server's prompt cache in RAM (`--cache-ram`). Repeated prefixes (system prompts, tool lists) are not re-read. Counted in the automatic cap. `0` = off. |
| `ctx_checkpoints` | `4` | Context checkpoints per slot (`--ctx-checkpoints`) for sliding-window and hybrid models (Gemma 3n, Qwen3.5). Counted in the automatic cap. |
| `pooling` | none | Embedding models only: `none`, `mean`, `cls`, `last`, `rank`. Unset = the model's own (MiniLM's GGUF says mean). |
| `adapters` | `[]` | LoRA adapter files loaded with the model at scale 0. Requests turn them on by id (see [LoRA](#lora-adapters)). |
| `chat_template_kwargs` | none | Added to every request that does not set its own. Use `{ enable_thinking = false }` for Qwen3 models; nothing for Gemma. |
| `gpu_layers` | unset | 28 Nov. GPU offload (`-ngl`): `"0"` = CPU only, a layer count, `"auto"` or `"all"`. Needs a GPU build of llama-server in `llama_server` (for example Vulkan for Iris Xe). Unset = llama-server's default (a GPU build may then offload everything). Not measured yet: speed, and whether the load estimate and memory cap fit an integrated GPU (its memory is shared system RAM). |
| `device` | unset | 28 Nov. Offload device (`--device`, for example `"Vulkan0"`). |
| `extra_args` | `[]` | Extra llama-server arguments, passed as-is. |
| `log_file` | see meaning | llama-server's log, rewritten at each start. Default `llama-server.log` with one model, `llama-server-<name>.log` with several. `""` = discard. |
| `load_timeout_secs` | `180` | Give up if the model is not ready after this long. |
| `memory_cap_mb`, `load_estimate_mb`, `idle_unload_secs` | from `[budget]` | Per-model overrides of the `[budget]` keys of the same name. |

### `[budget]`

| Key | Default | Meaning |
|---|---|---|
| `memory_cap_mb` | `"auto"` | Job Object commit (private bytes) cap on each child. `"auto"` = computed from the model file and ctx (below). A number = that many MiB. `0` = no cap. It does not count the mmap'd weights. |
| `reserve_free_mb` | `1536` | Refuse to load a model if free RAM after loading it would be below this. |
| `load_estimate_mb` | `0` | RAM a loaded model is expected to take. `0` = estimate it ([below](#load-estimate)). |
| `pool_budget_mb` | `0` | The load estimates of all loaded models together may not exceed this. `0` = no pool limit; only the free-RAM check applies. |
| `pressure_free_mb` | `1024` | Memory pressure: free RAM below this on two checks in a row, or Windows' low-memory signal. Idle models are unloaded at once; a model that is generating **finishes first** (for at most `pressure_finish_max_ms`), then it is unloaded. |
| `pressure_finish_max_ms` | `15000` | v0.3. How long a generation may run on under memory pressure. If free RAM is still under `pressure_free_mb` after this, the runtime cancels the requests on that model and unloads it. The client gets `503 memory_pressure` (for a stream: an error event with that code, then `[DONE]`). If RAM recovers in the meantime, the generation may finish. `0` = no limit (v0.2: wait for the end). |
| `critical_free_mb` | `500` | Free RAM below this on **one** check: unload every model at once, even mid-generation (those requests fail with `engine_error`). `0` = never. Must be below `pressure_free_mb`. |
| `idle_unload_secs` | `90` | Unload a model after this many seconds without requests. `0` = never. Pinned models are never idle-unloaded. |
| `watch_interval_ms` | `1000` | How often the watchdog checks idle time, memory and the children. |

#### How a load is decided

Loads happen one at a time. Before loading model M:

1. **Pool budget** (if set): (load estimates of the loaded models) + (M's estimate) ≤ `pool_budget_mb`.
2. **Free RAM:** free RAM ≥ M's estimate + `reserve_free_mb`.
3. If either fails, the runtime evicts the least recently used model that is loaded, **not pinned and not in use** (no request is loading it or generating on it), and checks again.
4. If nothing can be evicted: `503 insufficient_memory`. The message names the models that are in the way and why (`pinned`, `in use`).

#### Load estimate

What the free-RAM check and the pool budget count for one model (shown in `/stats` → `models[].load_estimate_mb`).
v0.3 made it closer to the real use. It is no longer the memory cap.

```
estimate = private + mapped (+ mapped again if no_repack = false)

private  = KV cache + recurrent state + compute buffers + prompt cache (cache_ram_mb)
           + checkpoints + overhead, rounded up to 16 MiB
overhead = 96 MiB + 5% of the weights (tensor bytes in the file)
mapped   = the part of the memory-mapped file that will be in RAM:
           the whole file, minus tensors llama.cpp leaves on disk.
           Today that is Gemma 3n's Per-Layer Embeddings (per_layer_token_embd, 1440 MiB of its 2887 MiB):
           llama.cpp reads only the rows of the tokens it sees.
```

- The prompt cache (`cache_ram_mb`, 256 MB) is counted in full. It fills with use: Qwen3.5-0.8B had 656 MB private after one answer and 849 MB at its peak after the smoke test.
- Qwen3.5 (`qwen35`): its weights end up in private memory even with mmap and `no_repack`. Measured: working set 714 MB, private 656 MB. So for this architecture the weights count as private, and mapped is 0. The total is the same.
- A `memory_cap_mb` number below the private estimate replaces it (the child cannot use more).
- Without GGUF metadata: the cap from the config, the fallback cap (1536 MiB), or 1024 MiB if there is no cap.
- `load_estimate_mb` (per model or in `[budget]`) still replaces the whole estimate.

Estimate vs real use, default config (ctx 4096, f16 KV, no repack). "Real" is the child after a 128-token answer ([results/06-low-ram.md](results/06-low-ram.md)). "Peak" is the 0.8B after the whole smoke test (22 Nov 2026), and Qwen3-4B's peak in [results/02-gemma3n.md](results/02-gemma3n.md). MB here means MiB, as in `/stats`.

| Model | File | v0.2 estimate | v0.3 estimate (private + mapped) | Real working set (private) | v0.3 ÷ real |
|---|---|---|---|---|---|
| Gemma 3 270M router | 241 MB | 1265 MB | **705 MB** (464 + 241) | 334 MB (146) | 2.1× |
| Qwen3.5-0.8B | 508 MB | 1596 MB | **1068 MB** (1068 + 0) | 714 MB (656); peak 909 MB (849) | 1.5×; peak 1.18× |
| Gemma 3n E2B | 2887 MB | 2663 MB | **2279 MB** (832 + 1447) | 1579 MB (371) | 1.44× |
| Qwen3-4B | 2382 MB | 3981 MB | **3534 MB** (1152 + 2382) | peak 3077 MB (841) | 1.15× |

- The estimate is above the real working set and the real private bytes for all four. A unit test checks this against the real files.
- Without the prompt cache the estimate would be 256 MB lower, but then it is below the 0.8B's peak. Set `cache_ram_mb` lower (or 0) to lower both the estimate and the real use.
- With the default `reserve_free_mb = 1536`, Gemma 3n now needs about 3.7 GB free (v0.2: 4.1 GB). Qwen3-4B needs about 5.0 GB (v0.2: 5.4 GB).

### `[pacer]`

See [Pacer](#pacer) for how it works.

| Key | Default | Meaning |
|---|---|---|
| `mode` | `"auto"` | `auto` = level from the signals; `always` = heavy level all the time; `off` = never cap (the signals are still shown in `/stats`). |
| `poll_ms` | `300` | How often the pacer samples. |
| `smooth_ms` | `1500` | Other programs' CPU is averaged over this long. |
| `light_cpu_pct` | `20.0` | Other programs above this share of all logical processors: light level. `0` = never. v0.2 default: 15 (an idle PC at 12 to 13% then tended to stay at light). |
| `heavy_cpu_pct` | `40.0` | Above this: heavy level. `0` = never. |
| `low_free_mb` | `0` | Free RAM below this: heavy level. `0` = `budget.pressure_free_mb`. |
| `light_cap_pct` | `75` | Light level: each model may use this share of the CPU it uses at full speed. |
| `heavy_cap_pct` | `50` | Heavy level. Must not be above `light_cap_pct`. |
| `hysteresis_pct` | `5.0` | To go down a level, other programs' CPU must be this many points below the threshold (RAM: 256 MB above `low_free_mb`). |
| `release_after_ms` | `5000` | A level holds at least this long before it may go down. Going up is immediate. (With 2000 the level flapped up to 10 times in 30 s under a light load; see results/05.) |
| `heavy_hysteresis_pct` | `10.0` | v0.3. To leave heavy, other programs' CPU must be this many points below `heavy_cpu_pct` (so under 30%). |
| `heavy_release_after_ms` | `10000` | v0.3. Heavy goes down only after CPU (under `heavy_cpu_pct - heavy_hysteresis_pct`) and RAM (256 MB above `low_free_mb`) have stayed clear for this long **in a row**. Any sample above that line restarts the wait. |
| `busy_priority` | `"unchanged"` | `idle` = also drop the children to idle priority while capped. |
| `inproc_lever` | `"pause"` | 28 Nov. How `engine = "inproc"` models are slowed. `pause`: no CPU cap; the worker pauses after each token so it computes `light_cap_pct` / `heavy_cap_pct` of the time, and uses one thread less at heavy (`GET /pace` on the worker shows it). `cap`: the Job Object CPU cap, as for llama-server. |
| `exclude_client_cpu` | `true` | 28 Nov. With `--parent-pid`, that process and its other descendants (for example jitmo's model-worker) are left out of "other programs' CPU", so the assistant's own router and tagger work does not cap its model. `/stats` → `pacer.client_pid`, `pacer.client_processes`. |

The v0.1 keys `input_active_ms`, `others_cpu_busy_pct` and `busy_cpu_pct` are accepted and ignored (a warning is logged).

### What the memory cap limits

The cap is a Windows Job Object `JOB_OBJECT_LIMIT_PROCESS_MEMORY` on each child. It limits the child's **commit charge (private bytes)** only.

- It does **not** count the model weights when they are memory-mapped (`mmap = true`, the default). Those pages are file-backed; Windows can drop them and read them back.
- It **does** count the KV cache, recurrent state, compute buffers, the prompt cache, context checkpoints and any private copy of the weights.
- With `no_repack = false`, llama.cpp copies the weights into private memory: **the automatic cap is then too small**. Set `memory_cap_mb` by hand or to `0`.
- The real RAM use is the **working set** (the touched part of the file + private bytes). `/stats` shows both.

**The automatic cap** (`memory_cap_mb = "auto"`), computed at startup from the GGUF metadata:

```
cap = round_up_64MiB( KV + recurrent + compute + cache_ram + checkpoints + 640 MiB )

KV          = sum over layers that keep a KV cache of  cells x n_head_kv x (key_length + value_length) x bytes(kv_cache_type)
              cells = ctx_size, or min(ctx_size, sliding_window + 512) for sliding-window layers
              layers without KV: recurrent layers of hybrid models (full_attention_interval),
              and the last shared_kv_layers layers (Gemma 3n reuses earlier KV)
recurrent   = per recurrent layer: (conv_kernel - 1) x (inner_size + 2 x group_count x state_size) x 4 + inner_size x state_size x 4
compute     = 16 x ubatch x (max feed_forward_length + embedding_length x altup.num_inputs)   (ubatch = 512; ctx_size for embedding models)
cache_ram   = cache_ram_mb
checkpoints = ctx_checkpoints x (SWA part of KV + recurrent), only for SWA/hybrid models
640 MiB     = fixed overhead (tokenizer, server, heap); measured 128 to 517 MiB on the three v0.1 test models.
              The cap keeps this large overhead and the prompt cache: it is a hard limit. The load estimate (above) does not.
```

bytes(kv type): f16/bf16 2, f32 4, q8_0 34/32, q4_0 18/32, q5_0 22/32, q5_1 24/32. If the metadata cannot be read, the cap falls back to 1536 MiB.
The KV and recurrent numbers equal what llama.cpp prints for Qwen3-4B, Qwen3.5-0.8B and Gemma 3n E2B (unit tests check this against the real files).

If a child goes over its cap, an allocation fails and llama-server exits. The runtime answers `503 model_load_failed` and says the memory cap was hit.

## Headers

| Header | Direction | Meaning |
|---|---|---|
| `X-Jitmo-Request-Id` | request (optional) | Your own id for this request: 1 to 128 characters, letters, digits, `.`, `_`, `:`, `-`. Must not be in use by another running request (`409 request_id_in_use`). **Send your own id if you may need to cancel a non-streamed request**: its response headers only arrive when it is done. |
| `X-Jitmo-Request-Id` | response | The request's id (yours, or a new one like `req-6b0284a9-7`). On every response from chat, chains and embeddings, also on errors. Chain reports also have it as `request_id`. |
| `X-Jitmo-Priority` | request (optional) | `interactive` (default) or `background`. Waiting interactive requests run before waiting background ones; within one priority, first come first served. With `server.preempt_background` (default on), an interactive request that has to wait stops a running background request, which gets `503 preempted`. Anything else: `400`. |

## Endpoints

All bodies are JSON, sent with `Content-Type: application/json`. Errors look like OpenAI errors:

```json
{ "error": { "message": "not enough free RAM to start model 'writer': ...", "type": "insufficient_memory", "code": "insufficient_memory" } }
```

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request_error` | Bad JSON, missing fields, a missing chain variable, a bad header, a `lora` id that does not exist. |
| 400 | `wrong_model_kind` | Chat on an embedding model, or embeddings on a chat model. |
| 404 | `model_not_found` / `chain_not_found` / `request_not_found` | Unknown model, chain, or request id (it may have finished). |
| 409 | `request_id_in_use` | Your `X-Jitmo-Request-Id` belongs to another running request. |
| 422 | `chain_step_failed` | A chain template could not be filled (for example a JSON field is missing). |
| 403 | `forbidden` | Refused by the [local-only guard](#local-only-guard): an `Origin` header not in `allowed_origins`, a `Host` that is not this machine, or a POST body without `Content-Type: application/json`. |
| 499 | `request_cancelled` | The request was cancelled with `DELETE /v1/requests/{id}`. |
| 502 | `engine_error` | llama-server failed or went away during the request. |
| 503 | `insufficient_memory` | Loading would break the pool budget or leave less than `reserve_free_mb` free, and nothing could be evicted. Retry later. |
| 503 | `model_load_failed` | llama-server did not start (bad file, memory cap hit, timeout). |
| 503 | `queue_full` | Too many requests waiting. |
| 503 | `preempted` | 28 Nov. Your background request was stopped so an interactive one could run. Retry it later. A stream gets an error event with this code, then `[DONE]`. |

### Local-only guard

The API has no login, so web pages the user visits must not reach it (28 Nov):

- A request with an `Origin` header (browsers add it to every cross-site POST and DELETE) is refused unless `server.allowed_origins` lists it. This stops a page's "simple" `text/plain` POST, which a browser sends without asking.
- The `Host` must be `127.0.0.1`, `localhost`, `[::1]` or `server.host`. This stops DNS rebinding (a page on a name that resolves to 127.0.0.1).
- A POST with a body must say `Content-Type: application/json`.

Normal clients (jitmo, curl, scripts) pass without changes.

Errors from llama-server itself (for example a prompt longer than the context) are passed through with llama-server's status and body.

### `GET /v1/models`

```json
{ "object": "list", "data": [
  { "id": "router", "object": "model", "created": 1795000000, "owned_by": "jitmo-runtime", "kind": "chat" },
  { "id": "minilm", "object": "model", "created": 1795000000, "owned_by": "jitmo-runtime", "kind": "embedding" },
  { "id": "chain:qa", "object": "model", "created": 1795000000, "owned_by": "jitmo-runtime", "kind": "chain" }
] }
```

### `POST /v1/chat/completions`

OpenAI chat completions. The request goes to the model's llama-server almost unchanged, so llama-server's extra fields work too
(`response_format` with a JSON schema, `grammar`, `cache_prompt`, `top_k`, `ignore_eos`, ...).
The runtime only adds `chat_template_kwargs` from the model's config when the request has none, and completes the `lora` list (see [LoRA](#lora-adapters)).

`model` is a model name, or `chain:<name>` to run a chain. Without `model`, the default (first chat) model is used.

```json
{ "model": "router",
  "messages": [ { "role": "system", "content": "Pick a tool." }, { "role": "user", "content": "open calculator" } ],
  "max_tokens": 1, "grammar": "root ::= [0-9]", "cache_prompt": true }
```

The response is llama-server's OpenAI response, including its `timings` object:

```json
{ "id": "chatcmpl-...", "object": "chat.completion", "model": "router",
  "choices": [ { "index": 0, "message": { "role": "assistant", "content": "3" }, "finish_reason": "length" } ],
  "usage": { "prompt_tokens": 24, "completion_tokens": 1, "total_tokens": 25 },
  "timings": { "prompt_n": 24, "prompt_ms": 41.2, "predicted_n": 1, "predicted_ms": 9.5, "predicted_per_second": 105.3 } }
```

**Streaming:** with `"stream": true` the answer comes as server-sent events (`data: {...}` lines, then `data: [DONE]`), passed through as they are made.
If the request is cancelled mid-stream, the stream ends with one error event and `[DONE]`:

```text
data: {"error":{"code":"request_cancelled","message":"request st-1 was cancelled","type":"request_cancelled"}}

data: [DONE]
```

**A chain through this endpoint:** `"model": "chain:qa"`. The last user message is the chain input; an optional top-level `"vars": {...}` gives the variables.
The answer is the last step's output. The response has an extra `chain` field with every step (same as `/v1/chains/{name}/run`).
With `"stream": true` the whole answer arrives in one chunk (chains are not streamed token by token).

### `POST /v1/embeddings`

OpenAI embeddings, for a model with `kind = "embedding"`. `input` is a string or an array of strings. Without `model`, the first embedding model is used.

```json
{ "model": "minilm", "input": ["open the calculator", "launch the calc app", "what is the weather in Paris"] }
```

```json
{ "model": "minilm", "object": "list",
  "data": [ { "object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, ...] }, ... ],
  "usage": { "prompt_tokens": 21, "total_tokens": 21 } }
```

Vectors are L2-normalized (llama-server's default), so a dot product is the cosine. Measured with all-MiniLM-L6-v2 Q8_0: 384 dimensions, cosine 0.567 between the two calculator phrases, 0.006 to the weather phrase; 3.8 ms for one short input with the model loaded.
Each input must fit in `ctx_size` tokens. Embedding requests take a generation slot like chat requests (they use the same CPU).

### `GET /v1/chains`

Lists the chains, their steps (with each step's model) and the request variables they need.

```json
{ "chains": [ { "name": "qa", "description": "", "vars": ["passage", "question"],
  "steps": [ { "name": "topic", "model": "router", "json_schema": false, "lora": null },
             { "name": "answer", "model": "writer", "json_schema": false, "lora": { "path": "...", "id": 0, "scale": 1.0 } } ] } ] }
```

### `POST /v1/chains/{name}/run`

```json
{ "input": "optional text, the {input} placeholder",
  "vars": { "question": "Who built the lighthouse?", "passage": "The lighthouse on Kell Island was built by Mara Voss in 1871." },
  "system": { "answer": "Reply in at most six words. Passage: {passage}" } }
```

| Field | Meaning |
|---|---|
| `input` | The `{input}` placeholder. Optional if `vars` is given. The v0.1 form `{"input": "..."}` still works. |
| `vars` | Request variables: `{question}`, `{passage}`, ... in any template. A string is used as it is; other JSON values as JSON; `{meta.lang}` reads a field of an object var. A variable may not have the name of a step, `input` or `prev`. Every variable a template uses must be sent: otherwise `400` naming the missing ones, before anything runs. |
| `system` | Per-request system prompt per step: `{ "<step>": "<template>" }`. It replaces that step's configured `system` for this request and may use the same placeholders. |

Response: each step's output, model and timings. `json` is there for steps with a `json_schema`.

```json
{ "chain": "qa", "request_id": "req-6b0284a9-7", "output": "Mara Voss.",
  "steps": [
    { "name": "topic", "model": "router", "output": "...", "finish_reason": "length",
      "timing": { "kind": "chain_step", "request_id": "req-6b0284a9-7", "model": "router", "label": "qa/topic", "queue_ms": 0.0, "load_ms": 911.3,
                  "first_token_ms": 990.1, "prompt_tokens": 20, "prompt_ms": 40.2, "prompt_tokens_per_sec": 497.5, "completion_tokens": 8,
                  "generation_ms": 110.5, "tokens_per_sec": 72.4, "total_ms": 1068.7 } },
    { "name": "answer", "model": "writer", "output": "Mara Voss.", "finish_reason": "stop", "timing": { ... } } ],
  "queue_ms": 0.0, "load_ms": 911.3, "total_ms": 1607.1 }
```

A chain holds one generation slot from its first step to its last; other requests wait. Each step holds its own model only while it runs, so a step on another model may load that model and evict an idle one.

### `DELETE /v1/requests/{id}`

Cancels a chat, chain or embeddings request.

```json
{ "id": "bar-42", "cancelled": true, "state": "running" }
```

| State when cancelled | What happens |
|---|---|
| `queued` (waiting for a slot) | It leaves the queue; the client gets `499 request_cancelled` at once. |
| `running` (loading its model, or generating) | The runtime closes its connection to llama-server, which stops the generation (llama-server logs `cancel task`; its CPU use stops at once, measured: +0.02 s CPU over the next 3 s). The client gets `499`, or for a stream the error event above. A model that was loading finishes loading. |

`404 request_not_found` if no such request is queued or running.

**Closing the connection** has the same effect as a cancel: measured for a stream and for a non-streamed request, llama-server stopped the task within a second and the request left the queue. So pressing Esc can simply close the HTTP request; use `DELETE` when the request runs elsewhere (another thread or process) and you only have its id.

### `GET /v1/requests`

The queued and running requests: `{ "requests": [ { "id", "kind", "model", "priority", "state", "age_ms" } ] }`.

### `GET /health`

Cheap status; never loads a model. `model`, `model_state` and `child_*` are about the default model (as in v0.1); `models` lists all.

```json
{ "status": "ok", "model": "router", "model_state": "loaded", "child_working_set_mb": 327.0, "child_private_mb": 137.5,
  "models": [ { "name": "router", "kind": "chat", "state": "loaded", "active": 0, "pinned": true, "last_used_secs_ago": 3.2, "child_working_set_mb": 327.0 },
              { "name": "writer", "kind": "chat", "state": "unloaded", "active": 0, "pinned": false, "last_used_secs_ago": null, "child_working_set_mb": null } ],
  "system_free_mb": 6026.0, "requests_served": 5, "requests_in_flight": 0, "last_tokens_per_sec": 19.8, "pacer_capped": false, "uptime_secs": 120.5 }
```

`state` is `unloaded`, `loading` or `loaded`.

### `GET /stats`

Everything. Main parts:

```json
{ "uptime_secs": 120.5,
  "models": [ { "name": "writer", "kind": "chat", "state": "loaded", "pinned": false, "path": "...", "file_mb": 2886.6,
                "ctx_size": 4096, "threads": 3, "kv_cache_type": "f16", "flash_attn": "on", "mmap": true, "no_repack": true,
                "cache_ram_mb": 256, "ctx_checkpoints": 4, "pooling": null, "adapters": [ { "id": 0, "path": "..." } ],
                "active": 0, "unload_pending": false, "idle_secs": 3.1, "last_used_secs_ago": 3.1, "idle_unload_secs": 90,
                "memory_cap_mb": 1216, "memory_cap_source": "auto",
                "memory_estimate_mb": { "kv": 64.0, "recurrent": 0.0, "compute": 128.0, "cache_ram": 256.0, "checkpoints": 128.0, "overhead": 640.0, "total": 1216.0,
                                        "load_overhead": 240.3, "load_private": 832.0, "weights_private": false, ... },
                "load_estimate_mb": { "total": 2278.7, "private": 832.0, "mapped": 1446.7, "repack": 0.0,
                                      "on_disk": [ { "tensor": "per_layer_token_embd", "mb": 1440.0 } ], "source": "estimate" },
                "counters": { "loads": 1, "load_failures": 0, "evictions": 0, "last_load_ms": 3497.9, "generations": 5 },
                "child": { "pid": 1234, "port": 51234, "loaded_secs": 40.2, "load_ms": 3497.9, "working_set_mb": 1571.7, "private_mb": 369.8,
                           "peak_working_set_mb": 1595.0, "peak_private_mb": 388.0, "job_peak_commit_mb": 388.0 } } ],
  "pool": { "budget_mb": 4500, "resident_estimate_mb": 3864.0, "loaded": ["router", "writer"], "evictions": 0 },
  "queue": { "generating": false, "running": 0, "waiting": 0, "waiting_interactive": 0, "waiting_background": 0, "max_concurrent": 1, "max_queue": 8 },
  "requests": [],
  "system": { "total_mb": 16000.0, "free_mb": 6026.0, "memory_load_pct": 62, "low_memory_signal": false },
  "budget": { "reserve_free_mb": 1536, "pool_budget_mb": 4500, "pressure_free_mb": 1024, "critical_free_mb": 500, "idle_unload_secs": 90, ... },
  "pacer": { "mode": "auto", "state": "busy", "level": "light", "reason": "other processes use 23% CPU (> 15%)", "capped": true,
             "cap_pct": 75, "cap_pct_of_system": 28.13, "others_cpu_pct": 23.4, "free_mb": 6026, "mem_latency_ns": null,
             "thresholds": { "light_cpu_pct": 20.0, "heavy_cpu_pct": 40.0, "low_free_mb": 1024, "light_cap_pct": 75, "heavy_cap_pct": 50, ... },
             "secs_at_level": { "none": 50.1, "light": 12.3, "heavy": 0.0 }, "level_changes": 2, "capped_secs_total": 12.3, "cap_on_count": 1, "last_error": null,
             "generations_capped": 2, "generations_uncapped": 3, "last_tokens_per_sec_capped": 6.1, "last_tokens_per_sec_uncapped": 7.6 },
  "counters": { "requests_served": 5, "requests_failed": 0, "refused_insufficient_memory": 0, "refused_queue_full": 0, "cancelled": 0,
                "loads": 2, "load_failures": 0, "last_load_ms": 3497.9, "unloads": { "idle": 1, "evicted": 0, "memory_pressure": 0, "memory_critical": 0 },
                "peak_child_working_set_mb": 1595.0, "peak_child_private_mb": 388.0, "peak_job_commit_mb": 388.0,
                "last_tokens_per_sec": 7.6, "last": { ... }, "recent": [ ... ] },
  "model_state": "loaded", "model": { "name": "router", "threads": 3 }, "child": { ... } }
```

The last three fields are the v0.1 fields, for the default model. `unloads` reasons: `idle`, `evicted`, `memory_pressure`, `memory_critical`, `admin`, `crashed`, `shutdown`.

Timing fields (per generation, in `last` and `recent`):

| Field | Meaning |
|---|---|
| `kind`, `request_id`, `model`, `label` | `chat`, `chat_stream`, `chain_step` or `embedding`; the request; the model; the model or `chain/step`. |
| `queue_ms` | Waiting for a generation slot. |
| `load_ms` | Waiting for the model to be ready for this request (0 if it was loaded). |
| `first_token_ms` | Request arrival to first token. Measured for streams. For non-stream requests: queue + load + prompt time. |
| `prompt_tokens`, `prompt_ms`, `prompt_tokens_per_sec` | Reading the prompt (llama-server's numbers). |
| `completion_tokens`, `generation_ms`, `tokens_per_sec` | Writing the answer (llama-server's numbers). |
| `total_ms` | Whole request inside the runtime. |

### `POST /admin/load`

Warm-up: loads a model and returns when it is ready. Body optional: `{"model": "writer"}`; no body = the default model.
It may evict an idle model, exactly like a request would. It does not take a generation slot, so it does not wait behind running generations.

```json
{ "model": "writer", "loaded": true, "was_loaded": false, "load_ms": 1076.0, "total_ms": 1076.3, "model_state": "loaded" }
```

If the model was already loaded: `"was_loaded": true, "load_ms": 0.0`. Errors as for requests (`404`, `503 insufficient_memory`, `503 model_load_failed`).
A warm-up counts as a use: it makes the model the most recently used one.

### `POST /admin/unload`

Stops models now and frees their RAM. `?model=<name>` = one model; without it, all models.
By default it waits until no request uses the model. `?force=true` kills it at once; a running request on it then fails with `engine_error`.

```json
{ "unloaded": true, "models": ["router"], "model_state": "unloaded" }
```

The next request that needs the model loads it again.

## Pacer

"Don't freeze the PC", v2. Code: `runtime/src/pacer.rs`. Measured in [results/05-pacer-v2.md](results/05-pacer-v2.md) (v1: [results/03-pacer-v1.md](results/03-pacer-v1.md)).

Every `poll_ms` (300 ms) the pacer reads two signals:

| Signal | Source |
|---|---|
| Other programs' CPU | `GetSystemTimes` minus the CPU time of every model's Job Object, as a share of all logical processors, **averaged over `smooth_ms` (1.5 s)** so that one short spike does not count. |
| Free RAM | `GlobalMemoryStatusEx` |

They set a **level**, and the level sets a **Job Object CPU rate hard cap** on every loaded child:

| Level | When | Cap on each model |
|---|---|---|
| `none` | other programs ≤ 20% | none |
| `light` | other programs 20 to 40% | 75% of the CPU the model uses at full speed (3 of 8 threads: 28% of the whole CPU) |
| `heavy` | other programs > 40%, or free RAM < `pressure_free_mb` | 50% (18.75% of the whole CPU) |

- **Going up is immediate** (after the 1.5 s average). **Going down** needs the signal below the threshold by `hysteresis_pct` (5 points; RAM: 256 MB) **and** the level held for `release_after_ms` (5 s). So a signal near a threshold does not make the cap flap.
- **Leaving heavy is slower (v0.3).** CPU must be under 30% (`heavy_hysteresis_pct` = 10 points), RAM must be 256 MB above `low_free_mb`, and both must stay so for `heavy_release_after_ms` (10 s) in a row. In v0.2 the level changed 3 to 9 times per 40 s around the 40% line. In a unit test with noisy CPU around 40% (±8 to ±14 points, 40 s), the v0.2 rules gave 10 to 14 changes and the v0.3 rules 1 or 2.
- **The reason says when a level is holding (v0.3).** For example `holding: RAM was below 1024 MB, now 1225 MB`, `holding: other processes used over 40% CPU, now 35%`, or `holding heavy for 6 s more; now ...`. v0.2 said "1225 MB RAM free (< 1024 MB)" in this case.
- The cap is set, changed and removed on the running children, with no restart.
- **The keyboard and mouse are not a signal any more** (owner decision, v0.2): typing alone does not slow the model; a program that needs the CPU does.
- Measured (on battery): a light ~20% background load gives the light level about 80% of the time, the model runs at 5.0 to 5.3 instead of 6.1 to 6.9 tok/s; on this laptop the PC's own background is 12 to 13%, close to the 15% line. See [results/05-pacer-v2.md](results/05-pacer-v2.md).
- `/stats` → `pacer` shows the level, its reason, the smoothed CPU, free RAM, the thresholds and the seconds spent at each level.
- **Extension point:** a memory-bandwidth signal is planned (a tiny RAM-latency probe; the v1 measurements showed the model and a user app fight for RAM bandwidth, which CPU % does not show). `/stats` already has the field `mem_latency_ns` (null). In the code a new signal is one field in `Signals` and one rule in `proposals()`.

Why this lever: llama-server has no per-request thread count (a restart costs a model load), and it keeps generating at full speed even if the runtime reads its stream slowly. A CPU cap is the one lever that works on the running process.
The cost: a capped model is slower. `/stats` → `pacer` shows the tokens/s of capped and uncapped generations.

## Chains

A chain is a list of steps in the config.

```toml
[[chains]]
name = "qa"
description = "Pick, then answer from a passage."

[[chains.steps]]
name = "pick"
model = "router"                 # optional; default = the first chat model
system = "Pick one tool for the request."
input = "Request: {question}"
max_tokens = 1
temperature = 0.0
grammar = "root ::= [0-9]"

[[chains.steps]]
name = "answer"
model = "writer"
system = "Answer from this passage only: {passage}"
input = "{question} (tool {pick})"
max_tokens = 200
temperature = 0.3
# lora = { path = 'adapters/answer.gguf', scale = 1.0 }
```

Step keys:

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Letters, digits, `_`, `-`. Not `input` or `prev`. Unique in the chain. |
| `model` | first chat model | The chat model this step runs on. |
| `system` | `""` | System prompt template. Empty = no system message. Can be replaced per request (`"system"` in the run request). |
| `input` | `{input}` for step 1, `{prev}` after | User message template. |
| `max_tokens` | `256` | |
| `temperature` | `0.2` | |
| `json_schema` | none | Output must match this JSON schema. A TOML table, or a string holding JSON. |
| `grammar` | none | GBNF grammar for the output. Not together with `json_schema`. |
| `lora` | none | `{ path = "...", scale = 1.0 }`. A LoRA adapter for this step, on the step's model. |

Template placeholders:

| Placeholder | Becomes |
|---|---|
| `{input}` | The chain input. |
| `{prev}` | The previous step's output. |
| `{stepname}` | That earlier step's output. |
| `{stepname.field}` | A field of that step's JSON output. Nested: `{stepname.a.b}`; arrays: `{stepname.items.0}`. |
| `{anyname}` | A request variable from `"vars"`; `{anyname.field}` for a field of an object variable. |

Only `{name}` and `{name.path}` are placeholders. Other braces, like a JSON example in a prompt, stay as they are.
A placeholder naming the step itself or a later step is a config error at startup. `GET /v1/chains` lists the variables each chain needs.

## LoRA adapters

Adapters are loaded with their model (`adapters = [...]` in the model, plus every `lora` of a chain step on that model), all at scale 0. The id of an adapter is its index in that list; `/stats` → `models[].adapters` shows the ids.

- **In a chat request:** `"lora": [{"id": 0, "scale": 1.0}]`. Switching the scale per request needs no reload (measured: the same router child served base, scale 1, 0.3 and 0 in turn).
- **The runtime always sends the complete list**, with scale 0 for every adapter the request does not name. Reason: llama.cpp build 12248 applies an adapter that a request leaves out at its load-time scale (1.0), even with `--lora-init-without-apply`. A v0.1 chain step without `lora` would have run with the adapter on; this is fixed.
- A `lora` id the model does not have: `400`.
- Tested with a synthetic rank-1 adapter for the 270M router model (`V:\jitmo-sandbox\adapters\router-synthetic-r1.gguf`, made with the `gguf` Python library; see results/04). A trained adapter has not been tested yet.

## How to use it from jitmo agent

- Start it as a child process (`--port 0 --parent-pid --port-file --log-file`) and point an OpenAI client at `http://127.0.0.1:<port>/v1`. No API key is checked.
- Call `POST /admin/load` with your bar's model when the bar opens. It returns when the model is ready.
- Send `X-Jitmo-Priority: background` for work the user is not waiting for, and your own `X-Jitmo-Request-Id` for anything the user may cancel. On Esc: close the request, or `DELETE /v1/requests/{id}`.
- For the writer with two inputs: a chain with `{question}` and `{passage}`, or simply a direct chat request with your own prompt.
- Several models: give each its `[[models]]` entry; pin the small one you always need (the router) so it is never evicted. Set `pool_budget_mb` to your RAM rule.
- Send `Content-Type: application/json` with every POST body.
- Handle `503 preempted` on background requests (retry later), `503 insufficient_memory` (tell the user, or retry later), `503 queue_full` and `502 engine_error` (try again), `499` (your own cancel), and `503 memory_pressure` (v0.3: the runtime stopped the generation because RAM stayed short; tell the user, retry later).
- Expect generation to be slower while other programs use the CPU (pacer levels). `GET /health` → `pacer_capped` and `/stats` → `pacer.level` say so.
- Expect a model switch to cost a model load if the other model had to be evicted: measured 1.2 s for the 270M router and 1.8 to 2.1 s for Gemma 3n E2B (warm file cache); about 0 if both stay loaded.

## Limits of v0.3

- One generation at a time across the pool by default; no batching inside a model (`--parallel 1`). A preempted background request starts again from the beginning when retried.
- Chain output is not streamed token by token.
- No authentication. It listens on 127.0.0.1 only by default, and the local-only guard refuses web pages. Other programs on the same PC can still call it, and can call the llama-server children's ports directly.
- Token tagging (NER) and classification heads are not supported (llama.cpp has none). Embeddings are.
- The pacer has one lever, a CPU cap per child. It cannot change threads per request and does not pause between tokens. Its signals are CPU and free RAM; memory-bandwidth contention is not measured yet.
- The load estimate counts the whole prompt cache (`cache_ram_mb`, 256 MB), so for a small model that is used little it is about twice the real use. Override it with `load_estimate_mb` per model, or lower `cache_ram_mb`.
- The memory cap does not cover memory-mapped weights, and the automatic cap does not cover a repacked copy (`no_repack = false`).
- Memory pressure waits for a running generation (or chain) for at most `pressure_finish_max_ms` (15 s), then cancels it. Measured in a real low-RAM run: it cancels after 15 to 17 s (the watchdog checks once a second), see [results/07-v0.3-low-ram.md](results/07-v0.3-low-ram.md).
- A model that is loading cannot be cancelled; cancelling its request lets the load finish.
- Windows only (Job Objects, Win32 memory and CPU APIs).

## Measured

- v0.2 (child mode, cancel, priority, pool switching, embeddings, LoRA, cold load, load estimate): [results/04-v0.2.md](results/04-v0.2.md).
- Pacer v2: [results/05-pacer-v2.md](results/05-pacer-v2.md).
- Low free RAM (loading, pacer at 15% vs 20%, memory-pressure unload): [results/06-low-ram.md](results/06-low-ram.md).
- v0.3 under low free RAM (load estimates, pacer, forced 4B, 15 s pressure limit): [results/07-v0.3-low-ram.md](results/07-v0.3-low-ram.md).
- v0.1: Gemma 3n vs Qwen3-4B [results/02-gemma3n.md](results/02-gemma3n.md); pacer v1 [results/03-pacer-v1.md](results/03-pacer-v1.md); v0 baseline numbers [results/01-baseline.md](results/01-baseline.md).
