# 08. Our own engine host: llama.cpp as a library

Started 28 Nov 2026. Status: **P0–P3 done (28 Nov).** Measured: [results/09](results/09-engine-compare.md). Today each model is a `llama-server` child; this replaces it step by step, behind a per-model switch. The HTTP API does not change.

## Why

`llama-server` is a black box we can only steer from outside. Owning the generation loop gives:

| Gain | How | Today |
|---|---|---|
| Pause between tokens when the user needs the PC (EdgePace / SERENO idea) | Our loop decides when to run the next decode | Only a Job Object CPU cap, which stalls llama.cpp's threads mid-step |
| Change threads live | `llama_set_n_threads` | Needs a restart (a model load) |
| One base model, many jobs | One `llama_model` (weights once), one `llama_context` + LoRA per job | One full model per job |
| Prompt cache we control | Save/restore the KV state of fixed prompt prefixes (`llama_state_seq_*`) | Whatever llama-server keeps in `--cache-ram` |
| Less overhead | No HTTP hop to a child, no second server per model | One extra process + HTTP per model |

## How

- **Load the prebuilt `llama.dll`** (and its ggml DLLs) at run time from a llama.cpp build folder: CPU or Vulkan, chosen per model. No C/C++ compiler or cmake is needed.
- **Hand-written FFI** for the part of `llama.h` we use, pinned to one llama.cpp tag (b12606). Structs passed by value (`llama_model_params`, `llama_context_params`) must match that tag exactly, so the runtime checks the build before using it.
- **What libllama lacks**, and we write in Rust:
  - OpenAI request/response format and streaming (we already parse and produce it).
  - Chat templates: `llama_chat_apply_template` covers the built-in ones (Gemma, ChatML/Qwen, ...), but not Jinja.
  - JSON schema → grammar (`response_format: json_schema`). jitmo's schemas are simple (objects of strings, enums), so a small converter is enough.

## Risks

- **A crash in llama.dll kills the process.** Today a `llama-server` crash is contained. Plan: run the engine in **our own worker process** (`jitmo-runtime.exe --engine-worker`), so we keep isolation, Job Objects and memory caps while owning the loop.
- **ABI drift.** A different llama.cpp build can change struct layouts. Pin the tag and check it at load.
- **Feature gap with llama-server** (templates, grammar, multimodal). Keep `llama-server` as the default until the new engine passes the same tests and jitmo's sets.

## How P1 works

- The runtime starts `jitmo-runtime.exe --engine-worker ...` instead of `llama-server` for a model with `engine = "inproc"`. That is already P4's isolation: the worker is its own process, in the model's Job Object.
- The worker serves `GET /health` and `POST /v1/chat/completions` (plain, stream, `grammar`, `response_format`, `stop`, sampling settings) with llama-server-style `timings` (`prompt_n`, `cache_n`, ...).
- The KV cache of the prefix shared with the previous prompt is reused (`cache_n`).
- The context reserves logits for **one** output token (`n_outputs_max = 1`). With Gemma's 262k vocabulary this cut the compute buffer from 521 MB to 21 MB; llama-server reserves outputs for a whole batch.
- A closed connection or `DELETE /v1/requests/{id}` stops the generation between tokens.
- **Prompt cache (P3):** before the live KV state is overwritten, it is saved (`llama_state_seq_get_data`, sequences of 32+ tokens) in an LRU store within `cache_ram_mb`. A new prompt restores the saved state with the longest shared prefix if it beats the live cache by more than 8 tokens. States are keyed by their LoRA scales too.
- **LoRA (P3):** adapters load once with the model; each request's `lora` scales are applied with `llama_set_adapters_lora`. A scale change drops the live KV cache (it was made with other scales).

## Phases

| Phase | What | Done when |
|---|---|---|
| P0 | Proof of concept: load, tokenize, template, prefill, greedy decode, a pause hook | Same text and similar tok/s as llama-server, same model and threads |
| P1 | `engine = "inproc"` per model; chat completions with grammar, simple JSON schema, streaming, cancel | **Done 28 Nov**: `runtime/scripts/smoke_inproc.py` 15/15 (CPU and Vulkan), old smoke test still passes |
| P2 | Pacer v3: pause between tokens (and between 64-token prompt chunks) and live thread changes | **Built and measured 28 Nov** ([results/09](results/09-engine-compare.md)): answers faster than the CPU cap at the same throughput (4.1 vs 4.9 s), but neither lever cuts the foreground slowdown much (memory bandwidth). Next lever: pause only while the user interacts |
| P3 | One base model shared by jobs, LoRA per job, prompt cache by KV state | **Done 28 Nov**: LoRA per request (scale 0 = base output exactly; unknown id → 400); prompt cache A, B, A → 75 tokens restored. One base for many jobs = one inproc model + per-request LoRA (jitmo already uses one base). Several contexts per model: not needed yet |
| P4 | Engine in our own worker process | A crash in the engine leaves the runtime running |
