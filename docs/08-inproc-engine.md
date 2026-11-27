# 08. Our own engine host: llama.cpp as a library

Started 28 Nov 2026. Status: **design + proof of concept.** Today each model is a `llama-server` child; this replaces it step by step, behind a per-model switch. The HTTP API does not change.

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

## Phases

| Phase | What | Done when |
|---|---|---|
| P0 | Proof of concept: load, tokenize, template, prefill, greedy decode, a pause hook | Same text and similar tok/s as llama-server, same model and threads |
| P1 | `engine = "inproc"` per model; chat completions with grammar, simple JSON schema, streaming, cancel | Smoke test passes on the new engine |
| P2 | Pacer v3: pause between tokens and live thread changes | UI-lag test better than the CPU cap at the same model speed |
| P3 | One base model shared by jobs, LoRA per job, prompt cache by KV state | RAM and switch-time measured |
| P4 | Engine in our own worker process | A crash in the engine leaves the runtime running |
