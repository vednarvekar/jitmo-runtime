# 04. Plan: how we move forward

Principle: **measure first, then build.** Every step ends with real numbers from the owner's laptop.

## Steps

| Step | What | Done when |
|---|---|---|
| **0. Save the work** | Docs into git, pushed to https://github.com/vednarvekar/jitmo-runtime | Repo has the docs |
| **1. Baseline** | Measure laptop RAM and SSD speed. Run 2 to 4B models with plain llama.cpp. Record speed, RAM use and how much the PC lags. | [results/01-baseline.md](results/01-baseline.md) has the numbers |
| **2. Ready tricks** | Apply the known tricks (below) one by one and measure each. | We know what each trick gives on this laptop |
| **3. Decide what to build** | Pick the first new part from the real numbers. Likely candidate: "don't freeze the PC" scheduling. | The owner chooses |

Next steps to a ready runtime (v1.0): [07 Roadmap](07-roadmap.md).

## Runtime v0 (decided 21 Nov 2026)

The first thing we build. Written in **Rust**, on top of llama.cpp at first.

| Part | What it does |
|---|---|
| **One base model** | Qwen3-4B (Q4_K_M) loaded once |
| **Chain** | A request runs through steps (for example router, planner, tool-caller, writer). Each step has its own instructions and an optional LoRA add-on. |
| **RAM budget** | Watches free RAM, caps the model's memory (for example 2.5 GB), unloads when idle, refuses to load when RAM is short |
| **Don't freeze the PC** | Limits threads; backs off when the PC is busy |
| **OpenAI-style API** | `/v1/chat/completions` and `/v1/models`, so jitmo agent (and OpenJarvis) can use it directly |

Code lives in `runtime/`.

## Step 1 setup

- **Laptop:** Intel i5-1135G7 (4 cores, 8 threads), 16 GB RAM, Windows 11, Iris Xe integrated GPU.
- **Engine:** llama.cpp CPU build 12248 (already in `V:\jitmo-sandbox\llama`).
- **Models:** Qwen3.5-0.8B, Gemma 3 1B (already on disk), a 3 to 4B model (Qwen3-4B, fallback Llama 3.2 3B), Gemma 3n E2B (built for low RAM).
- **8 GB simulation:** the laptop has 16 GB, so the model is limited to about 2.5 GB, roughly what an 8 GB laptop has free.
- **Lag test:** a small "fake user app" measures how late it runs while the model works.

## Step 2: the ready tricks

| Trick | Expected effect |
|---|---|
| 4-bit weights | Already used in the baseline |
| KV cache in 8-bit | Less RAM on long chats |
| Prompt caching | Repeated instructions are not re-read |
| Fewer threads | Less lag for other apps, maybe a little slower |
| Short, structured outputs | Fewer tokens, less work |
| One base + LoRA adapters | Many jobs for the RAM of one model |
| Speculative decoding | 1.5 to 2× faster, if a good draft model exists |

## Tracking

| Date | Event |
|---|---|
| 21 Nov 2026 | Step 1 started: measurement agent and research agent launched |
| 21 Nov 2026 | Research round 2 done (see 02 Research, sections 11 and 12) |
| 21 Nov 2026 | Owner decided: build both layers (runtime + assistant) |
| 21 Nov 2026 | Baseline paused: laptop on battery; waiting for AC power |
| 21 Nov 2026 | Baseline resumed on AC; first speeds: Qwen3-4B ~10.8 words/s at 4 threads (final numbers in results/01-baseline.md) |
| 21 Nov 2026 | Owner approved runtime v0 (Rust, on llama.cpp); build started |
| 21 Nov 2026 | Step 1 baseline done: see [results/01-baseline.md](results/01-baseline.md). Key findings: Qwen3-4B 11.2 tok/s (81% of RAM bandwidth); weight repacking doubles RAM (`--no-repack` halves it, ~20% slower); under a 2.5 GB cap speed falls to ~1.2 tok/s; a memory-heavy app slows 44 to 60% while the model runs |
| 21 Nov 2026 | Runtime v0 built and tested (18 unit tests, smoke test, chain on Qwen3-4B); pushed to GitHub |
| 21 Nov 2026 | Owner: go with suggested defaults (finish the running answer before unloading on pressure; threads = cores − 1 capped at half the logical threads; memory cap computed from model and ctx; KV f16). v0.1 started: these defaults, Gemma 3n test, pacer v1 ("don't freeze the PC") |
| 21 Nov 2026 | Owner approved v0.2 = jitmo agent's requests (see [jitmo-agent-response.md](jitmo-agent-response.md)), in order: run as child process, warm-up, cancel, priority, multi-input chains, several models at once, embeddings, LoRA and cold-load tests. Starts after v0.1 |
| 22 Nov 2026 | Owner decided the pacer design. "PC is busy" is based on other apps' CPU use plus free RAM; keyboard and mouse are not used. Adaptive cap: no cap under 15% other-app CPU, 75% at 15 to 40%, 50% above 40% or when RAM is low. Added to v0.2 |
| 21 Nov 2026 | v0.1 done: owner's defaults in code (finish-then-unload with 500 MB critical abort, threads min(cores − 1, logical/2), memory cap computed from GGUF metadata), pacer v1 (Job Object CPU cap while busy: app slowdown +29% → +2%, model 5.3 → 3.2 tok/s while capped), Gemma 3n tested (half the RAM of Qwen3-4B, 1.55 × faster). See [results/02-gemma3n.md](results/02-gemma3n.md), [results/03-pacer-v1.md](results/03-pacer-v1.md) |
| 22 Nov 2026 | v0.2 done: child-process mode (`--parent-pid`, port 0, `--port-file`), `POST /admin/load`, request ids + cancel (`DELETE /v1/requests/{id}`; closing the connection also stops llama-server), priority header, chain `vars` + per-request `system`, model pool (`[[models]]`, pool budget, LRU eviction, pinned), embeddings (`/v1/embeddings`, MiniLM tested), LoRA per-request scale (synthetic adapter; fixed llama.cpp default-scale quirk), looser load estimate (Gemma 3n 4.1 → 2.7 GB), pacer v2 (CPU levels 75% / 50%, no input signal). Router ↔ Gemma 3n switch: ~0 s both resident, 1.2 / 1.9 s with reload; cold load +0.3 to 0.7 s. Measured on battery. See [results/04-v0.2.md](results/04-v0.2.md), [results/05-pacer-v2.md](results/05-pacer-v2.md) |
| 22 Nov 2026 | Low-RAM test on AC with Docker running (1 to 4.4 GB free): Gemma 3n, 270m router and 0.8B load; Qwen3-4B refused (estimates 1.7 to 3.8 × real use: safe but too strict). Pacer v2 on Gemma 3n: light threshold 20 beat 15 (app +34% vs +41%, model 6.7 vs 6.1 tok/s); still 3 to 9 level changes per 40 s. A real memory-pressure unload worked (finished the generation, then unloaded). Forced-4B and some background-load runs not done (owner started a game). See [results/06-low-ram.md](results/06-low-ram.md) |
| 22 Nov 2026 | v0.3 done (5 fixes from the low-RAM test, owner approved): closer load estimate (KV + buffers + prompt cache + 96 MiB + 5% of the weights, no 640 MiB overhead; router 1265 → 705 MB, 0.8B 1596 → 1068, Gemma 3n 2663 → 2279, Qwen3-4B 3981 → 3534, all still above the real use), `light_cpu_pct` 15 → 20, slower release from heavy (`heavy_hysteresis_pct` 10, `heavy_release_after_ms` 10000; noisy CPU at 40% in a unit test: 10 to 14 changes per 40 s → 1 or 2), "holding: ..." reason text, and a 15 s limit on "finish, then unload" (`pressure_finish_max_ms`; then the request gets `503 memory_pressure`). 49 unit tests, smoke test passed (44 checks, incl. a live `memory_pressure` stop on the 0.8B). Load and pacer not re-measured in a low-RAM run yet |
| 22 Nov 2026 | Roadmap step 2 done (v0.3 low-RAM tests, AC, overlapping a jitmo audit run): v0.3 estimates 1.4 to 2.1 × real use (MiniLM 4.3 ×); forcing Qwen3-4B into ~3 GB free hit the 300 MB safety stop in ~8 s during load (refusal was right); 15 s pressure limit cancelled with `503 memory_pressure` and the PC stayed responsive; critical unload fired twice in real runs; pacer flapping 3–9 → 0–2 changes per 40 s. 20% vs 15% not measurable (the audit kept the pacer at heavy). See [results/07-v0.3-low-ram.md](results/07-v0.3-low-ram.md) |
