# 09. Our engine vs llama-server, CPU vs Iris Xe, and the pacer's levers (28 Nov 2026)

Bench: `bench/engine_compare.py` (through the runtime, pacer off for speed). Raw data: `bench/out/engine_compare/`. PC fairly quiet (jitmo bake-off idle, ~7.6 GB free), AC power. llama.cpp b12606 for both engines. Single runs: treat differences under ~10% as noise.

## Speed and RAM

Tool pick = jitmo's pick prompt (8 tools, ~70 tokens) + one grammar digit; 8 different requests × reps, same system prompt. Generation = 96 tokens.

**Gemma 3 270M (jitmo's tuned router)**

| Config | Load | Tool pick p50 | Prompt p50 | Gen tok/s | Working set | Private |
|---|---|---|---|---|---|---|
| llama-server, CPU | 2323 ms | 208 ms | 202 ms | 82 | 339 MB | 149 MB |
| **inproc, CPU** | **565 ms** | **89 ms** | **12 ms** | 79 | 357 MB | 171 MB |
| llama-server, Vulkan | 996 ms | 156 ms | 143 ms | 63 | 741 MB | 765 MB |
| inproc, Vulkan | 1077 ms | 80 ms | 17 ms | 68 | 732 MB | 766 MB |

**Gemma 3n E2B**

| Config | Load | Tool pick p50 | Prompt p50 | Gen tok/s | Working set | Private |
|---|---|---|---|---|---|---|
| llama-server, CPU | 1594 ms | 1283 ms | 1273 ms | 11.8 | 1586 MB | 364 MB |
| **inproc, CPU** | 1067 ms | **491 ms** | **316 ms** | 10.8 | 1633 MB | 416 MB |
| llama-server, Vulkan | 5517 ms | 914 ms | 905 ms | 10.7 | 3733 MB | 3859 MB |
| inproc, Vulkan | 2199 ms | 417 ms | 310 ms | 12.1 | 3760 MB | 3885 MB |

- **Our engine picks a tool 2.3× (270M) to 2.6× (Gemma 3n) faster on the CPU**, at the same RAM and generation speed. The win is prompt reuse: our worker keeps the KV cache of the shared system prompt; llama-server re-reads it for these SWA models.
- jitmo's answers are short (1 token, ~20 tokens of JSON), so **prompt reading is most of the cost**. Generation speed matters little.
- **Vulkan on Iris Xe is not worth it for these models**, given the RAM: +400 MB for the 270M, and for Gemma 3n 3.8 GB instead of 1.6 GB (the GPU copy loses Gemma 3n's on-disk Per-Layer Embeddings). With prompt reuse the CPU is nearly as fast. Vulkan may still pay off for models without that trick and with long fresh prompts.
- Bug found and fixed: the automatic memory cap did not count GPU-offloaded weights (on an iGPU they are private memory), so Gemma 3n on Vulkan failed to load under the cap. The cap and the load estimate now add the weights when `gpu_layers` is set.

## UI lag: CPU cap vs pause between tokens

Gemma 3n E2B, CPU, inproc unless noted. Load = a loop of jitmo-like requests with **fresh** prompts (pick + JSON fill; worst case, no reuse). `probe.exe` 12 s each: a 16 ms frame loop with a fixed CPU task, then a memory-bound pass over 200 MB.

| Case | Frame late p50/p99 | CPU task p50 | Memory pass p50 | Requests/s | Request p50 |
|---|---|---|---|---|---|
| no model | 0.39 / 3.52 ms | 6.73 ms | 17.2 ms | – | – |
| no pacer | 0.34 / 0.64 | 9.18 (+36%) | 26.2 (+52%) | 0.25 | 3.2 s |
| CPU cap 50% | 0.31 / 0.58 | 10.03 | 23.9 | 0.12 | 4.9 s |
| pause 50% (P2) | 0.31 / 0.61 | 10.28 | 25.7 | 0.12 | 4.1 s |
| llama-server, CPU cap 50% | 0.32 / 0.60 | 10.34 | 24.5 | 0.12 | 5.3 s |

- **Neither lever protects the foreground app much.** Halving the model's compute share halves its throughput but leaves the other app 40–50% slower. The contention is memory bandwidth, as the research said; a CPU-share lever can't fix it.
- The pause lever answers faster than the CPU cap at the same throughput (4.1 vs 4.9 s p50), so it is the better lever, but it is not a UI win.
- Frame lateness stays under 1 ms in every case: no freezing.
- What should help instead: fewer bytes read per answer (prompt reuse, smaller models), and pausing only while the user is actually interacting rather than whenever other CPU use is high.
