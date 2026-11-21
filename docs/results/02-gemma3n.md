# 02. Gemma 3n E2B vs Qwen3-4B

Measured on the owner's laptop on 21 Nov 2026, 21:46 to 22:03 IST, on AC power (see `bench\out\power.log`).
i5-1135G7, 16 GB, Windows 11, llama.cpp CPU build 12248. Other apps were open: **only 4.5 to 4.7 GB of RAM was free** before each run, and the owner was using the PC.
Raw outputs: `bench\out\lb2_*.json` (llama-bench) and `bench\out\rt_*.json` (runtime). Scripts: `bench\run_gemma3n_lb.sh`, `bench\run_runtime_model.ps1`.

Model: `gemma-3n-E2B-it-Q4_K_M.gguf` (unsloth), 3,026,881,888 bytes, SHA-256 verified. 30 layers, 4.46 B parameters, of which the Per-Layer Embeddings (PLE) table `per_layer_token_embd` is 7680 × 262144 in Q5_1 = **1440 MiB, about half the file**.

## llama-bench (pp512, tg128, 3 repeats, `-fa on`)

Median (min–max) tokens/s. "repack on" is llama.cpp's default; "repack off" (`--no-repack`) is the runtime's default.

| Model | Repack | Threads | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| Gemma 3n E2B | on | 3 | 68.1 (66.5–71.7) | 14.86 (14.02–15.07) |
| | on | 4 | 76.0 (75.9–77.5) | 15.64 (14.32–16.81) |
| | off | 3 | 81.8 (78.6–87.0) | 12.92 (12.89–13.00) |
| | off | 4 | 98.1 (95.1–100.1) | 12.86 (12.76–13.25) |
| Qwen3-4B | on | 3 | 38.3 (36.7–40.4) | 8.38 (6.82–9.02) |
| | on | 4 | 51.1 (50.8–52.3) | 9.84 (9.74–10.01) |
| | off | 3 | 57.6 (57.4–59.1) | 7.56 (6.89–7.72) |
| | off | 4 | 64.9 (60.5–65.7) | 8.88 (8.69–8.96) |

- Qwen3-4B with repack on is **17% slower than in the baseline** at 3 threads (8.38 vs 10.14 tg, 38.3 vs 50.9 pp). Likely cause (not proven): with repacking, Qwen3-4B needs about 4.6 GB of RAM and only about 4.7 GB was free, so it ran at the edge of memory. That would also explain why "repack on" read prompts slower than "repack off" here, the opposite of what repacking is for. Treat the "repack on" rows as numbers under memory pressure.

## Through the runtime (ctx 4096, f16 KV, mmap, 3 threads, pacer off)

One load request, then 3 streamed requests of 128 tokens (same prompt, 26 to 28 tokens). Requests 2 and 3 reuse the cached prompt, so their first token is faster. Single runs.

| | Gemma 3n, repack off | Gemma 3n, repack on | Qwen3-4B, repack off | Qwen3-4B, repack on |
|---|---|---|---|---|
| Load (warm file cache) | 3.4 s | 2.6 s | 3.1 s | not measured, see below |
| First token, new prompt / cached prompt | 636 ms / 270–313 ms | 482 ms / 166–181 ms | 620 ms / 141–207 ms | |
| Generation tok/s (3 runs) | 11.7–12.0 | 13.6–14.2 | 7.5–7.8 | |
| Child private bytes after load → peak | 370 → **388 MB** | 1522 → 1540 MB | 794 → 841 MB | baseline: 2211 MB |
| Child working set after load → peak | 1570 → **1595 MB** | 2727 → 2752 MB | 3028 → 3077 MB | baseline: 4201 MB |
| Memory cap | auto, 1216 MB | none (auto cap does not cover repacking) | auto, 1600 MB | |

- **Qwen3-4B with repack on did not fit.** During its load free RAM fell to 311 MB, and the runtime's new critical-memory rule (below 500 MB) unloaded it at once. The request got `502 engine_error`. So that column has the baseline numbers (llama-completion, ctx 512) instead.

## Does llama.cpp keep the PLE weights out of RAM?

**Yes, mostly.** llama.cpp memory-maps the whole file (`CPU_Mapped model buffer size = 2880.40 MiB`), but it only touches the PLE rows of the tokens it actually sees.

| | MiB |
|---|---|
| File, mapped | 2880 |
| PLE table inside it | 1440 |
| Everything else | 1440 |
| Child working set after load (repack off) | 1570 |
| … of which private | 370 |
| … so mapped file pages in RAM | about 1200 |
| Working set growth over 4 × 128 generated tokens | +25 |

The mapped pages in RAM (about 1200 MiB) are fewer than the non-PLE part of the file (1440 MiB), so the PLE table is not resident. Each token reads one PLE row: 7680 values in Q5_1 = 5.6 KB, about 2 pages.
Caveat: pages of the file that Windows read earlier (for example during the SHA-256 check) can sit in the file cache (standby list). They count as free RAM and are not charged to the process, but they are not "on disk only" either. On a PC that is really short of RAM, a PLE row would be read from the SSD when needed: a few KB per token.

With repack on, the repacked layer weights add about 1150 MB of private memory, but the PLE table is still not loaded (Q5_1 is not a repacked type).

## What this means

- **Gemma 3n E2B needs about half the RAM of Qwen3-4B**: 1.6 GB working set vs 3.1 GB in the runtime, and 388 MB vs 841 MB of private memory. Its PLE half stays on disk.
- **It is also faster**: 11.7 to 12.0 vs 7.5 to 7.8 tok/s in the runtime (about 1.55 ×), and it reads prompts faster (82 vs 58 t/s pp512 at 3 threads, repack off).
- For Gemma 3n, repacking gives 15% faster generation but 1150 MB more private memory. With 3 vs 4 threads, generation barely changes (12.9 vs 12.9 t/s, repack off): 3 threads is enough.
- **Answer quality was not compared.** Gemma 3n "E2B" has 4.46 B parameters but only about 2 B effective ones; whether it answers as well as Qwen3-4B for the jitmo agent is the next question.
- The runtime's free-RAM check is too strict for Gemma 3n: it counts the whole 2.9 GB file plus the cap (4.1 GB), while the model really uses 1.6 GB. With 4.5 GB free the default reserve refused to load it; the runtime runs above used `reserve_free_mb = 0`.
