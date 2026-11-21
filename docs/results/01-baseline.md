# 01. Baseline (Step 1)

These numbers were measured on the owner's laptop on 21 Nov 2026, between 19:49 and 20:43 IST.
They are real measurements. Labels show where something is a single run, an interpretation, or still pending.
Raw outputs are in `bench\out\` and the tools are in `bench\`.

## Setup

| Item | Value |
|---|---|
| CPU | Intel i5-1135G7 (4 cores, 8 threads, AVX-512 and VNNI) |
| RAM | 2 × 8 GB DDR4-3200 (Micron + SK Hynix). On paper the peak is 51.2 GB/s |
| SSD (V: and C:) | NVMe Intel SSDPEKNU512GZ (670p, QLC). V: and C: are on the same drive (disk 1) |
| D: | A different drive: HP SSD S700 (SATA, disk 0) |
| OS | Windows 11 Home, build 26300, Balanced plan |
| Power | On AC for every block (checked before each one; see `bench\out\power.log`). The battery was charging from 59 to 75 % |
| llama.cpp | build 12248, commit 3837bf0bc, Clang 20.1.8. It loads the `ggml-cpu-icelake` backend with REPACK on |
| Bench tools | Rust 1.99.1, release build with `target-cpu=native`. No external crates |

**The laptop was busy during the tests.** Other apps (Claude, Chrome, VS Code and others) were open, and only about 4 to 5 GB of RAM was free at the start.
Two downloads also ran the whole time:
- My Gemma 3n download, writing to V: at about 0.4 to 1 MB/s.
- The owner's archive.org download, writing to D:.

So the SSD numbers are **with background writes** on the same NVMe drive. Every test shows its spread, so you can see the noise.

### Model files

| File | Bytes | SHA-256 | Status |
|---|---|---|---|
| Qwen3.5-0.8B-Q4_K_M.gguf | 532,517,120 | bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517 | matches the unsloth repo |
| gemma-3-1b-it-Q4_K_M.gguf | 806,058,272 | 8270790f3ab69fdfe860b7b64008d9a19986d8df7e407bb018184caa08798ebd | matches the unsloth repo |
| Qwen3-4B-Instruct-2507-Q4_K_M.gguf (new, from unsloth) | 2,497,281,120 | 3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597 | matches the unsloth repo |
| gemma-3n-E2B-it-Q4_K_M.gguf (unsloth) | expected 3,026,881,888 | expected 189d42b4…dc415 | **Pending.** The download was still running at 20:43 (2.26 GB done). Not tested |
| Qwen3.5-2B-Q4_K_M.gguf | 867,934,208 (repo: 1,280,835,840) | c0555bc7… | **Incomplete.** llama.cpp says: `tensor 'blk.12.ffn_down.weight' data is not within the file bounds`. Not re-downloaded |

## 1. Hardware speed

### RAM bandwidth (`bench\target\release\membw.exe --size-mb 2048 --reps 5 --passes 4 --threads 1,2,4,8`)

This test used a 2 GiB buffer, 5 repeats and 4 passes per repeat. GB/s means 10⁹ bytes per second.

| Threads | Read GB/s, median (min–max) | Write (fill) GB/s | Copy GB/s (read + write bytes) |
|---|---|---|---|
| 1 | 20.5 (18.9–22.0) | 11.3 (11.0–11.4) | 28.0 (27.9–28.3) |
| 2 | 24.5 (21.8–32.3) | 10.7 (9.2–10.9) | 30.6 (29.3–31.6) |
| 4 | 34.5 (32.2–36.6) | 12.5 (11.8–13.1) | 29.9 (27.9–31.2) |
| 8 | 35.8 (35.1–36.5) | 11.1 (9.3–11.7) | 28.2 (27.2–28.5) |

- Read tops out at about **36 GB/s**, which is 70 % of the 51.2 GB/s paper peak.
- The first run had a bug: the compiler merged repeated passes, so it reported numbers 4 times too high. That run is kept as `membw_INVALID_cse_bug.txt` and is not used. The fix was `black_box`.
- A second read kernel (a 16-lane array version, shown as "read" in `membw.txt`) compiles to slow code (4.9 GB/s on 1 thread). It is ignored. "Read" above is the `readfold` kernel.

### SSD read speed on V: (unbuffered, with background writes)

Command: `diskbench.exe run <4 GiB file> --dur 5 --reps 3`.
- Reads use FILE_FLAG_NO_BUFFERING with page-aligned buffers, so the Windows file cache is bypassed.
- Queue depth (QD) is set with an I/O completion port.
- The 3 repeats are interleaved.
- MB/s means 10⁶ bytes per second.

| Pattern, block, QD | MB/s median (min–max) | IOPS | p50 latency µs | p99 latency µs |
|---|---|---|---|---|
| seq 1 MiB QD1 | 1544 (1406–1725) | 1472 | 617 | 1529 |
| seq 1 MiB QD32 | 1832 (1742–1863) | 1747 | 17919 | 38118 |
| seq 4 MiB QD1 | 1772 (1526–1780) | 423 | 2249 | 11605 |
| seq 4 MiB QD16 | 1875 (1872–1934) | 447 | 35342 | 68216 |
| rand 4 KiB QD1 | 30.5 (12.8–36.0) | 7445 | 91 | 298 |
| rand 4 KiB QD32 | 181 (171–193) | 44236 | 510 | 4875 |
| rand 64 KiB QD1 | 257 (252–269) | 3919 | 208 | 443 |
| rand 64 KiB QD32 | 931 (910–937) | 14213 | 2101 | 3192 |
| rand 1 MiB QD1 | 1175 (1164–1206) | 1120 | 847 | 1362 |
| rand 1 MiB QD16 | 1712 (1689–1735) | 1633 | 9246 | 24964 |

- Writing the 4 GiB test file (write-through, QD1, 4 MiB blocks) ran at only **68 MB/s**. This is a single run.
- The test file and the smoke file were deleted afterwards.

## 2. Model speed (llama-bench, CPU, Q4_K_M, 3 repeats each)

Command: `llama-bench -m <model> -p 512 -n 128 -t 1,2,3,4,8 -r 3 -o json`. Flash attention was left on auto.
"Effective GB/s" = file bytes × generation tokens/s. It shows how close each model gets to the RAM read wall (about 34.5 GB/s at 4 threads and 35.8 GB/s at 8).

| Model (file MB) | Threads | pp512 t/s, median (min–max) | tg128 t/s, median (min–max) | Effective GB/s | Share of measured RAM read |
|---|---|---|---|---|---|
| Qwen3.5-0.8B (533) | 1 | 131.5 (130.0–133.4) | 20.56 (20.36–21.77) | 10.9 | 53 % |
| | 2 | 236.0 (232.3–238.9) | 33.55 (33.41–33.90) | 17.9 | 73 % |
| | 3 | 242.1 (225.8–262.3) | 35.04 (34.98–36.57) | 18.7 | – |
| | 4 | 310.7 (275.6–313.3) | 43.23 (41.11–43.47) | 23.0 | 67 % |
| | 8 | 367.6 (366.2–371.2) | 46.65 (46.60–47.55) | 24.8 | 69 % |
| gemma-3-1b (806) | 1 | 62.7 (61.8–63.3) | 13.74 (11.89–13.99) | 11.1 | 54 % |
| | 2 | 103.7 (101.7–106.6) | 22.84 (22.15–23.67) | 18.4 | 75 % |
| | 3 | 121.3 (115.4–125.3) | 27.60 (27.47–27.85) | 22.3 | – |
| | 4 | 137.7 (127.3–143.6) | 30.68 (27.35–32.44) | 24.7 | 72 % |
| | 8 | 130.1 (112.5–163.1) | 12.26 (5.21–15.51) | 9.9 | unstable |
| **Qwen3-4B (2497)** | 1 | 22.7 (22.5–22.9) | 4.70 (4.28–4.76) | 11.7 | 57 % |
| | 2 | 41.0 (39.0–41.5) | 8.52 (8.50–8.62) | 21.3 | 87 % |
| | 3 | 50.9 (50.6–51.3) | 10.14 (9.95–10.50) | 25.3 | – |
| | 4 | 57.8 (53.9–58.2) | 11.23 (9.76–11.27) | 28.1 | 81 % |
| | 8 | 66.4 (65.7–69.5) | 13.03 (12.75–13.04) | 32.5 | 91 % |

Gemma 3n E2B is **pending**: its download was not finished.

## 3. Memory (Qwen3-4B, single runs: trimmed on the owner's request)

Command: `runmon.exe --csv … -- llama-completion.exe -m Qwen3-4B… -f <prompt> -c <ctx> -n 64 -no-cnv -t 4 -fa on -ctk <kv> -ctv <kv> -lm mmap --seed 42 --temp 0`.
- `runmon` samples every 100 ms. "Private" is private bytes, which equals commit charge. "Peak WS" is peak working set.
- The prompts are real text: the project docs plus the llama.cpp LICENSE file, in `bench\prompts\`.

| Context, load, KV | Prompt tokens | Peak private (= commit) MB | Peak working set MB | Prompt t/s | Gen t/s | Lowest free RAM on the PC |
|---|---|---|---|---|---|---|
| 2048, mmap, f16 (single run, from the aborted full matrix) | 1763 | 2437 | 4498 | 46.7 | 4.51 | 695 MB |
| 4096, mmap, f16 | 3688 | 2728 | 4790 | 34.0 | 2.12 | 2109 MB |
| 4096, mmap, q8_0 | 3688 | 2461 | 4523 | 11.4 | 3.84 | 2022 MB |

- **q8_0 KV saves 267 MB at 4096.** That matches the size of the f16 KV cache: 576 MiB, which llama.cpp itself reported as 603,979,776 bytes in the failed run below.
- **Memory is about 1.9 × the file, even with mmap.** A quick extra check (single run each, ctx 512, short prompt):

  | REPACK setting | Private MB | Peak working set MB | Gen t/s (4 tokens, rough) |
  |---|---|---|---|
  | REPACK on (default) | 2211 | 4201 | 12.2 |
  | `--no-repack` | 524 | 2518 | 9.7 |

  The CPU backend copies ("repacks") the weights into a faster private layout. The mapped file pages also stay resident, so the weights sit in RAM twice.
- Generation with a long prompt was much slower than llama-bench's tg128 (2.1 to 4.5 vs 11.2 t/s). **I did not find the cause.** These are single runs; deeper context and heat are possible reasons. This is a follow-up.
- Prompt speed with q8_0 KV (11.4 t/s) was 3 × slower than with f16. This is a single run and needs a repeat.
- The full matrix (contexts 2048 and 8192, mmap vs `-lm none`, f16 vs q8_0, 3 repeats each) was cut. One overlapping run happened by mistake; its results are in `bench\out\discarded_overlap\` and are not used.

## 4. Does the PC lag? (single runs, 60 s each)

Command: `probe.exe --secs 30` runs two 30-second phases:
- **Frame:** sleep 16 ms, then do a fixed CPU task. It records how late each wake-up was and how long the task took.
- **Membuf:** touch a 200 MB buffer once per pass and record the time per pass.

The model is Qwen3-4B on `llama-server -c 2048 -t T -fa on`, generating non-stop. `stream_client.py` timestamps every token.

| Condition | Wake lateness p50 / p95 / p99 / max ms | CPU task p50 ms (slowdown) | 200 MB pass p50 ms (slowdown) | Pass p99 ms | Model t/s in the window (frame phase / membuf phase) |
|---|---|---|---|---|---|
| Probe alone | 0.39 / 0.69 / 0.82 / 3.1 | 6.82 | 18.0 | 22.4 | – |
| + model at 4 threads | 0.34 / 0.58 / 0.73 / 22.5 | 9.03 (+32 %) | 28.8 (+60 %) | 33.3 | 7.50 (10.06 / 4.93) |
| + model at 2 threads | 0.38 / 0.75 / 5.05 / 73.7 | 7.98 (+17 %) | 25.9 (+44 %) | 41.0 | 4.99 (6.99 / 3.00) |

- Timer wake-ups barely change at the median. Only rare spikes appear, up to 22 to 74 ms.
- The memory-bound app gets 44 to 60 % slower. When that app is busy, the **model also halves its speed** (10.1 → 4.9 t/s), because both share the same RAM bandwidth.

## 5. "8 GB laptop" simulation: 2.5 GB cap (single runs)

Same command as in section 3 (ctx 4096, f16 KV, 3688-token prompt), run under `runmon` limits.

| Method | What it actually limits | Result | Prompt t/s | Gen t/s | Peak WS MB | Page faults |
|---|---|---|---|---|---|---|
| Hard max working set 2560 MiB (`--ws-max-mb 2560`, SetProcessWorkingSetSizeEx HARDWS_MAX), mmap | Resident RAM, including the mapped weight pages | **Works, but slow** | 41.4 | **1.20** (vs 2.12 uncapped) | 2560 | 20.9 M (vs 1.25 M uncapped) |
| Job Object commit limit 2560 MiB (`--commit-limit-mb 2560`), `-lm none` | Private commit only | **Fails at load** after 2.3 s: `failed to allocate buffer for kv cache` (603,979,776 bytes) | – | – | 2445 | – |

- **A commit limit does not count mmap'd file pages,** so with mmap it would not cap the weights. That is why the commit test used `-lm none`, which puts the weights in private memory.
- **The working-set cap is optimistic.** Pages pushed out of the working set stay in RAM on the standby list, because this PC has 16 GB. Each fault is cheap: the process read only 11.5 MB from disk. On a real 8 GB laptop many of those faults could go to the SSD, which would be slower still.

## What this means

- **The 4B model already runs at the memory wall.** At 4 threads it makes 11 words/s and moves about 28 GB/s, which is 81 % of what this RAM can deliver. Faster code alone won't help much. Reading fewer bytes per word will.
- **By default llama.cpp keeps the weights in RAM twice.** The 2.5 GB model used 4.5 to 4.8 GB of RAM. Turning off repacking cut that to about 2.5 GB, at roughly 20 % lower speed (rough, single run). This is the cheapest big win for an 8 GB laptop.
- **Under a 2.5 GB RAM cap the model still answers, but slowly:** 1.2 words/s instead of 2.1, even in this optimistic test.
- **The PC does not freeze, but it slows down.** A memory-heavy app got 44 to 60 % slower, and the model got 2 × slower at the same time. They fight over RAM speed, not CPU cores.
- **The SSD reads large blocks at about 1.5 to 1.9 GB/s but small random reads at only 30 to 180 MB/s.** Streaming weights from disk only works with big sequential reads, and even then it is about 20 × slower than RAM.

## Pending and not done

- Gemma 3n E2B: the download was not finished, so it has no llama-bench numbers yet.
- The memory matrix at contexts 2048 and 8192, and the mmap vs `-lm none` comparison: cut to save time.
- 3 repeats for sections 3 to 5: cut. These are single runs.
- The cause of slow generation with long prompts in llama-completion (2.1 to 4.5 vs 11.2 t/s).
