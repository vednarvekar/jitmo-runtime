# 03. Pacer v1: "don't freeze the PC"

Measured on the owner's laptop on 21 Nov 2026, 22:15 to 22:34 IST, on AC power (`bench\out\power.log`).
i5-1135G7, 16 GB, Windows 11, llama.cpp build 12248. Qwen3-4B Q4_K_M through jitmo-runtime v0.1 (3 threads, ctx 4096, no repack). Other apps were open; 3.2 to 4.0 GB free.
Script: `bench\run_pacer.sh` (3 repeats per condition). Raw data and per-repeat numbers: `bench\out\pacer\`, summary `bench\out\pacer\summary.txt` (`bench\summarize_pacer.py`).
An earlier attempt ran three copies of the script at once by mistake; its data is in `bench\out\discarded_pacer_overlap\` and is not used.

## Which lever

| Lever | Works on a running llama-server? | Finding |
|---|---|---|
| Threads per request | No | llama-server has no per-request thread setting (`-t`/`-tb` are start-up flags). Changing threads means a restart = a model load (3 s). |
| Pause between tokens | No | The slot keeps generating into its queue; reading the stream slowly does not slow the compute. The only per-request limit is `t_max_predict_ms`, which ends the answer. |
| Priority class | Yes | Measured below (`idle priority`): no effect here. |
| Job Object CPU rate hard cap | Yes | Set and removed on the child's existing job with `SetInformationJobObject`, no restart. **The v1 lever.** |

## Method

The probe (`bench\target\release\probe.exe --secs 20`) is the "user app": 20 s of a frame loop (sleep 16 ms, then a fixed CPU task) and 20 s of passes over a 200 MB buffer (memory-bound). The model streams 1500 tokens non-stop through the runtime (`stream_client.py --chat`); the probe starts 8 s after the stream.
Model tok/s is counted from the stream timestamps inside the probe's 40 s window.

| Condition | Pacer config |
|---|---|
| Probe alone | no model |
| No pacer | `mode = "off"` |
| Cap 50% | `mode = "always"`, `busy_cpu_pct = 50` (hard cap 18.75% of the CPU) |
| Cap 25% | `mode = "always"`, `busy_cpu_pct = 25` (9.4%) |
| Idle priority | `mode = "always"`, `busy_cpu_pct = 100` (no CPU cap), `busy_priority = "idle"` |
| Auto (v1 default) | `mode = "auto"`, defaults |

## Results (median of 3 repeats)

| Condition | CPU task p50 ms (slowdown) | 200 MB pass p50 ms (slowdown) | Pass p99 ms | Model tok/s in window (frame / membuf phase) |
|---|---|---|---|---|
| Probe alone | 6.85 | 20.14 | 25.1 | – |
| No pacer | 8.96 (+31%) | 25.94 (+29%) | 34.3 | 5.34 (6.75 / 3.95) |
| **Cap 50%** | 7.22 (+5%) | 20.51 (+2%) | 29.4 | **3.17** (4.05 / 2.30) |
| Cap 25% | 7.15 (+4%) | 20.62 (+2%) | 30.1 | 1.55 (2.00 / 1.10) |
| Idle priority | 8.81 (+29%) | 25.88 (+29%) | 31.1 | 5.71 (7.35 / 4.05) |
| **Auto** | 7.21 (+5%) | 21.51 (+7%) | 30.9 | 3.17 (4.15 / 2.20) |

- The three repeats agree closely (pass p50 per repeat: no pacer 27.6 / 25.9 / 25.1 ms; cap 50% 20.5 / 20.2 / 20.7 ms; per-repeat values in `summary.txt`).
- Timer lateness (p99 0.7 to 0.9 ms) did not change in any condition; rare spikes up to 32 ms happened in all conditions, including some with the cap.
- **Auto mode capped the whole window in all 3 repeats.** The trigger was the CPU signal ("other processes use 10 to 43% CPU"): the probe itself uses one of 8 logical CPUs (12.5%). The owner did not touch the PC during these windows (last input 34 to 215 s before the end of each window), so the input signal was not tested here. Between repeats, with only the stream running, auto mode removed the cap (the runtime log shows 9 cap on/off changes over the run).
- The baseline (llama-server alone, 4 threads) measured +60% for the memory-bound pass and +32% for the CPU task. With the runtime's 3 threads it is +29% and +31% today.

## What this means

- **A 50% CPU cap removes almost all of the lag**: the memory-heavy app goes from 29% slower to 2% slower, the CPU task from 31% to 5%.
- **The model pays for it**: 5.3 → 3.2 tok/s while capped (−40%). Uncapped it returns to full speed at once.
- **25% gives no extra benefit** for the user app and halves the model again (1.6 tok/s). 50% is the better default.
- **Lowering the priority does nothing here**: the model and the app do not fight for CPU cores (3 + 1 threads on 8 logical CPUs), they fight for RAM bandwidth. Priority only decides who runs when cores are short.
- **Auto mode works for a busy app** (CPU signal), but its 10% threshold is crude: on this PC the background alone measured 6 to 7% between runs, and one busy app thread crosses it. The keyboard/mouse signal is implemented but was not exercised in this test.

## Not tested

- The input signal (GetLastInputInfo) during a real typing session.
- Prompt reading (prefill) under the cap; only generation was measured.
- A foreground app that is busy but below 10% CPU (for example a browser scrolling).
- The 8 GB laptop.
