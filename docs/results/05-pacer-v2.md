# 05. Pacer v2: adaptive levels from CPU and RAM

Owner decision (22 Nov 2026): drop the keyboard/mouse signal; base "busy" on other programs' CPU (smoothed over 1 to 2 s) and free RAM; adaptive cap instead of a fixed 50%; hysteresis; room for a later memory-bandwidth signal. Design: [06-runtime-api.md, Pacer](../06-runtime-api.md#pacer). Code: `runtime/src/pacer.rs`.

Measured on the owner's laptop on 22 Nov 2026, 11:48 to 12:20 IST, **on battery** (47% → about 35%, `bench\out\power.log`). v1 (results/03) was measured on AC, so **absolute numbers are not comparable with v1**; compare within this file.
i5-1135G7, 16 GB, Windows 11, llama.cpp build 12248, Qwen3-4B Q4_K_M through jitmo-runtime v0.2 (3 threads, ctx 4096, no repack). Other apps were open; about 6 GB free.
Scripts: `bench\run_pacer2.sh` (probe runs, 3 repeats per condition), `bench\run_pacer2_bg.sh` (light load, no probe), summaries `bench\summarize_pacer.py out\pacer2`, `bench\summarize_pacer2.py`, `bench\summarize_pacer2_bg.py`. Raw data: `bench\out\pacer2\`, `bench\out\pacer2_hold5\`.

Method as in v1: the probe (`probe.exe --secs 20`) is the "user app", 20 s of a frame loop with a fixed CPU task, then 20 s of passes over a 200 MB buffer (memory-bound); the model streams 1500 tokens non-stop and the probe starts 8 s in. The pacer's state was read once a second (a small `curl | python` poller, which itself uses some CPU, in every condition).

## Probe runs (median of 3 repeats)

Levels: share of the once-a-second samples in each probe phase.

| Condition | CPU task p50 ms (vs alone) | 200 MB pass p50 ms (vs alone) | Pass p99 ms | Model tok/s in window (frame / membuf) | Pacer level during the probe |
|---|---|---|---|---|---|
| Probe alone | 14.89 | 18.40 | 22.5 | – | – |
| No pacer (`mode = "off"`) | 23.17 (+56%) | 31.50 (**+71%**) | 55.2 | **5.26** (6.75 / 3.80) | (off; others' CPU 21–36%) |
| Adaptive (v2 defaults, hold 2 s) | 26.77 (+80%) | 29.73 (**+62%**) | 33.8 | **4.08** (5.14 / 3.25) | frame: light 80–90%; membuf: light 40–82%, heavy 18–60% |
| Adaptive + light background load (~20% CPU) | 10.61 (−29%) | 24.09 (**+31%**) | 28.6 | **3.07** (3.45 / 2.80) | mostly heavy (others' CPU 28–41%) |

- **The CPU-task column cannot be trusted on battery.** With the extra background load the probe's CPU task was 29% *faster* than with the probe alone. The likely cause (not proven) is the battery power plan: the CPU clock rises with total load, so a busier PC runs the probe's task at a higher clock. The memory-bound pass depends much less on the clock, so it is the column to read here.
- **The memory-heavy app** went from +71% slower (no pacer) to +62% at the light level (75% cap) and +31% at mostly the heavy level (50% cap). v1 on AC measured +2% under a 50% cap; on battery the 50% cap helps much less.
- **The model paid** 5.26 → 4.08 tok/s at mostly light, 3.07 tok/s at mostly heavy (with the extra load competing too).
- **The probe itself sets the level:** it keeps one of 8 logical CPUs busy (12.5%), and the PC's own background is about 12%, so "probe + background" is 21–36%: the light level. The 40% line was crossed now and then in the memory phase.
- **Level changes:** 5 to 7 per 40 s probe window with the first default (`release_after_ms = 2000`). That is a change every 6 to 8 s, mostly light ↔ heavy around 40%.

## Light background load alone (no probe)

`bgload.py 2 0.8` (two processes busy 80% of every 10 ms = about 20% of the CPU) while the model generates. 20 samples over about 29 s per run, after 4 s to settle.

| Hold (`release_after_ms`) | Load | Model tok/s (3 runs) | Others' CPU, median | Levels (share of samples) | Level changes per run |
|---|---|---|---|---|---|
| 2000 | none | 7.60 / 6.87 / 6.12 | 11.7–13.2% | none 20–95%, rest light | 1 / 4 / 3 |
| 2000 | ~20% | 5.55 / 3.55 / 4.93 | 31–50% | light 70–85% (run 2: heavy 85%) | 3 / 1 / **10** |
| **5000 (new default)** | none | 6.85 / 6.63 / 6.12 | 11.8–13.4% | none 15–55%, rest light | 4 / 4 / 1 |
| **5000 (new default)** | ~20% | 4.99 / 5.33 / 5.17 | 32.6–32.8% | **light 80%**, heavy 20% | **2 / 2 / 1** |

- **The light level (75% cap) kicks in under a light load**, as intended: about 80% of the time with the 20% load (others' CPU 33% including the PC's own 12%).
- The model then runs at 5.0 to 5.3 tok/s instead of 6.1 to 6.9 (about −22%).
- **Flapping:** with a 2 s hold one run changed level 10 times in 29 s. With a 5 s hold, 1 to 2 changes per run under load. **`release_after_ms` is now 5000 by default.**
- **The 15% threshold is close to this laptop's own background** (12 to 13% with only the usual apps and the poller). With no extra load the pacer spent 5 to 85% of the time at the light level. A threshold of 20% would keep an idle PC at `none`; the owner's numbers were kept as defaults.

## What this means

- The adaptive pacer does what was asked: no cap on a quiet PC, 75% under a light load, 50% under a heavy one, with a reason in `/stats`, and with a 5 s hold it no longer flaps.
- **On battery the CPU cap helps a memory-heavy app much less than on AC** (+31% slowdown at 50% vs +2% in v1). Run `bench\run_pacer2.sh` again on AC to compare with v1 before tuning the levels.
- CPU % is a weak proxy for what hurts the user app most here, RAM bandwidth: the memory-heavy probe phase raised others' CPU only to 28–37%, mostly below the heavy line. The planned RAM-latency signal (`mem_latency_ns` in `/stats`, one rule in `proposals()`) is the next step.

## Not tested

- The same runs on AC power (the v1 conditions).
- A real foreground app (browser, IDE) instead of the probe.
- The RAM signal of v2 (free RAM below `pressure_free_mb` → heavy): covered by unit tests only.
- `busy_priority = "idle"` with v2.
