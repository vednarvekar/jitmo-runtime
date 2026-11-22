# 06. Low free RAM: loading, pacer v2 at 15% vs 20%, memory-pressure unload

Owner decision (22 Nov 2026): keep Docker running. Low free RAM is the real case on the 8 GB laptops we target, so test the runtime in it.

Measured on the owner's laptop on 22 Nov 2026, 13:45 to 14:23 IST, **on AC power** (checked before every run, `bench\out\power.log`).
i5-1135G7, 16 GB, Windows 11, llama.cpp build 12248, jitmo-runtime v0.2 (3 threads, ctx 4096, no repack).
Free RAM was logged once a second in every run (`bench\ramlog.py`, GlobalMemoryStatusEx). Raw data: `bench\out\lowram\`.

**Measured** = numbers from these runs. **Reasoned** = read from the code or inferred, not measured.

## Setup

- Docker Desktop's WSL VM (`vmmemWSL`) used 2.5 GB at 13:47 and about 1 GB by 14:25. Chrome and other apps were open as usual.
- Free RAM moved a lot during the session, from 2.7 GB to 4.5 GB, and later down to 87 MB.
- At 14:21:29 the owner started a game (F1 2020, 5.7 GB). Benches stopped after that.

| Run | Start (IST) | Free RAM at start |
|---|---|---|
| Idle CPU counter (no model) | 13:45 | not logged (about 2.7 GB at 13:47) |
| Load test, 4 models | 13:55 | 4439 MB |
| Probe runs, light threshold 15 | 13:57 | 4359 MB |
| Probe runs, light threshold 20 | 14:09 | 4145 MB |
| Background-load runs, threshold 15 | 14:15 | 3431 MB |
| Background-load runs, threshold 20 | 14:23 | 1062 MB (model refused) |

## Idle CPU on AC (measured)

The Windows total-CPU counter, 120 samples over 2 minutes, with nothing of ours running:

| Median | p10 | p90 | Mean | Max |
|---|---|---|---|---|
| 12.8% | 8.9% | 62.4% | 21.3% | 100% |

- The 12 to 13% idle figure from the battery runs holds on AC.
- Short spikes are common, and they lift the mean to 21%.

## Load behaviour (measured)

The runtime ran with its default budget (`reserve_free_mb = 1536`). "Real" is the child process right after a 128-token answer. `bench\run_lowram_load.sh`.

| Model | File MB | Result | Load estimate MB (private part) | Real working set MB | Real private MB | Estimate ÷ real working set | tok/s |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | 2382 | **refused** | 3981 (1600) | – | – | – | – |
| Gemma 3n E2B | 2887 | loaded in 3.1 s | 2663 (1216) | 1579 | 371 | 1.7× | 12.9 |
| gemma-3-270m router | 241 | loaded in 0.8 s | 1265 (1024) | 334 | 146 | 3.8× | 86.3 |
| Qwen3.5-0.8B | 508 | loaded in 1.5 s | 1596 (1088) | 714 | 656 | 2.2× | 33.5 |

- The refusal for Qwen3-4B said: 4437 MB free, the model needs about 3981 MB, and 1536 MB must stay free.
- The message is clear and says what to do ("Close some apps and try again").
- Free RAM fell by 1555 MB when Gemma 3n loaded and generated. That matches its working set.
- **The estimate is always too high, never too low.** The private part has a 640 MB fixed overhead plus the cap. The real private memory was 146 to 656 MB, while the private estimates were 1024 to 1600 MB.
- **It is too strict in practice.** At 14:17 to 14:22 Gemma 3n was refused with 1.8 to 2.6 GB free (estimate 2662 + 512 reserve). It really needs about 1.6 GB.
- The Qwen3-4B refusal at 4.4 GB free is probably right (**reasoned**). Its weights alone are 2.4 GB, and it needs room for the KV cache. This is unproven, because the forced test did not run.

## Pacer v2 at 15% vs 20% (measured, model Gemma 3n E2B)

Same method as [05-pacer-v2.md](05-pacer-v2.md): the probe is the "user app", with 20 s of a CPU task and then 20 s of passes over a 200 MB buffer. The model streams non-stop. Each condition has 3 repeats, and the table shows medians. The light threshold was set in the bench config only (`LIGHT_PCT`). `runtime/jitmo-runtime.toml` was not changed.

Probe alone: CPU task 6.74 ms, 200 MB pass 18.2 ms.

| Condition | CPU task slowdown | 200 MB pass slowdown | Model tok/s in the probe window | Level changes per 40 s window |
|---|---|---|---|---|
| No pacer | +24% | +54% | 7.89 | 0 |
| Threshold 15 | +16% | +41% | 6.14 | 3 / 3 / 5 |
| Threshold 20 | +19% | +34% | 6.69 | 3 / 5 / 6 |
| Threshold 15 + ~20% background load | +24% | +34% | 4.96 | 2 / 7 / 9 |
| Threshold 20 + ~20% background load | +31% | +34% | 4.54 | 4 / 3 / 6 |

Levels during the probe (share of the once-a-second samples):

| Condition | CPU phase | Memory phase |
|---|---|---|
| Threshold 15 | light 93–100% | light 25–54%, heavy 46–75% |
| Threshold 20 | none 21–36%, the rest light (one repeat heavy 46%) | light 54–100%, heavy 0–46% |
| Either threshold + background load | heavy 62–93% | heavy 75–92% |

- On AC the probe's CPU task is steady: 6.74 ms alone, against 14.89 ms on battery. Unlike on battery, the CPU-task column can be trusted.
- The battery anomaly, where the probe ran *faster* with extra load, is gone.
- With threshold 20 the model got more speed (6.69 vs 6.14 tok/s) and the memory-heavy app suffered less (+34% vs +41%). The CPU task was about the same (+19% vs +16%).
- **Battery vs AC, memory pass slowdown:** no pacer +71% battery vs +54% AC; adaptive +62% vs +41%; with background load +31% vs +34%. The models differ (Qwen3-4B on battery, Gemma 3n here), so this is a rough comparison only.
- **The probe still sets the level.** Probe plus the PC's own load is 16 to 31% CPU, which is above both thresholds. So 20 helps a little, but it does not keep a working PC at `none`.
- **Flapping is still there with the 5 s hold:** 3 to 9 level changes per 40 s window, mostly light ↔ heavy around the 40% line.
- The low-RAM rule (free RAM under 1024 MB gives heavy) **never fired** in the probe runs. Free RAM stayed at 1.3 GB or more.

### Why 15% gives a false "busy" at idle (reasoned, from `runtime/src/pacer.rs`)

- To go down a level, others' CPU must be `hysteresis_pct` (5 points) under the threshold.
- With 15, leaving light needs under 10% CPU. This PC idles at 12.8%, so one spike puts it at light and it tends to stay there. The battery runs fit this: 5 to 85% of the time at light with no load.
- With 20, leaving light needs under 15%, so an idle PC drops back to `none` after the 5 s hold.
- The background-load runs that would have measured this directly did not finish (see below).

## Background-load runs (mostly invalid)

The plan was no load, about 20% load and about 45% load (`bgload.py 4 0.9`), at 15 and 20, plus a no-pacer set. `bench\run_pacer2_bg.sh`.

- **Threshold 15, first repeat (valid but not comparable).** This ran from 14:15:45. Free RAM was 950 to 1260 MB, and others' CPU read 52 to 67% (median) even with no extra load.
  - With no load the model ran at 3.5 tok/s at the heavy level. With the 20% load it ran at 2.6 tok/s at the heavy level.
  - The reasons alternated between CPU and "N MB RAM free (< 1024 MB)", so the low-RAM rule did fire here.
  - The high CPU is likely Windows trimming and compressing memory (**reasoned**, not measured).
- **Threshold 15, repeats 2 and 3 (invalid).** The model had been unloaded for memory pressure (next section), and reloads were refused. The client got HTTP 503 and logged no tokens, and the pacer showed `none`.
- **Threshold 20 (not run).** The load was refused at 14:23 with 1062 MB free.
- **No pacer (not run).** It was not started, because the game had begun.

### A misleading reason text (measured)

While the heavy level holds because of RAM, the reason can read "1225 MB RAM free (< 1024 MB)". That is the 256 MB hysteresis margin at work. The level is right, but the message reads as if the rule was broken.

## Memory-pressure unload (measured, happened by itself)

This happened in the threshold-15 background-load run. The runtime used its default budget: below 1024 MB free twice in a row, finish the generation and then unload. Below 500 MB once, unload at once. Times are IST; the runtime log shows them in UTC.

| Time | Event |
|---|---|
| 14:15:45 | Gemma 3n loaded (4.1 s), generation starts, 3431 MB free at the start of the run |
| 14:15:45 to 14:16:40 | Free RAM falls to 1.0 to 1.3 GB, and the pacer goes to the heavy level |
| 14:16:40 | "memory pressure (969 MB free …): model may finish its generation, then it is unloaded" |
| 14:16:40 to 14:17:23 | The generation keeps running, with free RAM at 950 to 1000 MB |
| 14:17:23 | The bench client ends the stream |
| 14:17:24 | "unloading model now", then "unloaded (memory_pressure); child peak working set 1587 MB, peak private 380 MB" |
| 14:17:29 | Free RAM is back to 2580 MB |
| 14:17:31 to 14:22:09 | Every reload is refused ("not enough free RAM … 2584 MB free, the model needs about 2662 MB"), and clients get 503 |
| 14:21:29 | The game starts. Free RAM hits its low of 234 MB at 14:21:35, with no model loaded |

- The pressure path works as designed. It waited for the running generation, unloaded 1 s after it ended, and gave the RAM back. It said why at every step.
- **It waited 43 s while RAM stayed short.** This matches the known limitation in [06-runtime-api.md](../06-runtime-api.md): a long generation keeps its model loaded until free RAM drops under `critical_free_mb`.
- The critical rule (under 500 MB) never fired while a model was loaded, so it is still tested by unit tests only.
- The refusals after the unload were too strict (2.4 to 2.6 GB free, a 1.6 GB real need). See the load section.

## Not run

| Test | Why |
|---|---|
| Forced Qwen3-4B ("would it freeze?") | When it was due, free RAM was 87 MB (the game). The safety floor is 300 MB. Script ready: `bench\run_lowram_forced.sh` (reserve 0, kill under 300 MB, hard faults logged) |
| Background-load runs at 20%, repeats 2 to 3 at 15%, no-pacer set | The model was unloaded for pressure, then refused. The game started. Owner: no more benches for now |
| Pressure test on purpose (`bench\memhog.py`) | Not needed. The real event above covered the finish-then-unload path |
| Critical unload (under 500 MB mid-generation) | Not reached with a model loaded |

## Recommendations

1. **Make the load estimate less strict.** Lower or scale the 640 MB fixed overhead, and count only the KV cache plus buffers as private. As it is, Gemma 3n and the 0.8B are refused when they would fit, which matters on 8 GB laptops.
2. **Set `light_cpu_pct` to 20.** It measured better than 15 on the memory-heavy app (+34% vs +41%), with more model speed. By the code, 15 keeps an idle PC at the light level. It is the owner's call; the default is unchanged.
3. **Reduce light ↔ heavy flapping.** It still changes 3 to 9 times per 40 s at the 40% line. Options: a longer hold from heavy to light, or more hysteresis there.
4. **Fix the RAM reason text** so it does not claim "< 1024 MB" when free RAM is above that during hysteresis.
5. **Consider a time limit on "finish, then unload"** under pressure. Here it waited 43 s with under 1 GB free.
6. **Run the forced Qwen3-4B test and the remaining background-load sets** when the PC is not in use.
