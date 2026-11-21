# 04. Plan: how we move forward

Principle: **measure first, then build.** Every step ends with real numbers from the owner's laptop.

## Steps

| Step | What | Done when |
|---|---|---|
| **0. Save the work** | Docs into git, pushed to https://github.com/vednarvekar/jitmo-runtime | Repo has the docs |
| **1. Baseline** | Measure laptop RAM and SSD speed. Run 2 to 4B models with plain llama.cpp. Record speed, RAM use and how much the PC lags. | [results/01-baseline.md](results/01-baseline.md) has the numbers |
| **2. Ready tricks** | Apply the known tricks (below) one by one and measure each. | We know what each trick gives on this laptop |
| **3. Decide what to build** | Pick the first new part from the real numbers. Likely candidate: "don't freeze the PC" scheduling. | The owner chooses |

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
