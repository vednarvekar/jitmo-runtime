# 07. Roadmap: from v0.2 to a ready runtime (v1.0)

Written 22 Nov 2026. The owner decides the order; status is updated as steps finish.
Done so far: v0, v0.1 and v0.2 (see [04 Plan](04-plan.md), tracking table).

## Steps to v1.0 ("ready")

| # | Step | What it means | Done when | Status |
|---|---|---|---|---|
| 1 | **v0.3 fixes** | Five fixes from the low-RAM tests: a load estimate close to real use, "busy" threshold 20%, less pacer flapping, clearer reason text, a 15 s limit on "finish, then unload" | Tests pass, docs updated | Done 22 Nov (49 tests; not yet re-measured under real low RAM) |
| 2 | **Finish the low-RAM tests** | The forced Qwen3-4B "does it freeze?" test, plus the background-load pacer runs that were cut short (see [results/06](results/06-low-ram.md)) | Numbers in `docs/results/` | Done 22 Nov, see [results/07](results/07-v0.3-low-ram.md): forced 4B hit the 300 MB stop during load (refusal was right); 15 s pressure limit and the critical unload work; flapping 0 to 2 per 40 s. Runs overlapped a jitmo audit, so 20% vs 15% is still not measured on a quiet PC |
| 3 | **Define "ready" in numbers** | For example: a 4B model at 5 tok/s or more on 8 GB, other apps slow by 30% at most, never under 300 MB free | Owner sets the numbers in [00 Vision](00-vision.md) | Owner to decide |
| 4 | **Pick the base model** | Compare Gemma 3n E2B and Qwen3-4B on answer quality. Gemma 3n already uses half the RAM and is about 1.5× faster. | Quality numbers and an owner choice | Test set ready (96 items, [quality-test.md](quality-test.md)); not run yet |
| 5 | **Speed tricks** | Step 2 of the plan, not done yet: KV cache in 8 bits, prompt caching, speculative decoding / MTP | Each trick measured on the laptop | Not started |
| 6 | **Real user test** | jitmo agent runs its test sets (held-out 262, day-in-the-life 119, audit 712) on the runtime and compares them with its own backend | Same pass counts; time and RAM compared | Waiting on jitmo agent |
| 7 | **8 GB laptop test** | Run on a real 8 GB laptop, or a strict 8 GB simulation on this one | Numbers in `docs/results/` | Not started |
| 8 | **Reliability** | A run lasting several hours, automatic restart if llama-server crashes, clean logs | No crash and no memory growth over the long run | Not started |
| 9 | **Packaging** | One download with llama.cpp included. It picks a model that fits the laptop's RAM. | Installs and runs on a clean PC | Not started |

When all nine steps are done, the runtime is **v1.0**.

## After v1.0: going beyond llama.cpp

These are the research paths from [02 Research](02-research.md). Each one starts only once measurements show llama.cpp is the limit.

- **SSD offloading** (LLM in a Flash, PowerInfer-2): bigger models than the free RAM.
- **Sparsity** (Deja Vu, PowerInfer): read fewer weights for each word.
- **Base model + LoRA hot-swap at scale**: many jobs for the RAM of one model.
- **Our own engine parts**, where llama.cpp can't do what we need.
