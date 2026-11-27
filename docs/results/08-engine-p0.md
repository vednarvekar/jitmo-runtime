# 08. In-process engine P0 and the Iris Xe GPU (28 Nov 2026)

Design: [08-inproc-engine.md](../08-inproc-engine.md). Command: `jitmo-runtime.exe --engine-test --model <gguf> [--lib-dir <llama build>] [--ngl 99] ...`.

**Caveat:** measured while a jitmo bake-off was using the CPU and about 6 GB of RAM. CPU numbers swing a lot (2 to 31 tok/s on the same test). Re-measure on a quiet PC.

## P0 works

| Check | Result |
|---|---|
| Load llama.dll b12606 (CPU and Vulkan release zips), ABI check of default params | Pass (my first check had a wrong expected value; the layout matched) |
| Chat template (Gemma 3 270M, built-in template) | Same answer as llama-server |
| Grammar `root ::= [0-3]` | Picks `3` (Play music) for "play some lofi", 1 token |
| Pause hook between tokens (20 ms) | 400 ms paused over 20 tokens, exactly as asked |
| Generation speed vs llama-bench, CPU, 3 threads, no repack | Same within noise: 30.7 vs 29.8 tok/s in the quietest round |

## Iris Xe (Vulkan, all layers), Gemma 3 270M, 245-token jitmo-style pick prompt

Interleaved runs CPU / Vulkan / CPU / Vulkan, warm (first run dropped):

| Device | Prompt reading | Generation |
|---|---|---|
| CPU, 3 threads | 712 to 961 ms | 15 to 28 tok/s |
| Vulkan Iris Xe | **161 to 169 ms** (about 4.5× faster) | **43 to 45 tok/s**, steady |

- The GPU is also steadier because it doesn't fight the busy CPU.
- The first GPU request is slow (about 4 s: shader compile). Warm it up at load.
- The Vulkan compute buffer is 521 MB at batch 2048. It comes out of shared system RAM; a smaller `n_ubatch` should shrink it. Not measured yet: process RAM and the whole-PC impact.
- Timing needs `llama_synchronize` after the prompt: without it a GPU decode returns early (it showed a false 7 ms).

## Next

- Re-measure on a quiet PC, with Gemma 3n E2B and a 1–2B model.
- RAM of the Vulkan path (private bytes, shared GPU memory) and UI lag while it runs.
- P1: `engine = "inproc"` per model in the pool.
