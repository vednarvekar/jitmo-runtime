# 02. Research: what others are doing

Web research, 18 Nov 2026. Numbers are what each paper's authors report, usually on hardware different from ours. Treat them as hints, not promises.

## The big picture

1. **Everything comes back to memory bandwidth.** Writing each word means reading the weights. Every serious approach either reads fewer bytes, reads them from a faster place, or hides the waiting.
2. **The PC freezes from memory pressure and bandwidth fights, not mainly from CPU.** Recent work (SERENO, OSDI '26, and an Apple Silicon study) shows that a background LLM badly hurts foreground apps, while the LLM itself barely slows down.
3. **Most results are on GPUs or phones.** Very little targets an 8 GB Windows laptop running on CPU or integrated GPU. That gap is ours.

## Approaches, grouped

### 1. Smaller weights (quantization)

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| 4-bit (GGUF Q4 and similar) | 4 bits per weight | Standard; quality close to full precision | **Proven.** Our baseline. |
| QuIP#, AQLM, QTIP (2024) | Clever 2 to 3-bit codes | QTIP 2-bit on Llama 2 70B: C4 perplexity 5.48 (QuIP# 5.71, AQLM 5.62) | **Promising but unproven for 2 to 4B.** Mostly tested on 7B to 70B. |
| BitNet b1.58 2B4T + bitnet.cpp (Microsoft, 2025) | Model trained with weights of only -1, 0, +1 | ~0.4 GB non-embedding memory; 2.37× to 6.17× faster on x86 CPUs; quality competitive with other 2B models | **Very effective, few models.** Only works with models trained this way, and only fast with bitnet.cpp. |

### 2. Read only what each word needs (sparsity)

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| Deja Vu (ICML 2023) | Predicts which heads and neurons each word needs and skips the rest | OPT-175B: over 2× faster than FasterTransformer, no quality loss | **Effective on big ReLU models.** Less tested on small modern models. |
| PowerInfer (SOSP 2024) | "Hot" neurons on the fast chip, "cold" ones on the CPU | Up to 11.69× over llama.cpp (Falcon-40B ReLU), 2.89× average for INT4 models, RTX 4090 | **Effective only on sparse (ReLU) models.** Standard SiLU models gain little. |
| ProSparse (2024) | Retrains models to be sparse (~89% sparsity) | Up to 4.52× with PowerInfer | **Makes sparsity work, but needs retraining.** |
| TEAL (2024) | Training-free sparsity on any model | 1.53× at 40% and 1.8× at 50% sparsity; works with quantization | **Practical.** No retraining, modest gain. |

### 3. Keep the model on the SSD, load parts (offloading)

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| LLM in a Flash (Apple, 2023) | Weights on flash, load only needed rows; windowing and row-column bundling | Models up to 2× the DRAM size; 4 to 5× faster than naive loading on CPU | **Works, depends on sparsity.** Useful for keeping RAM use low. |
| PowerInfer-2 (2024) | Flash offloading plus sparsity on phones; NPU for dense work, CPU for sparse | TurboSparse-Mixtral-47B at 11.68 words/s on a phone; ~40% less memory for models that fit | **Strongest proof of the idea on weak devices.** Needs sparse models. |
| Mixtral offloading (2023), ProMoE, Fiddler | Cache and prefetch MoE experts | Mixtral at 2 to 3 words/s on consumer hardware; ProMoE 2.84× decode over older offloading | **For MoE models only.** Transfer speed is the limit. |
| FlexInfer, MNN-LLM | Async prefetch, overlapping reads with compute; DRAM plus flash storage | Survey summaries only | **Same idea, to check.** |

### 4. More words per weight read

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| Speculative decoding (Google, DeepMind, 2022 to 2023) | Small model drafts, big model verifies; output is identical | 2 to 3× on GPUs | **Unclear on small local models.** An MLX test found little gain for small 4-bit models. Needs a same-family draft. |
| EAGLE-3 (2025) | Trained draft head instead of a draft model | 3.0 to 3.4× on Llama-3.3-70B, H100 | **GPU-proven, model-specific.** Unknown on CPU. |

### 5. Stop the chip waiting

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| Megakernels, "Look Ma, No Bubbles" (Stanford Hazy Research, May 2025) | Fuse the whole Llama-1B forward pass into one GPU program, no idle gaps between steps | 78% memory-bandwidth use on H100, over 1.5× faster than existing systems | **The idea applies, the code does not.** Datacentre GPUs only. |

### 6. Smaller working memory (KV cache)

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| KIVI (ICML 2024) | 2-bit KV cache | 2.6× lower peak memory on Llama-2-7B | **4-bit KV is safe. 2-bit hurts newer models** (Kitty, 2025: big drops on Qwen3-8B and Llama3-8B in maths). |
| llama.cpp `--cache-type-k/v` | 8-bit or 4-bit KV cache | q8 near lossless, q4 noticeable (anecdotal) | **Available today.** |

### 7. Models designed for low RAM

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| Gemma 3n (Google, 2025) | Per-Layer Embeddings kept off fast memory; MatFormer lets you cut sub-models | 5B and 8B raw parameters running in ~2 GB and ~3 GB | **Very relevant.** The model itself is built for our problem. |

### 8. Other chips

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| Laptop NPUs (Intel OpenVINO, AMD Ryzen AI, Qualcomm) | Run models on the AI chip | Intel NPU 18.55 words/s in MLPerf Client; mostly vendor claims | **Good for low power, not speed.** Decode is still bandwidth-bound; many tools skip the NPU for the iGPU. |

### 9. Not freezing the PC

| Work | What it does | Reported result | Verdict for us |
|---|---|---|---|
| SERENO (OSDI '26) | LLM yields memory bandwidth to foreground apps on demand (phones) | Foreground apps suffer badly from LLM co-running; LLM throughput barely changes | **Directly on our "never freeze" goal.** |
| "The Gap Is the Loop" (NeurIPS 2026, Apple Silicon) | Measures background work against LLM decode | Bandwidth contention slows decode by 72% when saturated; lowering priority can make it worse | **Warns that priority settings alone don't fix freezing.** |
| arXiv 2505.15030 | Largest model that keeps a consumer device responsive while multitasking | Trade-offs between quantization and model size | **To read fully.** |

### 10. Practical tricks with today's tools

| Trick | What it does | Verdict for us |
|---|---|---|
| **Shared base + LoRA adapters** | One base model in RAM; tiny 10 to 50 MB adapters per job (router, planner, tool-caller, writer). Swapping takes milliseconds. | **Very effective.** The real version of "many small models". |
| **Fewer tokens** | Cost = weights × tokens. Short, structured output (for example a 20-token JSON tool call) cuts work directly. | **Free and effective.** |
| **KV and prompt caching** | Past tokens and shared prompt prefixes are not recomputed. | **Effective.** In llama.cpp today. |
| **Early exit / layer skipping** (LayerSkip) | Easy tokens leave after a few layers. | **Needs a model trained for it.** |

### What today's engines support

| Technique | llama.cpp | ONNX Runtime |
|---|---|---|
| mmap lazy loading | Yes | Partial |
| LoRA hot-swap | Yes, per request in the server | Yes, less flexible |
| KV / prompt cache save and restore | Yes | Manual |
| Speculative decoding | Yes | No, build it yourself |
| Grammar-constrained output | Yes | Limited |
| MoE models | Yes | Only if exported that way |
| Early exit | No | Only if baked into the exported graph |
| KV cache quantization | Yes (`--cache-type-k/v`) | Limited |
| "Don't freeze the PC" mode | No | No |

llama.cpp is the stronger starting point. Neither engine protects the rest of the PC.

## What is actually effective (for a 2 to 4B model on an 8 GB laptop)

| Works today | Works with conditions | Not for us (yet) |
|---|---|---|
| 4-bit weights | Sparsity (needs sparse or retrained models) | Megakernel code (GPU only) |
| 4 to 8-bit KV cache | Flash offloading (needs sparsity to be fast) | MoE offloading (our target models are dense and small) |
| Low-RAM model designs (Gemma 3n, BitNet) | 2 to 3-bit weights (small-model quality unproven) | NPU for speed |
| Bandwidth-aware scheduling (SERENO idea) | Speculative decoding (unclear gain on small models) | |

## Gap nobody fills

No single engine combines low-bit weights, sparsity, KV compression, partial loading and "never freeze the PC" scheduling, built for 8 GB Windows laptops on CPU and integrated GPU.

## Sources

- LLM in a Flash: https://arxiv.org/abs/2312.11514 · https://machinelearning.apple.com/research/efficient-large-language
- PowerInfer: https://arxiv.org/abs/2312.12456 · https://ipads.se.sjtu.edu.cn/_media/publications/song-sosp24.pdf
- PowerInfer-2: https://arxiv.org/abs/2406.06282
- Deja Vu: https://proceedings.mlr.press/v202/liu23am.html · https://github.com/FMInference/DejaVu
- ProSparse: https://arxiv.org/abs/2402.13516
- TEAL: https://arxiv.org/abs/2408.14690
- BitNet b1.58 2B4T: https://arxiv.org/abs/2504.12285 · https://techcrunch.com/2025/04/16/microsoft-researchers-say-theyve-developed-a-hyper-efficient-ai-model-that-can-run-on-cpus
- QuIP#: https://arxiv.org/abs/2402.04396 · AQLM: https://arxiv.org/abs/2401.06118 · QTIP: https://arxiv.org/abs/2406.11235
- Megakernels: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- Speculative decoding and EAGLE-3: https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding · https://github.com/ml-explore/mlx-lm/discussions/890
- Mixtral offloading: https://arxiv.org/abs/2312.17238 · ProMoE: https://arxiv.org/abs/2410.22134 · CPU-GPU MoE (vs Fiddler): https://arxiv.org/abs/2512.16473
- KIVI: https://arxiv.org/abs/2402.02750 · Kitty: https://arxiv.org/abs/2511.18643
- Gemma 3n: https://developers.googleblog.com/en/introducing-gemma-3n/
- Laptop NPUs: https://newsroom.intel.com/client-computing/intel-achieves-first-only-full-npu-support-mlperf-client-v0-6-benchmark · https://www.tomshardware.com/pc-components/cpus/ryzen-ai-300-takes-big-wins-over-intel-in-llm-ai-performance-up-to-27-percent-faster-token-generation-than-lunar-lake-in-lm-studio
- SERENO: https://www.usenix.org/system/files/osdi26-xin.pdf
- The Gap Is the Loop: https://neurips.cc/virtual/2026/171323
- Consumer device responsiveness: https://arxiv.org/abs/2505.15030
- Edge LLM inference survey: https://www.sciopen.com/article/10.26599/TST.2025.9010166
