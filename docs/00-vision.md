# 00. Vision

## One line

**Run useful AI models locally on ordinary laptops, smartly and cheaply, so people can build on them instead of paying for cloud LLMs.**

## The problem

- Cloud LLMs cost a lot of money for every request.
- Local models are too heavy today. On an 8 GB laptop they eat RAM and CPU, and the PC can freeze.
- So people keep paying for the cloud, even for small tasks a local model could do.

## What we want

1. **2 to 4 billion parameter models running well on an 8 GB RAM laptop.** Not trillion-parameter models. Small, useful ones.
2. **The PC never freezes.** It may slow down a little while the model works. It must never stop everything else.
3. **Several models when needed**, without each one costing a full share of RAM.
4. **Light on resources.** RAM, CPU and power are used smartly and efficiently. RAM is used, not avoided.
5. **A base to build on.** Assistants and tools run on it locally instead of calling the cloud.

## What this is not

- Loading and unloading models fast was a **starting idea**, not the goal. We use whatever approach works best.
- Not a cloud service. Everything runs on the device.
- Not a new model. We run open models, and may convert or compress them.

## Attitude

- **Effort is not a limit.** If we must rebuild something like llama.cpp or ONNX Runtime from scratch, we do it.
- **New approaches are welcome.** We study current research and go beyond it where we can.
- **Measure, don't guess.** Every claim gets a number from a real 8 GB laptop.

## Research to learn from

The main problem across this research is the **memory wall**: the chip waits on memory more than it computes. It is the same problem whether the memory is HBM on a datacentre GPU or RAM on a laptop.
Full findings, numbers, verdicts and sources: [02 Research](02-research.md).

| Work | Who, when | Core idea | Why it matters to us |
|---|---|---|---|
| **LLM in a Flash** | Apple, 2023 | Keep the model on flash storage (SSD). Load only the rows a word needs. Reuse recently loaded rows ("windowing"). Store rows that are read together side by side ("row-column bundling"). | Runs models bigger than the RAM you have. |
| **PowerInfer** | Shanghai Jiao Tong Univ., 2023 | A few "hot" neurons fire for most words. Keep them on the fast chip and leave the "cold" ones on the slower one. | Keep only the useful part of the model in fast memory. |
| **PowerInfer-2** | Shanghai Jiao Tong Univ., 2024 | The same idea on smartphones, with neurons streamed from storage. Ran a large MoE model on a phone. | Shows this works on weak, low-RAM devices. |
| **Deja Vu** | Liu et al., ICML 2023 | "Contextual sparsity": each word needs only part of the model. A small predictor guesses which part and skips the rest. | Read fewer bytes per word, so it runs faster. |
| **BitNet b1.58** | Microsoft, 2024 (2B model in 2025, runs with `bitnet.cpp`) | Weights are only -1, 0 or +1 (about 1.58 bits). Multiplications turn into additions. | A 2B model in about 0.4 GB, fast on a plain CPU. |
| **Speculative decoding** | Google and DeepMind, 2022 to 2023 | A tiny model drafts several words. The big model checks them all in one pass. The output is identical to the big model's. | One read of the weights gives several words. |
| **Megakernels ("Look Ma, No Bubbles")** | Stanford Hazy Research, 2025 | Fuse the whole model into one GPU program, so the chip never sits idle between steps waiting on memory. Related: ThunderKittens. | Cut the waiting time, not just the work. |
| **FlexGen** | Stanford and others, 2023 | Splits weights and KV cache across GPU, CPU RAM and disk, scheduled for the best throughput. | Plan which data lives where. |
| **MoE offloading** | Eliseev and Mazur, 2023 | Runs Mixtral on small GPUs by caching recently used experts and guessing the next ones. | Cache and prefetch the parts of the model in use. |
| **KV cache compression** | Many papers (for example KIVI, 2024) | Store the model's memory of earlier words in fewer bits. | Long prompts stop eating RAM. |

## Open questions (owner to decide)

- Who is it for: only jitmo, any developer, or both?
- Which laptops first: Windows only, or Mac and Linux too?
- What counts as "doesn't freeze" and "runs well", in numbers?
