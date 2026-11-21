# 01. Blockers

What stops a 2 to 4B model from running well on an 8 GB laptop without freezing it.
*est.* = estimate, still to be measured on a real laptop.

## 1. The memory wall (the main one)

To write one word, the model reads all its weights from RAM. The CPU mostly waits on memory instead of computing.

> words per second ≈ RAM speed ÷ bytes read per word

| Model | Size at 4-bit *est.* | Laptop RAM at ~40 to 60 GB/s *est.* |
|---|---|---|
| 2B | ~1.2 GB | ~30 to 50 words/s at best |
| 4B | ~2.4 GB | ~15 to 25 words/s at best |

Real speed is lower. This is the same problem datacentres have with HBM, on slower memory.
**The fix is reading fewer bytes per word**, not faster code alone.

## 2. The PC freezes

- Engines like llama.cpp use every CPU core by default, so nothing else gets a turn.
- When RAM runs out, Windows pages to disk and everything stalls.
- An 8 GB laptop has only about 2 to 3 GB free *est.* once Windows and a browser are running.

## 3. The KV cache grows

The model's memory of earlier words grows with prompt length. Measured in the jitmo agent repo: a 0.8B model used 937 MB at context 4096 and 580 MB at 768.

## 4. Shrinking costs quality

Fewer bits per weight (3-bit, 2-bit) means a smaller, faster model, but a dumber one. Models trained for low bits (like BitNet) avoid this, but there are only a few of them.

## 5. Other chips sit unused

Most laptops have an integrated GPU, and newer ones have an NPU (AI chip). Most engines run only on the CPU. Each chip needs its own code.

## 6. Several models fight for RAM

Each loaded model costs its full size in RAM. Today, loading one takes 1.2 to 2.7 s (measured: 270M to 0.8B models), and private memory is about 2× the file size.

## 7. Existing engines are not built for this

llama.cpp and ONNX Runtime assume the model sits fully in RAM. They have no RAM budget, no "be nice to the PC" mode, and no partial loading.

## 8. Laptops vary a lot

RAM speed, SSD speed, CPU features (AVX2, AVX-512, VNNI), GPU and NPU all differ. Something fast on one laptop can be slow on another.

## 9. Solo builder, big scope

An engine is years of work for a team. Scope has to be cut hard: one model, one laptop, one number at a time.

## Not blockers

- **SSD wear.** Reading does not wear out an SSD.
- **Model size alone.** A 3B model at 4-bit already fits in 8 GB. Freezing comes from blockers 2 and 3, not from size.
