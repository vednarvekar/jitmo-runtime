# 03. Concepts: the basics, explained simply

A learning guide for the ideas behind this project. Read top to bottom.

## The kitchen analogy

| Real thing | Kitchen |
|---|---|
| **SSD** | Storeroom in the basement. Big, far, slow. |
| **RAM** | Kitchen counter. Smaller, close, fast. |
| **CPU cache** | The chef's hands. Tiny (~8 MB on our laptop), instant. |
| **CPU** | The chef. Can only cook with what is on the counter. |
| **Memory bandwidth** | How much the chef can carry from counter to hands per second. |
| **Other apps** | Other chefs sharing the same counter and the same path. |

## What a model is

- A model is a huge list of numbers called **weights** (also called **parameters**).
- A 3B model has 3 billion weights.
- Size on disk = number of weights × bytes per weight.
  - 16-bit: 3B × 2 bytes = **6 GB**
  - 4-bit: 3B × 0.5 bytes = **1.5 GB**
- The count of weights never changes. Only how much space each one takes.

## How a model writes text

- It writes one word (**token**) at a time.
- For **each** token it reads **every** weight once.
- For the next token, it reads them **all again**.
- A 100-token answer from a 1.5 GB model means reading 1.5 GB a hundred times.

## The memory wall

Each token needs two things:

- **Reading:** fetch all the weights from RAM (slow).
- **Maths:** about 2 operations per weight (fast).

Each weight is fetched, used once, and thrown away. So the CPU spends most of its time **waiting** for data, not computing. That waiting is the **memory wall**.

> tokens per second ≤ memory bandwidth ÷ bytes of weights read per token

Example: 50 GB/s ÷ 1.5 GB ≈ at most 33 tokens per second.

**Reading the prompt (prefill)** is different. All prompt tokens are processed together, so each weight read serves many tokens. That's why prompts are processed much faster than answers are written (**decode**).

**The deep insight: use each weight read for many tokens, not one.**

## Why the PC freezes

- **RAM fills up.** Windows then moves memory to the SSD (**paging**). Every app now waits on the slow storeroom. Everything crawls.
- **Bandwidth is hogged.** The model pulls data non-stop. Other apps wait for their turn on the same path. The mouse lags and the browser stutters.
- Research (SERENO, OSDI '26) shows the foreground apps suffer badly while the model itself barely slows down.

## The three ways to beat the memory wall (no new hardware)

| Way | Techniques |
|---|---|
| **1. Carry fewer bytes** | Quantization, sparsity, low-RAM model designs |
| **2. Use each byte for more tokens** | Speculative decoding, multi-token prediction, diffusion models |
| **3. Keep data closer to the CPU** | Use CPU cache for small, hot parts |

## Quantization

- Storing each weight with fewer bits by **rounding** it.
- 4 bits = only 16 possible values. `0.739` becomes the nearest level, say `0.75`.
- Each small block of weights also stores a **scale**, so the 16 levels fit that block's range.
- 4-bit: almost as smart as the original. 2-bit: noticeably dumber.
- **BitNet** goes further: weights are only -1, 0 or +1. The model must be **trained** that way.

## Sparsity

- Inside the model, many neurons output zero (or almost zero) for a given token.
- A zero output means its weights don't matter for this token, so **we don't need to read them**.
- The hard part: guess **which** neurons will be zero **before** reading them. A tiny predictor network does this (Deja Vu, PowerInfer).
- Works best on models trained to be sparse (ReLU activations).

## KV cache

- When writing token 500, the model looks back at tokens 1 to 499.
- It stores a summary of each past token instead of recomputing it. That store is the **KV cache**.
- It grows with conversation length and takes RAM.
- Fixes: store it in fewer bits (8 or 4-bit), keep only recent and important tokens (StreamingLLM, H2O), or use model designs that need less of it (GQA, Mamba-style hybrids).

## SSD offloading

- Keep rarely used parts of the model on the SSD, load them only when needed.
- The SSD is 10 to 20 times slower than RAM, so this saves **RAM**, not **time**.
- Only good for parts that are rarely used.

## Speculative decoding

- A tiny "draft" model quickly guesses the next few tokens.
- The big model checks all of them in **one** pass (one weight read).
- Correct guesses are kept. The output is exactly what the big model would write.
- Typical gain: 1.5 to 2× on CPU, less for very small models.

## Diffusion language models

- Like image generation: start with a rough draft of the **whole** answer, then sharpen it over several passes.
- Each pass reads the weights once and improves **all** tokens together.
- 200 tokens in maybe 20 passes instead of 200. That attacks the memory wall directly.
- Examples: LLaDA, Mercury (Inception Labs), Gemini Diffusion. Still early research.

## Shared base + LoRA adapters

- **LoRA adapter:** a small add-on file (10 to 50 MB) that changes how a base model behaves.
- Load **one** base model, plus tiny adapters for each job (router, planner, tool-caller, writer).
- Swapping adapters takes milliseconds. RAM stays at about one model.
- A better version of "load and unload many small models".

## Glossary

- **Token**: a word or piece of a word.
- **Parameter / weight**: one number inside the model.
- **Prefill**: reading the prompt. **Decode**: writing the answer.
- **GGUF**: llama.cpp's model file format.
- **mmap**: letting Windows load file parts into RAM on demand.
- **MoE (mixture of experts)**: a model split into "experts"; each token uses only a few.
