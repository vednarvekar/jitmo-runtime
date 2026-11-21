# jitmo-runtime: docs index

Run useful AI models locally on ordinary laptops, smartly and cheaply, instead of paying for cloud LLMs.
Repo: https://github.com/vednarvekar/jitmo-runtime · owner: Ved Narvekar · started 18 Nov 2026

| Doc | What it holds |
|---|---|
| [00 Vision](00-vision.md) | The goal, the problem, what we want, research to learn from, open questions |
| [01 Blockers](01-blockers.md) | What stops 2 to 4B models running well on an 8 GB laptop |
| [02 Research](02-research.md) | What others are doing, what actually works, papers and sources |
| [03 Concepts](03-concepts.md) | Learning guide: memory wall, quantization, sparsity, KV cache, diffusion, LoRA |
| [04 Plan](04-plan.md) | How we move forward: steps, setup, tracking |
| [results/](results/) | Measured numbers from the owner's laptop |

## Rules for these docs

- Short, plain sentences. One fact, one place. Link instead of repeating.
- Every claim about speed or memory needs a number measured on the target laptop. Estimates are marked *est.*
- When a target fails, write the number and say so. Do not lower the target quietly.

## Background (in the old repo, `V:\Code\jitmo\docs`)

jitmo is a local assistant (a desktop command bar). How it relates to this project is still open (see the open questions in [00 Vision](00-vision.md)).
Relevant old docs: `00-core/` (an earlier draft of this vision, superseded by these docs),
`03-delivery/15 - Low-Memory Engine Brief.md` (the 1.2 GB hard rule, measured model memory),
`00-vision/00.5 Models and Runtime.md` (the old whole-model scheduler).
