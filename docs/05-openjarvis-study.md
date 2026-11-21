# 05. OpenJarvis study

Studied from the local clone `V:\jitmo-sandbox\openjarvis` (commit 5a9b5268, 21 Nov 2026) on 21 Nov 2026.
Repo: https://github.com/open-jarvis/OpenJarvis · Stanford Hazy Research + Scaling Intelligence Lab · Apache-2.0 · ~17.7k stars.

## What it is

"Personal AI, on personal devices." A local-first framework for building personal AI agents. It is mainly a **research platform**, part of Stanford's "Intelligence Per Watt" work.

**Size:** about 190,000 lines of Python plus about 28,000 lines of Rust (an optional speed-up layer). Also a Tauri desktop app and a web frontend.

## How it works

Five building blocks:

| Block | What it does |
|---|---|
| **Intelligence** | A catalog of models: size, context length, memory needed, which engines can run them |
| **Engine** | Runs the model. **It does not run models itself.** It calls Ollama, llama.cpp, vLLM, MLX, SGLang, LM Studio or cloud APIs, all through one interface (`generate`, `stream`, `list_models`, `health`) |
| **Agents** | 9 agent types: simple chat, tool-calling loop, ReAct, code-running agents, scheduled long-running agents |
| **Memory** | Document search: SQLite full-text, FAISS vectors, BM25, ColBERT, or a hybrid |
| **Learning** | Records every run as a "trace" and learns which model, agent and tool work best per query type |

**One request, step by step:**
1. Load config and find which engines are running.
2. Pick a model (a router can choose per query).
3. The agent searches memory and adds relevant text to the prompt.
4. It calls the engine. If the model asks for a tool, it runs the tool and calls the engine again.
5. It records speed, energy, cost and the full trace.

## How it picks a model for your laptop

`core/config.py`: usable memory = (total RAM − 4 GB) × 0.8. Then a fixed table:

| Usable memory | Model |
|---|---|
| ≤ 8 GB | Qwen3.5 2B |
| ≤ 16 GB | Qwen3.5 4B |
| ≤ 32 GB | Qwen3.5 9B |
| More | Qwen3.5 27B |

An 8 GB laptop gets (8 − 4) × 0.8 = 3.2 GB, so Qwen3.5 2B. This is a **one-time guess at setup**, not live memory management.

## Where it is better than us

- **The whole assistant layer:** agents, more than 40 tools (files, shell, browser, web search, PDF, calendar, code runner), connectors (Gmail, Calendar, Drive, Notion, Slack, Telegram, WhatsApp…), MCP support, skills, scheduling.
- **Voice:** speech-to-text (faster-whisper, Whisper, Deepgram) and text-to-speech (Kokoro, ElevenLabs, Cartesia, OpenAI).
- **Measurement:** benchmarks that count energy, speed and dollar cost, not just accuracy, plus a leaderboard.
- **Learning from use:** the trace-driven router improves model choice over time.
- **Security:** Docker sandbox for agents, approval store for risky actions.
- **Team and reach:** a Stanford lab, sponsors and a big community.

## Where it is weak (our opening)

| Gap | Detail |
|---|---|
| **No RAM management** | Nothing loads, unloads or limits models by free RAM. That is left to Ollama or llama.cpp. |
| **No chain of small models** | One model per request. The router picks a model; it doesn't chain specialists. |
| **No "don't freeze the PC"** | Nothing throttles or pauses when other apps need the machine. |
| **Heavy for 8 GB** | Python, many dependencies, plus Ollama in the background. The framework itself costs RAM on a small laptop. |
| **Static model choice** | The RAM table above is fixed at setup. |
| **Analytics** | Ships PostHog analytics (`analytics/client.py`). Check what it sends and how to turn it off before using it. |

## What we can reuse

License is Apache-2.0, so code can be reused with attribution (keep their copyright and license notice).

| Reuse | How | Value |
|---|---|---|
| **Plug our runtime in as an engine** | Our runtime offers an OpenAI-compatible API (like `llama-server` at `/v1`). OpenJarvis already speaks that (`engine/openai_compat_engines.py`). | **High.** A real assistant on top of our runtime, almost for free. |
| **Their benchmarks** | `evals/` and `bench/` measure energy, speed and cost. | **High.** Ready-made tests for our runtime. |
| **Tools and connectors** | `tools/`, `connectors/`, `channels/`, `mcp/` | **High** for the assistant layer. |
| **Speech** | `speech/` wrappers for faster-whisper and Kokoro (local) | Medium |
| **Memory search** | SQLite FTS5 + BM25 + hybrid ranking | Medium |
| **Trace-driven router** | `learning/routing/` | Medium. Their router picks a model per query; ours could pick a chain. |
| **Agent loops** | Orchestrator and ReAct agents | Medium |

## Bottom line

- OpenJarvis is the **assistant layer** done well. It is **not** a runtime.
- Our runtime fills the gap underneath it: live RAM management, a chain of small models, and no freezing on 8 GB laptops.
- Fastest path: build our runtime with an OpenAI-compatible API, then test it under OpenJarvis as a real user.
