# Quality test: picking the base model

For roadmap step 4 ([07 Roadmap](07-roadmap.md)): compare **Gemma 3n E2B** and **Qwen3-4B-Instruct-2507** (both Q4_K_M) on answer quality, with **Qwen3.5-0.8B** as a floor.
Speed and RAM are already measured ([results/02](results/02-gemma3n.md)): Gemma 3n uses about half the RAM and is about 1.5× faster. This test adds the missing part: does it answer as well?

Files: `bench/quality/` (`tasks.jsonl`, `make_tasks.py`, `run_quality.py`, `quality.toml`). Results go to `bench/out/quality/` (git-ignored); copy the numbers into `docs/results/` when they are final.

## Method

- **The tasks are jitmo agent's real calls.** The prompts are copied from jitmo agent's `crates/llm/src/lib.rs` (`pick_system`, `pick_schema`, `pick_compact_system`, `args_system`, `args_schema`, `answer_from`). Tool titles, examples and argument descriptions come from its `skills/*.yaml`. The requests are written in the style of its held-out set, day-in-the-life set (`testset/vision_corpus.txt`) and audit corpus: the same kinds (natural, worker, student, typo, Hinglish, decline, follow-up) and its known hard cases. None are taken from its training data.
- **Three candidate tools per pick**, as jitmo's `flow.rs` sends them (the router's best two plus the best word-overlap match), so the wrong choices are the confusable ones.
- **Passages are written for this test** (fictional places and notices), so the model must read the passage, not remember a fact.
- **Every request:** `temperature 0`, `top_k 1`, `seed 42`, `cache_prompt`, `chat_template_kwargs {enable_thinking: false}` (as jitmo sends; Gemma's template ignores it), plus the item's `grammar` or `response_format` json_schema. One request at a time, `X-Jitmo-Priority: interactive`.
- **Each item has a known-good `reference` answer.** `run_quality.py --self-test` scores all references; all 96 pass, so every check accepts a right answer.

## Categories

| Category | Items | What jitmo uses it for | What is sent | Scored by |
|---|---|---|---|---|
| `tool_pick` | 23 | Tool-caller pick for an untuned model | `pick_system` (3 tools with examples), JSON schema `{"tool": enum}`, 40 tokens. 3 Hinglish, 3 typos, 3 "none" (poem, unsafe delete, out of scope), 1 injection, 1 follow-up with `(Earlier: ...)` | Valid JSON for the schema, `tool` equals the expected id |
| `tool_pick_digit` | 10 | Tool-caller pick, the fine-tune's format | `pick_compact_system` (numbered titles), grammar `root ::= [0-N]`, 1 token. 2 "none" (0) | The digit equals the expected one |
| `json_fill` | 19 | Tool-caller values (`fill`) | `args_system`, JSON schema of the tool's string arguments, 120 tokens | Valid JSON for the schema, then per field: contains the words the user said, or **empty** when the user did not say it (no invention). Includes the known failure "1 hour 30 minutes" cut to "1 hour" |
| `answer_passage` | 16 | Writer (`answer_from`) | `answer_from` system prompt, passage + question, 160 tokens. 4 unanswerable, 2 yes/no, 1 sum, 1 with a distractor number | Contains the expected facts, does not contain the wrong one, at most 3 sentences. Unanswerable: starts with "I'm not sure" (jitmo treats exactly that as "no answer") |
| `instruction` | 10 | Short replies the assistant writes | One-word, yes/no, 3-line list, word limits, lowercase, date format, extract an email, a prompt injection inside text to summarize | Regex, word / line / sentence counts, contains |
| `reasoning` | 8 | Quick sums in answers | Percent, °F to °C, pens, date + 10 days, end time, GST, speed, 0.9 vs 0.11. "Think briefly, then end with: Answer: <answer>", 300 tokens | The value after the last `Answer:` (numbers with a tolerance) |
| `multilingual` | 10 | Hindi / Hinglish users | 2 tool picks (Devanagari, Hinglish "none"), 2 fills (Hinglish, Devanagari), 2 Hindi passage questions (one unanswerable), Hindi answer, en→hi, hi→en, en→fr | As above, plus "mostly Devanagari letters" where the answer must be in Hindi |
| **Total** | **96** | | | |

An item **passes** when all its checks pass. The summary also shows **Checks %** (share of checks passed, partial credit), errors, p50 latency, mean prompt and output tokens, and mean generation tok/s.

**How big a difference means something:** with 10 to 23 items per category, one item is 4 to 10 points. Read the **all** row and the list of items where the models differ; a category gap of one or two items is noise.

## How to run

The PC must be otherwise idle (the latency columns are meaningless while another benchmark runs).

```powershell
# 1. Build and start the runtime with the quality config
cd V:\Code\jitmo-runtime\runtime
cargo build --release
.\target\release\jitmo-runtime.exe --config ..\bench\quality\quality.toml

# 2. In a second terminal, one run per model (each takes a few minutes)
cd V:\Code\jitmo-runtime\bench\quality
python run_quality.py --model gemma3n
python run_quality.py --model qwen3-4b
python run_quality.py --model qwen35-08b --no-grammar

# 3. Side by side
python run_quality.py --compare ..\out\quality\gemma3n-*.jsonl ..\out\quality\qwen3-4b-*.jsonl ..\out\quality\qwen35-08b-*.jsonl
```

PowerShell does not expand `*` for Python: pass the real file names (the run prints them at the end).

Other options: `--category tool_pick` (repeatable), `--id fill-04`, `--limit 5` (quick check), `--dry-run` (print the request bodies, send nothing), `--self-test` (check the checks, no server), `--no-schema` (JSON from the prompt alone), `--url` (default `http://127.0.0.1:8090`), `--tag` (extra word in the file names).

- **Qwen3.5-0.8B with `--no-grammar`:** jitmo agent found that a grammar made llama-server use about 6 GB with Qwen3.5 (its `LlmConfig.grammar` note). Without the grammar the digit items are scored on the first digit of the one token. The JSON-schema items still send `response_format`, which llama-server also turns into a grammar: watch `/stats` during that run, and add `--no-schema` if the memory grows.
- The runner calls `POST /admin/load` first, so the model load is not in the first item's latency; the load time is in the summary title.
- Each output line has the answer, every check with a note (for example `length='1 hour'`), status, latency, tokens, `timings` and `cached_tokens`.

To change the items, edit `make_tasks.py` and run `python make_tasks.py`, then `python run_quality.py --self-test`.

## Not covered

- Long answers, summaries of long documents, multi-turn chat: jitmo agent does not use the base model for them today.
- Speed under load and RAM: already in `docs/results/`.
- The fine-tuned 270M router: it is jitmo agent's own model, measured in its own test sets (roadmap step 6).
