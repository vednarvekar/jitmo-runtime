#!/usr/bin/env bash
# v0.1 Task 2: llama-bench for Gemma 3n E2B vs Qwen3-4B (pp512, tg128, threads 3 and 4, 3 repeats),
# with weight repacking on (llama.cpp default) and off (the runtime default). One run at a time.
source /v/Code/jitmo-runtime/bench/common.sh
for model in G3N Q4B; do
  path=${!model}
  for rp in 1 0; do
    power_check "llama-bench $model repack=$rp"
    "$L/llama-bench.exe" -m "$path" -p 512 -n 128 -t 3,4 -r 3 -fa on --repack $rp -o json \
      > "$OUT/lb2_${model}_repack$rp.json" 2> "$OUT/lb2_${model}_repack$rp.err"
    sleep 5
  done
done
echo DONE
