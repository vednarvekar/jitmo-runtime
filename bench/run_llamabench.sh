#!/usr/bin/env bash
# Task 3: llama-bench pp512 / tg128 at threads 1,2,3,4,8, 3 repetitions each, JSON output (has per-rep samples).
source /v/Code/jitmo-runtime/bench/common.sh
for pair in ${PAIRS:-"q08:$Q08" "g1b:$G1B" "q4b:$Q4B"}; do
  name=${pair%%:*}; model=${pair#*:}
  power_check "llama-bench $name"
  "$L/llama-bench.exe" -m "$model" -p 512 -n 128 -t 1,2,3,4,8 -r 3 -o json \
      > "$OUT/lb_$name.json" 2> "$OUT/lb_$name.err"
  echo "$name rc=$?"
  sleep 5
done
power_check "llama-bench done"
