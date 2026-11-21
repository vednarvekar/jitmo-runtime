#!/usr/bin/env bash
# Task 4: memory of the 3-4B model with a real prompt. runmon samples every 100 ms.
# Matrix: ctx {2048 (prompt_2k), 8192 (prompt_8k)} x load {mmap (default), none (= no mmap)} x KV {f16, q8_0}; 3 reps.
# Flash attention is forced on for every run so KV type is the only change (q8_0 V-cache needs it).
source /v/Code/jitmo-runtime/bench/common.sh
P=$B/prompts
for rep in 1 2 3; do
for ctx in 2048 8192; do
  pf=$P/prompt_2k.txt; [ $ctx = 8192 ] && pf=$P/prompt_8k.txt
  for lm in mmap none; do
    for kv in f16 q8_0; do
      tag=mem_c${ctx}_${lm}_${kv}_r${rep}
      power_check "$tag"
      "$BIN/runmon.exe" --csv "$OUT/$tag.csv" --interval-ms 100 -- \
        "$L/llama-completion.exe" -m "$Q4B" -f "$pf" -c $ctx -n 64 -no-cnv -t 4 -fa on \
        -ctk $kv -ctv $kv -lm $lm --seed 42 --temp 0 --no-display-prompt \
        > "$OUT/$tag.txt" 2>&1 < /dev/null
      grep RUNMON "$OUT/$tag.txt"
      sleep 3
    done
  done
done
done
