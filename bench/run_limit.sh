#!/usr/bin/env bash
# Task 6: "8 GB laptop" simulation. Qwen3-4B, ctx 2048, real ~1.7k-token prompt, 64 generated tokens, -t 4, f16 KV, -fa on.
# A: Job Object private-commit limit 2560 MiB, default mmap      (file-backed weight pages are NOT charged to commit)
# B: Job Object private-commit limit 2560 MiB, -lm none (no mmap) (weights are copied into private memory -> charged)
# C: hard max working set 2560 MiB, mmap                          (caps resident pages incl. mapped weights)
# D: hard max working set 1536 MiB, mmap                          (below the model size: forces constant re-faulting)
source /v/Code/jitmo-runtime/bench/common.sh
P=$B/prompts
run() { # tag, runmon-limit-args, lm
  local tag=$1 lim=$2 lm=$3
  power_check "$tag"
  "$BIN/runmon.exe" --csv "$OUT/$tag.csv" --interval-ms 100 --timeout-s 900 $lim -- \
    "$L/llama-completion.exe" -m "$Q4B" -f "$P/prompt_2k.txt" -c 2048 -n 64 -no-cnv -t 4 -fa on \
    -ctk f16 -ctv f16 -lm $lm --seed 42 --temp 0 --no-display-prompt \
    > "$OUT/$tag.txt" 2>&1 < /dev/null
  grep RUNMON "$OUT/$tag.txt"
  sleep 3
}
for rep in 1 2 3; do
  run lim_A_commit2560_mmap_r$rep "--commit-limit-mb 2560" mmap
  run lim_B_commit2560_nommap_r$rep "--commit-limit-mb 2560" none
  run lim_C_ws2560_mmap_r$rep "--ws-max-mb 2560" mmap
  run lim_D_ws1536_mmap_r$rep "--ws-max-mb 1536" mmap
done
