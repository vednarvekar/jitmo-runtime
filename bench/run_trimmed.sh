#!/usr/bin/env bash
# Trimmed Step 1 (owner asked to finish fast): single runs only.
# Memory: Qwen3-4B, ctx 4096, default mmap, KV f16 vs q8_0 (-fa on), real 3703-token prompt, 64 generated tokens.
# Lag: probe alone, then with llama-server generating at -t 4 and -t 2 (probe = 30 s frame loop + 30 s membuf loop).
# Cap: hard working-set max 2560 MiB (mmap), and Job Object commit limit 2560 MiB with -lm none.
source /v/Code/jitmo-runtime/bench/common.sh
P=$B/prompts
mem() { # tag, runmon-limit-args, kv, lm
  local tag=$1 lim=$2 kv=$3 lm=$4
  power_check "$tag"
  "$BIN/runmon.exe" --csv "$OUT/$tag.csv" --interval-ms 100 --timeout-s 900 $lim -- \
    "$L/llama-completion.exe" -m "$Q4B" -f "$P/prompt_4k.txt" -c 4096 -n 64 -no-cnv -t 4 -fa on \
    -ctk $kv -ctv $kv -lm $lm --seed 42 --temp 0 --no-display-prompt \
    > "$OUT/$tag.txt" 2>&1 < /dev/null
  grep -E "RUNMON|prompt eval time|  eval time" "$OUT/$tag.txt"
  sleep 3
}
mem mem_c4096_mmap_f16 "" f16 mmap
mem mem_c4096_mmap_q8_0 "" q8_0 mmap

PORT=8090
power_check "lag base"
"$BIN/probe.exe" --secs 30 --tag base > "$OUT/lag_base.txt" 2>&1
sleep 3
for T in 4 2; do
  power_check "lag t$T"
  "$L/llama-server.exe" -m "$Q4B" -c 2048 -np 1 -t $T -tb $T -fa on --port $PORT --host 127.0.0.1 --no-webui \
      > "$OUT/lag_server_t$T.log" 2>&1 &
  SPID=$!
  until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 1; done
  python -I "$B/stream_client.py" --port $PORT --n 1500 --out "$OUT/lag_t${T}_tokens.json" > /dev/null 2>&1 &
  CPID=$!
  sleep 8
  "$BIN/probe.exe" --secs 30 --tag t$T > "$OUT/lag_t$T.txt" 2>&1
  sleep 6
  kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
  python -I "$B/window_tps.py" "$OUT/lag_t${T}_tokens.json" "$OUT/lag_t$T.txt" | tee "$OUT/lag_t${T}_tps.json"
  kill $SPID; wait $SPID 2>/dev/null
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  sleep 5
done

mem lim_ws2560_mmap "--ws-max-mb 2560" f16 mmap
mem lim_commit2560_nommap "--commit-limit-mb 2560" f16 none
power_check "trimmed done"
