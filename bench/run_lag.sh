#!/usr/bin/env bash
# Task 5: "does the PC lag?" probe alone, then while Qwen3-4B generates non-stop via llama-server at -t 4 and -t 2.
# Model tokens are streamed by stream_client.py with wall-clock stamps; window_tps.py counts tokens inside the probe window.
source /v/Code/jitmo-runtime/bench/common.sh
PY="python -I"
PORT=8090
for r in 1 2 3; do
  power_check "lag base r$r"
  "$BIN/probe.exe" --secs 20 --tag base_r$r > "$OUT/lag_base_r$r.txt" 2>&1
  sleep 3
done
for T in 4 2; do
  power_check "lag server t$T"
  "$L/llama-server.exe" -m "$Q4B" -c 2048 -np 1 -t $T -tb $T -fa on --port $PORT --host 127.0.0.1 --no-webui \
      > "$OUT/lag_server_t$T.log" 2>&1 &
  SPID=$!
  until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 1; done
  for r in 1 2 3; do
    power_check "lag t$T r$r"
    $PY "$B/stream_client.py" --port $PORT --n 1500 --out "$OUT/lag_t${T}_r${r}_tokens.json" > /dev/null 2>&1 &
    CPID=$!
    sleep 8   # model is past prompt processing and generating steadily
    "$BIN/probe.exe" --secs 20 --tag t${T}_r$r > "$OUT/lag_t${T}_r$r.txt" 2>&1
    sleep 8
    kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
    $PY "$B/window_tps.py" "$OUT/lag_t${T}_r${r}_tokens.json" "$OUT/lag_t${T}_r$r.txt" | tee "$OUT/lag_t${T}_r${r}_tps.json"
    sleep 8   # server notices the disconnect and goes idle
  done
  kill $SPID; wait $SPID 2>/dev/null
  taskkill //F //IM llama-server.exe > /dev/null 2>&1
  sleep 5
done
