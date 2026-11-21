#!/usr/bin/env bash
# v0.1 Task 3: pacer v1. Foreground probe alone, then while Qwen3-4B generates non-stop through
# jitmo-runtime (default threads = 3) with different pacer settings. One run at a time.
# Usage: bash run_pacer.sh [reps] [conditions...]   conditions: alone off cap50 cap25 idleprio auto
source /v/Code/jitmo-runtime/bench/common.sh
PY="python -I"
PORT=8092
RT=/v/Code/jitmo-runtime/runtime/target/release/jitmo-runtime.exe
REPS=${1:-3}
shift || true
CONDS=${*:-"alone off cap50 cap25 idleprio auto"}
P=$OUT/pacer
mkdir -p "$P"

write_cfg() { # name mode busy_cpu_pct busy_priority
  cat > "$P/rt_$1.toml" <<EOF
[server]
port = $PORT
[model]
path = '$(cygpath -w "$Q4B")'
chat_template_kwargs = { enable_thinking = false }
log_file = 'rt_$1.llama.log'
[budget]
idle_unload_secs = 0
reserve_free_mb = 512
[pacer]
mode = "$2"
busy_cpu_pct = $3
busy_priority = "$4"
EOF
}

stats_line() { curl -s "http://127.0.0.1:$PORT/stats" | $PY -c "import json,sys; d=json.load(sys.stdin); p=d['pacer']; print(json.dumps({k:p[k] for k in ('mode','state','capped','cap_pct_of_system','input_idle_ms','others_cpu_pct','free_mb','capped_secs_total','cap_on_count','last_error')}))"; }

for C in $CONDS; do
  if [ "$C" = alone ]; then
    for r in $(seq 1 $REPS); do
      power_check "pacer alone r$r"
      "$BIN/probe.exe" --secs 20 --tag alone_r$r > "$P/alone_r$r.txt" 2>&1
      sleep 3
    done
    continue
  fi
  case $C in
    off) write_cfg off off 50 unchanged ;;
    cap50) write_cfg cap50 always 50 unchanged ;;
    cap25) write_cfg cap25 always 25 unchanged ;;
    idleprio) write_cfg idleprio always 100 idle ;;
    auto) write_cfg auto auto 50 unchanged ;;
    *) echo "unknown condition $C"; exit 2 ;;
  esac
  power_check "pacer $C start"
  "$RT" --config "$(cygpath -w "$P/rt_$C.toml")" > "$P/rt_$C.out.log" 2>&1 &
  RPID=$!
  until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 0.5; done
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"jitmo-base","max_tokens":4,"messages":[{"role":"user","content":"Hi"}]}' | grep -q '"choices"'     || { echo "LOAD FAILED for $C"; tail -3 "$P/rt_$C.out.log"; kill $RPID; exit 3; }
  for r in $(seq 1 $REPS); do
    power_check "pacer $C r$r"
    $PY "$B/stream_client.py" --chat --port $PORT --n 1500 --out "$P/${C}_r${r}_tokens.json" > "$P/${C}_r${r}_client.txt" 2>&1 &
    CPID=$!
    sleep 8
    stats_line > "$P/${C}_r${r}_pacer_before.json"
    "$BIN/probe.exe" --secs 20 --tag ${C}_r$r > "$P/${C}_r$r.txt" 2>&1
    stats_line > "$P/${C}_r${r}_pacer_after.json"
    sleep 4
    kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
    $PY "$B/window_tps.py" "$P/${C}_r${r}_tokens.json" "$P/${C}_r$r.txt" > "$P/${C}_r${r}_tps.json"
    cat "$P/${C}_r${r}_tps.json"
    sleep 6
  done
  kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
  taskkill //F //IM jitmo-runtime.exe > /dev/null 2>&1
  sleep 3
done
echo DONE
