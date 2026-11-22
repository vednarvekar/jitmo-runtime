#!/usr/bin/env bash
# v0.2 item 10: pacer v2 (adaptive levels from smoothed other-process CPU + free RAM).
# Foreground probe alone, then while Qwen3-4B generates non-stop through jitmo-runtime:
#   off      = no pacer
#   adaptive = pacer v2 defaults (15% / 40%, caps 75% / 50%)
#   bgload   = adaptive, plus a light background load (bgload.py, about 20% CPU)
# One run at a time. Usage: bash run_pacer2.sh [reps] [conditions...]
# Set ALLOW_BATTERY=1 to run on battery (the power state is logged either way).
source /v/Code/jitmo-runtime/bench/common.sh
PY="python -I"
PORT=8093
RT=/v/Code/jitmo-runtime/runtime/target/release/jitmo-runtime.exe
REPS=${1:-3}
shift || true
CONDS=${*:-"alone off adaptive bgload"}
P=${PDIR:-$OUT/pacer2}
mkdir -p "$P"
$PY "$B/ramlog.py" "$P/ram.jsonl" 7200 & RAMPID=$!  # free RAM once a second
trap 'kill $RAMPID 2>/dev/null' EXIT

power() {
  if [ "${ALLOW_BATTERY:-0}" = 1 ]; then
    local s
    s=$(powershell -NoProfile -c "Add-Type -AssemblyName System.Windows.Forms; \$p=[System.Windows.Forms.SystemInformation]::PowerStatus; '{0} {1}' -f \$p.PowerLineStatus, \$p.BatteryLifePercent" | tr -d '\r')
    echo "$(date '+%F %T') power: $s ($1)" | tee -a "$OUT/power.log"
  else
    power_check "$1"
  fi
}

write_cfg() { # name mode
  cat > "$P/rt_$1.toml" <<EOF
[server]
port = $PORT
[model]
path = '$(cygpath -w "${MODEL:-$Q4B}")'
chat_template_kwargs = { enable_thinking = false }
log_file = 'rt_$1.llama.log'
[budget]
idle_unload_secs = 0
reserve_free_mb = 512
[pacer]
mode = "$2"
${LIGHT_PCT:+light_cpu_pct = $LIGHT_PCT}
EOF
}

# The pacer's state once a second while the probe runs.
watch_pacer() { # out
  while true; do
    curl -s "http://127.0.0.1:$PORT/stats" | $PY -c "import json,sys,time; p=json.load(sys.stdin)['pacer']; print(json.dumps({'t':round(time.time()*1000),'level':p['level'],'cap_pct':p['cap_pct'],'others_cpu_pct':p['others_cpu_pct'],'free_mb':p['free_mb'],'reason':p['reason'],'level_changes':p['level_changes'],'secs_at_level':p['secs_at_level']}))" >> "$1" 2>/dev/null
    sleep 1
  done
}

for C in $CONDS; do
  if [ "$C" = alone ]; then
    for r in $(seq 1 $REPS); do
      power "pacer2 alone r$r"
      "$BIN/probe.exe" --secs 20 --tag alone_r$r > "$P/alone_r$r.txt" 2>&1
      sleep 3
    done
    continue
  fi
  case $C in
    off) write_cfg off off ;;
    adaptive|bgload) write_cfg $C auto ;;
    *) echo "unknown condition $C"; exit 2 ;;
  esac
  power "pacer2 $C start"
  "$RT" --config "$(cygpath -w "$P/rt_$C.toml")" > "$P/rt_$C.out.log" 2>&1 &
  RPID=$!
  until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 0.5; done
  curl -s -X POST "http://127.0.0.1:$PORT/admin/load" | grep -q '"loaded":true' || { echo "LOAD FAILED for $C"; kill $RPID; exit 3; }
  for r in $(seq 1 $REPS); do
    power "pacer2 $C r$r"
    BGPID=
    if [ "$C" = bgload ]; then $PY "$B/bgload.py" 2 0.8 36 & BGPID=$!; fi
    $PY "$B/stream_client.py" --chat --port $PORT --n 1500 --out "$P/${C}_r${r}_tokens.json" > "$P/${C}_r${r}_client.txt" 2>&1 &
    CPID=$!
    sleep 8
    rm -f "$P/${C}_r${r}_pacer.jsonl"
    watch_pacer "$P/${C}_r${r}_pacer.jsonl" &
    WPID=$!
    "$BIN/probe.exe" --secs 20 --tag ${C}_r$r > "$P/${C}_r$r.txt" 2>&1
    kill $WPID 2>/dev/null; wait $WPID 2>/dev/null
    sleep 4
    kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
    if [ -n "$BGPID" ]; then wait $BGPID 2>/dev/null; fi  # bgload stops by itself after 36 s
    $PY "$B/window_tps.py" "$P/${C}_r${r}_tokens.json" "$P/${C}_r$r.txt" > "$P/${C}_r${r}_tps.json"
    cat "$P/${C}_r${r}_tps.json"
    sleep 6
  done
  kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
  sleep 3
done
kill $RAMPID 2>/dev/null
echo DONE
