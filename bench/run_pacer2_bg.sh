#!/usr/bin/env bash
# v0.2 item 10, extra run: a light background load (bgload.py, ~20% CPU) while Qwen3-4B generates,
# no probe. Shows which pacer level the light load alone gives, and the model's tok/s at that level
# vs no load. Usage: bash run_pacer2_bg.sh [reps]   (ALLOW_BATTERY=1 to run on battery)
source /v/Code/jitmo-runtime/bench/common.sh
PY="python -I"
PORT=8094
RT=/v/Code/jitmo-runtime/runtime/target/release/jitmo-runtime.exe
REPS=${1:-3}
P=${PDIR:-$OUT/pacer2}
mkdir -p "$P"
s=$(powershell -NoProfile -c "Add-Type -AssemblyName System.Windows.Forms; \$p=[System.Windows.Forms.SystemInformation]::PowerStatus; '{0} {1}' -f \$p.PowerLineStatus, \$p.BatteryLifePercent" | tr -d '\r')
echo "$(date '+%F %T') power: $s (pacer2 bgonly)" | tee -a "$OUT/power.log"
case "$s" in Online*) ;; *) [ "${ALLOW_BATTERY:-0}" = 1 ] || { echo "not on AC"; exit 9; } ;; esac

cat > "$P/rt_bgonly.toml" <<EOF
[server]
port = $PORT
[model]
path = '$(cygpath -w "$Q4B")'
chat_template_kwargs = { enable_thinking = false }
log_file = 'rt_bgonly.llama.log'
[budget]
idle_unload_secs = 0
reserve_free_mb = 512
EOF
"$RT" --config "$(cygpath -w "$P/rt_bgonly.toml")" > "$P/rt_bgonly.out.log" 2>&1 &
RPID=$!
until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 0.5; done
curl -s -X POST "http://127.0.0.1:$PORT/admin/load" | grep -q '"loaded":true' || { echo "LOAD FAILED"; kill $RPID; exit 3; }

sample() { # out secs
  for i in $(seq 1 $2); do
    curl -s "http://127.0.0.1:$PORT/stats" | $PY -c "import json,sys,time; p=json.load(sys.stdin)['pacer']; print(json.dumps({'t':round(time.time()*1000),'level':p['level'],'cap_pct':p['cap_pct'],'others_cpu_pct':p['others_cpu_pct'],'reason':p['reason']}))" >> "$1"
    sleep 1
  done
}

for r in $(seq 1 $REPS); do
  for load in noload bg; do
    tag=bgonly_${load}_r$r
    rm -f "$P/${tag}_pacer.jsonl"
    $PY "$B/stream_client.py" --chat --port $PORT --n 1500 --out "$P/${tag}_tokens.json" > "$P/${tag}_client.txt" 2>&1 &
    CPID=$!
    sleep 5
    BG=
    if [ $load = bg ]; then $PY "$B/bgload.py" 2 0.8 30 & BG=$!; fi
    sleep 4  # let the 1.5 s average and the level settle
    T0=$(date +%s%3N)
    sample "$P/${tag}_pacer.jsonl" 20
    T1=$(date +%s%3N)
    kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
    [ -n "$BG" ] && wait $BG 2>/dev/null
    sleep 1  # let stream_client write its token file
    sleep 6
  done
done
kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
$PY "$B/summarize_pacer2_bg.py" "$P"
echo DONE
