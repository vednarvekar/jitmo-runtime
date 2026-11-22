#!/usr/bin/env bash
# "Would it freeze?": force Qwen3-4B into low free RAM (budget.reserve_free_mb = 0, bench-only config;
# pressure/critical unload left at their defaults), generate while the foreground probe runs.
# Logs free RAM every second (ramlog.py) and hard faults (typeperf: Page Reads/sec, Pages Input/sec).
# Safety: ramlog kills the runtime (and its llama-server) if free RAM drops under 300 MB.
# HOG_TARGET=<MB>: first lower free RAM to about this with memhog.py (held for the whole run), to act
# like an 8 GB laptop. The load estimate is bypassed (load_estimate_mb = 1) so the model is forced in.
# A probe run alone (with the hog holding) comes first, as the lag baseline.
# Usage: [HOG_TARGET=3000] bash run_lowram_forced.sh [probe_secs]
source /v/Code/jitmo-runtime/bench/common.sh
PY="python -I"
PORT=8096
RT=/v/Code/jitmo-runtime/runtime/target/release/jitmo-runtime.exe
PS=${1:-10}
P=${PDIR:-$OUT/lowram/forced}
mkdir -p "$P"
power_check "lowram forced 4B"
cat > "$P/rt_forced.toml" <<EOF
[server]
port = $PORT
[model]
path = '$(cygpath -w "$Q4B")'
chat_template_kwargs = { enable_thinking = false }
log_file = 'rt_forced.llama.log'
[budget]
idle_unload_secs = 0
reserve_free_mb = 0
load_estimate_mb = 1
[pacer]
mode = "off"
EOF
HOGPID=; RAMPID=; TPID=; PROCPID=; SPID=
trap 'kill $RAMPID $TPID $PROCPID $HOGPID $SPID 2>/dev/null' EXIT
if [ -n "${HOG_TARGET:-}" ]; then
  $PY "$B/memhog.py" $HOG_TARGET 900 300 > "$P/memhog.jsonl" 2>&1 &
  HOGPID=$!
  for i in $(seq 1 120); do grep -q '"holding": true' "$P/memhog.jsonl" 2>/dev/null && break; sleep 1; done
  echo "memhog holding: $(tail -n 1 "$P/memhog.jsonl")"
fi
"$BIN/probe.exe" --secs $PS --tag forced_alone > "$P/probe_alone.txt" 2>&1
typeperf "\\Memory\\Available MBytes" "\\Memory\\Page Reads/sec" "\\Memory\\Pages Input/sec" -si 1 -o "$(cygpath -w "$P/faults.csv")" -y > /dev/null 2>&1 &
TPID=$!
"$RT" --config "$(cygpath -w "$P/rt_forced.toml")" > "$P/rt_forced.out.log" 2>&1 &
RPID=$!
WIN=$(cat /proc/$RPID/winpid)
$PY "$B/ramlog.py" "$P/ram.jsonl" 600 --kill-below 300 --kill-pid $WIN > "$P/ramlog.txt" 2>&1 &
RAMPID=$!
start_proclog "$P/procs.jsonl" 900
trap 'kill $RAMPID $TPID $PROCPID $HOGPID $SPID 2>/dev/null' EXIT
( while true; do curl -s "http://127.0.0.1:$PORT/stats" | $PY -c "import json,sys,time; s=json.load(sys.stdin); m=s['models'][0]; print(json.dumps({'t':round(time.time()*1000),'state':m['state'],'free_mb':s['system']['free_mb'],'ws_mb':(m['child'] or {}).get('working_set_mb'),'priv_mb':(m['child'] or {}).get('private_mb')}))" >> "$P/stats.jsonl" 2>/dev/null; sleep 1; done ) &
SPID=$!
until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 0.5; done
sleep 3
T0=$(date +%s%3N)
curl -s -X POST "http://127.0.0.1:$PORT/admin/load" > "$P/load.json"
T1=$(date +%s%3N)
echo "load $((T1-T0)) ms: $(cat "$P/load.json")"
grep -q '"loaded":true' "$P/load.json" || { kill $RPID; exit 3; }
cat "$P/ramlog.txt"
curl -s "http://127.0.0.1:$PORT/stats" > "$P/stats_loaded.json"
$PY "$B/stream_client.py" --chat --port $PORT --n ${NTOK:-300} --out "$P/tokens.json" > "$P/client.txt" 2>&1 &
CPID=$!
sleep 6
"$BIN/probe.exe" --secs $PS --tag forced > "$P/probe.txt" 2>&1
cat "$P/probe.txt"
curl -s "http://127.0.0.1:$PORT/stats" > "$P/stats_gen.json"
for i in $(seq 1 120); do kill -0 $CPID 2>/dev/null || break; sleep 1; done
kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
curl -s "http://127.0.0.1:$PORT/stats" > "$P/stats_end.json"
kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
sleep 3
cat "$P/ramlog.txt"
sed 's/\x1b\[[0-9;]*m//g' "$P/rt_forced.out.log" | grep -i -E 'pressure|critical|unload|memory' | cut -c12-260
echo DONE
