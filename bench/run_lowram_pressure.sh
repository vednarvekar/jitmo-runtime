#!/usr/bin/env bash
# Memory-pressure path: Gemma 3n generates through jitmo-runtime (default budget: pressure 1024 MB
# twice in a row = finish, then unload; critical 500 MB once = unload at once), while memhog.py
# lowers free RAM to TARGET MB and holds it. Logs /stats once a second.
# Usage: bash run_lowram_pressure.sh <target_free_mb> [tag]
source /v/Code/jitmo-runtime/bench/common.sh
PY="python -I"
PORT=8097
RT=/v/Code/jitmo-runtime/runtime/target/release/jitmo-runtime.exe
TARGET=$1
TAG=${2:-t$1}
P=${PDIR:-$OUT/lowram/pressure}
mkdir -p "$P"
power_check "lowram pressure $TAG"
cat > "$P/rt_$TAG.toml" <<EOF
[server]
port = $PORT
[model]
path = '$(cygpath -w "$G3N")'
log_file = 'rt_$TAG.llama.log'
[budget]
idle_unload_secs = 0
reserve_free_mb = 512
[pacer]
mode = "off"
EOF
"$RT" --config "$(cygpath -w "$P/rt_$TAG.toml")" > "$P/rt_$TAG.out.log" 2>&1 &
RPID=$!
until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 0.5; done
curl -s -X POST "http://127.0.0.1:$PORT/admin/load" > "$P/${TAG}_load.json"
grep -q '"loaded":true' "$P/${TAG}_load.json" || { echo "LOAD FAILED $(cat "$P/${TAG}_load.json")"; kill $RPID; exit 3; }
rm -f "$P/${TAG}_stats.jsonl"
( for i in $(seq 1 90); do
    curl -s "http://127.0.0.1:$PORT/stats" | $PY -c "import json,sys,time; s=json.load(sys.stdin); m=s['models'][0]; print(json.dumps({'t':round(time.time()*1000),'state':m['state'],'active':m['active'],'unload_pending':m['unload_pending'],'free_mb':s['system']['free_mb'],'low_signal':s['system']['low_memory_signal'],'generating':s['queue']['generating'],'ws_mb':(m['child'] or {}).get('working_set_mb')}))" >> "$P/${TAG}_stats.jsonl" 2>/dev/null
    sleep 1
  done ) &
SPID=$!
$PY "$B/stream_client.py" --chat --port $PORT --n 600 --out "$P/${TAG}_tokens.json" > "$P/${TAG}_client.txt" 2>&1 &
CPID=$!
sleep 8
$PY "$B/memhog.py" $TARGET 25 300 > "$P/${TAG}_memhog.jsonl" 2>&1
for i in $(seq 1 60); do kill -0 $CPID 2>/dev/null || break; sleep 1; done
kill $CPID 2>/dev/null; wait $CPID 2>/dev/null
sleep 5
kill $SPID 2>/dev/null
curl -s "http://127.0.0.1:$PORT/stats" > "$P/${TAG}_stats_end.json"
kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
grep -i -E 'pressure|critical|unload|memory' "$P/rt_$TAG.out.log" | sed 's/\x1b\[[0-9;]*m//g'
echo DONE
