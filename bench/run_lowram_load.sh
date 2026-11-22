#!/usr/bin/env bash
# Low free RAM: which models the runtime loads with its default budget (reserve_free_mb 1536),
# the refusal message and estimate for the others, and the real memory of what loads
# (after a 128-token generation). Free RAM logged every second. One run at a time, AC only.
# Usage: bash run_lowram_load.sh [model-files...]
source /v/Code/jitmo-runtime/bench/common.sh
PY="python -I"
PORT=8095
RT=/v/Code/jitmo-runtime/runtime/target/release/jitmo-runtime.exe
P=${PDIR:-$OUT/lowram/load}
mkdir -p "$P"
MODELS=${*:-"Qwen3-4B-Instruct-2507-Q4_K_M.gguf gemma-3n-E2B-it-Q4_K_M.gguf gemma-3-270m-it-router-Q4_K_M.gguf Qwen3.5-0.8B-Q4_K_M.gguf all-MiniLM-L6-v2-Q8_0.gguf"}
$PY "$B/ramlog.py" "$P/ram.jsonl" 3600 & RAMPID=$!
start_proclog "$P/procs.jsonl" 7200
trap 'kill $RAMPID $PROCPID 2>/dev/null' EXIT
for f in $MODELS; do
  tag=${f%.gguf}
  KIND=chat; case $f in *MiniLM*) KIND=embedding ;; esac
  power_check "lowram load $tag"
  cat > "$P/rt_$tag.toml" <<EOF
[server]
port = $PORT
[model]
path = '$(cygpath -w "$M/$f")'
kind = "$KIND"
chat_template_kwargs = { enable_thinking = false }
log_file = 'rt_$tag.llama.log'
[budget]
idle_unload_secs = 0
[pacer]
mode = "off"
EOF
  free0=$($PY -c "import json;print(json.loads(open(r'$(cygpath -w "$P/ram.jsonl")').readlines()[-1])['free_mb'])")
  "$RT" --config "$(cygpath -w "$P/rt_$tag.toml")" > "$P/rt_$tag.out.log" 2>&1 &
  RPID=$!
  until curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; do sleep 0.5; done
  T0=$(date +%s%3N)
  curl -s -X POST "http://127.0.0.1:$PORT/admin/load" > "$P/${tag}_load.json"
  T1=$(date +%s%3N)
  echo "$tag free_before=$free0 load_ms=$((T1-T0)) $(cat "$P/${tag}_load.json")"
  if grep -q '"loaded":true' "$P/${tag}_load.json"; then
    if [ $KIND = embedding ]; then
      for i in 1 2 3 4 5; do curl -s -X POST "http://127.0.0.1:$PORT/v1/embeddings" -H 'Content-Type: application/json'         -d '{"model":"jitmo-base","input":["The quick brown fox jumps over the lazy dog.","Low free RAM test for the runtime."]}' > "$P/${tag}_embed.json"; done
    else
      $PY "$B/stream_client.py" --chat --port $PORT --n 128 --out "$P/${tag}_tokens.json" > "$P/${tag}_client.txt" 2>&1
    fi
  fi
  curl -s "http://127.0.0.1:$PORT/stats" > "$P/${tag}_stats.json"
  kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
  sleep 5
done
kill $RAMPID 2>/dev/null
echo DONE
