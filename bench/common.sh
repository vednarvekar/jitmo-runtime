# Shared settings for the Step 1 baseline scripts (Git Bash). Source this file.
set -u
B=/v/Code/jitmo-runtime/bench
OUT=$B/out
BIN=$B/target/release
L=/v/jitmo-sandbox/llama
M=/v/jitmo-sandbox/models
Q4B=$M/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
G3N=$M/gemma-3n-E2B-it-Q4_K_M.gguf
Q08=$M/Qwen3.5-0.8B-Q4_K_M.gguf
G1B=$M/gemma-3-1b-it-Q4_K_M.gguf
mkdir -p "$OUT"

# Abort the block if the laptop is not on AC power. Logs power state with a timestamp.
power_check() {
  local s
  s=$(powershell -NoProfile -c "Add-Type -AssemblyName System.Windows.Forms; \$p=[System.Windows.Forms.SystemInformation]::PowerStatus; '{0} {1}' -f \$p.PowerLineStatus, \$p.BatteryLifePercent" | tr -d '\r')
  echo "$(date '+%F %T') power: $s ($1)" | tee -a "$OUT/power.log"
  case "$s" in
    Online*) return 0 ;;
    *) echo "POWER NOT ONLINE - stopping"; exit 9 ;;
  esac
}

# Log the concurrent jitmo audit run (process count, CPU, RAM) once a second into $1, for $2 seconds.
# Sets PROCPID; kill it when done (scripts do this in their EXIT trap).
start_proclog() {
  powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$B/proclog.ps1")" -Out "$(cygpath -w "$1")" -Secs ${2:-3600} > /dev/null 2>&1 &
  PROCPID=$!
}

# POST /admin/load on port $1, retrying every 10 s for up to 2 minutes while the runtime refuses for
# free RAM (other apps' RAM moves a lot). Returns 0 once loaded. Logs each refusal.
load_retry() {
  local i r
  for i in $(seq 1 12); do
    r=$(curl -s -X POST "http://127.0.0.1:$1/admin/load")
    echo "$r" | grep -q '"loaded":true' && return 0
    echo "$(date '+%T') load refused (try $i): $r"
    sleep 10
  done
  return 1
}
