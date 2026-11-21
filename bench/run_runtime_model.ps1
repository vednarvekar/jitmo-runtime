# v0.1 Task 2: one model through jitmo-runtime. Load time, first token, tok/s, child peak memory.
# Starts the runtime with a temporary config (pacer off), sends 1 load request + 3 streamed
# requests (128 tokens each), reads /stats, stops the runtime. One run at a time.
#
#   powershell -ExecutionPolicy Bypass -File bench\run_runtime_model.ps1 -Model <gguf> -NoRepack $true -Tag g3n_nr
param(
    [Parameter(Mandatory)] [string]$Model,
    [bool]$NoRepack = $true,
    [string]$Tag = 'run',
    [int]$Port = 8091,
    [int]$Tokens = 128,
    [string]$Exe = 'V:\Code\jitmo-runtime\runtime\target\release\jitmo-runtime.exe',
    [string]$OutDir = 'V:\Code\jitmo-runtime\bench\out'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$power = [System.Windows.Forms.SystemInformation]::PowerStatus
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') power: $($power.PowerLineStatus) $($power.BatteryLifePercent) (runtime $Tag)" | Tee-Object -Append "$OutDir\power.log"

$cfgPath = Join-Path $OutDir "rt_$Tag.toml"
$nr = if ($NoRepack) { 'true' } else { 'false' }
# The automatic cap does not include a repacked private copy of the weights: no cap when repacking.
$capLine = if ($NoRepack) { 'memory_cap_mb = "auto"' } else { 'memory_cap_mb = 0' }
$freeMb = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1KB)
"free RAM before: $freeMb MB" 
@"
[server]
port = $Port
[model]
path = '$Model'
no_repack = $nr
chat_template_kwargs = { enable_thinking = false }
log_file = 'rt_$Tag.llama.log'
[budget]
idle_unload_secs = 0
reserve_free_mb = 0
$capLine
[pacer]
mode = "off"
"@ | Set-Content -Encoding ascii $cfgPath

$base = "http://127.0.0.1:$Port"
$proc = Start-Process -FilePath $Exe -ArgumentList @('--config', "`"$cfgPath`"") -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$OutDir\rt_$Tag.out.log" -RedirectStandardError "$OutDir\rt_$Tag.err.log"
try {
    for ($i = 0; $i -lt 50; $i++) { try { Invoke-RestMethod "$base/health" -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Milliseconds 200 } }

    function Chat($content, $n, [bool]$stream) {
        $body = @{ model = 'jitmo-base'; max_tokens = $n; temperature = 0; stream = $stream
            messages = @(@{ role = 'user'; content = $content }) } | ConvertTo-Json -Depth 5 -Compress
        $req = [System.Net.HttpWebRequest]::Create("$base/v1/chat/completions")
        $req.Method = 'POST'; $req.ContentType = 'application/json'; $req.Timeout = 600000; $req.ReadWriteTimeout = 600000
        $bytes = [Text.Encoding]::UTF8.GetBytes($body)
        $t0 = [Diagnostics.Stopwatch]::StartNew()
        $s = $req.GetRequestStream(); $s.Write($bytes, 0, $bytes.Length); $s.Close()
        $resp = $req.GetResponse()
        $rd = New-Object IO.StreamReader($resp.GetResponseStream())
        $first = $null; $text = ''
        while (-not $rd.EndOfStream) {
            $line = $rd.ReadLine()
            if ($stream -and $line -like 'data: {*') {
                $j = $line.Substring(6) | ConvertFrom-Json
                $c = $j.choices[0].delta.content
                if ($c) { if ($null -eq $first) { $first = $t0.Elapsed.TotalMilliseconds }; $text += $c }
            } elseif (-not $stream) { $text += $line }
        }
        $rd.Close()
        [pscustomobject]@{ ms = $t0.Elapsed.TotalMilliseconds; first_ms = $first; text = $text }
    }

    $load = Chat 'Say hello in five words.' 16 $false
    $st = Invoke-RestMethod "$base/stats"
    $afterLoad = @{ ws = $st.child.working_set_mb; private = $st.child.private_mb }
    $runs = @()
    $prompt = 'Write a detailed paragraph about how a lighthouse works, its history, and the life of a keeper.'
    for ($r = 1; $r -le 3; $r++) {
        $x = Chat $prompt $Tokens $true
        $t = (Invoke-RestMethod "$base/stats").counters.last
        $runs += [pscustomobject]@{ first_ms_client = [math]::Round($x.first_ms, 1); prompt_tokens = $t.prompt_tokens
            prompt_tps = $t.prompt_tokens_per_sec; gen_tokens = $t.completion_tokens; tps = $t.tokens_per_sec; total_ms = $t.total_ms }
    }
    $st = Invoke-RestMethod "$base/stats"
    $result = [pscustomobject]@{
        tag = $Tag; model = (Split-Path $Model -Leaf); no_repack = $NoRepack; threads = $st.model.threads
        load_ms = $st.child.load_ms; first_request_ms = [math]::Round($load.ms, 0)
        ws_after_load_mb = $afterLoad.ws; private_after_load_mb = $afterLoad.private
        peak_ws_mb = $st.child.peak_working_set_mb; peak_private_mb = $st.child.peak_private_mb; job_peak_commit_mb = $st.child.job_peak_commit_mb
        cap_mb = $st.budget.memory_cap_mb; cap_source = $st.budget.memory_cap_source; estimate = $st.budget.memory_estimate_mb
        free_mb_before = $freeMb; system_free_mb = $st.system.free_mb; runs = $runs; sample_text = $x.text.Substring(0, [math]::Min(160, $x.text.Length))
    }
    $json = $result | ConvertTo-Json -Depth 6
    $json | Set-Content -Encoding utf8 "$OutDir\rt_$Tag.json"
    $json
}
finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Milliseconds 800
    Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -like 'V:\jitmo-sandbox\llama\*' } | Stop-Process -Force
}
