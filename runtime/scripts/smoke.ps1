# Smoke test for jitmo-runtime v0.1 (Windows PowerShell 5.1+).
# Starts the runtime with a small model, calls every endpoint, checks the llama-server child
# goes away on /admin/unload and when the runtime is killed. Part 2 checks the v0.1 rules:
# memory pressure lets a running generation finish, critical memory aborts it, the pacer caps
# the child. Exit code 0 = all checks passed.
#
#   powershell -ExecutionPolicy Bypass -File runtime\scripts\smoke.ps1 [-Model <gguf>] [-Port 8090]
param(
    [string]$Model = 'V:\jitmo-sandbox\models\Qwen3.5-0.8B-Q4_K_M.gguf',
    [int]$Port = 8090,
    [string]$Exe = (Join-Path $PSScriptRoot '..\target\release\jitmo-runtime.exe'),
    [string]$Config = (Join-Path $PSScriptRoot '..\jitmo-runtime.toml')
)
$ErrorActionPreference = 'Stop'
$base = "http://127.0.0.1:$Port"
$script:failed = 0
$logDir = Join-Path $env:TEMP 'jitmo-runtime-smoke'
New-Item -ItemType Directory -Force $logDir | Out-Null
$outLog = Join-Path $logDir 'runtime.out.log'
$errLog = Join-Path $logDir 'runtime.err.log'

function Check([string]$name, [bool]$ok, [string]$detail = '') {
    if ($ok) { Write-Host "PASS  $name $detail" -ForegroundColor Green }
    else { Write-Host "FAIL  $name $detail" -ForegroundColor Red; $script:failed++ }
}

function Post([string]$path, $obj, [int]$timeoutSec = 600) {
    $json = $obj | ConvertTo-Json -Depth 20 -Compress
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base$path" -Body $bytes `
        -ContentType 'application/json; charset=utf-8' -TimeoutSec $timeoutSec
}

function PostJson([string]$path, $obj, [int]$timeoutSec = 600) {
    $r = Post $path $obj $timeoutSec
    [Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
}

function Get-Json([string]$path) {
    Invoke-RestMethod -UseBasicParsing -Uri "$base$path" -TimeoutSec 30
}

function Child-Alive([int]$childPid) {
    return [bool](Get-Process -Id $childPid -ErrorAction SilentlyContinue)
}

if (-not (Test-Path $Exe)) { throw "runtime not built: $Exe (run: cargo build --release)" }
Write-Host "runtime: $Exe"
Write-Host "model:   $Model"

$proc = Start-Process -FilePath $Exe -PassThru -WindowStyle Hidden `
    -ArgumentList @('--config', "`"$Config`"", '--model', "`"$Model`"", '--port', $Port) `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$childPid = 0
try {
    # 1. The runtime answers, and the model is not loaded yet (lazy start).
    $up = $false
    for ($i = 0; $i -lt 50; $i++) {
        try { $h = Get-Json '/health'; $up = $true; break } catch { Start-Sleep -Milliseconds 200 }
    }
    Check 'runtime up' $up
    if (-not $up) { throw 'runtime did not start; see logs' }
    Check 'lazy start (model unloaded before first request)' ($h.model_state -eq 'unloaded') "state=$($h.model_state)"

    # 2. /v1/models lists the base model and the chain.
    $m = Get-Json '/v1/models'
    $ids = @($m.data | ForEach-Object { $_.id })
    Check '/v1/models' (($ids -contains 'jitmo-base') -and ($ids -contains 'chain:classify-answer')) ($ids -join ', ')

    # 3. Non-stream chat (this request loads the model).
    $t0 = Get-Date
    $r = PostJson '/v1/chat/completions' @{ model = 'jitmo-base'; max_tokens = 40; temperature = 0
        messages = @(@{ role = 'user'; content = 'Say hello in five words.' }) }
    $secs = ((Get-Date) - $t0).TotalSeconds
    $text = $r.choices[0].message.content
    Check 'chat (non-stream)' ($text.Length -gt 0) ("{0:N1}s: {1}" -f $secs, $text)

    # 4. Streamed chat: SSE chunks and [DONE].
    $s = Post '/v1/chat/completions' @{ model = 'jitmo-base'; stream = $true; max_tokens = 30; temperature = 0
        messages = @(@{ role = 'user'; content = 'Count from 1 to 5.' }) }
    $sse = [Text.Encoding]::UTF8.GetString($s.RawContentStream.ToArray())
    $chunks = ([regex]::Matches($sse, '(?m)^data: ')).Count
    Check 'chat (stream)' (($s.Headers['Content-Type'] -like 'text/event-stream*') -and ($chunks -gt 2) -and ($sse -match '\[DONE\]')) "$chunks SSE events"

    # 5. Unknown model is refused.
    $code = 0
    try { Post '/v1/chat/completions' @{ model = 'nope'; messages = @(@{ role = 'user'; content = 'x' }) } | Out-Null }
    catch { $code = [int]$_.Exception.Response.StatusCode }
    Check 'unknown model -> 404' ($code -eq 404) "status=$code"

    # 6. The example chain.
    $c = PostJson '/v1/chains/classify-answer/run' @{ input = 'What is the capital of France?' }
    $kind = $c.steps[0].json.kind
    Check 'chain run' (($c.steps.Count -eq 2) -and ($kind -in @('question', 'task', 'chat')) -and ($c.output.Length -gt 0)) `
        ("classify={0} | answer={1} | {2:N0} ms" -f $kind, $c.output, $c.total_ms)

    # 7. A chain through the OpenAI endpoint.
    $cc = PostJson '/v1/chat/completions' @{ model = 'chain:classify-answer'; messages = @(@{ role = 'user'; content = 'Hi there!' }) }
    Check 'chain via chat/completions' ($cc.choices[0].message.content.Length -gt 0) $cc.choices[0].message.content

    # 8. /stats shows the loaded child and timings.
    $st = Get-Json '/stats'
    $childPid = [int]$st.child.pid
    Check 'auto memory cap' (($st.budget.memory_cap_source -eq 'auto') -and ($st.budget.memory_cap_mb -gt $st.child.peak_private_mb)) `
        ("cap={0}MB peak_private={1}MB kv={2} recurrent={3} compute={4}" -f $st.budget.memory_cap_mb, $st.child.peak_private_mb, `
            $st.budget.memory_estimate_mb.kv, $st.budget.memory_estimate_mb.recurrent, $st.budget.memory_estimate_mb.compute)
    Check '/stats pacer fields' (($null -ne $st.pacer.mode) -and ($null -ne $st.pacer.input_idle_ms) -and ($null -eq $st.pacer.last_error)) `
        ("mode={0} state={1} input_idle={2}ms others_cpu={3}%" -f $st.pacer.mode, $st.pacer.state, $st.pacer.input_idle_ms, $st.pacer.others_cpu_pct)
    Check 'threads = min(cores-1, logical/2)' ($st.model.threads -eq [math]::Max(1, [math]::Min(((Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum - 1), [int]([Environment]::ProcessorCount / 2)))) "threads=$($st.model.threads)"
    Check '/stats' (($st.model_state -eq 'loaded') -and ($childPid -gt 0) -and ($st.counters.requests_served -ge 4)) `
        ("pid={0} ws={1}MB private={2}MB peak_ws={3}MB peak_private={4}MB job_peak={5}MB load={6}ms served={7}" -f `
            $childPid, $st.child.working_set_mb, $st.child.private_mb, $st.child.peak_working_set_mb, `
            $st.child.peak_private_mb, $st.child.job_peak_commit_mb, $st.child.load_ms, $st.counters.requests_served)
    foreach ($t in $st.counters.recent) {
        Write-Host ("      {0,-12} {1,-28} queue={2}ms load={3}ms first_token={4}ms prompt={5}tok@{6}tok/s gen={7}tok@{8}tok/s total={9}ms" -f `
            $t.kind, $t.label, $t.queue_ms, $t.load_ms, $t.first_token_ms, $t.prompt_tokens, $t.prompt_tokens_per_sec, `
            $t.completion_tokens, $t.tokens_per_sec, $t.total_ms)
    }
    $hl = Get-Json '/health'
    Check '/health' (($hl.status -eq 'ok') -and ($hl.model_state -eq 'loaded') -and ($null -ne $hl.last_tokens_per_sec)) `
        ("last_tokens_per_sec={0} free={1}MB" -f $hl.last_tokens_per_sec, $hl.system_free_mb)

    # 9. /admin/unload stops the child.
    Check 'child alive before unload' (Child-Alive $childPid) "pid=$childPid"
    $u = PostJson '/admin/unload' @{}
    Start-Sleep -Milliseconds 500
    Check '/admin/unload' ($u.unloaded -and ($u.model_state -eq 'unloaded')) ($u | ConvertTo-Json -Compress)
    Check 'child process gone after unload' (-not (Child-Alive $childPid)) "pid=$childPid"

    # 10. The next request loads it again; then killing the runtime must kill the child too.
    $r2 = PostJson '/v1/chat/completions' @{ model = 'jitmo-base'; max_tokens = 8; messages = @(@{ role = 'user'; content = 'Hi' }) }
    $childPid = [int](Get-Json '/stats').child.pid
    Check 'reload after unload' (($r2.choices.Count -eq 1) -and (Child-Alive $childPid)) "new pid=$childPid"
}
finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    Start-Sleep -Milliseconds 800
    if ($childPid -gt 0) {
        Check 'child gone after the runtime is killed (Job Object kill-on-close)' (-not (Child-Alive $childPid)) "pid=$childPid"
    }
    Write-Host "runtime logs: $outLog , $errLog"
}
# ---------- Part 2: v0.1 memory pressure rules and the pacer ----------
function Start-Rt([string]$name, [string]$extra) {
    $cfg = Join-Path $logDir "smoke_$name.toml"
    @"
[server]
port = $Port
[model]
path = '$Model'
log_file = '$(Join-Path $logDir "smoke_$name.llama.log")'
[budget]
idle_unload_secs = 0
$extra
"@ | Set-Content -Encoding ascii $cfg
    $p = Start-Process -FilePath $Exe -PassThru -WindowStyle Hidden -ArgumentList @('--config', "`"$cfg`"") `
        -RedirectStandardOutput (Join-Path $logDir "smoke_$name.out.log") -RedirectStandardError (Join-Path $logDir "smoke_$name.err.log")
    for ($i = 0; $i -lt 50; $i++) { try { Get-Json '/health' | Out-Null; break } catch { Start-Sleep -Milliseconds 200 } }
    return $p
}
function Stop-Rt($p) { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; Start-Sleep -Milliseconds 800 }
$long = @{ model = 'jitmo-base'; max_tokens = 400; temperature = 0; ignore_eos = $true
    messages = @(@{ role = 'user'; content = 'Write a long story about a lighthouse.' }) }

# 11. Pressure during a generation: it finishes, then the model is unloaded.
#     (Pressure is simulated with a floor above the free RAM; critical is off.)
$p = Start-Rt 'pressure' "pressure_free_mb = 60000`ncritical_free_mb = 0`nreserve_free_mb = 0`nwatch_interval_ms = 300"
try {
    $job = Start-Job -ScriptBlock { param($u, $b) Invoke-WebRequest -UseBasicParsing -Method Post -Uri $u -Body $b -ContentType 'application/json' -TimeoutSec 600 } `
        -ArgumentList "$base/v1/chat/completions", ($long | ConvertTo-Json -Depth 5 -Compress)
    $pendingSeen = $false
    for ($i = 0; $i -lt 100 -and $job.State -eq 'Running'; $i++) {
        Start-Sleep -Milliseconds 300
        try { $st = Get-Json '/stats'; if ($st.budget.unload_pending -and $st.model_state -eq 'loaded') { $pendingSeen = $true } } catch {}
    }
    $res = Receive-Job $job -Wait -ErrorAction SilentlyContinue
    $ok = $res -and ($res.StatusCode -eq 200) -and (($res.Content | ConvertFrom-Json).usage.completion_tokens -eq 400)
    Check 'pressure: running generation finishes' $ok "pending_seen_while_generating=$pendingSeen"
    Start-Sleep -Milliseconds 1200
    $st = Get-Json '/stats'
    Check 'pressure: unloaded after the generation' (($st.model_state -eq 'unloaded') -and ($st.counters.unloads.memory_pressure -ge 1)) `
        ("state={0} unloads={1}" -f $st.model_state, ($st.counters.unloads | ConvertTo-Json -Compress))
} finally { Stop-Rt $p }

# 12. Critical memory: the running generation is aborted at once.
$p = Start-Rt 'critical' "pressure_free_mb = 60001`ncritical_free_mb = 60000`nreserve_free_mb = 0`nwatch_interval_ms = 300"
try {
    $code = 0
    try { Post '/v1/chat/completions' $long | Out-Null; $code = 200 } catch { if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode } else { $code = -1 } }
    Start-Sleep -Milliseconds 800
    $st = Get-Json '/stats'
    Check 'critical: generation aborted, model unloaded' (($code -ne 200) -and ($st.counters.unloads.memory_critical -ge 1)) `
        ("status={0} unloads={1}" -f $code, ($st.counters.unloads | ConvertTo-Json -Compress))
} finally { Stop-Rt $p }

# 13. Pacer: mode always caps the loaded child; off does not.
$p = Start-Rt 'pacer' "[pacer]`nmode = `"always`"`nbusy_cpu_pct = 50"
try {
    PostJson '/v1/chat/completions' @{ model = 'jitmo-base'; max_tokens = 8; messages = @(@{ role = 'user'; content = 'Hi' }) } | Out-Null
    Start-Sleep -Milliseconds 800
    $st = Get-Json '/stats'
    Check 'pacer always: child capped' (($st.pacer.state -eq 'busy') -and $st.pacer.capped -and ($null -eq $st.pacer.last_error)) `
        ("cap={0}% of CPU err={1}" -f $st.pacer.cap_pct_of_system, $st.pacer.last_error)
} finally { Stop-Rt $p }
$p = Start-Rt 'pacer_off' "[pacer]`nmode = `"off`""
try {
    PostJson '/v1/chat/completions' @{ model = 'jitmo-base'; max_tokens = 8; messages = @(@{ role = 'user'; content = 'Hi' }) } | Out-Null
    Start-Sleep -Milliseconds 800
    $st = Get-Json '/stats'
    Check 'pacer off: child not capped' ((-not $st.pacer.capped) -and ($st.pacer.state -eq 'idle')) "state=$($st.pacer.state)"
} finally { Stop-Rt $p }

if ($script:failed -gt 0) { Write-Host "$($script:failed) check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host 'ALL CHECKS PASSED' -ForegroundColor Green
exit 0
