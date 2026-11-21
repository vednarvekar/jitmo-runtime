# Smoke test for jitmo-runtime v0 (Windows PowerShell 5.1+).
# Starts the runtime with a small model, calls every endpoint, checks the llama-server child
# goes away on /admin/unload and when the runtime is killed. Exit code 0 = all checks passed.
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
if ($script:failed -gt 0) { Write-Host "$($script:failed) check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host 'ALL CHECKS PASSED' -ForegroundColor Green
exit 0
