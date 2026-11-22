# Smoke test for jitmo-runtime v0.3 (Windows PowerShell 5.1+).
# Starts the runtime with a small model, calls every endpoint, checks the llama-server child
# goes away on /admin/unload and when the runtime is killed. Part 2 checks the v0.1 rules:
# memory pressure lets a running generation finish, critical memory aborts it, the pacer caps
# the child. Part 3 checks v0.2: child-process mode (port 0, --port-file, --parent-pid),
# /admin/load, request ids, cancel (running, queued, client closes a stream), priority, chain
# vars over two models with pool eviction, embeddings and LoRA (skipped if the 270M router,
# all-MiniLM-L6-v2 or the synthetic adapter are not on disk). v0.3: pressure_finish_max_ms stops a
# generation that runs too long under pressure (503 memory_pressure). Exit code 0 = all checks passed.
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
            $st.models[0].memory_estimate_mb.kv, $st.models[0].memory_estimate_mb.recurrent, $st.models[0].memory_estimate_mb.compute)
    Check '/stats pacer fields' (($null -ne $st.pacer.mode) -and ($null -ne $st.pacer.level) -and ($null -ne $st.pacer.thresholds) -and ($null -eq $st.pacer.last_error)) `
        ("mode={0} level={1} cap={2}% reason={3} others_cpu={4}%" -f $st.pacer.mode, $st.pacer.level, $st.pacer.cap_pct, $st.pacer.reason, $st.pacer.others_cpu_pct)
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
$p = Start-Rt 'pressure' "pressure_free_mb = 60000`ncritical_free_mb = 0`nreserve_free_mb = 0`nwatch_interval_ms = 300`npressure_finish_max_ms = 0"
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

# 11b. Pressure with a time limit (v0.3): after pressure_finish_max_ms the generation is stopped
#     with 503 memory_pressure, and the model is unloaded.
$p = Start-Rt 'pressure_limit' "pressure_free_mb = 60000`ncritical_free_mb = 0`nreserve_free_mb = 0`nwatch_interval_ms = 300`npressure_finish_max_ms = 2000"
try {
    $code = 0; $body = ''
    $t0 = Get-Date
    try { Post '/v1/chat/completions' $long | Out-Null; $code = 200 } catch {
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode; $body = "$($_.ErrorDetails.Message)" } else { $code = -1 } }
    $secs = ((Get-Date) - $t0).TotalSeconds
    Start-Sleep -Milliseconds 800
    $st = Get-Json '/stats'
    Check 'pressure limit: generation stopped with 503 memory_pressure, model unloaded' `
        (($code -eq 503) -and ($body -match 'memory_pressure') -and ($st.model_state -eq 'unloaded') -and ($st.counters.unloads.memory_pressure -ge 1)) `
        ("status={0} after {1:N1} s state={2} body={3}" -f $code, $secs, $st.model_state, $body)
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
$p = Start-Rt 'pacer' "[pacer]`nmode = `"always`""
try {
    PostJson '/v1/chat/completions' @{ model = 'jitmo-base'; max_tokens = 8; messages = @(@{ role = 'user'; content = 'Hi' }) } | Out-Null
    Start-Sleep -Milliseconds 800
    $st = Get-Json '/stats'
    Check 'pacer always: child capped at the heavy level' (($st.pacer.state -eq 'busy') -and $st.pacer.capped -and ($st.pacer.level -eq 'heavy') -and ($st.pacer.cap_pct -eq 50) -and ($null -eq $st.pacer.last_error)) `
        ("level={0} cap={1}% = {2}% of the system err={3}" -f $st.pacer.level, $st.pacer.cap_pct, $st.pacer.cap_pct_of_system, $st.pacer.last_error)
} finally { Stop-Rt $p }
$p = Start-Rt 'pacer_off' "[pacer]`nmode = `"off`""
try {
    PostJson '/v1/chat/completions' @{ model = 'jitmo-base'; max_tokens = 8; messages = @(@{ role = 'user'; content = 'Hi' }) } | Out-Null
    Start-Sleep -Milliseconds 800
    $st = Get-Json '/stats'
    Check 'pacer off: child not capped' ((-not $st.pacer.capped) -and ($st.pacer.state -eq 'idle')) "state=$($st.pacer.state)"
} finally { Stop-Rt $p }

# ---------- Part 3: v0.2 (child process, warm-up, cancel, priority, chain vars, pool, embeddings, LoRA) ----------
$router = 'V:\jitmo-sandbox\models\gemma-3-270m-it-router-Q4_K_M.gguf'
$minilm = 'V:\jitmo-sandbox\models\all-MiniLM-L6-v2-Q8_0.gguf'
$adapter = 'V:\jitmo-sandbox\adapters\router-synthetic-r1.gguf'
$haveRouter = Test-Path $router
$haveEmb = Test-Path $minilm
$haveLora = $haveRouter -and (Test-Path $adapter)
$cfg3 = Join-Path $logDir 'smoke_v02.toml'
$pf = Join-Path $logDir 'smoke_v02.port'
$routerModel = ''
if ($haveRouter) {
    $routerModel = @"
[[models]]
name = "router"
path = '$router'
ctx_size = 2048
$(if ($haveLora) { "adapters = ['$adapter']" })
"@
}
$embModel = ''
if ($haveEmb) {
    $embModel = @"
[[models]]
name = "minilm"
kind = "embedding"
path = '$minilm'
ctx_size = 512
"@
}
$firstStep = if ($haveRouter) { 'router' } else { 'chat' }
@"
[server]
port = 0
[budget]
idle_unload_secs = 0
reserve_free_mb = 512
pool_budget_mb = 1500      # v0.3 estimates: chat 1068 + router 705 > 1500, so a step on the other model evicts
[pacer]
mode = "off"
[[models]]
name = "chat"
path = '$Model'
chat_template_kwargs = { enable_thinking = false }
$routerModel
$embModel
[[chains]]
name = "qa"
[[chains.steps]]
name = "topic"
model = "$firstStep"
input = "Question: {question}"
max_tokens = 6
temperature = 0.0
[[chains.steps]]
name = "answer"
model = "chat"
system = "Answer from this passage only: {passage}"
input = "{question}"
max_tokens = 24
temperature = 0.0
"@ | Set-Content -Encoding ascii $cfg3

function Curl-Async([string]$path, [string]$json, [string]$tag, [string[]]$hdr = @()) {
    $bodyFile = Join-Path $logDir "smoke_$tag.req.json"
    [IO.File]::WriteAllText($bodyFile, $json)
    $a = @('-s', '-o', (Join-Path $logDir "smoke_$tag.resp"), '-w', '%{http_code}', '-H', 'content-type: application/json')
    foreach ($h in $hdr) { $a += @('-H', $h) }
    $a += @('-d', "@$bodyFile", "$base$path")
    $a = $a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }  # Start-Process does not quote
    Start-Process -FilePath curl.exe -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "smoke_$tag.code")
}
function Async-Code([string]$tag) { [int](Get-Content (Join-Path $logDir "smoke_$tag.code")) }
function Async-Body([string]$tag) { Get-Content -Raw (Join-Path $logDir "smoke_$tag.resp") }
function Cancel-Request([string]$id) { Invoke-RestMethod -Method Delete -Uri "$base/v1/requests/$id" -TimeoutSec 10 }

$parent = Start-Process -FilePath powershell -ArgumentList '-NoProfile', '-Command', 'Start-Sleep 900' -PassThru -WindowStyle Hidden
Remove-Item $pf -ErrorAction SilentlyContinue
$rt = Start-Process -FilePath $Exe -PassThru -WindowStyle Hidden -ArgumentList @('--config', "`"$cfg3`"", '--parent-pid', $parent.Id, '--port-file', "`"$pf`"") `
    -RedirectStandardOutput (Join-Path $logDir 'smoke_v02.out.log') -RedirectStandardError (Join-Path $logDir 'smoke_v02.err.log')
$null = $rt.Handle  # keep a handle so the exit code can be read later
$kids = @()
try {
    for ($i = 0; $i -lt 100 -and -not ((Test-Path $pf) -and (Get-Item $pf).Length -gt 0); $i++) { Start-Sleep -Milliseconds 100 }
    $line = (Get-Content $pf -ErrorAction SilentlyContinue) | Select-Object -First 1
    $stdout = @(Get-Content (Join-Path $logDir 'smoke_v02.out.log'))
    $Port3 = [int]($line -replace '.*port=', '')
    $base = "http://127.0.0.1:$Port3"
    Check 'port 0: one LISTENING line on stdout and in --port-file' (($line -match '^JITMO_RUNTIME_LISTENING port=\d+$') -and ($stdout.Count -eq 1) -and ($stdout[0] -eq $line)) $line

    # Warm-up.
    $w = PostJson '/admin/load' @{ model = 'chat' }
    $w2 = PostJson '/admin/load' @{ model = 'chat' }
    Check 'POST /admin/load loads, then reports already loaded' ($w.loaded -and ($w.load_ms -gt 0) -and $w2.was_loaded -and ($w2.load_ms -eq 0)) "load_ms=$($w.load_ms)"
    $code = 0; try { Post '/admin/load' @{ model = 'nope' } | Out-Null } catch { $code = [int]$_.Exception.Response.StatusCode }
    Check '/admin/load unknown model -> 404' ($code -eq 404) "status=$code"

    # Request id header (client-chosen).
    $r = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base/v1/chat/completions" -ContentType 'application/json' `
        -Headers @{ 'X-Jitmo-Request-Id' = 'smoke-id-1' } -Body '{"model":"chat","max_tokens":4,"messages":[{"role":"user","content":"Hi"}]}'
    Check 'X-Jitmo-Request-Id echoed' ($r.Headers['X-Jitmo-Request-Id'] -eq 'smoke-id-1')
    $r = Post '/v1/chat/completions' @{ model = 'chat'; max_tokens = 4; messages = @(@{ role = 'user'; content = 'Hi' }) }
    Check 'X-Jitmo-Request-Id generated' ($r.Headers['X-Jitmo-Request-Id'] -like 'req-*') $r.Headers['X-Jitmo-Request-Id']

    # Cancel a running request and a queued one.
    $chatLog = Join-Path $logDir 'llama-server-chat.log'
    $longJson = '{"model":"chat","max_tokens":1500,"ignore_eos":true,"messages":[{"role":"user","content":"Write a long story."}]}'
    $kids += Curl-Async '/v1/chat/completions' $longJson 'run' @('X-Jitmo-Request-Id: smoke-run')
    Start-Sleep -Milliseconds 2500
    $kids += Curl-Async '/v1/chat/completions' '{"model":"chat","max_tokens":4,"messages":[{"role":"user","content":"Hi"}]}' 'queued' @('X-Jitmo-Request-Id: smoke-queued')
    Start-Sleep -Milliseconds 800
    $list = Get-Json '/v1/requests'
    $states = ($list.requests | ForEach-Object { "$($_.id)=$($_.state)" }) -join ' '
    $dq = Cancel-Request 'smoke-queued'
    $dr = Cancel-Request 'smoke-run'
    $kids | ForEach-Object { $_.WaitForExit(15000) | Out-Null }
    Check 'DELETE a queued request -> 499' (($dq.state -eq 'queued') -and ((Async-Code 'queued') -eq 499)) "$states"
    Start-Sleep -Milliseconds 800
    $cancelLines = @(Select-String -Path $chatLog -Pattern 'cancel task').Count
    Check 'DELETE a running request -> 499, llama-server stops the task' (($dr.state -eq 'running') -and ((Async-Code 'run') -eq 499) -and ($cancelLines -ge 1)) "cancel_task_lines=$cancelLines"
    $code = 0; try { Cancel-Request 'smoke-run' | Out-Null } catch { $code = [int]$_.Exception.Response.StatusCode }
    Check 'DELETE a finished request -> 404' ($code -eq 404)

    # A client that closes a stream: llama-server stops too.
    $before = @(Select-String -Path $chatLog -Pattern 'cancel task').Count
    $streamJson = '{"model":"chat","stream":true,"max_tokens":1500,"ignore_eos":true,"messages":[{"role":"user","content":"Write a long story."}]}'
    [IO.File]::WriteAllText((Join-Path $logDir 'smoke_stream.json'), $streamJson)
    & curl.exe -s -N -m 3 -o NUL -H 'content-type: application/json' -d "@$(Join-Path $logDir 'smoke_stream.json')" "$base/v1/chat/completions"
    Start-Sleep -Milliseconds 1000
    $after = @(Select-String -Path $chatLog -Pattern 'cancel task').Count
    Check 'client closes a stream -> llama-server stops the task' ($after -gt $before) "cancel_task_lines $before -> $after"

    # Priority: one running, then background, background, interactive queued.
    $order = @()
    $kids = @()
    $kids += Curl-Async '/v1/chat/completions' '{"model":"chat","max_tokens":150,"ignore_eos":true,"messages":[{"role":"user","content":"story"}]}' 'p0' @('X-Jitmo-Priority: background')
    Start-Sleep -Milliseconds 1200
    $short = '{"model":"chat","max_tokens":3,"messages":[{"role":"user","content":"Hi"}]}'
    foreach ($t in @(@('bg1', 'background'), @('bg2', 'background'), @('ia1', 'interactive'))) {
        $kids += Curl-Async '/v1/chat/completions' $short $t[0] @("X-Jitmo-Priority: $($t[1])", "X-Jitmo-Request-Id: prio-$($t[0])")
        Start-Sleep -Milliseconds 400
    }
    $q = (Get-Json '/stats').queue
    $kids | ForEach-Object { $_.WaitForExit(60000) | Out-Null }
    $done = (Get-Json '/stats').counters.recent | Where-Object { $_.request_id -like 'prio-*' } | ForEach-Object { $_.request_id -replace 'prio-', '' }
    Check 'priority: interactive goes before queued background' ((($done -join ',') -eq 'ia1,bg1,bg2') -and ($q.waiting_background -eq 2)) ("order={0} queue={1}" -f ($done -join ','), ($q | ConvertTo-Json -Compress))
    $code = 0; try { Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$base/v1/chat/completions" -ContentType 'application/json' -Headers @{ 'X-Jitmo-Priority' = 'urgent' } -Body $short | Out-Null } catch { $code = [int]$_.Exception.Response.StatusCode }
    Check 'bad X-Jitmo-Priority -> 400' ($code -eq 400)

    # Chain with vars, a system override, and steps on two models.
    $ev0 = (Get-Json '/stats').pool.evictions
    $c = PostJson '/v1/chains/qa/run' @{ vars = @{ question = 'Who built the lighthouse?'; passage = 'The lighthouse on Kell Island was built by Mara Voss in 1871.' }
        system = @{ answer = 'Reply in at most six words. Passage: {passage}' } }
    $ev1 = (Get-Json '/stats').pool.evictions
    Check 'chain with vars + system override' (($c.output -match 'Mara|Voss') -and ($c.steps.Count -eq 2) -and ($c.request_id -like 'req-*')) `
        ("output={0} models={1}" -f $c.output, (($c.steps | ForEach-Object { $_.model }) -join ','))
    if ($haveRouter) {
        Check 'pool budget: a step on another model evicts the idle one' ($ev1 -gt $ev0) "evictions $ev0 -> $ev1"
    }
    $code = 0; try { Post '/v1/chains/qa/run' @{ vars = @{ question = 'x' } } | Out-Null } catch { $code = [int]$_.Exception.Response.StatusCode }
    Check 'chain with a missing var -> 400' ($code -eq 400)
    $old = PostJson '/v1/chains/qa/run' @{ input = 'unused'; vars = @{ question = 'Who?'; passage = 'Ann did it.' } }
    Check 'chain: input + vars' ($old.output.Length -gt 0) $old.output

    # Embeddings.
    if ($haveEmb) {
        $e = PostJson '/v1/embeddings' @{ model = 'minilm'; input = @('open the calculator', 'launch the calc app', 'weather in Paris') }
        $v = $e.data | ForEach-Object { , [double[]]$_.embedding }
        $dot = { param($a, $b) $s = 0.0; for ($k = 0; $k -lt $a.Length; $k++) { $s += $a[$k] * $b[$k] }; $s }
        $near = & $dot $v[0] $v[1]; $far = & $dot $v[0] $v[2]
        Check 'POST /v1/embeddings' (($e.data.Count -eq 3) -and ($v[0].Length -eq 384) -and ($near -gt $far)) ("dim={0} cos(near)={1:N3} cos(far)={2:N3}" -f $v[0].Length, $near, $far)
        $code = 0; try { Post '/v1/embeddings' @{ model = 'chat'; input = 'x' } | Out-Null } catch { $code = [int]$_.Exception.Response.StatusCode }
        Check 'embeddings on a chat model -> 400' ($code -eq 400)
    } else { Write-Host "SKIP  embeddings ($minilm not found)" }

    # LoRA: no lora = scale 0 = base; scale 1 changes the output; no reload.
    if ($haveLora) {
        $ask = { param($lora) $b = @{ model = 'router'; max_tokens = 10; temperature = 0; messages = @(@{ role = 'user'; content = 'Name three colors.' }) }
            if ($null -ne $lora) { $b.lora = @(@{ id = 0; scale = $lora }) }
            (PostJson '/v1/chat/completions' $b).choices[0].message.content }
        $a0 = & $ask $null; $a1 = & $ask 1.0; $az = & $ask 0.0; $a2 = & $ask 1.0
        $loads = ((Get-Json '/stats').models | Where-Object { $_.name -eq 'router' }).counters.loads
        Check 'LoRA: per-request scale switch' (($a0 -ne $a1) -and ($a1 -eq $a2) -and ($az.Trim('.') -eq $a0.Trim('.'))) "base='$a0' scale1='$a1' scale0='$az' router_loads=$loads"
    } else { Write-Host "SKIP  LoRA ($adapter not found)" }

    # /stats lists every model.
    $st = Get-Json '/stats'
    Check '/stats lists every model with state and memory' (($st.models.Count -ge 1) -and ($null -ne $st.models[0].load_estimate_mb.total) -and ($null -ne $st.pool)) `
        (($st.models | ForEach-Object { "$($_.name)=$($_.state)" }) -join ' ')

    # The parent exits: the runtime stops its models and exits.
    $pids = @($st.models | Where-Object { $_.child } | ForEach-Object { [int]$_.child.pid })
    Stop-Process -Id $parent.Id -Force
    $exited = $rt.WaitForExit(10000)
    Start-Sleep -Milliseconds 500
    $alive = @($pids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count
    Check '--parent-pid: runtime exits (code 0), children gone, port file removed' ($exited -and ($rt.ExitCode -eq 0) -and ($alive -eq 0) -and -not (Test-Path $pf)) `
        "exit=$($rt.ExitCode) children_alive=$alive"
}
finally {
    if (-not $rt.HasExited) { Stop-Process -Id $rt.Id -Force }
    if (-not $parent.HasExited) { Stop-Process -Id $parent.Id -Force }
    $kids | Where-Object { -not $_.HasExited } | ForEach-Object { Stop-Process -Id $_.Id -Force }
    $base = "http://127.0.0.1:$Port"
}

if ($script:failed -gt 0) { Write-Host "$($script:failed) check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host 'ALL CHECKS PASSED' -ForegroundColor Green
exit 0
