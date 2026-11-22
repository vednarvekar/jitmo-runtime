# Other programs' load once a second, as JSON lines: process count, CPU % (of all logical CPUs) and
# working set for each name in -Names. Used to log a concurrent run (the jitmo audit) next to benches.
#   powershell -NoProfile -File proclog.ps1 -Out <file.jsonl> [-Secs 3600] [-Names a,b]
param(
    [Parameter(Mandatory)] [string]$Out,
    [int]$Secs = 3600,
    [string[]]$Names = @('jitmo-model-worker', 'audit-copy', 'vmmemWSL')
)
$n = [Environment]::ProcessorCount
$prev = @{}
$tPrev = Get-Date
$end = (Get-Date).AddSeconds($Secs)
while ((Get-Date) -lt $end) {
    $now = Get-Date
    $dt = ($now - $tPrev).TotalSeconds
    $row = [ordered]@{ t = [int64](([DateTimeOffset]$now).ToUnixTimeMilliseconds()) }
    $cur = @{}
    foreach ($name in $Names) {
        $ps = @(Get-Process -Name $name -ErrorAction SilentlyContinue)
        $cpu = 0.0
        foreach ($p in $ps) {
            $c = [double]$p.CPU
            $cur[$p.Id] = $c
            if ($prev.ContainsKey($p.Id) -and $dt -gt 0) { $cpu += ($c - $prev[$p.Id]) }
        }
        $key = $name -replace '-', '_'
        $row["${key}_n"] = $ps.Count
        $row["${key}_cpu_pct"] = if ($dt -gt 0) { [math]::Round($cpu / $dt / $n * 100, 1) } else { 0 }
        $row["${key}_ws_mb"] = [int](($ps | Measure-Object WorkingSet64 -Sum).Sum / 1MB)
    }
    # total CPU of the whole PC over the next second (also catches short-lived processes); doubles as the 1 s wait
    try { $row['total_cpu_pct'] = [math]::Round((Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue, 1) } catch { Start-Sleep -Milliseconds 1000 }
    $prev = $cur
    $tPrev = $now
    ($row | ConvertTo-Json -Compress) | Add-Content -Encoding ascii $Out
}
