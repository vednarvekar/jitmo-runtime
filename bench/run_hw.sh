#!/usr/bin/env bash
# Task 1: RAM bandwidth and SSD read speed. Run alone, nothing else benchmarking.
source /v/Code/jitmo-runtime/bench/common.sh
power_check "membw"
"$BIN/membw.exe" --size-mb 2048 --reps 5 --passes 4 --threads 1,2,4,8 > "$OUT/membw.txt" 2>&1
power_check "disk create"
TF=/v/jitmo-sandbox/disktest/testfile_4g.bin
"$BIN/diskbench.exe" create "$TF" 4 > "$OUT/disk_create.txt" 2>&1
sleep 10   # let the SSD settle after the big write
power_check "disk run"
"$BIN/diskbench.exe" run "$TF" --dur 5 --reps 3 > "$OUT/disk.txt" 2>&1
power_check "hw done"
