#!/usr/bin/env bash
set -euo pipefail

# Auto-chain: ensure 30-iteration run completes, then start 200-iteration run.
# Writes progress to run_30.out, run_200.out, and this script's own stdout.

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

start_run() {
  local iters="$1"; shift
  local pidfile="run_${iters}.pid"
  local outfile="run_${iters}.out"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null; then
    echo "Run $iters already active (PID $(cat "$pidfile"))."
    return 0
  fi
  echo "Starting $iters-iteration run..."
  nohup bash run_evolution.sh "$iters" > "$outfile" 2>&1 & echo $! > "$pidfile"
  sleep 2
  echo "Run $iters started. PID $(cat "$pidfile"). Output: $outfile"
}

wait_for_checkpoint() {
  local target_iters="$1"; shift
  local logfile
  # Latest evolve log file (sorted by mtime)
  logfile=$(ls -1t openevolve_output/logs/openevolve_*.log 2>/dev/null | head -n1 || true)
  echo "Waiting for checkpoint at iteration $target_iters. Using log: ${logfile:-<none>}"

  while true; do
    # Refresh logfile in case a new one appears
    local latest
    latest=$(ls -1t openevolve_output/logs/openevolve_*.log 2>/dev/null | head -n1 || true)
    if [[ -n "$latest" && "$latest" != "$logfile" ]]; then
      logfile="$latest"
      echo "Switched to newest log: $logfile"
    fi

    if [[ -n "$logfile" ]] && rg -n "Saved checkpoint at iteration ${target_iters}\b" -S "$logfile" >/dev/null 2>&1; then
      echo "Detected checkpoint at iteration $target_iters in $logfile"
      break
    fi

    # Also break if best_program saved at the end and last_iteration >= target
    if [[ -f openevolve_output/checkpoints/checkpoint_${target_iters}/metadata.json ]]; then
      echo "Found checkpoint directory for iteration $target_iters"
      break
    fi

    sleep 60
  done
}

echo "[auto] Ensuring 30-iteration run, then chaining 200-iteration run..."

# 1) Ensure 30-iteration run is active (or already finished)
if [[ -d openevolve_output/checkpoints/checkpoint_30 ]]; then
  echo "30-iteration checkpoint already exists."
else
  start_run 30
  wait_for_checkpoint 30
fi

# 2) Start 200-iteration run (always ensure running)
start_run 200
wait_for_checkpoint 200

echo "[auto] 200-iteration run reached checkpoint 200. Done."

