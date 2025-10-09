#!/bin/bash
set -euo pipefail

# Auto-detect repo root
if [ -d "$HOME/sky_workdir" ] && [ -f "$HOME/sky_workdir/openevolve-run.py" ]; then
  REPO_ROOT="$HOME/sky_workdir"
elif [ -d "$HOME/RealSysBench" ] && [ -f "$HOME/RealSysBench/openevolve-run.py" ]; then
  REPO_ROOT="$HOME/RealSysBench"
else
  echo "ERROR: Cannot find RealSysBench repository" >&2
  exit 1
fi

echo "Using REPO_ROOT: $REPO_ROOT"
SCRIPT_DIR="$REPO_ROOT/examples/cant-be-late"

# Allow override; default to greedy seed to break from UP
INITIAL_PROGRAM="${INITIAL_PROGRAM:-$SCRIPT_DIR/initial_greedy.py}"
OUTPUT_DIR="${OUTPUT_DIR:-openevolve_output_greedy_start}"
ITERATIONS=${1:-100}
shift 2>/dev/null || true
EXTRA_ARGS=("$@")

# should always use real30 evaluator
EVALUATOR_PATH="$SCRIPT_DIR/evaluator_real30.py"

# Load repo-level .env if present (optional)
if [ -f "$REPO_ROOT/.env" ]; then
  echo "Loading environment variables from $REPO_ROOT/.env"
  set -a; source "$REPO_ROOT/.env"; set +a
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export LOG_LEVEL="INFO"

mkdir -p "$OUTPUT_DIR"/{best,checkpoints,logs}

echo "Starting single-region strategy evolution (iterations: $ITERATIONS)..."

uv run "$REPO_ROOT/openevolve-run.py" \
  "$INITIAL_PROGRAM" \
  "$EVALUATOR_PATH" \
  --config config.yaml \
  --output "$OUTPUT_DIR" \
  --iterations "$ITERATIONS" \
  --log-level INFO \
  "${EXTRA_ARGS[@]}"

echo
echo "Evolution complete! Check $OUTPUT_DIR/best/ for the best evolved strategy."
