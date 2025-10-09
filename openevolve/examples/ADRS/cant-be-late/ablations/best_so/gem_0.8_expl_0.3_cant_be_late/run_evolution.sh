#!/bin/bash

# Auto-detect repo root - try sky_workdir first, then RealSysBench
if [ -d "$HOME/sky_workdir" ] && [ -f "$HOME/sky_workdir/openevolve-run.py" ]; then
    REPO_ROOT="$HOME/sky_workdir"
elif [ -d "$HOME/RealSysBench" ] && [ -f "$HOME/RealSysBench/openevolve-run.py" ]; then
    REPO_ROOT="$HOME/RealSysBench"
else
    echo "ERROR: Cannot find RealSysBench repository"
    exit 1
fi

echo "Using REPO_ROOT: $REPO_ROOT"
SCRIPT_DIR="$REPO_ROOT/examples/cant-be-late"
# Allow override from environment; default to initial_up.py
INITIAL_PROGRAM="${INITIAL_PROGRAM:-$SCRIPT_DIR/initial_up.py}"

# Default to 100 iterations if not set argv[1]
ITERATIONS=${1:-100}

# Load environment variables from repo-level .env file if present
if [ -f "$REPO_ROOT/.env" ]; then
    echo "Loading environment variables from $REPO_ROOT/.env"
    set -a
    # shellcheck disable=SC1090
    source "$REPO_ROOT/.env"
    set +a
fi

# Set up environment variables
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export LOG_LEVEL="INFO"

OUTPUT_DIR="${OUTPUT_DIR:-openevolve_output}"
# Create output directories
mkdir -p "$OUTPUT_DIR"/{best,checkpoints,logs}

# Run the evolution
echo "Starting single-region strategy evolution (iterations: $ITERATIONS)..."

uv run $REPO_ROOT/openevolve-run.py \
    $INITIAL_PROGRAM \
    $SCRIPT_DIR/evaluator.py \
    --config config.yaml \
    --output "$OUTPUT_DIR" \
    --iterations $ITERATIONS \
    --log-level INFO

echo ""
echo "Evolution complete! Check $OUTPUT_DIR/best/ for the best evolved strategy."
