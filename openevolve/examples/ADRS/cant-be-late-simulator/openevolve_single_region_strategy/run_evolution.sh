#!/bin/bash

# Resolve repository root (parent of this script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from repo-level .env file if present
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$REPO_ROOT/.env"
    set +a
fi

# Set up environment variables
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export LOG_LEVEL="INFO"

# Create output directories
mkdir -p openevolve_output/{best,checkpoints,logs}

# Run the evolution
echo "Starting single-region strategy evolution..."
echo "This will evolve strategies to beat rc_cr_threshold baseline"
echo "Target: >60% cost savings (baseline achieves ~55%)"
echo ""

# Run OpenEvolve using local checkout under @temp
OPENEVO_RUN="$REPO_ROOT/@temp/openevolve/openevolve-run.py"
if [ ! -f "$OPENEVO_RUN" ]; then
    echo "Error: $OPENEVO_RUN not found. Please clone OpenEvolve into @temp/openevolve." >&2
    echo "  git clone https://github.com/andylizf/openevolve.git @temp/openevolve" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
python "$OPENEVO_RUN" \
    initial_program.py \
    evaluator.py \
    --config config.yaml \
    --output openevolve_output \
    --iterations 100 \
    --log-level INFO

echo ""
echo "Evolution complete! Check openevolve_output/best/ for the best evolved strategy."