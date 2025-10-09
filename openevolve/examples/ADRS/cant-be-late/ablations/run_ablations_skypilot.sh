#!/bin/bash

# Script to run all ablation experiments using SkyPilot
# This distributes the evolution jobs across remote instances to avoid OOM issues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
SKYPILOT_CONFIG="$SCRIPT_DIR/skypilot_evolve.yaml"

# Default iterations if not provided
ITERATIONS=${1:-100}

echo "Running ablation experiments with SkyPilot"
echo "Repository root: $REPO_ROOT"
echo "Iterations per job: $ITERATIONS"
echo ""

# Function to launch a SkyPilot cluster for a specific config
launch_job() {
    local config_dir="$1"
    local job_name="$2"
    
    echo "Launching cluster: $job_name for config: $config_dir"
    
    # Launch the cluster with the specific config directory in detached mode
    sky launch -c "$job_name" \
        --env CONFIG_DIR="$config_dir" \
        --env ITERATIONS="$ITERATIONS" \
        -y \
        --detach-run \
        "$SKYPILOT_CONFIG"
    
    echo "Cluster $job_name launched successfully"
    echo ""
}

# Find all config directories and launch jobs
echo "Scanning for ablation configurations..."

# Exploration ratio ablations
# for config_dir in exploration_ratio/cfg_*; do
#     if [ -d "$config_dir" ] && [ -f "$config_dir/config.yaml" ]; then
#         # Extract job name from directory path
#         job_name="evolve-$(basename "$config_dir")"
#         # Convert to full path for SkyPilot
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         launch_job "$full_config_path" "$job_name"
#     fi
# done

# # Model ensemble ablations
# for config_dir in model_ensemble/cfg_*; do
#     if [ -d "$config_dir" ] && [ -f "$config_dir/config.yaml" ]; then
#         # Extract job name from directory path
#         job_name="evolve-$(basename "$config_dir")"
#         # Convert to full path for SkyPilot
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         launch_job "$full_config_path" "$job_name"
#     fi
# done

# Elite selection ratio ablations
# for config_dir in elite_selection_ratio/elite_*; do
#     if [ -d "$config_dir" ] && [ -f "$config_dir/config.yaml" ]; then
#         # Extract job name from directory path
#         job_name="evolve-$(basename "$config_dir")"
#         # Convert to full path for SkyPilot
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         launch_job "$full_config_path" "$job_name"
#     fi
# done

# # Migration ratio ablations
# for config_dir in migration_ratio/migration_*; do
#     if [ -d "$config_dir" ] && [ -f "$config_dir/config.yaml" ]; then
#         # Extract job name from directory path
#         job_name="evolve-$(basename "$config_dir")"
#         # Convert to full path for SkyPilot
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         launch_job "$full_config_path" "$job_name"
#     fi
# done

# Best so far ablations (excluding gem_0.8_expl_0.3)
for config_dir in best_so/gem_0.8_expl_0.3_cant_be_late; do
    if [ -d "$config_dir" ] && [ -f "$config_dir/config.yaml" ]; then
        # Extract job name from directory path
        job_name="evolve-$(basename "$config_dir")"
        # Convert to full path for SkyPilot
        full_config_path="examples/cant-be-late/ablations/$config_dir"
        launch_job "$full_config_path" "$job_name"
    fi
done

echo "All ablation clusters have been launched!"
echo ""
echo "To monitor cluster status, run:"
echo "  sky status"
echo ""
echo "To view logs for a specific cluster, run:"
echo "  sky logs <cluster_name>"
echo ""
echo "To download results when clusters complete, run:"
echo "  sky down <cluster_name>  # This will sync results and terminate the cluster"
echo ""
echo "To terminate all clusters if needed, run:"
echo "  sky down -a"