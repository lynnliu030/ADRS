#!/bin/bash

# Script to sync results back from SkyPilot clusters to local directories
# Run this after SkyPilot jobs complete to get the results locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# Function to sync results for a specific config
sync_config_results() {
    local config_dir="$1"
    local cluster_name="$2"
    
    echo "Syncing results for: $config_dir"
    
    if [ ! -d "$REPO_ROOT/$config_dir" ]; then
        echo "Warning: Config directory $REPO_ROOT/$config_dir does not exist locally"
        return 1
    fi
    
    # Check if cluster exists
    if ! sky status | grep -q "$cluster_name"; then
        echo "Warning: Cluster $cluster_name not found. It may have been terminated."
        echo "Checking if results are already in skypilot_results backup..."
        
        # Check for backup results
        if [ -d "$REPO_ROOT/skypilot_results" ]; then
            backup_pattern="$(basename "$config_dir")_*"
            backup_dirs=$(find "$REPO_ROOT/skypilot_results" -name "$backup_pattern" -type d 2>/dev/null | sort -r | head -1)
            
            if [ -n "$backup_dirs" ]; then
                echo "Found backup results: $backup_dirs"
                echo "Copying to local config directory..."
                cp -r "$backup_dirs"/* "$REPO_ROOT/$config_dir/" 2>/dev/null || echo "No files to copy"
                echo "Results synced from backup successfully"
                return 0
            fi
        fi
        
        echo "No backup results found for $config_dir"
        return 1
    fi
    
    # Sync results using rsync from the cluster
    echo "Downloading results from cluster: $cluster_name"
    
    # Use the correct rsync command format from SkyPilot docs
    rsync -Pavz "$cluster_name:~/sky_workdir/$config_dir/openevolve_output/" "$REPO_ROOT/$config_dir/openevolve_output/" || echo "Failed to sync openevolve_output"
    rsync -Pavz "$cluster_name:~/sky_workdir/$config_dir/run.log" "$REPO_ROOT/$config_dir/" || echo "Failed to sync run.log"
    
    echo "Results synced successfully for $config_dir"
    echo ""
}

echo "Syncing results from SkyPilot jobs to local directories..."
echo ""

# Sync exploration ratio ablations
# for config_dir in exploration_ratio/cfg_*; do
#     if [ -d "$config_dir" ]; then
#         cluster_name="evolve-$(basename "$config_dir")"
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         sync_config_results "$full_config_path" "$cluster_name"
#     fi
# done

# # Sync model ensemble ablations  
# for config_dir in model_ensemble/cfg_*; do
#     if [ -d "$config_dir" ]; then
#         cluster_name="evolve-$(basename "$config_dir")"
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         sync_config_results "$full_config_path" "$cluster_name"
#     fi
# done

# Sync elite selection ratio ablations
# for config_dir in elite_selection_ratio/elite_*; do
#     if [ -d "$config_dir" ]; then
#         cluster_name="evolve-$(basename "$config_dir")"
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         sync_config_results "$full_config_path" "$cluster_name"
#     fi
# done

# # Sync migration ratio ablations
# for config_dir in migration_ratio/migration_*; do
#     if [ -d "$config_dir" ]; then
#         cluster_name="evolve-$(basename "$config_dir")"
#         full_config_path="examples/cant-be-late/ablations/$config_dir"
#         sync_config_results "$full_config_path" "$cluster_name"
#     fi
# done

# Sync best so far ablations (excluding gem_0.8_expl_0.3)
for config_dir in best_so/gem_0.8_expl_0.3_cant_be_late; do
    if [ -d "$config_dir" ]; then
        cluster_name="evolve-$(basename "$config_dir")"
        echo $cluster_name
        full_config_path="examples/cant-be-late/ablations/$config_dir"
        sync_config_results "$full_config_path" "$cluster_name"
    fi
done

echo "Results sync completed!"
echo ""
echo "Check the local config directories and skypilot_results/ for results."
echo "To view the status of all clusters, run:"
echo "  sky status"
echo ""
echo "To terminate clusters after syncing, run:"
echo "  sky down <cluster_name>  # or 'sky down -a' for all clusters"