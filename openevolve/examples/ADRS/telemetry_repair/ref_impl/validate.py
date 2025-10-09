#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import pandas as pd
import json
import sys
import os
# Add parent directory to path for imports from root level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add current directory to path for common_utils import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Tuple
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import plot_validation_results
from scripts.snapshot_cache import SnapshotCache
from common_utils import NetworkTopology, perc_equal, compute_predicted_counters, load_paths_for_timestamp

@dataclass
class ValidatorConfig:
    confidence_cutoff: float = 0.0
    threshold: float = 0.03  # For perc_equal comparisons
    default_confidence: float = 0.0
    paths_perturbed_num_nodes: int = 0  # For evaluation only
    disable_cache: bool = False

class Validator:
    def __init__(self, topology: Dict, config: ValidatorConfig = ValidatorConfig()):
        self.config = config
        self.network_topology = NetworkTopology(topology)
        
        if self.config.paths_perturbed_num_nodes > 0:
            print(f"Warning: paths_perturbed_num_nodes is set to {self.config.paths_perturbed_num_nodes}, which may affect validation results. Intended for buggy eval purpose only!")
        
        self.cache = SnapshotCache("cache/validation", disable_cache=self.config.disable_cache)

    def _create_corrected_snapshot(self, row: pd.Series) -> Tuple[Dict, Dict, int, int]:
        """Create a snapshot from a row, preferring corrected values over perturbed values (over ground_truth)."""
        snapshot = {}
        confidences = {}  # Store confidence values for each interface
        corrected_count = 0
        perturbed_count = 0
        
        # Copy non-dictionary columns as is
        snapshot['timestamp'] = row['timestamp']
        snapshot['telemetry_perturbed_type'] = row['telemetry_perturbed_type']
        snapshot['input_perturbed_type'] = row['input_perturbed_type']
        snapshot['true_detect_inconsistent'] = row['true_detect_inconsistent']
        
        # Extract values from all columns
        for col in row.index:
            if col not in snapshot:  # Skip the columns we already processed
                val = row[col]
                if isinstance(val, dict):
                    # For dictionary columns (low-level counters), prefer corrected over perturbed
                    if 'corrected' in val and val['corrected'] is not None and isinstance(val['corrected'], (int, float)):
                        snapshot[col] = val['corrected']
                        corrected_count += 1
                        # Store confidence for each interface
                        confidences[col] = val.get('confidence', self.config.default_confidence)
                    # Fall back to perturbed value
                    elif 'perturbed' in val and val['perturbed'] is not None:
                        snapshot[col] = val['perturbed']
                        confidences[col] = self.config.default_confidence
                        perturbed_count += 1
                    # If both are None or invalid, skip this value
                    elif 'ground_truth' in val and val['ground_truth'] is not None:
                        snapshot[col] = val['ground_truth']
                        confidences[col] = self.config.default_confidence
                        # Treat this as a perturbed value
                        perturbed_count += 1
                    else:
                        print(f"Error: no value at all for {col}")
                elif col.startswith('high_'):
                    # For high-level demands, use the value directly if it's numeric
                    # high-level demands are np.float64, so we need to check for that
                    if not pd.isna(val):
                        # convert val to float from np.float64
                        snapshot[col] = float(val)
        
        return snapshot, confidences, corrected_count, perturbed_count
    
    def get_expected_values(self, snapshot: Dict, paths: Dict = None):
        """Get expected values for links in the network based on demand and path information if available."""
        expected_values = {} # key is (src_id, dst_id), value is expected value

        for node_id in self.network_topology.get_all_node_ids():
            expected_values[(-1, node_id)] = 0
            expected_values[(node_id, -1)] = 0
        
        for src_id in self.network_topology.get_all_node_ids():
            for dst_id in self.network_topology.get_all_node_ids():
                if src_id == dst_id:
                    continue
                
                demand_key = f"high_{self.network_topology.get_node_name(src_id)}_{self.network_topology.get_node_name(dst_id)}"
                demand = snapshot[demand_key]
                expected_values[(-1, src_id)] += demand
                expected_values[(dst_id, -1)] += demand
        
        if paths is not None:
            path_predicted_values = compute_predicted_counters(
                snapshot, paths, self.network_topology, 
                paths_perturbed_num_nodes=self.config.paths_perturbed_num_nodes
            )
            for link_id, val in expected_values.items():
                # Note: this is a *sanity check*, as path-predicted external link values should match.
                assert val == path_predicted_values[link_id], f"Expected {val} but got {path_predicted_values[link_id]} for link {link_id}"
            return path_predicted_values
        
        return expected_values
    
    
    def validate_row(self, row: pd.Series, paths: Dict = None):
        """Internal method to validate a snapshot using path-based predictions."""
        snapshot, confidences, _, _ = self._create_corrected_snapshot(row)
        expected_values = self.get_expected_values(snapshot, paths)
        
        # Compare each counter with its prediction
        total_checks = 0
        satisfied_checks = 0
        
        for (src_id, dst_id), predicted_value in expected_values.items():
            send_key, recv_key = self.network_topology.get_snapshot_keys_for_link((src_id, dst_id))
            
            for key in (v for v in [send_key, recv_key] if v is not None):
                if confidences[key] >= self.config.confidence_cutoff:
                    total_checks += 1
                    if perc_equal(snapshot[key], predicted_value, threshold=self.config.threshold):
                        satisfied_checks += 1
                        
        satisfaction_rate = satisfied_checks / total_checks if total_checks > 0 else None
        
        # Update row with validation results
        output_row = row.copy()
        
        output_row['validation_type'] = "w/ paths" if paths is not None else "w/o paths"
        output_row['validation_result'] = None  # As specified
        output_row['validation_confidence'] = satisfaction_rate
        
        return output_row
        
    
    def validate_df(self, df: pd.DataFrame, paths_path: str = None, paths_dict: Dict = None) -> pd.DataFrame:
        """
        Validate an entire DataFrame with automatic caching and parallelization.
        
        Args:
            df: DataFrame containing snapshot data to validate
            paths_path: Optional path to directory/file containing paths data
            paths_dict: Optional paths dictionary (used for all rows if provided)
            
        Returns:
            DataFrame with validation results
        """
        # Check cache first
        result = self.cache.get_result(df, "validate_df", self.network_topology.get_uuid(), paths_path, self.config)
        if result is not None:
            return result
        
        # Parallel execution using df.iloc[i] so we pass Series objects
        with ProcessPoolExecutor() as executor:
            if paths_dict is not None:
                # Use the same paths dict for all rows
                futures = [
                    executor.submit(self.validate_row, df.iloc[i], paths_dict)
                    for i in range(len(df))
                ]
            elif paths_path is not None:
                # Load paths per timestamp
                futures = []
                for i in range(len(df)):
                    row = df.iloc[i]
                    try:
                        paths = load_paths_for_timestamp(paths_path, row['timestamp'])
                        futures.append(executor.submit(self.validate_row, row, paths))
                    except ValueError as e:
                        print(f"Warning: {e}, skipping row {i}")
                        futures.append(None)
            else:
                # Demand invariant validation (no paths)
                futures = [
                    executor.submit(self.validate_row, df.iloc[i], None)
                    for i in range(len(df))
                ]
            
            # Collect results
            validated_rows = []
            for i, future in enumerate(tqdm(futures, desc="Validating rows")):
                if future is not None:
                    try:
                        validated_rows.append(future.result())
                    except Exception as e:
                        print(f"Warning: Error validating row {i}: {e}")
        
        # Convert Series objects to DataFrame properly
        if validated_rows:
            result_df = pd.concat(validated_rows, axis=1).T.reset_index(drop=True)
        else:
            result_df = pd.DataFrame()
        result_df.attrs['config'] = self.config
        # Store in cache
        self.cache.store_result(df, result_df, "validate_df", self.network_topology.get_uuid(), paths_path, self.config)
        
        return result_df


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Validate network data using demand invariants')
    parser.add_argument('input_path', help='Path to the .pkl file containing network data')
    # Output name is optional, defaulting to 'validated_'
    parser.add_argument('--output_name', help='String to prepend to the output filename', default='validated')
    # Skip invariant checks for interfaces with counter confidence below this value. Default is 0.0 (no skipping).
    parser.add_argument('--confidence_cutoff', type=float, default=0.0, help='Confidence cutoff for invariant checking - which counters to not use due to confidence too low.')
    # Get timestamp for output file
    parser.add_argument('--no_paths', action='store_true', help='Do not use paths for validation')
    parser.add_argument('--plot_results', action='store_true', help='Generate validation plots')
    parser.add_argument('--same_dir', action='store_true', help='Save output files in same directory as input file')
    parser.add_argument('--paths_path', help='Directory containing paths files, or path to a single paths file')
    parser.add_argument('--topology_file', help='Path to the topology file (JSON format)', default='ref_data/abilene/topology.json')
    parser.add_argument('--paths_perturbed_num_nodes', type=int, default=0, help='Number of nodes in the perturbed paths (default: 0)')
    args = parser.parse_args()
    
    # Get timestamp for output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config = ValidatorConfig(
        confidence_cutoff=args.confidence_cutoff,
        paths_perturbed_num_nodes=args.paths_perturbed_num_nodes,
        disable_cache=False
    )
    
    # Load data & topology
    print(f"Reading data from {args.input_path}")
    df = pd.read_pickle(args.input_path)

    with open(args.topology_file) as f:
        topology = json.load(f)
    
    # Validate using Validator.
    validator = Validator(topology, config)
    validated_df = validator.validate_df(df, args.paths_path, paths_dict=None)
    
    # Determine output directory
    if args.same_dir:
        results_dir = os.path.dirname(args.input_path)
    else:
        results_dir = os.path.join('results', f'validation_run_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
    
    # Get input filename without extension
    input_name = os.path.splitext(os.path.basename(args.input_path))[0]
    output_path = os.path.join(results_dir, f"{args.output_name}_{input_name}_cutoff={args.confidence_cutoff}{'' if args.no_paths else '+paths'}_{timestamp}.pkl")
    validated_df.to_pickle(output_path)
    print(f"\nValidated data saved to {output_path}")
    
    # Generate plots if requested
    if args.plot_results:
        print("\nGenerating validation plots...")
        plot_validation_results.process_results(output_path, results_dir)

if __name__ == "__main__":
    main()
