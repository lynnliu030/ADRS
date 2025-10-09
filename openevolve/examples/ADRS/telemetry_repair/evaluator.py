"""
Enhanced evaluator for network input validation repair.

Uses the Dataset class from input-validation-eval to load abilene_clean data,
applies comprehensive perturbations (zeroing, scaling up/down, correlated, random),
and measures how well the repair algorithm can restore the original values.
"""

import importlib.util
import os
import sys
import json
import csv
import ast
import random
from typing import Dict, Any, List, Tuple
import traceback
import pandas as pd
import numpy as np


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Enhanced evaluation function using Dataset class and comprehensive perturbation scenarios.
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary with performance metrics across multiple perturbation scenarios
    """
    
    try:
        # Import the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Import Dataset class from input-validation-eval
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "input-validation-eval", "eval")
        sys.path.insert(0, dataset_path)
        
        try:
            from dataset import Dataset
        except ImportError:
            print("Could not import Dataset class. Falling back to old evaluation method.")
            return evaluate_legacy(program_path)
        
        # Define paths to abilene data
        telemetry_path = os.path.join(current_dir, "input-validation-eval", "ref_data", "abilene", "clean", "abilene_telem_clean.pkl")
        topology_path = os.path.join(current_dir, "input-validation-eval", "ref_data", "abilene", "topology.json")
        paths_path = os.path.join(current_dir, "input-validation-eval", "ref_data", "abilene", "paths.json")
        
        # Check if files exist
        if not all(os.path.exists(p) for p in [telemetry_path, topology_path, paths_path]):
            print("Abilene data files not found. Falling back to old evaluation method.")
            return evaluate_legacy(program_path)
        
        print("Loading abilene_clean dataset...")
        
        # Test multiple perturbation scenarios
        scenarios = [
            {
                "name": "zeroing_random",
                "method": "random", 
                "scale_factor": (0.0, 0.0),
                "percent_counters": 25.0,
                "description": "Random zeroing of 25% counters"
            },
            {
                "name": "zeroing_correlated", 
                "method": "correlated",
                "scale_factor": (0.0, 0.0),
                "percent_counters": 30.0,
                "description": "Correlated zeroing of 30% nodes"
            },
            {
                "name": "scaling_down_random",
                "method": "random",
                "scale_factor": (0.5, 0.8),
                "percent_counters": 20.0,
                "description": "Random scaling down (50-80%) of 20% counters"
            },
            {
                "name": "scaling_up_random",
                "method": "random", 
                "scale_factor": (1.2, 1.5),
                "percent_counters": 15.0,
                "description": "Random scaling up (120-150%) of 15% counters"
            },
            {
                "name": "scaling_down_correlated",
                "method": "correlated",
                "scale_factor": (0.6, 0.7),
                "percent_counters": 25.0,
                "description": "Correlated scaling down (60-70%) of 25% nodes"
            },
            {
                "name": "scaling_up_correlated",
                "method": "correlated",
                "scale_factor": (1.3, 1.4),
                "percent_counters": 20.0,
                "description": "Correlated scaling up (130-140%) of 20% nodes"
            }
        ]
        
        scenario_results = {}
        total_score = 0.0
        total_scenarios = 0
        
        random.seed(42)
        np.random.seed(42)
        
        for scenario in scenarios:
            print(f"\n--- Testing scenario: {scenario['description']} ---")
            
            # Create fresh dataset for each scenario
            try:
                dataset = Dataset(telemetry_path, topology_path, paths_path)
            except Exception as e:
                print(f"Failed to load dataset for scenario {scenario['name']}: {e}")
                scenario_results[scenario["name"]] = {'score': 0.0, 'num_tests': 0}
                continue
            
            # Apply perturbation
            try:
                dataset.perturb_telemetry(
                    method=scenario["method"],
                    scale_factor=scenario["scale_factor"],
                    percent_rows=0.8,  # Perturb 80% of rows
                    percent_counters=scenario["percent_counters"],
                    random_seed=42 + total_scenarios  # Different seed per scenario
                )
            except Exception as e:
                print(f"Failed to apply perturbation for scenario {scenario['name']}: {e}")
                scenario_results[scenario["name"]] = {'score': 0.0, 'num_tests': 0}
                continue
            
            # Get perturbed data
            df = dataset.get_data()
            
            # Test on a sample of rows for comprehensive evaluation
            sample_size = min(500, len(df))
            sample_rows = df.sample(n=sample_size, random_state=42)
            
            scenario_score = 0.0
            scenario_tests = 0
            test_results = []
            
            for idx, row in sample_rows.iterrows():
                # Convert row to interface format
                interfaces = convert_dataframe_row_to_interfaces(row, dataset)
                
                if not interfaces:
                    continue
                
                # Evaluate this scenario
                score_dict = evaluate_interface_scenario(program, interfaces)
                test_results.append(score_dict)
                scenario_score += score_dict.get('combined_score', 0.0)
                scenario_tests += 1
            
            if scenario_tests > 0:
                avg_scenario_score = scenario_score / scenario_tests
                scenario_results[scenario["name"]] = {
                    'score': avg_scenario_score,
                    'num_tests': scenario_tests,
                    'counter_repair_accuracy': sum(r.get('counter_repair_accuracy', 0) for r in test_results) / len(test_results),
                    'status_repair_accuracy': sum(r.get('status_repair_accuracy', 0) for r in test_results) / len(test_results),
                    'confidence_calibration': sum(r.get('confidence_calibration', 0) for r in test_results) / len(test_results)
                }
                total_score += avg_scenario_score
                total_scenarios += 1
                print(f"Scenario '{scenario['name']}' average score: {avg_scenario_score:.3f}")
            else:
                scenario_results[scenario["name"]] = {'score': 0.0, 'num_tests': 0}
                print(f"Scenario '{scenario['name']}' had no valid tests")
        
        # Calculate overall metrics
        overall_score = total_score / total_scenarios if total_scenarios > 0 else 0.0
        
        # Weighted average of detailed metrics across scenarios
        total_counter_accuracy = sum(r['counter_repair_accuracy'] for r in scenario_results.values() if r['num_tests'] > 0)
        total_status_accuracy = sum(r['status_repair_accuracy'] for r in scenario_results.values() if r['num_tests'] > 0)
        total_confidence_calibration = sum(r['confidence_calibration'] for r in scenario_results.values() if r['num_tests'] > 0)
        valid_scenarios = sum(1 for r in scenario_results.values() if r['num_tests'] > 0)
        
        result = {
            'combined_score': overall_score,
            'counter_repair_accuracy': total_counter_accuracy / valid_scenarios if valid_scenarios > 0 else 0.0,
            'status_repair_accuracy': total_status_accuracy / valid_scenarios if valid_scenarios > 0 else 0.0,
            'confidence_calibration': total_confidence_calibration / valid_scenarios if valid_scenarios > 0 else 0.0,
            'num_scenarios': total_scenarios,
            'scenario_breakdown': scenario_results
        }
        
        print(f"\nOverall evaluation score: {overall_score:.3f}")
        return result
        
    except Exception as e:
        print(f"Enhanced evaluation failed: {str(e)}")
        traceback.print_exc()
        print("Falling back to legacy evaluation...")
        return evaluate_legacy(program_path)


def convert_dataframe_row_to_interfaces(row: pd.Series, dataset) -> Dict[str, Dict[str, Any]]:
    """
    Convert a DataFrame row with perturbed telemetry data to interface format.
    
    Args:
        row: Pandas Series representing one row from the dataset
        dataset: Dataset instance for accessing metadata
        
    Returns:
        Dictionary in interface format expected by repair algorithms
    """
    
    interfaces = {}
    
    # Process telemetry columns (low_* columns)
    for col in dataset.measurement_cols:
        if col.startswith('low_') and ('_egress_to_' in col or '_ingress_from_' in col):
            try:
                # Get telemetry data
                telemetry_data = row[col]
                
                if not isinstance(telemetry_data, dict) or 'ground_truth' not in telemetry_data:
                    continue
                
                # Extract interface information from column name
                if '_egress_to_' in col:
                    # This is TX data: low_SOURCE_egress_to_DEST
                    parts = col.replace('low_', '').split('_egress_to_')
                    if len(parts) != 2:
                        continue
                    source, dest = parts
                    
                    # Create or update interface entry for source
                    if_id = f"{source}_to_{dest}"
                    if if_id not in interfaces:
                        interfaces[if_id] = {
                            'interface_status': 'up',
                            'rx_rate': 0.0,
                            'tx_rate': 0.0,
                            'connected_to': f"{dest}_to_{source}",
                            'local_router': source,
                            'remote_router': dest
                        }
                    
                    # Use perturbed value if available, otherwise ground truth
                    tx_rate = telemetry_data.get('perturbed', telemetry_data['ground_truth'])
                    interfaces[if_id]['tx_rate'] = float(tx_rate) if tx_rate is not None else 0.0
                    interfaces[if_id]['_ground_truth_tx'] = float(telemetry_data['ground_truth'])
                    
                elif '_ingress_from_' in col:
                    # This is RX data: low_DEST_ingress_from_SOURCE  
                    parts = col.replace('low_', '').split('_ingress_from_')
                    if len(parts) != 2:
                        continue
                    dest, source = parts
                    
                    # Create or update interface entry for dest
                    if_id = f"{dest}_to_{source}"
                    if if_id not in interfaces:
                        interfaces[if_id] = {
                            'interface_status': 'up',
                            'rx_rate': 0.0,
                            'tx_rate': 0.0,
                            'connected_to': f"{source}_to_{dest}",
                            'local_router': dest,
                            'remote_router': source
                        }
                    
                    # Use perturbed value if available, otherwise ground truth
                    rx_rate = telemetry_data.get('perturbed', telemetry_data['ground_truth'])
                    interfaces[if_id]['rx_rate'] = float(rx_rate) if rx_rate is not None else 0.0
                    interfaces[if_id]['_ground_truth_rx'] = float(telemetry_data['ground_truth'])
                
            except (ValueError, KeyError) as e:
                # Skip malformed data
                print(f"Skipping malformed data in column {col}: {e}")
                continue
    
    return interfaces


def evaluate_legacy(program_path: str) -> Dict[str, float]:
    """
    Legacy evaluation function (renamed from original evaluate function).
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary with performance metrics
    """
    
    try:
        # Import the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Load raw data from CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_data, topology = load_network_data(current_dir)
        
        if not csv_data:
            print("No data loaded from CSV")
            return {"combined_score": 0.0, "error": 1.0}
        
        # Test multiple scenarios
        total_score = 0.0
        num_tests = 0
        test_results = []
        
        random.seed(42)
        # Test 30 random rows to keep evaluation fast
        test_rows = random.sample(csv_data, min(30, len(csv_data)))
        
        for row in test_rows:
            # Convert CSV row to interface format
            interfaces = convert_csv_row_to_interfaces(row)
            
            if not interfaces:
                continue
            
            # Evaluate this scenario
            score_dict = evaluate_interface_scenario(program, interfaces)
            test_results.append(score_dict)
            total_score += score_dict.get('combined_score', 0.0)
            num_tests += 1
            
            print(f"Test scenario: {score_dict.get('combined_score', 0.0):.3f}")
        
        avg_score = total_score / num_tests if num_tests > 0 else 0.0
        
        # Calculate additional metrics
        counter_repair_accuracy = sum(r.get('counter_repair_accuracy', 0) for r in test_results) / len(test_results) if test_results else 0.0
        status_repair_accuracy = sum(r.get('status_repair_accuracy', 0) for r in test_results) / len(test_results) if test_results else 0.0
        confidence_calibration = sum(r.get('confidence_calibration', 0) for r in test_results) / len(test_results) if test_results else 0.0
        
        return {
            'combined_score': avg_score,
            'counter_repair_accuracy': counter_repair_accuracy,
            'status_repair_accuracy': status_repair_accuracy,
            'confidence_calibration': confidence_calibration,
            'num_tests': num_tests
        }
        
    except Exception as e:
        print(f"Legacy evaluation failed: {str(e)}")
        traceback.print_exc()
        return {
            'combined_score': 0.0,
            'counter_repair_accuracy': 0.0,
            'status_repair_accuracy': 0.0,
            'confidence_calibration': 0.0,
            'num_tests': 0,
            'error': 1.0
        }


def load_network_data(base_dir: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Load network data from CSV and topology files.
    
    Args:
        base_dir: Base directory containing ref_data folder
        
    Returns:
        Tuple of (csv_rows, topology_dict)
    """
    
    data_file = os.path.join(base_dir, "ref_data", "evaluation_data.csv")
    topology_file = os.path.join(base_dir, "ref_data", "topology.json")
    
    # Load CSV data
    csv_data = []
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f)
            csv_data = list(reader)
    else:
        print(f"Data file not found: {data_file}")
    
    # Load topology
    topology = {}
    if os.path.exists(topology_file):
        with open(topology_file, 'r') as f:
            topology = json.load(f)
    
    return csv_data, topology


def convert_csv_row_to_interfaces(data_row: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Adapter function: Convert a CSV row to the interface format expected by repair algorithms.
    
    Args:
        data_row: Single row from the evaluation CSV
        
    Returns:
        Dictionary in interface format: 
        {
            'interface_id': {
                'interface_status': 'up'/'down',
                'rx_rate': float,
                'tx_rate': float, 
                'connected_to': 'other_interface_id',
                'local_router': 'router_name',
                'remote_router': 'router_name'
            }
        }
    """
    
    interfaces = {}
    
    # Parse telemetry columns that have the format: low_SOURCE_egress_to_DEST or low_SOURCE_ingress_from_DEST
    for col_name, col_value in data_row.items():
        if col_name.startswith('low_') and ('_egress_to_' in col_name or '_ingress_from_' in col_name):
            try:
                # Parse the JSON-like string value
                telemetry_data = ast.literal_eval(col_value)
                
                if 'ground_truth' not in telemetry_data:
                    continue
                
                # Extract interface information from column name
                if '_egress_to_' in col_name:
                    # This is TX data: low_SOURCE_egress_to_DEST
                    parts = col_name.replace('low_', '').split('_egress_to_')
                    if len(parts) != 2:
                        continue
                    source, dest = parts
                    
                    # Create or update interface entry for source
                    if_id = f"{source}_to_{dest}"
                    if if_id not in interfaces:
                        interfaces[if_id] = {
                            'interface_status': 'up',
                            'rx_rate': 0.0,
                            'tx_rate': 0.0,
                            'connected_to': f"{dest}_to_{source}",
                            'local_router': source,
                            'remote_router': dest
                        }
                    
                    # Use perturbed value if available, otherwise ground truth
                    tx_rate = telemetry_data['perturbed'] if telemetry_data['perturbed'] is not None else telemetry_data['ground_truth']
                    interfaces[if_id]['tx_rate'] = float(tx_rate) if tx_rate is not None else 0.0
                    interfaces[if_id]['_ground_truth_tx'] = float(telemetry_data['ground_truth'])
                    
                elif '_ingress_from_' in col_name:
                    # This is RX data: low_DEST_ingress_from_SOURCE  
                    parts = col_name.replace('low_', '').split('_ingress_from_')
                    if len(parts) != 2:
                        continue
                    dest, source = parts
                    
                    # Create or update interface entry for dest
                    if_id = f"{dest}_to_{source}"
                    if if_id not in interfaces:
                        interfaces[if_id] = {
                            'interface_status': 'up',
                            'rx_rate': 0.0,
                            'tx_rate': 0.0,
                            'connected_to': f"{source}_to_{dest}",
                            'local_router': dest,
                            'remote_router': source
                        }
                    
                    # Use perturbed value if available, otherwise ground truth
                    rx_rate = telemetry_data['perturbed'] if telemetry_data['perturbed'] is not None else telemetry_data['ground_truth']
                    interfaces[if_id]['rx_rate'] = float(rx_rate) if rx_rate is not None else 0.0
                    interfaces[if_id]['_ground_truth_rx'] = float(telemetry_data['ground_truth'])
                
            except (ValueError, SyntaxError, KeyError) as e:
                # Skip malformed data
                print(f"Skipping malformed data: {e}")
                continue
    
    return interfaces


def evaluate_interface_scenario(program, interfaces: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate repair algorithm on a single interface scenario.
    
    Args:
        program: The imported repair program module
        interfaces: Interface data in the expected format
        
    Returns:
        Dictionary with scenario metrics
    """
    
    try:
        topology = {}
        for interface_id, telemetry in interfaces.items():
            local_router = telemetry.get('local_router')
            if local_router not in topology:
                topology[local_router] = []
            topology[local_router].append(interface_id)
        # Apply some additional perturbations to test repair (beyond what's already in data)
        perturbed_interfaces, ground_truth = apply_test_perturbations(interfaces, topology)
        
        # Run the repair algorithm
        repaired_interfaces = program.run_repair(perturbed_interfaces, topology)
        
        # Calculate how well the repair worked
        counter_repair_score = calculate_counter_repair_quality(repaired_interfaces, ground_truth)
        status_repair_score = calculate_status_repair_quality(repaired_interfaces, ground_truth)
        confidence_calibration_score = calculate_confidence_calibration(repaired_interfaces, ground_truth)
        
        # Combined score weights counter repair quality more heavily, with status repair included
        combined_score = counter_repair_score * 0.75 + status_repair_score * 0.05 + confidence_calibration_score * 0.2
        
        return {
            'combined_score': combined_score,
            'counter_repair_accuracy': counter_repair_score,
            'status_repair_accuracy': status_repair_score,
            'confidence_calibration': confidence_calibration_score
        }
        
    except Exception as e:
        print(f"Interface scenario evaluation failed: {e}")
        return {'combined_score': 0.0, 'counter_repair_accuracy': 0.0, 'status_repair_accuracy': 0.0, 'confidence_calibration': 0.0}


def apply_test_perturbations(interfaces: Dict[str, Dict[str, Any]], topology: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Apply additional perturbations for testing and keep ground truth.
    
    Args:
        interfaces: Original interface data
        topology: Dictionary where key is router_id and value contains a list of interface_ids
    Returns:
        Tuple of (perturbed_interfaces, ground_truth_interfaces)
    """
    
    perturbed = {}
    ground_truth = {}
    
    # Bug type:
    # 1. Dropped counters (zeroed)
    # 2. Scaled counters (scaled, up or down)
    # 3. Correlated dropped counters (zero)
    # 4. Correlated scaled counters (scaled, up or down)
    
    bug_type = random.choice(['dropped', 'scaled', 'correlated_dropped', 'correlated_scaled'])
    
    # uncorrelated case:
    for if_id, if_data in interfaces.items():
        # Store ground truth
        ground_truth[if_id] = {
            'rx_rate': if_data.get('_ground_truth_rx', if_data['rx_rate']),
            'tx_rate': if_data.get('_ground_truth_tx', if_data['tx_rate']),
            'interface_status': if_data['interface_status'],
        }
        
        # Create perturbed version
        perturbed_data = if_data.copy()
        
        # Add some noise to all measurements (not bugs)
        perturbed_data['rx_rate'] *= random.uniform(0.98, 1.02)  # ±2% noise
        perturbed_data['tx_rate'] *= random.uniform(0.98, 1.02)  # ±2% noise
        

        if bug_type == 'dropped':
            if random.random() < 0.20:
                perturbed_data['rx_rate'] = 0
                perturbed_data['tx_rate'] = 0
        elif bug_type == 'scaled':
            if random.random() < 0.20:
                perturbed_data['rx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
                perturbed_data['tx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
        
        # Remove ground truth metadata
        perturbed_data.pop('_ground_truth_rx', None)
        perturbed_data.pop('_ground_truth_tx', None)
        
        perturbed[if_id] = perturbed_data
    
    # correlated case:
    for _, if_ids in topology.items():
        if random.random() < 0.80: # 20% chance of correlated router bug
            continue
        if bug_type == 'correlated_dropped':
            for if_id in if_ids:
                perturbed_data = interfaces[if_id].copy()
                perturbed_data['rx_rate'] = 0
                perturbed_data['tx_rate'] = 0
                perturbed[if_id] = perturbed_data
        elif bug_type == 'correlated_scaled':
            for if_id in if_ids:
                perturbed_data = interfaces[if_id].copy()
                perturbed_data['rx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
                perturbed_data['tx_rate'] *= random.choice([random.uniform(0.5, 0.8), random.uniform(1.2, 1.5)])
                perturbed[if_id] = perturbed_data
    
    return perturbed, ground_truth


def calculate_counter_repair_quality(repaired_interfaces: Dict[str, Dict], ground_truth: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate how well the repair algorithm restored the ground truth values.
    
    Args:
        repaired_interfaces: Output from repair algorithm  
        ground_truth: Original correct values
        
    Returns:
        Repair quality score (0.0 to 1.0)
    """
    
    total_error = 0.0
    num_measurements = 0
    
    for if_id in ground_truth:
        if if_id not in repaired_interfaces:
            continue
            
        gt = ground_truth[if_id]
        repaired = repaired_interfaces[if_id]
        
        # Compare RX rates
        if 'rx_rate' in gt:
            gt_rx = gt['rx_rate']
            # Extract repaired value from tuple format (original, repaired, confidence)
            if isinstance(repaired.get('rx_rate'), tuple) and len(repaired['rx_rate']) >= 2:
                rep_rx = repaired['rx_rate'][1]  # repaired value
            else:
                rep_rx = repaired.get('rx_rate', 0.0)
            
            if gt_rx > 0:  # Only measure error on non-zero ground truth
                error = abs(rep_rx - gt_rx) / gt_rx
                total_error += min(error, 1.0)  # Cap error at 100%
                num_measurements += 1
        
        # Compare TX rates
        if 'tx_rate' in gt:
            gt_tx = gt['tx_rate']
            if isinstance(repaired.get('tx_rate'), tuple) and len(repaired['tx_rate']) >= 2:
                rep_tx = repaired['tx_rate'][1]  # repaired value
            else:
                rep_tx = repaired.get('tx_rate', 0.0)
            
            if gt_tx > 0:  # Only measure error on non-zero ground truth
                error = abs(rep_tx - gt_tx) / gt_tx
                total_error += min(error, 1.0)  # Cap error at 100%
                num_measurements += 1
    
    if num_measurements == 0:
        return 0.0
    
    avg_error = total_error / num_measurements
    repair_quality = 1.0 - avg_error  # Convert error to quality score
    return max(0.0, repair_quality)


def calculate_status_repair_quality(repaired_interfaces: Dict[str, Dict], ground_truth: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate how well the repair algorithm restored interface status values.
    
    Args:
        repaired_interfaces: Output from repair algorithm  
        ground_truth: Original correct values
        
    Returns:
        Status repair quality score (0.0 to 1.0)
    """
    
    correct_status_repairs = 0
    total_status_measurements = 0
    
    for if_id in ground_truth:
        if if_id not in repaired_interfaces:
            continue
            
        gt = ground_truth[if_id]
        repaired = repaired_interfaces[if_id]
        
        # Check interface status repair
        if 'interface_status' in gt:
            gt_status = gt['interface_status']
            
            # Extract repaired status from tuple format (original, repaired, confidence)
            if isinstance(repaired.get('interface_status'), tuple) and len(repaired['interface_status']) >= 2:
                rep_status = repaired['interface_status'][1]  # repaired value
            else:
                rep_status = repaired.get('interface_status', 'unknown')
            
            total_status_measurements += 1
            if rep_status == gt_status:
                correct_status_repairs += 1
    
    if total_status_measurements == 0:
        return 1.0  # Perfect score if no status to evaluate
    
    return correct_status_repairs / total_status_measurements


def calculate_confidence_calibration(repaired_interfaces: Dict[str, Dict],
                                     ground_truth: Dict[str, Dict[str, float]]) -> float:
    """
    Evaluate how well confidence scores reflect repair accuracy.
    Penalizes cases where:
    - Counter is close to correct but we have low confidence (missed opportunity)
    - Counter is far off but we have high confidence (dangerous overconfidence)
    
    Args:
        repaired_interfaces: Output from repair algorithm
        ground_truth: Original correct values
        
    Returns:
        Confidence calibration quality score (0.0 to 1.0)
    """
    
    total_score = 0.0
    total_measurements = 0
    
    for if_id in ground_truth:
        if if_id not in repaired_interfaces:
            continue
            
        gt = ground_truth[if_id]
        repaired = repaired_interfaces[if_id]
        
        # Check RX rate confidence calibration
        if 'rx_rate' in gt:
            gt_rx = gt['rx_rate']
            
            if isinstance(repaired.get('rx_rate'), tuple) and len(repaired['rx_rate']) >= 3:
                _, rep_rx, confidence = repaired['rx_rate']
                
                # Calculate repair accuracy (0.0 = perfect, 1.0 = completely wrong)
                max_val = max(abs(gt_rx), abs(rep_rx), 1.0)  # Avoid division by zero
                repair_error = abs(gt_rx - rep_rx) / max_val
                repair_accuracy = 1.0 - repair_error
                
                # Score based on confidence-accuracy alignment
                if repair_accuracy > 0.8:  # Good repair
                    if confidence > 0.7:  # High confidence - perfect alignment
                        score = 1.0
                    else:  # Low confidence - missed opportunity
                        score = 0.5 + (confidence - 0.3) * 1.25  # Penalty for being too conservative
                else:  # Poor repair (repair_accuracy <= 0.8)
                    if confidence < 0.3:  # Low confidence - appropriate uncertainty
                        score = 0.8
                    else:  # High confidence - dangerous overconfidence
                        overconfidence_penalty = confidence * (1.0 - repair_accuracy)
                        score = max(0.0, 0.8 - overconfidence_penalty * 2.0)
                
                total_score += max(0.0, min(1.0, score))
                total_measurements += 1
        
        # Check TX rate confidence calibration
        if 'tx_rate' in gt:
            gt_tx = gt['tx_rate']
            
            if isinstance(repaired.get('tx_rate'), tuple) and len(repaired['tx_rate']) >= 3:
                _, rep_tx, confidence = repaired['tx_rate']
                
                max_val = max(abs(gt_tx), abs(rep_tx), 1.0)
                repair_error = abs(gt_tx - rep_tx) / max_val
                repair_accuracy = 1.0 - repair_error
                
                if repair_accuracy > 0.8:  # Good repair
                    if confidence > 0.7:  # High confidence - perfect alignment
                        score = 1.0
                    else:  # Low confidence - missed opportunity
                        score = 0.5 + (confidence - 0.3) * 1.25
                else:  # Poor repair
                    if confidence < 0.3:  # Low confidence - appropriate uncertainty  
                        score = 0.8
                    else:  # High confidence - dangerous overconfidence
                        overconfidence_penalty = confidence * (1.0 - repair_accuracy)
                        score = max(0.0, 0.8 - overconfidence_penalty * 2.0)
                
                total_score += max(0.0, min(1.0, score))
                total_measurements += 1
    
    if total_measurements == 0:
        return 1.0
    
    return total_score / total_measurements


# ============================================================================
# STANDALONE TESTING FUNCTIONS
# ============================================================================

def test_data_adapter():
    """
    Test the data adapter function independently.
    """
    print("Testing data adapter...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_data, topology = load_network_data(current_dir)
    
    if not csv_data:
        print("No data loaded!")
        return
        
    # Test converting first few rows
    for i, row in enumerate(csv_data[:3]):
        print(f"\n--- Row {i+1} ---")
        interfaces = convert_csv_row_to_interfaces(row)
        print(f"Extracted {len(interfaces)} interfaces:")
        
        for if_id, if_data in interfaces.items():
            print(f"  {if_id}: rx={if_data['rx_rate']:.1f}, tx={if_data['tx_rate']:.1f}")


def test_repair_evaluation():
    """
    Test the evaluation logic independently.
    """
    print("Testing repair evaluation...")
    
    # Create some mock interface data  
    interfaces = {
        'RouterA_to_RouterB': {
            'interface_status': 'up',
            'rx_rate': 100.0,
            'tx_rate': 95.0,
            'connected_to': 'RouterB_to_RouterA',
            'local_router': 'RouterA',
            'remote_router': 'RouterB',
            '_ground_truth_rx': 100.0,
            '_ground_truth_tx': 95.0
        },
        'RouterB_to_RouterA': {
            'interface_status': 'up', 
            'rx_rate': 95.0,
            'tx_rate': 100.0,
            'connected_to': 'RouterA_to_RouterB',
            'local_router': 'RouterB',
            'remote_router': 'RouterA',
            '_ground_truth_rx': 95.0,
            '_ground_truth_tx': 100.0
        }
    }
    
    topology = {}
    for interface_id, telemetry in interfaces.items():
        local_router = telemetry.get('local_router')
        if local_router not in topology:
            topology[local_router] = []
        topology[local_router].append(interface_id)
    
    # Test the evaluation without the actual repair program
    perturbed, ground_truth = apply_test_perturbations(interfaces, topology)
    
    print(f"Ground truth interfaces: {len(ground_truth)}")
    print(f"Perturbed interfaces: {len(perturbed)}")
    
    for if_id in ground_truth:
        gt = ground_truth[if_id]
        pert = perturbed[if_id]
        print(f"{if_id}:")
        print(f"  GT: rx={gt['rx_rate']:.1f}, tx={gt['tx_rate']:.1f}")
        print(f"  Perturbed: rx={pert['rx_rate']:.1f}, tx={pert['tx_rate']:.1f}")


if __name__ == "__main__":
    # Test individual components
    print("=== Testing Data Adapter ===")
    test_data_adapter()
    
    print("\n=== Testing Repair Evaluation ===")  
    test_repair_evaluation()
    
    print("\n=== Testing Full Integration ===")
    # Test the full evaluator
    current_dir = os.path.dirname(os.path.abspath(__file__))
    initial_program_path = os.path.join(current_dir, "initial_program.py")
    
    if os.path.exists(initial_program_path):
        print("Testing evaluator with real network data...")
        metrics = evaluate(initial_program_path)
        
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print("initial_program.py not found - cannot test full evaluation")
