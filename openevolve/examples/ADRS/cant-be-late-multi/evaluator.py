# openevolve_multi_region_strategy/evaluator.py

import os
import sys
import json
import subprocess
import logging
import re
import traceback
import random
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import time

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Go up one level

MAIN_SIMULATOR_PATH = os.path.join(PROJECT_ROOT, 'main.py')
DATA_PATH = os.path.join(PROJECT_ROOT, "data/converted_multi_region")
CACHE_DIR = Path(PROJECT_ROOT) / ".evaluator_cache"
CACHE_DIR.mkdir(exist_ok=True)

TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2
TIMEOUT_SECONDS = 600
WORST_POSSIBLE_SCORE = -1e9

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Dynamic trace count based on number of regions
def get_num_traces_for_scenario(num_regions: int) -> int:
    """Dynamically determine number of traces based on region count"""
    if num_regions <= 2:
        return 6   # 2 regions: 6 traces (slightly more than original 4)
    elif num_regions <= 3:
        return 5   # 3 regions: 5 traces (slightly more than original 3)
    elif num_regions <= 5:
        return 4   # 5 regions: 4 traces (double the original 2)
    else:
        return 3   # 9 regions: 3 traces

# Select traces from the available pool
def get_random_traces(num_traces: int, total_traces: int = 100) -> List[str]:
    """Select first N trace files (deterministic, not random)"""
    # Use fixed first N traces for reproducibility
    return [f"{i}.json" for i in range(min(num_traces, total_traces))]

# Full test scenarios for the final evaluation stage
FULL_TEST_SCENARIOS = [
    # Dynamic trace generation
    {"name": "2_zones_same_region", "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1"]},
    {"name": "2_regions_east_west", "regions": ["us-east-2a_v100_1", "us-west-2a_v100_1"]},
    {"name": "3_regions_diverse", "regions": ["us-east-1a_v100_1", "us-east-2b_v100_1", "us-west-2c_v100_1"]},
    {"name": "3_zones_same_region", "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1", "us-east-1d_v100_1"]},
    {"name": "5_regions_high_diversity", "regions": ["us-east-1a_v100_1", "us-east-1f_v100_1", "us-west-2a_v100_1", "us-west-2b_v100_1", "us-east-2b_v100_1"]},
    {"name": "all_9_regions", "regions": ["us-east-2a_v100_1", "us-west-2c_v100_1", "us-east-1d_v100_1", "us-east-2b_v100_1", "us-west-2a_v100_1", "us-east-1f_v100_1", "us-east-1a_v100_1", "us-west-2b_v100_1", "us-east-1c_v100_1"]}
]

# Add traces to each scenario
for scenario in FULL_TEST_SCENARIOS:
    num_regions = len(scenario["regions"])
    num_traces = get_num_traces_for_scenario(num_regions)
    scenario["traces"] = get_random_traces(num_traces)
    scenario["num_regions"] = num_regions

# A single, simple scenario for the quick first-stage evaluation
STAGE_1_SCENARIO = {
    "name": "stage_1_quick_check", 
    "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1"], 
    "traces": ["0.json"]
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cache_key(regions: List[str], trace_file: str, strategy: str = "quick_optimal") -> str:
    """Generate cache key"""
    content = f"{strategy}_{sorted(regions)}_{trace_file}_{TASK_DURATION_HOURS}_{DEADLINE_HOURS}"
    return hashlib.md5(content.encode()).hexdigest()

def get_or_create_union_trace(regions: List[str], trace_file: str) -> str:
    """Get or create union trace for optimal calculation"""
    cache_key = f"union_{'_'.join(sorted(regions))}_{trace_file}"
    union_path = CACHE_DIR / f"{cache_key}.json"
    
    if union_path.exists():
        return str(union_path)
    
    # Create union trace
    traces_data = []
    for region in regions:
        trace_path = Path(DATA_PATH) / region / trace_file
        if trace_path.exists():
            with open(trace_path) as f:
                data = json.load(f)
                traces_data.append(data["data"])
    
    if not traces_data:
        raise ValueError(f"No valid traces found for {regions} and {trace_file}")
    
    # Union: any region available = 1
    union_data = [int(any(traces_data[i][j] for i in range(len(traces_data)))) 
                  for j in range(len(traces_data[0]))]
    
    # Save union trace
    union_trace = {
        "data": union_data,
        "metadata": {
            "gap_seconds": 360,
            "regions": regions,
            "original_trace": trace_file
        }
    }
    
    with open(union_path, 'w') as f:
        json.dump(union_trace, f)
    
    return str(union_path)

def run_optimal_simulation(regions: List[str], trace_file: str, timeout: Optional[float] = None) -> float:
    """Run optimal strategy with cached results"""
    cache_key = get_cache_key(regions, trace_file)
    cache_file = CACHE_DIR / f"optimal_{cache_key}.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)["cost"]
    
    # Create union trace
    union_trace_path = get_or_create_union_trace(regions, trace_file)
    
    # Run optimal strategy
    cmd = [
        sys.executable,
        os.path.basename(MAIN_SIMULATOR_PATH),
        "--strategy=quick_optimal",
        "--env=trace",
        f"--trace-file={union_trace_path}",
        f"--task-duration-hours={TASK_DURATION_HOURS}",
        f"--deadline-hours={DEADLINE_HOURS}",
        f"--restart-overhead-hours={RESTART_OVERHEAD_HOURS}",
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout or TIMEOUT_SECONDS,
        cwd=PROJECT_ROOT,
    )
    
    output = result.stdout + result.stderr
    match = re.search(r"mean:\s*([\d.]+)", output)
    
    if match:
        cost = float(match.group(1))
        # Cache results
        with open(cache_file, 'w') as f:
            json.dump({"cost": cost, "regions": regions, "trace": trace_file}, f)
        return cost
    else:
        raise ValueError("Could not parse optimal cost")

def run_simulation(program_path: str, trace_files: List[str], timeout: Optional[float] = None) -> Dict[str, Union[str, float]]:
    """Run simulation with a specific strategy on given trace files."""
    cmd = [
        sys.executable,
        os.path.basename(MAIN_SIMULATOR_PATH),
        f"--strategy-file={program_path}",
        "--env=multi_trace",
        f"--task-duration-hours={TASK_DURATION_HOURS}",
        f"--deadline-hours={DEADLINE_HOURS}",
        f"--restart-overhead-hours={RESTART_OVERHEAD_HOURS}",
        "--trace-files",
    ] + trace_files

    try:
        # Using subprocess.run to execute the simulation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True, # Will raise CalledProcessError for non-zero exit codes
            timeout=timeout or TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
        )

        output = result.stdout + result.stderr
        match = re.search(r"mean:\s*([\d.]+)", output)
        
        if match:
            return {"status": "success", "cost": float(match.group(1)), "output": output}
        
        error_msg = f"Could not parse 'mean:' cost from simulation output."
        return {"status": "failure", "error": error_msg, "output": output}

    except subprocess.CalledProcessError as e:
        error_msg = f"Simulation failed with exit code {e.returncode}."
        return {"status": "failure", "error": error_msg, "stdout": e.stdout, "stderr": e.stderr}
    except subprocess.TimeoutExpired as e:
        error_msg = f"Simulation timed out after {TIMEOUT_SECONDS}s."
        return {"status": "failure", "error": error_msg, "stdout": e.stdout, "stderr": e.stderr}
    except Exception:
        # Catch any other unexpected errors during simulation execution
        error_msg = "An unexpected error occurred during simulation execution."
        return {"status": "failure", "error": error_msg, "traceback": traceback.format_exc()}

def evaluate_stage1(program_path: str) -> Dict[str, Union[float, str]]:
    """
    First-stage evaluation: A quick check to see if the program can run a single,
    simple scenario without crashing. This filters out basic syntax and runtime errors.
    """
    absolute_program_path = os.path.abspath(program_path)
    logger.info(f"--- Stage 1: Quick Check for {os.path.basename(program_path)} ---")

    try:
        trace_files = [os.path.join(DATA_PATH, region, STAGE_1_SCENARIO["traces"][0]) for region in STAGE_1_SCENARIO["regions"]]
        
        if not all(os.path.exists(p) for p in trace_files):
            return {"runs_successfully": 0.0, "error": "Missing trace files for Stage 1."}

        sim_result = run_simulation(absolute_program_path, trace_files)

        if sim_result["status"] == "success":
            logger.info("Stage 1 PASSED.")
            # IMPORTANT: Only return the metric that is being checked by the pass_metric config.
            # The framework's _passes_threshold function incorrectly averages all numeric metrics.
            # By returning only this, we ensure the average is 1.0, passing the check correctly.
            return {"runs_successfully": 1.0}
        else:
            logger.warning(f"Stage 1 FAILED. Reason: {sim_result.get('error')}")
            return {
                "runs_successfully": 0.0,
                "error": sim_result.get("error"),
                "stdout": sim_result.get("stdout"),
                "stderr": sim_result.get("stderr"),
                "traceback": sim_result.get("traceback"),
            }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Stage 1 evaluator itself failed: {tb}")
        return {"runs_successfully": 0.0, "error": "Evaluator script failure", "traceback": tb}

def evaluate_stage2(program_path: str) -> Dict[str, Union[float, str]]:
    """
    Second-stage evaluation: The full, comprehensive evaluation across all test scenarios.
    This is only run for programs that have passed Stage 1.
    """
    absolute_program_path = os.path.abspath(program_path)
    logger.info(f"--- Stage 2: Full Evaluation for {os.path.basename(program_path)} ---")
    
    # Track stage2 start time
    stage2_start_time = time.time()
    STAGE2_TOTAL_TIMEOUT = 600  # 10 minutes for entire stage2
    
    all_performance_ratios = []
    scenario_summaries = []
    last_error = "No scenarios were successfully evaluated in Stage 2."
    
    # Calculate ON_DEMAND baseline cost
    on_demand_cost = (TASK_DURATION_HOURS + RESTART_OVERHEAD_HOURS) * 3.06

    for scenario in FULL_TEST_SCENARIOS:
        # Check if we're running out of time
        elapsed = time.time() - stage2_start_time
        if elapsed > STAGE2_TOTAL_TIMEOUT:
            logger.warning(f"Stage 2 timeout reached after {elapsed:.1f}s, stopping evaluation")
            last_error = f"Stage 2 timeout after evaluating {len(scenario_summaries)} scenarios"
            break
            
        scenario_name = scenario["name"]
        scenario_performance_ratios = []
        scenario_costs = []
        scenario_optimal_costs = []
        
        # Calculate remaining time and adjust per-trace timeout
        remaining_time = STAGE2_TOTAL_TIMEOUT - elapsed
        traces_to_eval = len(scenario["traces"])
        per_trace_timeout = min(TIMEOUT_SECONDS, remaining_time / (traces_to_eval * 2))  # *2 for strategy + optimal
        
        logger.info(f"--- Evaluating Scenario: {scenario_name} ({scenario['num_regions']} regions, {traces_to_eval} traces, {per_trace_timeout:.1f}s per sim) ---")

        for trace_file_name in scenario["traces"]:
            # Check timeout again before each trace
            elapsed = time.time() - stage2_start_time
            if elapsed > STAGE2_TOTAL_TIMEOUT:
                logger.warning(f"Stage 2 timeout during scenario {scenario_name}")
                break
                
            trace_files = [os.path.join(DATA_PATH, region, trace_file_name) for region in scenario["regions"]]
            
            if not all(os.path.exists(p) for p in trace_files):
                last_error = f"Missing trace files for {scenario_name}, trace {trace_file_name}."
                logger.warning(last_error)
                continue

            # Run strategy with adjusted timeout
            sim_result = run_simulation(absolute_program_path, trace_files, timeout=per_trace_timeout)

            if sim_result["status"] == "failure":
                last_error = f"Error in scenario '{scenario_name}': {sim_result.get('error')}"
                logger.warning(f"Skipping trace {trace_file_name} due to error: {sim_result.get('error', 'Unknown error')}")
                # Log more details if available
                if 'stderr' in sim_result:
                    logger.debug(f"Stderr output: {sim_result['stderr'][:500]}...")  # First 500 chars
                continue
            
            strategy_cost = sim_result.get("cost", float('inf'))
            
            # Calculate optimal cost with adjusted timeout
            optimal_cost = run_optimal_simulation(scenario["regions"], trace_file_name, timeout=per_trace_timeout)
            
            # Calculate performance ratio
            performance_ratio = strategy_cost / optimal_cost if optimal_cost > 0 else float('inf')
            
            scenario_performance_ratios.append(performance_ratio)
            scenario_costs.append(strategy_cost)
            scenario_optimal_costs.append(optimal_cost)
            
            # Log details
            logger.debug(f"  Trace {trace_file_name}: Strategy=${strategy_cost:.2f}, Optimal=${optimal_cost:.2f}, Ratio={performance_ratio:.3f}")
        
        # Check if we have enough successful traces
        num_traces_expected = len(scenario["traces"])
        num_traces_evaluated = len(scenario_performance_ratios)
        
        if num_traces_evaluated < num_traces_expected * 0.5:  # Less than 50% success
            logger.error(f"Scenario '{scenario_name}' failed: only {num_traces_evaluated}/{num_traces_expected} traces succeeded")
            last_error = f"Too many trace failures in scenario {scenario_name}"
            continue
        
        if scenario_performance_ratios:
            # Calculate scenario statistics
            avg_ratio = sum(scenario_performance_ratios) / len(scenario_performance_ratios)
            avg_cost = sum(scenario_costs) / len(scenario_costs)
            avg_optimal = sum(scenario_optimal_costs) / len(scenario_optimal_costs)
            
            # Calculate savings vs ON_DEMAND
            savings_vs_ondemand = (on_demand_cost - avg_cost) / on_demand_cost * 100
            
            scenario_summary = {
                "name": scenario_name,
                "num_regions": scenario["num_regions"],
                "num_traces_evaluated": len(scenario_performance_ratios),
                "avg_performance_ratio": avg_ratio,
                "avg_cost": avg_cost,
                "avg_optimal_cost": avg_optimal,
                "savings_vs_ondemand": savings_vs_ondemand
            }
            scenario_summaries.append(scenario_summary)
            all_performance_ratios.extend(scenario_performance_ratios)
            
            logger.info(f"Scenario '{scenario_name}' Summary:")
            logger.info(f"  - Average Cost: ${avg_cost:.2f}")
            logger.info(f"  - Average Optimal: ${avg_optimal:.2f}")
            logger.info(f"  - Performance Ratio: {avg_ratio:.3f}")
            logger.info(f"  - Savings vs ON_DEMAND: {savings_vs_ondemand:.1f}%")
        else:
            logger.warning(f"Scenario '{scenario_name}' failed completely. Last error: {last_error}")

    # Overall performance calculation
    if len(scenario_summaries) == len(FULL_TEST_SCENARIOS) and all_performance_ratios:
        # ALL scenarios must complete successfully
        overall_avg_ratio = sum(all_performance_ratios) / len(all_performance_ratios)
        overall_std_ratio = (sum((r - overall_avg_ratio) ** 2 for r in all_performance_ratios) / len(all_performance_ratios)) ** 0.5
        combined_score = -(overall_avg_ratio + 0.1 * overall_std_ratio)
        
        logger.info(f"=== Overall Performance Ratio: {overall_avg_ratio:.3f} ± {overall_std_ratio:.3f} ===")
        logger.info(f"=== Final Combined Score: {combined_score:.3f} ===")
        
        # This full set of metrics is for the database, which will correctly prioritize combined_score.
        return {"runs_successfully": 1.0, "combined_score": combined_score}
    else:
        # ANY failure means complete failure - no partial credit
        logger.error(f"Stage 2 FAILED: Only {len(scenario_summaries)}/{len(FULL_TEST_SCENARIOS)} scenarios completed")
        # Return runs_successfully: 0.0 to indicate failure and stop evolution
        return {"runs_successfully": 0.0, "cost": float('inf'), "combined_score": WORST_POSSIBLE_SCORE, "error": last_error}

def evaluate(program_path: str) -> Dict[str, Union[float, str]]:
    """
    Main entry point for the evaluator, required by the OpenEvolve framework.
    When cascade evaluation is enabled, this function is effectively a placeholder,
    as the stages (`evaluate_stage1`, `evaluate_stage2`, etc.) are called directly.
    """
    return {"runs_successfully": 1.0, "overall_score": 0.0}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python openevolve_multi_region_strategy/evaluator.py <path_to_program_file>")
        sys.exit(1)
    
    test_program_path = sys.argv[1]
    if not os.path.exists(test_program_path):
        print(f"Error: Program file not found at {test_program_path}")
        sys.exit(1)

    print(f"Running evaluator in standalone mode with program: {test_program_path}...")
    
    # Simulating the cascade for standalone testing
    print("\n--- Running Stage 1 ---")
    stage1_result = evaluate_stage1(test_program_path)
    print(json.dumps(stage1_result, indent=2))

    if stage1_result.get("runs_successfully", 0.0) > 0:
        print("\n--- Running Stage 2 ---")
        stage2_result = evaluate_stage2(test_program_path)
        print("\n--- Final Result ---")
        print(json.dumps(stage2_result, indent=2))
    else:
        print("\n--- Stage 1 Failed. Skipping Stage 2. ---")