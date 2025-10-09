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
BATCH_COMPARISON_PATH = os.path.join(PROJECT_ROOT, 'scripts_multi/batch_strategy_comparison.py')
DATA_PATH = os.path.join(PROJECT_ROOT, "data/converted_multi_region_aligned")  # Use aligned traces
CACHE_DIR = Path(PROJECT_ROOT) / ".evaluator_cache"
CACHE_DIR.mkdir(exist_ok=True)

TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2
TIMEOUT_SECONDS = 600
WORST_POSSIBLE_SCORE = -1e9

RANDOM_SEED = 42
random.seed(RANDOM_SEED)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_temp_strategy_with_unique_name(original_path: str, strategy_name: str) -> str:
    """
    Create a temporary strategy file with unique NAME to avoid conflicts.
    Simply replaces the NAME attribute with our unique strategy name.
    """
    import re
    
    # Read the original strategy file
    with open(original_path, 'r') as f:
        content = f.read()
    
    # Replace NAME attribute with our unique strategy name
    content = re.sub(
        r'NAME\s*=\s*["\'][\w_]+["\']',
        f'NAME = "{strategy_name}"',
        content
    )
    
    # Create temporary file
    temp_dir = Path(PROJECT_ROOT) / ".temp_strategies"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / f"{strategy_name}.py"
    
    with open(temp_file, 'w') as f:
        f.write(content)
    
    return str(temp_file)

def run_batch_evaluation(program_path: str, num_traces: int, timeout: Optional[float] = None) -> Dict[str, Union[float, str]]:
    """
    Run evaluation using our robust batch_strategy_comparison.py tool.
    This replaces the custom simulation logic with our proven approach.
    """
    temp_strategy_path: Optional[str] = None
    try:
        # Create temporary strategy name based on program path
        strategy_name = f"evolved_strategy_{hashlib.md5(program_path.encode()).hexdigest()[:8]}"
        
        # Create a temporary strategy file with unique class name to avoid conflicts
        temp_strategy_path = create_temp_strategy_with_unique_name(program_path, strategy_name)
        
        # Run batch comparison with the evolved strategy vs baseline
        cmd = [
            sys.executable,
            BATCH_COMPARISON_PATH,
            str(num_traces),
            "--strategies", strategy_name, "multi_region_rc_cr_threshold",
            "--strategy-files", f"{strategy_name}={temp_strategy_path}",
            "--no-viz",  # Skip visualization for speed
            "--timeout", str(int(timeout or TIMEOUT_SECONDS))
        ]
        
        # Set environment to use the strategy file
        env = os.environ.copy()
        env["PYTHONPATH"] = PROJECT_ROOT
        
        logger.info(f"Running batch evaluation: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout or TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
            env=env
        )
        
        if result.returncode != 0:
            return {
                "status": "failure",
                "error": f"Batch evaluation failed with exit code {result.returncode}",
                "stderr": result.stderr,
                "stdout": result.stdout
            }
        
        # Parse results from the generated summary file
        output_dir = Path(PROJECT_ROOT) / f"outputs/multi_trace_comparison_{num_traces}traces"
        summary_file = output_dir / "comparison_summary.json"
        
        if not summary_file.exists():
            return {
                "status": "failure", 
                "error": "Summary file not found after batch evaluation",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # Extract strategy performance
        strategy_results = [r for r in summary_data if r['strategy'] == strategy_name and r['status'] == 'success']
        baseline_results = [r for r in summary_data if r['strategy'] == 'multi_region_rc_cr_threshold' and r['status'] == 'success']
        
        if not strategy_results:
            return {
                "status": "failure",
                "error": "No successful runs for evolved strategy",
                "summary_data": summary_data
            }
        
        if not baseline_results:
            return {
                "status": "failure", 
                "error": "No successful runs for baseline strategy",
                "summary_data": summary_data
            }
        
        # Calculate metrics - filter out non-numeric costs
        strategy_costs = [r['cost'] for r in strategy_results if isinstance(r['cost'], (int, float))]
        baseline_costs = [r['cost'] for r in baseline_results if isinstance(r['cost'], (int, float))]
        
        if not strategy_costs or not baseline_costs:
            return {
                "status": "failure",
                "error": "No valid numeric costs found for comparison"
            }
        
        strategy_avg = sum(strategy_costs) / len(strategy_costs)
        strategy_std = (sum((c - strategy_avg) ** 2 for c in strategy_costs) / len(strategy_costs)) ** 0.5
        
        baseline_avg = sum(baseline_costs) / len(baseline_costs)
        baseline_std = (sum((c - baseline_avg) ** 2 for c in baseline_costs) / len(baseline_costs)) ** 0.5
        
        # Calculate performance ratio (lower is better for costs)
        performance_ratio = strategy_avg / baseline_avg
        
        return {
            "status": "success",
            "strategy_cost": strategy_avg,
            "strategy_std": strategy_std,
            "baseline_cost": baseline_avg,
            "baseline_std": baseline_std,
            "performance_ratio": performance_ratio,
            "num_successful_traces": len(strategy_results),
            "total_traces": num_traces
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "failure",
            "error": f"Batch evaluation timed out after {timeout or TIMEOUT_SECONDS}s"
        }
    except Exception as e:
        return {
            "status": "failure",
            "error": f"Batch evaluation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
    finally:
        # Clean up temporary strategy file
        try:
            if temp_strategy_path is not None and os.path.exists(temp_strategy_path):
                os.remove(temp_strategy_path)
        except Exception:
            pass  # Ignore cleanup errors

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
    Stage 1: Quick validation using batch evaluation with minimal traces.
    Filters out basic syntax and runtime errors using our proven batch tool.
    """
    absolute_program_path = os.path.abspath(program_path)
    logger.info(f"--- Stage 1: Quick Check for {os.path.basename(program_path)} ---")

    try:
        # Use minimal traces for quick validation (realistic 5-region scenario)
        num_traces = 1
        timeout_per_evaluation = 120  # 2 minutes for stage 1
        
        batch_result = run_batch_evaluation(absolute_program_path, num_traces, timeout_per_evaluation)
        
        if batch_result["status"] == "failure":
            logger.warning(f"Stage 1 FAILED. Reason: {batch_result.get('error')}")
            return {
                "runs_successfully": 0.0,
                "error": batch_result.get("error", "Unknown batch evaluation error")
            }
        
        performance_ratio = batch_result["performance_ratio"]
        
        # Stage 1 success criteria: strategy must run and not be extremely bad
        if performance_ratio > 10.0:  # More than 10x worse than baseline is considered failure
            logger.warning(f"Stage 1 FAILED. Poor performance ratio: {performance_ratio:.3f}")
            return {
                "runs_successfully": 0.0,
                "error": f"Performance too poor: {performance_ratio:.3f}x worse than baseline"
            }
        
        logger.info(f"Stage 1 PASSED. Performance ratio: {performance_ratio:.3f}")
        return {"runs_successfully": 1.0}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Stage 1 evaluator itself failed: {tb}")
        return {"runs_successfully": 0.0, "error": "Evaluator script failure", "traceback": tb}

def evaluate_stage2(program_path: str) -> Dict[str, Union[float, str]]:
    """
    Stage 2: Comprehensive evaluation using batch evaluation with realistic scenarios.
    Uses our proven 5-region setup with multiple traces for statistical significance.
    Replaces complex scenario logic with robust batch_strategy_comparison.py tool.
    """
    absolute_program_path = os.path.abspath(program_path)
    logger.info(f"--- Stage 2: Full Evaluation for {os.path.basename(program_path)} ---")
    
    try:
        # Use multiple traces for comprehensive evaluation with statistical significance
        num_traces = 10  # More traces for robustness
        timeout_per_evaluation = 600  # 10 minutes total for stage 2
        
        batch_result = run_batch_evaluation(absolute_program_path, num_traces, timeout_per_evaluation)
        
        if batch_result["status"] == "failure":
            logger.error(f"Stage 2 FAILED: {batch_result.get('error')}")
            return {
                "runs_successfully": 0.0,
                "cost": float('inf'),
                "combined_score": WORST_POSSIBLE_SCORE,
                "error": batch_result.get("error", "Unknown batch evaluation error")
            }
        
        # Extract key metrics from batch evaluation
        performance_ratio = batch_result["performance_ratio"]
        strategy_cost = batch_result["strategy_cost"]
        strategy_std = batch_result["strategy_std"]
        baseline_cost = batch_result["baseline_cost"]
        baseline_std = batch_result["baseline_std"]
        success_rate = batch_result["num_successful_traces"] / batch_result["total_traces"]
        
        # Calculate OpenEvolve-style combined score with std penalty
        # Lower performance_ratio is better (closer to 1.0), lower std is better
        avg_ratio = performance_ratio
        std_ratio_penalty = strategy_std / strategy_cost if isinstance(strategy_cost, (int, float)) and strategy_cost > 0 else 1.0
        
        # Combined score: negative because OpenEvolve maximizes scores
        # Better strategies (lower cost ratio, lower variability) get higher (less negative) scores
        combined_score = -(avg_ratio + 0.1 * std_ratio_penalty)
        
        # Calculate savings vs ON_DEMAND baseline for context
        on_demand_cost = (TASK_DURATION_HOURS + RESTART_OVERHEAD_HOURS) * 3.06
        savings_vs_ondemand = (on_demand_cost - strategy_cost) / on_demand_cost if isinstance(strategy_cost, (int, float)) else 0.0
        
        # Require minimum success rate to pass Stage 2
        if success_rate < 0.8:  # At least 80% of traces must succeed
            logger.error(f"Stage 2 FAILED: Low success rate {success_rate:.1%}")
            return {
                "runs_successfully": 0.0,
                "cost": float('inf'),
                "combined_score": WORST_POSSIBLE_SCORE,
                "error": f"Success rate too low: {success_rate:.1%}"
            }
        
        logger.info(f"=== Stage 2 Results ===")
        logger.info(f"Performance Ratio: {performance_ratio:.3f}")
        logger.info(f"Strategy Cost: ${strategy_cost:.2f} ± ${strategy_std:.2f}")
        logger.info(f"Baseline Cost: ${baseline_cost:.2f} ± ${baseline_std:.2f}")
        logger.info(f"Success Rate: {success_rate:.1%}")
        logger.info(f"Combined Score: {combined_score:.3f}")
        logger.info(f"Savings vs ON_DEMAND: {savings_vs_ondemand:.1%}")
        
        # Return comprehensive metrics following OpenEvolve patterns
        return {
            "runs_successfully": 1.0,
            "combined_score": combined_score,
            "performance_ratio": performance_ratio,
            "strategy_cost": strategy_cost,
            "strategy_std": strategy_std,
            "baseline_cost": baseline_cost,
            "baseline_std": baseline_std,
            "success_rate": success_rate,
            "savings_vs_ondemand": savings_vs_ondemand,
            "num_traces_evaluated": batch_result["num_successful_traces"]
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Stage 2 evaluator itself failed: {tb}")
        return {
            "runs_successfully": 0.0,
            "cost": float('inf'),
            "combined_score": WORST_POSSIBLE_SCORE,
            "error": "Evaluator script failure",
            "traceback": tb
        }

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