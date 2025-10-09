"""
Evaluator for single-region strategies.
Simplified version that just computes average costs.
"""

import os
import sys
import json
import subprocess
import logging
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional

# Ensure Weights & Biases stays offline during eval to avoid auth prompts
os.environ.setdefault("WANDB_MODE", "offline")

# Ensure this directory is importable for worker module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use parent directory path for traces
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Evaluation configuration
NUM_TRACES_PER_TYPE_PER_REGION = 10  # Number of traces per accelerator type per region
START_TRACE_INDEX = 10

# Accelerator types to test
ACCELERATOR_TYPES = ["k80", "v100"]

# Regions to consider for traces (only these exist)
REGIONS = ["us-west-2a", "us-west-2b"]

# Job configurations - just a few typical ones to show robustness
JOB_CONFIGS = [
    {"duration": 48, "deadline": 56},   # fraction = 0.86 (17% slack) - moderate
    {"duration": 48, "deadline": 52},   # fraction = 0.92 (8% slack) - tight
    {"duration": 24, "deadline": 32},   # fraction = 0.75 (33% slack) - relaxed
]

# Changeover delays - typical values from paper
CHANGEOVER_DELAYS = [0.01, 0.2]  # Just the standard 0.2 hours (12 minutes)

def get_evaluation_traces():
    """Get a systematic set of evaluation traces covering all regions and accelerator types."""
    all_traces: list[str] = []
    
    for region in REGIONS:
        for accel_type in ACCELERATOR_TYPES:
            # Get traces for this specific region and accelerator combination
            pattern = os.path.join(
                PARENT_DIR,
                f"data/real/ping_based/random_start_time/{region}_{accel_type}_1/*.json"
            )
            matching_files = sorted(glob.glob(pattern))  # Sort for reproducibility
            
            # Take the specified number of traces for this combination
            selected = matching_files[START_TRACE_INDEX:START_TRACE_INDEX+NUM_TRACES_PER_TYPE_PER_REGION]
            
            if selected:
                logger.info(f"Found {len(selected)} traces for {region}_{accel_type}")
                all_traces.extend(selected)
            else:
                logger.warning(f"No traces found for {region}_{accel_type}")
    
    # Total expected: 3 regions × 2 types × 2 traces = 12 traces
    logger.info(f"Total evaluation traces: {len(all_traces)}")
    
    return all_traces

from sim_worker import run_single_simulation


def evaluate_stage1(program_path: str) -> dict:
    """Stage 1: Quick syntax and import check."""
    try:
        # Read the strategy code from file
        with open(program_path, 'r') as f:
            code = f.read()
        
        # Try to compile the code
        compile(code, program_path, 'exec')
        
        # Basic validation - check for required class structure
        if "class" not in code or "Strategy" not in code:
            return {
                "runs_successfully": 0.0,
                "error": "No Strategy class found"
            }
        
        if "_step" not in code:
            return {
                "runs_successfully": 0.0,
                "error": "No _step method found"
            }
        
        return {"runs_successfully": 1.0}
        
    except SyntaxError as e:
        return {
            "runs_successfully": 0.0,
            "error": f"Syntax error: {e}"
        }
    except Exception as e:
        return {
            "runs_successfully": 0.0,
            "error": str(e)
        }


def evaluate_stage2(program_path: str) -> dict:
    """Stage 2: Full evaluation on real traces - using average costs with parallelization."""
    # Ensure we have an absolute path
    program_path = os.path.abspath(program_path)
    
    # Get the evaluation traces
    selected_traces = get_evaluation_traces()
    
    if not selected_traces:
        return {
            "runs_successfully": 0.0,
            "score": 0.0,
            "combined_score": -1000.0,
            "error": "No trace files found"
        }
    
    # Build evaluation configurations
    eval_configs = []
    for job_config in JOB_CONFIGS:
        for delay in CHANGEOVER_DELAYS:
            eval_configs.append({
                "duration": job_config["duration"],
                "deadline": job_config["deadline"],
                "overhead": delay
            })
    
    logger.info(f"Testing on {len(selected_traces)} traces with {len(eval_configs)} configs")
    logger.info(f"Total evaluations: {len(selected_traces) * len(eval_configs)}")
    
    # Collect all costs from running the strategy
    all_costs: list[float] = []
    
    # Use ProcessPoolExecutor (CPU-bound). Worker lives in separate module for pickling.
    max_workers = min(16, os.cpu_count() or 4)  # Limit to 16 workers max
    logger.info(f"Running simulations in parallel with {max_workers} workers")
    
    # Try to prefer 'fork' on POSIX; ignore if unavailable (e.g., Windows)
    executor_kwargs = {}
    try:
        import multiprocessing
        if hasattr(multiprocessing, "get_context"):
            executor_kwargs["mp_context"] = multiprocessing.get_context("fork")
    except Exception:
        pass

    with ProcessPoolExecutor(max_workers=max_workers, **executor_kwargs) as executor:
        # Submit all tasks
        future_to_info = {}
        for config in eval_configs:
            for trace_file in selected_traces:
                future = executor.submit(run_single_simulation, program_path, trace_file, config)
                future_to_info[future] = (trace_file, config)

        # Collect results as they complete
        for future in as_completed(future_to_info):
            trace_file, config = future_to_info[future]
            # Extract clean trace name: us-west-2a_v100_1/0
            parts = trace_file.replace(".json", "").split("/")
            trace_name = f"{parts[-2]}/{parts[-1]}"
            
            try:
                success, cost, error_msg = future.result()
                if success:
                    all_costs.append(cost)
                    logger.info(f"✓ {trace_name} (d={config['duration']}, dl={config['deadline']}, o={config['overhead']}): ${cost:.2f}")
                else:
                    # Cancel all other futures
                    for f in future_to_info:
                        f.cancel()
                    logger.error(f"✗ {trace_name}: {error_msg}")
                    return {
                        "runs_successfully": 0.0,
                        "score": 0.0,
                        "combined_score": 0.0,
                        "error": f"Not all runs successful: {error_msg}"
                    }
            except Exception as e:
                # Cancel all other futures
                for f in future_to_info:
                    f.cancel()
                logger.error(f"✗ {trace_name}: Exception {e}")
                return {
                    "runs_successfully": 0.0,
                    "score": 0.0,
                    "combined_score": 0.0,
                    "error": f"Not all runs successful: {e}"
                }

    # All runs successful - calculate average cost
    avg_cost = sum(all_costs) / len(all_costs)
    
    # Use negative average cost as score (lower cost is better)
    # OpenEvolve will maximize score, so negative cost means minimizing cost
    score = -avg_cost
    
    logger.info(f"All {len(all_costs)} simulations completed successfully!")
    logger.info(f"Average cost: ${avg_cost:.2f}")
    logger.info(f"Score (negative cost): {score:.2f}")
    
    return {
        "runs_successfully": 1.0,
        "score": score,
        "combined_score": score,
        "avg_cost": avg_cost,
    }


def evaluate(_program_path: str) -> dict:
    """Main evaluation function for OpenEvolve.
    
    Args:
        program_path: Path to a Python file
    """
    raise NotImplementedError("This is just a placeholder for OpenEvolve cascade evaluation")


if __name__ == "__main__":
    # Test the evaluator with the initial program
    program_path = "examples/cant-be-late-simulator/openevolve_single_region_strategy/evaluator.py"
    
    print("Testing evaluator with initial program...")
    print("\nStage 1:")
    result1 = evaluate_stage1(program_path)
    print(json.dumps(result1, indent=2))
    
    if result1["runs_successfully"] == 1:
        print("\nStage 2:")
        result2 = evaluate_stage2(program_path)
        print(json.dumps(result2, indent=2))
