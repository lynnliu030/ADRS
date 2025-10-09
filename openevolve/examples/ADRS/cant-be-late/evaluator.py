"""
Evaluator for single-region strategies.
Simplified version that just computes average costs.
"""

import os
import sys
import json
import logging
import glob
import signal
import argparse
import numpy as np
from openevolve.evaluation_result import EvaluationResult
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure this directory is importable for worker module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Score for failed programs
FAILED_SCORE = -100000.0

# Use parent directory path for traces
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.join(PARENT_DIR, "cant-be-late-simulator")

os.environ["WANDB_MODE"] = "offline"

# Evaluation configuration
# Use a single trace per accelerator+region to increase evolutionary signal per unit of work
# and reduce prompt bloat from artifacts. Keep start index fixed for reproducibility.
NUM_TRACES_PER_TYPE_PER_REGION = 1
START_TRACE_INDEX = 10

ENV_PATHS = [
    "us-west-2a_k80_1",
    "us-west-2a_v100_1",
    "us-west-2b_v100_1",
    "us-west-2b_k80_8",
    "us-west-2a_k80_8",
    "us-west-2b_k80_1",
    # Add V100 8x environments to cover heavy GPU traces
    "us-west-2a_v100_8",
    "us-west-2b_v100_8",
]

# Job configurations - just a few typical ones to show robustness
JOB_CONFIGS = [
    {"duration": 48, "deadline": 52},   # fraction = 0.92 (8% slack) - tight
    {"duration": 48, "deadline": 70},   # fraction = 0.86 (17% slack) - moderate
    {"duration": 48, "deadline": 92},   # fraction = 0.75 (33% slack) - relaxed
]

# Changeover delays - typical values from paper
CHANGEOVER_DELAYS = [0.02, 0.2, 0.4]  # Just the standard 0.2 hours (12 minutes)

def get_evaluation_traces():
    """Get a systematic set of evaluation traces covering all regions and accelerator types."""
    all_traces: list[str] = []
    
    for env_path in ENV_PATHS:
        # Get traces for this specific region and accelerator combination
        pattern = os.path.join(
            PROJECT_ROOT,
            f"data/real/ping_based/random_start_time/{env_path}/*.json"
        )
        matching_files = sorted(glob.glob(pattern))  # Sort for reproducibility
        
        # Take the specified number of traces for this combination
        selected = matching_files[START_TRACE_INDEX:START_TRACE_INDEX+NUM_TRACES_PER_TYPE_PER_REGION]
        
        if selected:
            logger.info(f"Found {len(selected)} traces for {env_path}")
            all_traces.extend(selected)
        else:
            logger.warning(f"No traces found for {env_path}")

    # Total expected: 6 ENV_PATHS × 2 traces = 12 traces
    logger.info(f"Total evaluation traces: {len(all_traces)}")
    
    return all_traces

from sim_worker import run_single_simulation


def _run_baseline_comparison(selected_traces, eval_configs, max_workers=4):
    """Run the uniform progress baseline for comparison by scenario."""
    baseline_program = os.path.join(CURRENT_DIR, "uniform_progress_baseline.py")
    if not os.path.exists(baseline_program):
        logger.warning("Baseline program not found, skipping comparison")
        return None

    baseline_by_config = {}

    # Run baseline in parallel
    executor = ProcessPoolExecutor(max_workers=max_workers)
    future_to_info = {}

    try:
        # Submit baseline tasks (sample a subset for speed)
        sample_traces = selected_traces[::3]  # Every 3rd trace for speed

        for config in eval_configs:
            config_key = f"d{config['duration']}_dl{config['deadline']}_o{config['overhead']}"
            baseline_by_config[config_key] = []

            for trace_file in sample_traces:
                future = executor.submit(run_single_simulation, baseline_program, trace_file, config)
                future_to_info[future] = (trace_file, config, config_key)

        # Collect baseline results by config
        for future in as_completed(future_to_info):
            try:
                result = future.result()
                if len(result) >= 2:
                    success, cost = result[0], result[1]
                    if success:
                        _, _, config_key = future_to_info[future]
                        baseline_by_config[config_key].append(cost)
            except Exception:
                pass  # Skip failed baseline runs

    except Exception as e:
        logger.warning(f"Baseline comparison failed: {e}")
    finally:
        executor.shutdown(wait=True)

    # Calculate per-scenario baseline stats
    baseline_stats = {}
    for config_key, costs in baseline_by_config.items():
        if costs:
            baseline_stats[config_key] = {
                "baseline_avg": np.mean(costs),
                "baseline_std": np.std(costs),
                "sample_size": len(costs)
            }

    return baseline_stats if baseline_stats else None


def _analyze_spot_availability(traces_by_config):
    """Analyze SPOT availability patterns by reading trace files using real_cost.py approach."""
    import json
    import glob

    config_availability = {}

    for config_key, trace_infos in traces_by_config.items():
        region_lifetime_stats = {}
        region_availability_stats = {}

        for trace_info in trace_infos:
            trace_name = trace_info["trace_name"]
            # Extract region from trace name (e.g., "us-west-2a_k80_1/107")
            region = trace_name.split('/')[0]
            trace_id = trace_name.split('/')[1] if '/' in trace_name else '0'

            # Find the exact trace file used in the simulation
            trace_file_pattern = f"{PROJECT_ROOT}/data/real/ping_based/random_start_time/{region}/{trace_id}.json"

            try:
                with open(trace_file_pattern, 'r') as f:
                    trace_data = json.load(f)

                # Use the exact same approach as real_cost.py lines 78-83
                # In trace data: 0 = available, 1 = preempted
                availability_trace = 1 - np.array(trace_data["data"])  # Convert to: 1 = available, 0 = preempted
                gap_seconds = trace_data["metadata"]["gap_seconds"]

                # Calculate average lifetime (how long instances stay up)
                padded = np.array([0] + availability_trace.tolist() + [0])
                start_end = padded[1:] - padded[:-1]
                lengths = np.where(start_end == -1)[0] - np.where(start_end == 1)[0]
                if len(lengths) > 0:
                    avg_lifetime_hours = np.mean(lengths) * gap_seconds / 3600
                else:
                    avg_lifetime_hours = 0.0

                # Calculate availability fraction (how often instances are available)
                availability_fraction = np.mean(availability_trace)

                # Store per-region statistics
                if region not in region_lifetime_stats:
                    region_lifetime_stats[region] = []
                    region_availability_stats[region] = []

                region_lifetime_stats[region].append(avg_lifetime_hours)
                region_availability_stats[region].append(availability_fraction)

            except Exception as e:
                # Log warning when we can't read trace data and skip
                logger.warning(f"Could not read trace file {trace_file_pattern}: {e}")
                continue

        # Aggregate statistics for this config
        if region_lifetime_stats:
            config_availability[config_key] = {
                "region_lifetime_hours": {
                    region: {
                        "avg": np.mean(lifetimes),
                        "std": np.std(lifetimes),
                        "min": np.min(lifetimes),
                        "max": np.max(lifetimes)
                    } for region, lifetimes in region_lifetime_stats.items()
                },
                "region_availability_fraction": {
                    region: {
                        "avg": np.mean(fractions),
                        "std": np.std(fractions)
                    } for region, fractions in region_availability_stats.items()
                },
                "regions": list(region_lifetime_stats.keys())
            }

    return config_availability


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
            # Mark clear failure with a strongly negative combined_score so evolution never promotes it
            return {
                "runs_successfully": 0.0,
                "score": FAILED_SCORE,
                "combined_score": FAILED_SCORE,
                "error": "No Strategy class found",
            }
        
        if "_step" not in code:
            # Mark clear failure with a strongly negative combined_score so evolution never promotes it
            return {
                "runs_successfully": 0.0,
                "score": FAILED_SCORE,
                "combined_score": FAILED_SCORE,
                "error": "No _step method found",
            }
        
        return {"runs_successfully": 1.0}
        
    except SyntaxError as e:
        # Syntax failures must never be considered better than valid programs
        return {
            "runs_successfully": 0.0,
            "score": FAILED_SCORE,
            "combined_score": FAILED_SCORE,
            "error": f"Syntax error: {e}",
        }
    except Exception as e:
        # Any other failure should also be penalized heavily
        return {
            "runs_successfully": 0.0,
            "score": FAILED_SCORE,
            "combined_score": FAILED_SCORE,
            "error": str(e),
        }


def evaluate_stage2(program_path: str):
    """Stage 2: Full evaluation on real traces - using average costs with parallelization."""
    # Ensure we have an absolute path
    program_path = os.path.abspath(program_path)
    
    # Get the evaluation traces
    selected_traces = get_evaluation_traces()
    
    if not selected_traces:
        return {
            "runs_successfully": 0.0,
            "score": 0.0,
            "combined_score": FAILED_SCORE,
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
    all_detailed_info: list[dict] = []
    all_warnings: list[str] = []
    all_errors: list[str] = []
    cost_by_config: dict = {}
    traces_by_config: dict = {}
    
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

    executor = ProcessPoolExecutor(max_workers=max_workers, **executor_kwargs)
    future_to_info = {}
    
    # Only set up signal handler if we're in the main thread
    old_sigint = None
    old_sigterm = None
    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            # Set up signal handler for graceful shutdown
            def signal_handler(signum, frame):
                del signum, frame  # Suppress unused variable warnings
                logger.info("\nReceived interrupt signal, shutting down...")
                # Cancel all pending futures
                for future in future_to_info:
                    future.cancel()
                # Shutdown executor
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
            
            # Register signal handler
            old_sigint = signal.signal(signal.SIGINT, signal_handler)
            old_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    except Exception as e:
        logger.debug(f"Could not set up signal handler: {e}")
    
    try:
        # Submit all tasks
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
                result = future.result()
                if len(result) == 4:  # New format with detailed info
                    success, cost, error_msg, detailed_info = result
                else:  # Old format for backward compatibility
                    success, cost, error_msg = result
                    detailed_info = {}

                if success:
                    all_costs.append(cost)
                    all_detailed_info.append(detailed_info)

                    # Collect warnings and errors
                    if detailed_info.get("warnings"):
                        all_warnings.extend(detailed_info["warnings"])
                    if detailed_info.get("errors"):
                        all_errors.extend(detailed_info["errors"])

                    # Skip discrete time tracking

                    # Organize costs by config and collect trace info
                    config_key = f"d{config['duration']}_dl{config['deadline']}_o{config['overhead']}"
                    if config_key not in cost_by_config:
                        cost_by_config[config_key] = []
                        traces_by_config[config_key] = []
                    cost_by_config[config_key].append(cost)

                    # Extract trace info
                    trace_info = {
                        "trace_name": trace_name,
                        "cost": cost,
                        "config": config
                    }
                    traces_by_config[config_key].append(trace_info)

                    logger.info(f"✓ {trace_name} (d={config['duration']}, dl={config['deadline']}, o={config['overhead']}): ${cost:.2f}")
                else:
                    # Cancel all other futures
                    for f in future_to_info:
                        f.cancel()
                    logger.error(f"✗ {trace_name}: {error_msg}")

                    # Include detailed failure info in return
                    failure_details = {
                        "runs_successfully": 0.0,
                        "score": 0.0,
                        "combined_score": FAILED_SCORE,
                        "error": f"Not all runs successful: {error_msg}",
                        "failure_warnings": detailed_info.get("warnings", []),
                        "failure_errors": detailed_info.get("errors", []),
                        "likely_uses_discrete_time": detailed_info.get("likely_uses_discrete_time", False)
                    }
                    return failure_details
            except Exception as e:
                # Cancel all other futures
                for f in future_to_info:
                    f.cancel()
                logger.error(f"✗ {trace_name}: Exception {e}")
                return {
                    "runs_successfully": 0.0,
                    "score": 0.0,
                    "combined_score": FAILED_SCORE,
                    "error": f"Not all runs successful: {e}"
                }
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received, cancelling tasks...")
        # Cancel all pending futures
        for future in future_to_info:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return {
            "runs_successfully": 0.0,
            "score": 0.0,
            "combined_score": FAILED_SCORE,
            "error": "Evaluation interrupted by user"
        }
    finally:
        # Restore original signal handlers if they were set
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        # Ensure executor is properly shut down
        executor.shutdown(wait=True)

    # All runs successful - calculate comprehensive statistics
    avg_cost = np.mean(all_costs)
    std_cost = np.std(all_costs)
    min_cost = np.min(all_costs)
    max_cost = np.max(all_costs)

    # Base score: negative average cost (lower cost is better)
    score = -avg_cost
    # Tie-breaker: prefer lower variance across scenarios to avoid gaming only a subset
    # Keep the weight small to preserve primary objective dominance
    combined_score = score - 0.25 * std_cost

    logger.info(f"All {len(all_costs)} simulations completed successfully!")
    logger.info(f"Average cost: ${avg_cost:.2f}")
    logger.info(f"Score (negative cost): {score:.2f}")

    # Calculate cost breakdown by configuration with scenario details
    config_stats = {}
    for config_key, costs in cost_by_config.items():
        # Parse config details from config_key (e.g., "d48_dl56_o0.02")
        parts = config_key.split('_')
        duration = int(parts[0][1:])  # Remove 'd' prefix
        deadline = int(parts[1][2:])  # Remove 'dl' prefix
        overhead = float(parts[2][1:])  # Remove 'o' prefix

        config_stats[config_key] = {
            "avg": np.mean(costs),
            "std": np.std(costs),
            "count": len(costs),
            "config": {
                "duration_hours": duration,
                "deadline_hours": deadline,
                "restart_overhead_hours": overhead
            }
        }

    # Analyze SPOT availability patterns
    availability_stats = _analyze_spot_availability(traces_by_config)

    # Run baseline comparison
    baseline_stats = _run_baseline_comparison(selected_traces, eval_configs)

    # Compile warnings and errors summary
    warning_summary = {}
    for warning in all_warnings:
        warning_summary[warning] = warning_summary.get(warning, 0) + 1

    error_summary = {}
    for error in all_errors:
        error_summary[error] = error_summary.get(error, 0) + 1

    # Simplified feedback for LLM - focus on key metrics
    detailed_feedback = {
        "runs_successfully": 1.0,
        "score": score,
        "combined_score": combined_score,
        "avg_cost": avg_cost,
        "cost_std": std_cost,
        "cost_range": max_cost - min_cost,
        "cost_by_config": config_stats,
    }

    # Add error logs only if there are actual errors/warnings
    if all_warnings or all_errors:
        detailed_feedback.update({
            "warnings": list(set(all_warnings))[:3],  # Only unique warnings, max 3
            "errors": list(set(all_errors))[:3],      # Only unique errors, max 3
        })

    # Add real SPOT availability statistics to each config
    for config_key, stats in config_stats.items():
        if config_key in availability_stats:
            stats["spot_availability"] = availability_stats[config_key]


    # Add per-scenario baseline comparison if available
    if baseline_stats:
        # Add baseline comparison to each config
        for config_key, stats in config_stats.items():
            if config_key in baseline_stats:
                baseline_avg = baseline_stats[config_key]["baseline_avg"]
                improvement_ratio = stats["avg"] / baseline_avg
                stats["vs_baseline"] = {
                    "baseline_avg": baseline_avg,
                    "improvement_ratio": improvement_ratio,
                    "improvement_pct": (1 - improvement_ratio) * 100
                }

    # Prepare a compact artifact to help the evolution LLM without prescribing a policy
    # Show top-3 worst configs by average cost for directional feedback
    try:
        worst = sorted(
            config_stats.items(), key=lambda kv: kv[1]["avg"], reverse=True
        )[:3]
        lines = [
            "Worst-by-config summary (avg cost high → needs attention):"
        ]
        for key, stats in worst:
            cfg = stats.get("config", {})
            lines.append(
                f"- {key} (d={cfg.get('duration_hours')}, dl={cfg.get('deadline_hours')}, o={cfg.get('restart_overhead_hours')}): "
                f"avg=${stats['avg']:.2f}, std=${stats['std']:.2f}, n={stats['count']}"
            )
        artifact_text = "\n".join(lines)
    except Exception:
        artifact_text = ""

    # Return metrics plus artifacts to help prompts. Include the full detailed feedback JSON for transparency.
    artifacts = {}
    if artifact_text:
        artifacts["scenario_summary"] = artifact_text
    try:
        artifacts["raw_feedback_json"] = json.dumps(
            {
                **detailed_feedback,
                # Also include availability and baseline summaries if present
                "availability_stats": availability_stats,
                "baseline_stats": baseline_stats,
                "warnings": warning_summary,
                "errors": error_summary,
            },
            ensure_ascii=False,
        )
    except Exception:
        pass

    return EvaluationResult(metrics=detailed_feedback, artifacts=artifacts)


def evaluate(_program_path: str) -> dict:
    """Main evaluation function for OpenEvolve.
    
    Args:
        program_path: Path to a Python file
    """
    raise NotImplementedError("This is just a placeholder for OpenEvolve cascade evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path", type=str, default="initial_program.py", nargs='?', help="Path to the program to evaluate.")
    args = parser.parse_args()
    program_path = args.program_path

    print(f"Testing evaluator with program: {program_path}")
    
    print("\nStage 1:")
    result1 = evaluate_stage1(program_path)
    print(json.dumps(result1, indent=2))
    
    if result1["runs_successfully"] == 1:
        print("\nStage 2:")
        result2 = evaluate_stage2(program_path)
        print(json.dumps(result2, indent=2))
