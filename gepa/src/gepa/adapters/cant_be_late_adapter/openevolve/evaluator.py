"""
Evaluator for single-region strategies.
Simplified version that just computes average costs.
"""

import math
import os
import sys
import json
import logging
import signal
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from ..trace_config import TRACE_OVERHEADS, TRACE_SAMPLE_IDS

# Ensure this directory is importable for worker module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
CANTBELATE_DIR = os.path.join(SRC_ROOT, "cantbelate")
if os.path.isdir(CANTBELATE_DIR) and CANTBELATE_DIR not in sys.path:
    sys.path.insert(0, CANTBELATE_DIR)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Score for failed programs
FAILED_SCORE = -100000.0

# Use parent directory path for simulator and extracted traces
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "simulator")
TRACE_ARCHIVE_ROOT = Path(__file__).resolve().parents[4] / "exp" / "real"

os.environ["WANDB_MODE"] = "offline"

# Evaluation configuration
# Job configurations - just a few typical ones to show robustness
JOB_CONFIGS = [
    {"duration": 48, "deadline": 56},   # fraction = 0.86 (17% slack) - moderate
    {"duration": 48, "deadline": 52},   # fraction = 0.92 (8% slack) - tight
    {"duration": 24, "deadline": 32},   # fraction = 0.75 (33% slack) - relaxed
]

# Changeover delays - typical values from paper
CHANGEOVER_DELAYS = TRACE_OVERHEADS

def _normalize_trace_path(trace: str) -> str:
    if os.path.isabs(trace):
        return trace
    return os.path.join(PROJECT_ROOT, trace)


def get_evaluation_traces(custom_traces: list[str] | None = None) -> list[str]:
    """Return either a provided trace list or the default systematic set."""
    if custom_traces:
        normalized: list[str] = []
        missing: list[str] = []
        for trace in custom_traces:
            candidate = _normalize_trace_path(trace)
            if os.path.exists(candidate):
                normalized.append(candidate)
            else:
                missing.append(trace)
        if missing:
            logger.warning("Missing trace files: %s", ", ".join(missing))
        if normalized:
            return normalized

    if not TRACE_ARCHIVE_ROOT.exists():
        logger.error("Trace archive root %s does not exist", TRACE_ARCHIVE_ROOT)
        return []

    trace_paths: list[str] = []
    for overhead in TRACE_OVERHEADS:
        overhead_dir = TRACE_ARCHIVE_ROOT / f"ddl=search+task=48+overhead={overhead:.2f}" / "real"
        if not overhead_dir.exists():
            logger.warning("Missing overhead directory: %s", overhead_dir)
            continue

        for env_dir in sorted(overhead_dir.iterdir()):
            if not env_dir.is_dir() or env_dir.name.endswith(".json"):
                continue
            random_start_dir = env_dir / "traces" / "random_start"
            if not random_start_dir.exists():
                logger.warning("Missing random_start directory: %s", random_start_dir)
                continue
            for trace_id in TRACE_SAMPLE_IDS:
                candidate = random_start_dir / f"{trace_id}.json"
                if candidate.exists():
                    trace_paths.append(str(candidate))
                else:
                    logger.debug("Trace %s not found", candidate)

    logger.info("Total evaluation traces: %d", len(trace_paths))
    return trace_paths

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
                "error": "No Strategy class found",
                "score": FAILED_SCORE,
                "combined_score": FAILED_SCORE,
            }

        if "_step" not in code:
            return {
                "runs_successfully": 0.0,
                "error": "No _step method found",
                "score": FAILED_SCORE,
                "combined_score": FAILED_SCORE,
            }
        
        return {"runs_successfully": 1.0}
        
    except SyntaxError as e:
        return {
            "runs_successfully": 0.0,
            "error": f"Syntax error: {e}",
            "score": FAILED_SCORE,
            "combined_score": FAILED_SCORE,
        }
    except Exception as e:
        return {
            "runs_successfully": 0.0,
            "error": str(e),
            "score": FAILED_SCORE,
            "combined_score": FAILED_SCORE,
        }


def evaluate_stage2(program_path: str, trace_files: list[str] | None = None) -> dict:
    """Stage 2: Full evaluation on real traces - using average costs with parallelization."""
    # Ensure we have an absolute path
    program_path = os.path.abspath(program_path)
    
    # Get the evaluation traces
    selected_traces = get_evaluation_traces(trace_files)

    total_trace_count = len(selected_traces)
    default_ratio = 0.30
    trace_ratio_env = os.environ.get("GEPA_EVAL_TRACE_RATIO")
    trace_ratio = default_ratio
    if trace_ratio_env:
        try:
            trace_ratio = float(trace_ratio_env)
        except ValueError:
            logger.warning(
                "Invalid GEPA_EVAL_TRACE_RATIO value '%s'; falling back to %.2f",
                trace_ratio_env,
                default_ratio,
            )
            trace_ratio = default_ratio

    trace_ratio = min(1.0, max(trace_ratio, default_ratio))
    target_trace_count = max(1, math.ceil(total_trace_count * trace_ratio))
    if target_trace_count < total_trace_count:
        selected_traces = selected_traces[:target_trace_count]

    if not selected_traces:
        return {
            "runs_successfully": 0.0,
            "score": FAILED_SCORE,
            "combined_score": FAILED_SCORE,
            "error": "No trace files found",
            "trace_files": [],
            "scenario_stats": {},
            "scenario_summary": "",
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

    effective_ratio = (len(selected_traces) / total_trace_count) if total_trace_count else 0.0
    total_jobs = len(selected_traces) * len(eval_configs)
    logger.info(
        "Evaluating %d/%d traces (%.1f%%) across %d configs -> %d simulations",
        len(selected_traces),
        total_trace_count,
        effective_ratio * 100,
        len(eval_configs),
        total_jobs,
    )
    
    # Collect all costs from running the strategy
    all_costs: list[float] = []
    scenario_costs: defaultdict[str, list[float]] = defaultdict(list)

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

        log_successes = os.environ.get("GEPA_EVAL_LOG_SUCCESSES", "0") != "0"
        progress_every_env = os.environ.get("GEPA_EVAL_PROGRESS_EVERY")
        try:
            progress_every = max(1, int(progress_every_env)) if progress_every_env else 50
        except ValueError:
            logger.warning(
                "Invalid GEPA_EVAL_PROGRESS_EVERY value '%s'; defaulting to 50",
                progress_every_env,
            )
            progress_every = 50

        completed_jobs = 0

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
                    scenario_key = _scenario_key(trace_file, config)
                    scenario_costs[scenario_key].append(cost)
                    completed_jobs += 1

                    should_log_progress = (
                        completed_jobs == total_jobs
                        or completed_jobs % progress_every == 0
                    )

                    if should_log_progress:
                        if log_successes:
                            logger.info(
                                "✓ %s (d=%s, dl=%s, o=%.2f): $%.2f [%d/%d]",
                                trace_name,
                                config["duration"],
                                config["deadline"],
                                config["overhead"],
                                cost,
                                completed_jobs,
                                total_jobs,
                            )
                        else:
                            logger.info(
                                "Evaluation progress: %d/%d completed (latest $%.2f)",
                                completed_jobs,
                                total_jobs,
                                cost,
                            )
                else:
                    # Cancel all other futures
                    for f in future_to_info:
                        f.cancel()
                    logger.error(f"✗ {trace_name}: {error_msg}")
                    return {
                        "runs_successfully": 0.0,
                        "score": FAILED_SCORE,
                        "combined_score": FAILED_SCORE,
                        "error": f"Not all runs successful: {error_msg}",
                        "trace_files": selected_traces,
                        "scenario_stats": {},
                        "scenario_summary": "",
                    }
            except Exception as e:
                # Cancel all other futures
                for f in future_to_info:
                    f.cancel()
                logger.error(f"✗ {trace_name}: Exception {e}")
                return {
                    "runs_successfully": 0.0,
                    "score": FAILED_SCORE,
                    "combined_score": FAILED_SCORE,
                    "error": f"Not all runs successful: {e}",
                    "trace_files": selected_traces,
                    "scenario_stats": {},
                    "scenario_summary": "",
                }
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received, cancelling tasks...")
        # Cancel all pending futures
        for future in future_to_info:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return {
            "runs_successfully": 0.0,
            "score": FAILED_SCORE,
            "combined_score": FAILED_SCORE,
            "error": "Evaluation interrupted by user",
            "trace_files": selected_traces,
            "scenario_stats": {},
            "scenario_summary": "",
        }
    finally:
        # Restore original signal handlers if they were set
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        # Ensure executor is properly shut down
        executor.shutdown(wait=True)

    # All runs successful - calculate average cost
    avg_cost = sum(all_costs) / len(all_costs)
    cost_std = pstdev(all_costs) if len(all_costs) > 1 else 0.0

    # Use negative average cost as score (lower cost is better)
    # Apply variance penalty to combined score to discourage unstable programs
    score = -avg_cost
    combined_score = score - 0.25 * cost_std

    scenario_stats = _build_scenario_stats(scenario_costs)
    scenario_summary = _build_scenario_summary(scenario_stats)
    
    logger.info(
        "All simulations completed [%d/%d]. avg=$%.2f, std=$%.2f, score=%.2f, combined=%.2f",
        len(all_costs),
        total_jobs,
        avg_cost,
        cost_std,
        score,
        combined_score,
    )
    
    return {
        "runs_successfully": 1.0,
        "score": score,
        "combined_score": combined_score,
        "avg_cost": avg_cost,
        "cost_std": cost_std,
        "trace_files": selected_traces,
        "scenario_stats": scenario_stats,
        "scenario_summary": scenario_summary,
    }


def _scenario_key(trace_file: str, config: dict) -> str:
    path = Path(trace_file)
    env_name = path.parts[-4] if len(path.parts) >= 4 else path.stem
    return (
        f"{env_name} | duration={config['duration']}"
        f" | deadline={config['deadline']} | overhead={config['overhead']:.2f}"
    )


def _build_scenario_stats(scenario_costs: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for scenario, costs in scenario_costs.items():
        if not costs:
            continue
        mean_cost = mean(costs)
        std_cost = pstdev(costs) if len(costs) > 1 else 0.0
        stats[scenario] = {
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "num_samples": len(costs),
        }
    return stats


def _build_scenario_summary(stats: dict[str, dict[str, float]]) -> str:
    if not stats:
        return ""

    ranked = sorted(stats.items(), key=lambda item: item[1]["mean_cost"], reverse=True)
    top = ranked[:5]
    lines = []
    for idx, (scenario, info) in enumerate(top, start=1):
        lines.append(
            f"{idx}. {scenario} -> mean=${info['mean_cost']:.2f}, std=${info['std_cost']:.2f}, n={int(info['num_samples'])}"
        )
    return "\n".join(lines)


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
