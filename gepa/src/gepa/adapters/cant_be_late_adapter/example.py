from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Optional

from gepa.adapters.cant_be_late_adapter.open_evolve_program_adapter import CantBeLateAdapter
from gepa.adapters.cant_be_late_adapter.trace_dataset import load_trace_dataset

INITIAL_PROGRAM_SRC = """import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class EvolveSingleRegionStrategy(Strategy):
    NAME = 'evolve_single_region'
    
    def __init__(self, args):
        super().__init__(args)
    
    def reset(self, env, task):
        super().reset(env, task)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env
        
        # Task completion check
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE
        
        # Calculate remaining time until deadline
        remaining_time = self.deadline - env.elapsed_seconds
        
        # Simple deadline check: if we're running out of time, use ON_DEMAND
        # Add restart overhead to account for potential restart
        if remaining_task_time + self.restart_overhead >= remaining_time:
            # We need ON_DEMAND to guarantee completion
            return ClusterType.ON_DEMAND
        
        # Simple greedy logic: use SPOT if available, wait otherwise
        if has_spot:
            return ClusterType.SPOT
        else:
            # Just wait for SPOT to become available
            return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)"""
        
DATASET_ROOT = Path(__file__).resolve().parents[4] / "exp" / "real"
DEFAULT_TRACE_RATIO = 0.30


def load_dataset(
    max_traces_per_split: int | None = None,
    trace_ratio: float | None = None,
    include_test: bool = True,
):
    """Load train/val/test splits from extracted cant-be-late traces."""

    splits = load_trace_dataset(
        dataset_root=str(DATASET_ROOT),
        max_traces_per_split=max_traces_per_split,
        trace_ratio=trace_ratio,
    )
    train_set = splits["train"]
    val_set = splits["val"]
    test_set = splits["test"] if include_test else []
    return train_set, val_set, test_set

def _resolve_run_dir() -> Path:
    import os

    run_dir_env = os.environ.get("GEPA_RUN_DIR")
    if run_dir_env:
        run_dir = Path(run_dir_env)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs") / "cant_be_late" / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_checkpoints(
    run_dir: Path,
    gepa_result,
    base_score: Optional[float],
    optimized_score: Optional[float],
    best_candidate: dict[str, str],
):
    # Serialize the full GEPA result for later inspection
    result_path = run_dir / "gepa_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(gepa_result.to_dict(), f, indent=2)

    # Write the best program as a Python file
    best_program_path = run_dir / "best_program.py"
    best_program_path.write_text(best_candidate["program"], encoding="utf-8")

    # Record test metrics for quick reference
    metrics_path = run_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_test_score": base_score,
                "optimized_test_score": optimized_score,
                "best_candidate_index": gepa_result.best_idx,
            },
            f,
            indent=2,
        )

    # Snapshot every candidate for manual analysis
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    for idx, candidate in enumerate(gepa_result.candidates):
        program_path = candidates_dir / f"candidate_{idx:03d}.py"
        program_path.write_text(candidate["program"], encoding="utf-8")


if __name__ == "__main__":
    import os

    max_traces_env = os.environ.get("CANT_BE_LATE_MAX_TRACES")
    max_traces = int(max_traces_env) if max_traces_env else None
    max_metric_calls_env = os.environ.get("GEPA_MAX_METRIC_CALLS")
    max_metric_calls = int(max_metric_calls_env) if max_metric_calls_env else 20
    trace_ratio_env = os.environ.get("CANT_BE_LATE_TRACE_RATIO")
    trace_ratio = DEFAULT_TRACE_RATIO
    if trace_ratio_env:
        try:
            trace_ratio = float(trace_ratio_env)
        except ValueError:
            print(
                f"Invalid CANT_BE_LATE_TRACE_RATIO='{trace_ratio_env}', falling back to {DEFAULT_TRACE_RATIO:.2f}",
                flush=True,
            )
    trace_ratio = min(1.0, max(trace_ratio, DEFAULT_TRACE_RATIO))
    skip_test = os.environ.get("GEPA_SKIP_TEST", "0") == "1"

    run_dir = _resolve_run_dir()
    print(f"GEPA artifacts will be saved to: {run_dir}")
    print(f"Using {trace_ratio:.0%} of traces for train/val evaluation", flush=True)

    adapter = CantBeLateAdapter(
        # model="openai/gpt-4.1-mini", # TODO: do we need a task_lm or not 
        # model="openai/gpt-4.1-mini", # this is used for reflection LM 
        model="openai/o3"
    )
    
    # Load from train and test set
    train_set, val_set, test_set = load_dataset(
        max_traces_per_split=max_traces,
        trace_ratio=trace_ratio,
        include_test=not skip_test,
    )
    
    if skip_test:
        base_score: Optional[float] = None
        print("Base program score: skipped (GEPA_SKIP_TEST=1)")
    else:
        output_base = adapter.evaluate(test_set, {"program": INITIAL_PROGRAM_SRC})
        base_score = sum(output_base.scores)
        print(f"Base program score: {base_score}")

    # NOTE(core): GEPA optimization 
    from gepa import optimize
    gepa_result = optimize(
        seed_candidate={"program": INITIAL_PROGRAM_SRC},
        trainset=train_set,
        valset=val_set,
        adapter=adapter,
        reflection_lm="openai/o3",
        max_metric_calls=max_metric_calls,
        run_dir=str(run_dir),
    )
    best_candidate = gepa_result.best_candidate
    print(f"Best program from optimization: {best_candidate['program']}")

    if skip_test:
        optimized_score: Optional[float] = None
        print("Optimized program score: skipped (GEPA_SKIP_TEST=1)")
    else:
        output_optimized = adapter.evaluate(test_set, best_candidate)
        optimized_score = sum(output_optimized.scores)
        print(f"Optimized program score: {optimized_score}")

    _write_checkpoints(run_dir, gepa_result, base_score, optimized_score, best_candidate)
    print(f"Checkpoint artifacts written under {run_dir}")
