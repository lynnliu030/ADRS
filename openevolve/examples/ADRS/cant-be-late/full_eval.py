#!/usr/bin/env python3
"""Run the ADRS evaluator against every available trace and report averages."""

import argparse
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import evaluator as base_evaluator
from openevolve.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


@contextmanager
def _evaluator_context(cap: Optional[int]):
    """Less traces for faster evaluation."""
    original_trace = getattr(base_evaluator, "TRACE_TARGET", None)
    original_envs = getattr(base_evaluator, "ENV_PATHS", None)
    trace_overridden = False
    env_overridden = False
    try:
        if cap is not None:
            base_evaluator.TRACE_TARGET = cap
            trace_overridden = True
        if isinstance(original_envs, list) and len(original_envs) > 1:
            preferred_env = None
            k80_envs = [env for env in original_envs if "k80" in env]
            if k80_envs:
                preferred_env = sorted(k80_envs)[-1]
            else:
                preferred_env = original_envs[0]
            base_evaluator.ENV_PATHS = [preferred_env]
            env_overridden = True
        yield
    finally:
        if trace_overridden and original_trace is not None:
            base_evaluator.TRACE_TARGET = original_trace
        if env_overridden:
            base_evaluator.ENV_PATHS = original_envs


def run_full_evaluation(program_path: str, trace_cap: Optional[int] = None):
    """Run stage2 evaluation with the requested trace cap."""
    with _evaluator_context(trace_cap):
        return base_evaluator.evaluate_stage2(program_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "program_path",
        type=str,
        help="Path to the strategy implementation to evaluate",
    )
    parser.add_argument(
        "--trace-cap",
        type=int,
        default=None,
        help="Optional cap for traces per env (defaults to no cap)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level forwarded to the underlying evaluator (default: INFO)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a simple stderr progress counter during evaluation",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=None,
        help=(
            "Baseline strategy path for comparison. "
            "Specify multiple times to compare against several baselines "
            "(default: initial_greedy)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the detailed JSON comparison report",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    trace_cap = args.trace_cap
    if trace_cap is None:
        # Large number so the evaluator takes every eligible trace (all envs currently 100)
        trace_cap = 1_000_000

    if args.progress:
        os.environ["EVALUATOR_PROGRESS"] = "1"

    logger.info("Evaluating target strategy: %s", args.program_path)
    target_result = run_full_evaluation(args.program_path, trace_cap)

    baseline_paths = args.baseline
    if not baseline_paths:
        baseline_paths = [
            "openevolve/examples/ADRS/cant-be-late/initial_greedy.py"
        ]

    def _extract(result):
        if isinstance(result, EvaluationResult):
            return dict(result.metrics), dict(result.artifacts)
        return dict(result), {}

    target_metrics, target_artifacts = _extract(target_result)
    if target_metrics.get("runs_successfully") != 1.0:
        logger.error(
            "Target strategy evaluation failed: %s",
            target_metrics.get("error", "unknown error"),
        )
        print(json.dumps({"error": target_metrics}, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    def _sum_counts(scenario_stats):
        return sum(entry.get("count", 0) for entry in scenario_stats.values())

    target_scenario_stats = target_metrics.get("scenario_stats", {})
    total_runs = _sum_counts(target_scenario_stats)

    def _build_summary(metrics):
        return {
            "average_cost": metrics.get("avg_cost"),
            "cost_std": metrics.get("cost_std"),
            "min_cost": metrics.get("min_cost"),
            "max_cost": metrics.get("max_cost"),
            "cost_range": (
                (metrics.get("max_cost") - metrics.get("min_cost"))
                if metrics.get("max_cost") is not None
                and metrics.get("min_cost") is not None
                else None
            ),
            "score": metrics.get("score"),
            "combined_score": metrics.get("combined_score"),
        }

    summary = _build_summary(target_metrics)
    summary["total_runs"] = total_runs

    def _load_trace_costs(artifact_dict):
        payload = artifact_dict.get("trace_costs_json")
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to decode trace_costs_json artifact")
            return {}
    def _load_trace_costs_for(result):
        metrics, artifacts = result
        trace_costs = _load_trace_costs(artifacts)
        return metrics, trace_costs

    target_trace_costs = _load_trace_costs(target_artifacts)

    def _compute_comparison(base_trace_costs):
        improvements = []
        for scenario_key, entries in target_trace_costs.items():
            baseline_entries = base_trace_costs.get(scenario_key, [])
            baseline_map = {item["trace_name"]: item for item in baseline_entries}
            for entry in entries:
                trace_name = entry.get("trace_name")
                if trace_name not in baseline_map:
                    continue
                base_cost = baseline_map[trace_name].get("cost")
                new_cost = entry.get("cost")
                if base_cost is None or new_cost is None:
                    continue
                improvement = base_cost - new_cost
                ratio = None
                if base_cost not in (None, 0):
                    ratio = improvement / base_cost
                improvements.append(
                    {
                        "scenario": scenario_key,
                        "trace_name": trace_name,
                        "config": entry.get("config"),
                        "baseline_cost": base_cost,
                        "new_cost": new_cost,
                        "improvement": improvement,
                        "improvement_ratio": ratio,
                    }
                )
        if not improvements:
            logger.warning("No comparable traces matched")
        improvement_values = [item["improvement"] for item in improvements]
        ratio_values = [
            item["improvement_ratio"]
            for item in improvements
            if item["improvement_ratio"] is not None
        ]
        comparison = {
            "count": len(improvements),
            "avg_improvement": (
                float(sum(improvement_values) / len(improvement_values))
                if improvement_values
                else None
            ),
            "min_improvement": min(improvement_values) if improvement_values else None,
            "max_improvement": max(improvement_values) if improvement_values else None,
            "improvement_range": (
                max(improvement_values) - min(improvement_values)
                if len(improvement_values) >= 2
                else None
            ),
            "avg_improvement_ratio": (
                float(sum(ratio_values) / len(ratio_values))
                if ratio_values
                else None
            ),
            "min_improvement_ratio": min(ratio_values) if ratio_values else None,
            "max_improvement_ratio": max(ratio_values) if ratio_values else None,
            "improvement_ratio_range": (
                max(ratio_values) - min(ratio_values)
                if len(ratio_values) >= 2
                else None
            ),
            "per_trace": improvements,
        }
        return comparison

    def _overhead_stats(per_trace, overhead_value: float) -> dict:
        subset = [
            item for item in per_trace
            if item.get("config", {}).get("overhead") == overhead_value
            and item.get("baseline_cost") not in (None, 0)
        ]
        if not subset:
            return {}
        subset_ratios = [item["improvement"] / item["baseline_cost"] for item in subset]
        return {
            "count": len(subset),
            "avg_ratio": float(sum(subset_ratios) / len(subset_ratios)),
            "min_ratio": float(min(subset_ratios)),
            "max_ratio": float(max(subset_ratios)),
        }


    baseline_records = []
    for baseline_path in baseline_paths:
        logger.info("Evaluating baseline strategy: %s", baseline_path)
        baseline_result = run_full_evaluation(baseline_path, trace_cap)
        baseline_metrics, baseline_artifacts = _extract(baseline_result)
        if baseline_metrics.get("runs_successfully") != 1.0:
            logger.error(
                "Baseline strategy evaluation failed: %s",
                baseline_metrics.get("error", "unknown error"),
            )
            print(
                json.dumps({"error": baseline_metrics}, indent=2, ensure_ascii=False)
            )
            raise SystemExit(1)

        baseline_summary = _build_summary(baseline_metrics)
        baseline_summary["total_runs"] = _sum_counts(
            baseline_metrics.get("scenario_stats", {})
        )
        _, baseline_trace_costs = _load_trace_costs_for(
            (baseline_metrics, baseline_artifacts)
        )
        comparison = _compute_comparison(baseline_trace_costs)
        overhead_stats = _overhead_stats(comparison["per_trace"], 0.02)

        min_imp = comparison.get("min_improvement")
        max_imp = comparison.get("max_improvement")
        min_ratio = comparison.get("min_improvement_ratio")
        max_ratio = comparison.get("max_improvement_ratio")
        baseline_label = Path(baseline_path).name
        range_msg = ""
        if min_imp is not None and max_imp is not None:
            ratio_msg = ""
            if min_ratio is not None and max_ratio is not None:
                ratio_msg = f" ({min_ratio*100:.1f}% → {max_ratio*100:.1f}%)"
            range_msg = (
                f"Improvement range vs {baseline_label}: {min_imp:.2f} → {max_imp:.2f}{ratio_msg}"
            )
        if overhead_stats:
            avg_pct = overhead_stats["avg_ratio"] * 100.0
            min_pct = overhead_stats["min_ratio"] * 100.0
            max_pct = overhead_stats["max_ratio"] * 100.0
            range_msg += (
                f"; overhead=0.02 → avg {avg_pct:.2f}% (range {min_pct:.2f}% → {max_pct:.2f}%)"
            )
        if range_msg:
            logger.info(range_msg)
        else:
            logger.info("No effective improvement statistics (baseline=%s)", baseline_label)

        baseline_records.append(
            {
                "path": baseline_path,
                "summary": baseline_summary,
                "comparison": comparison,
                "overhead_0_02": overhead_stats,
            }
        )

    output = {
        "summary": summary,
        "baselines": baseline_records,
    }

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    first_record = baseline_records[0] if baseline_records else None
    if first_record:
        comp = first_record["comparison"]
        min_imp = comp.get("min_improvement")
        max_imp = comp.get("max_improvement")
        min_ratio = comp.get("min_improvement_ratio")
        max_ratio = comp.get("max_improvement_ratio")
        overhead_stats = first_record.get("overhead_0_02", {})
        baseline_label = Path(first_record["path"]).name
        if min_imp is not None and max_imp is not None:
            ratio_msg = ""
            if min_ratio is not None and max_ratio is not None:
                ratio_msg = f" ({min_ratio*100:.1f}% → {max_ratio*100:.1f}%)"
            overhead_msg = ""
            if overhead_stats:
                overhead_msg = (
                    f"; overhead=0.02 → avg {overhead_stats['avg_ratio']*100:.2f}% "
                    f"(range {overhead_stats['min_ratio']*100:.2f}% → "
                    f"{overhead_stats['max_ratio']*100:.2f}%)"
                )
            print(
                f"Improvement range vs {baseline_label}: {min_imp:.2f} → {max_imp:.2f}{ratio_msg}{overhead_msg}"
                f" (saved to {output_path})"
            )
        else:
            print(f"No comparable traces found; details saved to {output_path}")
    else:
        print(f"No baselines evaluated; details saved to {output_path}")


if __name__ == "__main__":
    main()
