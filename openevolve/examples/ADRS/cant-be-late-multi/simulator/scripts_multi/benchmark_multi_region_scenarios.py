import argparse
import json
import logging
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

# --- Configuration ---
# Use the new aligned data source
DATA_PATH = "data/converted_multi_region_aligned"
OUTPUT_DIR = Path("outputs/multi_region_scenario_analysis")
TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2
MAX_WORKERS = 8

# --- Experiment Definitions ---

# This single-region strategy will be run on each region within a scenario
# for comparison, if `compare_single_region` is True for that scenario.
SINGLE_REGION_HEURISTIC_TO_COMPARE = "rc_cr_threshold"

# Define display names and colors for strategies for consistent plotting
STRATEGY_DISPLAY_NAMES = {
    "multi_region_rc_cr_threshold": "Multi-Region Uniform Progress",
    "multi_region_rc_cr_quality_bar": "Multi-Region Uniform Progress (w/o Cond. 2)",
    "multi_region_rc_cr_threshold_optimized": "Multi-Region Uniform Progress (Random)",
    "quick_optimal": "Optimal (on Union)",
    SINGLE_REGION_HEURISTIC_TO_COMPARE: "Single-Region Uniform Progress",
    "best_single_region": "Best Single-Region",
    "average_single_region": "Average Single-Region",
}
STRATEGY_COLORS = {
    # "multi_region_rc_cr_threshold": "mediumseagreen",
    # "multi_region_rc_cr_quality_bar": "darkviolet",
    # "quick_optimal": "orangered",
    "multi_region_rc_cr_threshold_optimized": "dodgerblue",
    # "best_single_region": "gray",
    "average_single_region": "black",
}

ALL_REGIONS = [
    "us-east-2a_v100_1",
    "us-west-2c_v100_1",
    "us-east-1d_v100_1",
    "us-east-2b_v100_1",
    "us-west-2a_v100_1",
    "us-east-1f_v100_1",
    "us-east-1a_v100_1",
    "us-west-2b_v100_1",
    "us-east-1c_v100_1",
]


# Define the scenarios to run. Each scenario is a dictionary containing:
# - name: A descriptive name for the plot.
# - regions: A list of region names to include in this scenario.
# - strategies: A list of multi-region strategy identifiers to run.
# - compare_single_region: If True, runs the single-region heuristic on each
#   region individually and plots the 'best' and 'average' as baselines.
EXPERIMENT_SCENARIOS = [
    {
        "name": "2 Zones (in us-east-1 region)",
        "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1"],
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "2 Regions",
        "regions": ["us-east-1a_v100_1", "us-west-2a_v100_1"],
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "3 Regions",
        "regions": ["us-east-1a_v100_1", "us-east-2a_v100_1", "us-west-2b_v100_1"],
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "3 Zones (Same Region)",
        "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1", "us-east-1d_v100_1"],
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "5 Regions (High Diversity)",
        "regions": [
            "us-east-1a_v100_1",
            "us-east-1f_v100_1",
            "us-west-2a_v100_1",
            "us-west-2b_v100_1",
            "us-east-2b_v100_1",
        ],
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "5 Regions (1)",
        # random 5 regions from ALL_REGIONS
        "regions": random.sample(ALL_REGIONS, 5),
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "5 Regions (2)",
        # random 5 regions from ALL_REGIONS
        "regions": random.sample(ALL_REGIONS, 5),
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "5 Regions (3)",
        # random 5 regions from ALL_REGIONS
        "regions": random.sample(ALL_REGIONS, 5),
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
    {
        "name": "All 9 Regions",
        "regions": ALL_REGIONS,
        "strategies": [
            "multi_region_rc_cr_threshold",
            "multi_region_rc_cr_quality_bar",
            "multi_region_rc_cr_threshold_optimized",
            "quick_optimal",
        ],
        "compare_single_region": True,
    },
]


# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
UNION_TRACES_DIR = OUTPUT_DIR / "union_traces"
UNION_TRACES_DIR.mkdir(exist_ok=True)


def get_or_create_union_trace(regions: List[str], trace_index: int) -> str:
    """Creates or retrieves a union trace for a specific set of regions and trace index."""
    trace_idx_file = f"{trace_index}.json"
    regions_str = "+".join(sorted(regions))
    union_trace_filename = f"union_{regions_str}_{trace_index}.json"
    union_trace_path = UNION_TRACES_DIR / union_trace_filename

    if union_trace_path.exists():
        return str(union_trace_path)

    all_data, metadata_template, min_len = [], None, float("inf")

    for region_name in regions:
        trace_file_path = Path(DATA_PATH) / region_name / trace_idx_file
        if not trace_file_path.exists():
            raise FileNotFoundError(
                f"Required trace file does not exist: {trace_file_path}"
            )
        with open(trace_file_path, "r") as f:
            trace = json.load(f)
        all_data.append(trace["data"])
        min_len = min(min_len, len(trace["data"]))
        if metadata_template is None:
            metadata_template = trace["metadata"]

    trimmed_data = [data[: int(min_len)] for data in all_data]
    final_union = [
        1 if all(d[i] == 1 for d in trimmed_data) else 0 for i in range(int(min_len))
    ]
    union_trace_content = {"metadata": metadata_template, "data": final_union}

    with open(union_trace_path, "w") as f:
        json.dump(union_trace_content, f)
    logger.info(f"Created Union Trace for {len(regions)} regions at {union_trace_path}")
    return str(union_trace_path)


def generate_cache_filename(strategy: str, env: str, traces: list[str]) -> str:
    """Generates a cache filename, using hash if too long."""
    trace_descs = []
    for trace_path_str in traces:
        trace_path = Path(trace_path_str)
        if "union" in trace_path.name:
            # For union traces, the name itself is descriptive
            trace_descs.append(trace_path.stem)
        else:
            trace_descs.append(f"{trace_path.parent.name}_{trace_path.stem}")

    trace_identifier = "+".join(sorted(trace_descs))
    safe_strategy_name = strategy.replace("/", "_").replace("\\", "_")
    filename = f"{safe_strategy_name}_{env}_{trace_identifier}.json"
    
    # If filename is too long, use hash
    if len(filename) > 200:  # Conservative limit to avoid filesystem issues
        content_hash = hashlib.md5(f"{strategy}_{env}_{trace_identifier}".encode()).hexdigest()
        # Keep strategy name but shorten traces part
        short_strategy = safe_strategy_name[-50:] if len(safe_strategy_name) > 50 else safe_strategy_name
        filename = f"{short_strategy}_{env}_{content_hash}.json"
    
    return filename


def run_simulation(strategy: str, env_type: str, trace_paths: List[str], strategy_file: Optional[str] = None) -> float:
    """A general-purpose simulation runner with caching."""
    
    # Use strategy_file path for caching if it exists, otherwise use strategy name
    cache_key_strategy = strategy_file if strategy_file else strategy
    cache_filename = generate_cache_filename(cache_key_strategy, env_type, trace_paths)
    cache_file = CACHE_DIR / cache_filename

    if cache_file.exists():
        with open(cache_file, "r") as f:
            cost = json.load(f)["mean_cost"]
        logger.info(
            f"Cache HIT for {cache_key_strategy} on {len(trace_paths)} traces -> ${cost:.2f}"
        )
        return cost

    logger.info(f"Cache MISS for: {cache_filename}")
    cmd = [
        "python",
        "./main.py",
        f"--env={env_type}",
        f"--output-dir={OUTPUT_DIR / 'sim_temp'}",
        f"--task-duration-hours={TASK_DURATION_HOURS}",
        f"--deadline-hours={DEADLINE_HOURS}",
        f"--restart-overhead-hours={RESTART_OVERHEAD_HOURS}",
        # f"--env-start-hours={ENV_START_HOURS}",
    ]

    if strategy_file:
        cmd.append(f"--strategy-file={strategy_file}")
    else:
        cmd.append(f"--strategy={strategy}")

    if env_type == "trace":
        cmd.append(f"--trace-file={trace_paths[0]}")
    else:
        cmd.extend(["--trace-files"] + trace_paths)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=900
        )
        match = re.search(r"mean:\s*([\d.]+)", result.stdout + result.stderr)
        if not match:
            raise RuntimeError("Could not parse cost from simulation output.")

        mean_cost = float(match.group(1))
        with open(cache_file, "w") as f:
            json.dump({"mean_cost": mean_cost}, f, indent=2)
        return mean_cost
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        RuntimeError,
    ) as e:
        logger.error(
            f"Simulation failed for {strategy}. Stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}"
        )
        return np.nan


def create_simulation_task(scenario, strategy, trace_index, task_type, region=None, strategy_file=None):
    """Creates a dictionary representing a single simulation task."""
    return {
        "scenario_name": scenario["name"],
        "regions_in_scenario": scenario["regions"],
        "strategy": strategy,
        "trace_index": trace_index,
        "task_type": task_type,
        "region": region,
        "strategy_file": strategy_file,
    }


def execute_simulation_task(task):
    """Executes a task, handling different environment types."""
    try:
        trace_idx = task["trace_index"]
        strategy = task["strategy"]
        strategy_file = task.get("strategy_file")
        cost = np.nan

        if task["task_type"] == "multi_region":
            regions = task["regions_in_scenario"]
            if strategy == "quick_optimal":
                env_type = "trace"
                trace_paths = [get_or_create_union_trace(regions, trace_idx)]
            else:
                env_type = "multi_trace"
                trace_paths = [
                    str(Path(DATA_PATH) / r / f"{trace_idx}.json") for r in regions
                ]
                if not all(Path(p).exists() for p in trace_paths):
                    raise FileNotFoundError("One or more regional traces are missing.")
            cost = run_simulation(strategy, env_type, trace_paths, strategy_file=strategy_file)

        elif task["task_type"] == "single_region":
            region = task["region"]
            env_type = "trace"
            trace_path = Path(DATA_PATH) / region / f"{trace_idx}.json"
            if not trace_path.exists():
                raise FileNotFoundError(
                    f"Trace file not found for single region: {trace_path}"
                )
            cost = run_simulation(strategy, env_type, [str(trace_path)], strategy_file=strategy_file)

        else:
            raise ValueError(f"Unknown task type: {task['task_type']}")

        return {**task, "cost": cost}
    except Exception as e:
        logger.error(
            f"Failed task {task['scenario_name']}/{task['strategy']}/trace_{task['trace_index']}: {e}"
        )
        return {**task, "cost": np.nan}


def plot_scenario_results(df: pd.DataFrame, num_traces: int, custom_strategy_name: Optional[str] = None):
    """Generates and saves a grouped bar chart for the experiment scenarios."""

    multi_region_df = df[df["task_type"] == "multi_region"]
    single_region_df = df[df["task_type"] == "single_region"]

    # Calculate statistics for the main bar plot (multi-region strategies)
    stats_df = (
        multi_region_df.groupby(["scenario_name", "strategy"])["cost"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats_df["std"] = stats_df["std"].fillna(0)

    # Calculate single-region baselines and add them to the main stats_df as new bars
    if not single_region_df.empty:
        single_region_means = (
            single_region_df.groupby(["scenario_name", "region"])["cost"]
            .mean()
            .reset_index()
        )
        scenario_baselines = (
            single_region_means.groupby("scenario_name")["cost"]
            .agg(best_single_cost="min", avg_single_cost="mean")
            .reset_index()
        )

        baseline_dfs = []
        for _, row in scenario_baselines.iterrows():
            scenario_name = row["scenario_name"]
            baseline_dfs.append(
                pd.DataFrame(
                    [
                        {
                            "scenario_name": scenario_name,
                            "strategy": "best_single_region",
                            "mean": row["best_single_cost"],
                            "std": 0,
                        }
                    ]
                )
            )
            baseline_dfs.append(
                pd.DataFrame(
                    [
                        {
                            "scenario_name": scenario_name,
                            "strategy": "average_single_region",
                            "mean": row["avg_single_cost"],
                            "std": 0,
                        }
                    ]
                )
            )
        if baseline_dfs:
            stats_df = pd.concat([stats_df] + baseline_dfs, ignore_index=True)

    # --- Define a canonical order for scenarios and strategies ---
    scenario_order = [s["name"] for s in EXPERIMENT_SCENARIOS]
    # Filter for strategies that actually have results to avoid issues
    present_strategies = stats_df["strategy"].unique()
    strategy_order = sorted([s for s in STRATEGY_COLORS if s in present_strategies])

    # Apply categorical ordering to ensure plots follow the intended sequence
    stats_df["scenario_name"] = pd.Categorical(
        stats_df["scenario_name"], categories=scenario_order, ordered=True
    )
    stats_df = stats_df.sort_values("scenario_name")

    # --- Plotting ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.barplot(
        x="scenario_name",
        y="mean",
        hue="strategy",
        data=stats_df,
        palette=STRATEGY_COLORS,
        ax=ax,
        order=scenario_order,
        hue_order=strategy_order,
    )

    # --- Robustly Add Error Bars ---
    patch_map = []
    for strategy in strategy_order:
        for scenario in scenario_order:
            row = stats_df[
                (stats_df["scenario_name"] == scenario)
                & (stats_df["strategy"] == strategy)
            ]
            if not row.empty:
                patch_map.append(row.iloc[0])

    if len(ax.patches) != len(patch_map):
        logger.warning(
            "Patch map length does not match number of bars. Error bars may be incorrect."
        )

    for i, patch in enumerate(ax.patches):
        if i < len(patch_map):
            std_val = patch_map[i]["std"]
            if std_val > 0:  # Only draw error bars if there is variance
                ax.errorbar(
                    patch.get_x() + patch.get_width() / 2.0,
                    patch.get_height(),
                    yerr=std_val,
                    fmt="none",
                    c="black",
                    capsize=4,
                )

    # --- Final Plot Customization ---
    ax.set_title(
        f"Cost Comparison Across Multi-Region Scenarios (n={num_traces})",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xlabel("Scenario", fontsize=14)
    ax.set_ylabel(r"Mean Execution Cost \($\)", fontsize=14)
    ax.tick_params(axis="x", rotation=10, labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="x", which="minor", linestyle="--")

    # --- Custom Legend ---
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [STRATEGY_DISPLAY_NAMES.get(l, l) for l in labels],
        title="Strategy",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # Position below the plot
        ncol=5,  # Arrange in columns
        fontsize=11,
        frameon=False,
    )

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 1])
    plot_path = OUTPUT_DIR / f"scenario_comparison_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"📊 Plot saved to {plot_path}")
    plt.close(fig)

    # Calculate and print cost saving if custom strategy was used
    # Use the same stats_df that was used for plotting (the bar heights)
    if custom_strategy_name:
        baseline_strategy = "multi_region_rc_cr_threshold_optimized"
        cost_ratios = []
        
        print(f"\n--- Cost Saving Analysis: {custom_strategy_name} vs {baseline_strategy} ---")
        print(f"(Using aggregated mean values from the plot bars)\n")
        
        for scenario in EXPERIMENT_SCENARIOS:
            scenario_name = scenario["name"]
            
            # Get the mean values from stats_df (the same data used for plotting)
            custom_mean = stats_df[
                (stats_df["scenario_name"] == scenario_name) 
                & (stats_df["strategy"] == custom_strategy_name)
            ]["mean"]
            baseline_mean = stats_df[
                (stats_df["scenario_name"] == scenario_name) 
                & (stats_df["strategy"] == baseline_strategy)
            ]["mean"]
            
            if not custom_mean.empty and not baseline_mean.empty:
                ac = custom_mean.iloc[0]
                mc = baseline_mean.iloc[0]
                if mc > 0:
                    ratio = (mc - ac) / mc  # Positive means savings
                    cost_ratios.append(ratio)
                    print(f"  {scenario_name}: custom=${ac:.2f}, baseline=${mc:.2f}, saving={ratio:.4f} ({ratio*100:.2f}%)")
        
        if cost_ratios:
            avg_ratio = sum(cost_ratios) / len(cost_ratios)
            max_ratio = max(cost_ratios)
            print(f"\n📊 Average Cost Saving: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
            print(f"📊 Max Cost Saving: {max_ratio:.4f} ({max_ratio*100:.2f}%)")
        else:
            print("⚠️  Could not calculate cost ratios (missing data)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multi-region strategy scenarios."
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=10,
        help="Number of traces to average over for each run.",
    )
    parser.add_argument(
        "--strategy-file",
        type=str,
        default=None,
        help="Path to a custom strategy file to include in the benchmark."
    )
    args = parser.parse_args()

    all_tasks = []
    
    custom_strategy_name = None
    if args.strategy_file:
        custom_strategy_name = Path(args.strategy_file).stem
        STRATEGY_DISPLAY_NAMES[custom_strategy_name] = f"Custom ({custom_strategy_name})"
        STRATEGY_COLORS[custom_strategy_name] = "purple"

    for scenario in EXPERIMENT_SCENARIOS:
        strategies_to_run = scenario["strategies"][:] # Create a copy
        if custom_strategy_name:
            strategies_to_run.append(custom_strategy_name)

        for strategy in strategies_to_run:
            for i in range(args.num_traces):
                task_data = create_simulation_task(
                    scenario, 
                    strategy, 
                    i, 
                    "multi_region",
                    strategy_file=args.strategy_file if strategy == custom_strategy_name else None
                )
                all_tasks.append(task_data)

        if scenario.get("compare_single_region"):
            for region in scenario["regions"]:
                for i in range(args.num_traces):
                    all_tasks.append(
                        create_simulation_task(
                            scenario,
                            SINGLE_REGION_HEURISTIC_TO_COMPARE,
                            i,
                            "single_region",
                            region=region,
                        )
                    )

    logger.info(
        f"🚀 Found {len(all_tasks)} simulation tasks to run across {len(EXPERIMENT_SCENARIOS)} scenarios."
    )

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(execute_simulation_task, task): task for task in all_tasks
        }
        for future in as_completed(future_to_task):
            result = future.result()
            if result:
                results.append(result)

    if not results:
        logger.error("No simulation tasks completed successfully. Exiting.")
        return

    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / f"scenario_results_t{args.num_traces}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ All detailed results saved to: {csv_path}")

    plot_scenario_results(df, args.num_traces, custom_strategy_name)

    # Print summary
    summary = (
        df.groupby(["scenario_name", "strategy"])["cost"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print("\n--- Final Results Summary ---")
    print(summary)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
