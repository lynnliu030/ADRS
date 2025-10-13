import os
import json
import subprocess
import re
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns

# Define display names for strategies
STRATEGY_DISPLAY_NAMES = {
    "quick_optimal": "Optimal Cost",
    "rc_cr_threshold": "Single-Region Uniform Progress",
    "rc_cr_threshold_no_condition2": "Single-Region Uniform Progress (No Cond. 2)",
    "multi_region_rc_cr_threshold": "Multi-Region Uniform Progress",
    "cr_only_threshold": "CR Only",  # If you use this strategy
}

# --- Configuration ---
DATA_PATH = "data/converted_multi_region_aligned"
OUTPUT_DIR = Path("outputs/multi_region_analysis")

# DATA_PATH = f"data/real/ping_based/random_start_time"
# OUTPUT_DIR = Path("outputs/unified_instance_analysis")
TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2
OPTIMAL_STRATEGY_NAME = "quick_optimal"
HEURISTIC_STRATEGY_NAME = "rc_cr_threshold"
MULTI_ZONE_HEURISTIC_NAME = "multi_region_rc_cr_threshold"
MAX_WORKERS = 4  # Number of parallel workers

# --- Analysis Control Switches ---
RUN_OPTIMAL_ANALYSIS = False  # Core optimal strategy analysis
RUN_SINGLE_ZONE_HEURISTICS = True  # Run rc_cr_threshold for individual zones
RUN_UNION_ANALYSIS = False  # NEW: Controls analysis on the 'All Zones Union' trace
RUN_MULTI_ZONE_HEURISTICS = True  # Run multi_region_rc_cr_threshold for instance groups

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

# --- Heuristics to Evaluate ---
# List of single-zone heuristic strategies to run on each zone's trace individually.
SINGLE_ZONE_HEURISTICS = [
    "rc_cr_threshold",  # Uniform Progress
    # "rc_cr_threshold_no_condition2",  # Uniform Progress (No Cond. 2)
]

# List of multi-zone heuristic strategies to run on the combined multi-zone traces.
MULTI_ZONE_HEURISTICS = [
    "multi_region_rc_cr_threshold",
    # "multi_region_rc_cr_quality_bar",
]


def generate_human_readable_cache_filename(
    strategy: str, env: str, traces: list[str]
) -> str:
    """根据新规则生成可读的文件名"""
    trace_descs = []
    trace_index = ""
    for trace_path_str in traces:
        trace_path = Path(trace_path_str)
        trace_descs.append(trace_path.parent.name)
        if not trace_index:
            trace_index = trace_path.stem
    trace_identifier = "+".join(sorted(trace_descs))
    safe_strategy_name = strategy.replace("/", "_")
    return f"{safe_strategy_name}_{env}_{trace_identifier}_{trace_index}.json"


def find_instance_groups(data_path: str) -> Dict[str, List[str]]:
    # This function remains unchanged
    groups = defaultdict(list)
    pattern = re.compile(r"_(v\d+_\d+|k\d+_\d+)$")
    for dir_name in os.listdir(data_path):
        full_path = Path(data_path) / dir_name
        if full_path.is_dir():
            match = pattern.search(dir_name)
            if match:
                instance_type = match.group(1)
                group_key = f"us-{instance_type}"
                groups[group_key].append(str(full_path))
    logger.info(f"Found {len(groups)} unified instance group(s).")
    for name, paths in groups.items():
        logger.info(f"- Group '{name}' contains {len(paths)} zones.")
    return groups


def discover_available_trace_files(zone_dirs: List[str]) -> List[int]:
    """
    Discover all available trace file indices across all zones.
    Returns a sorted list of indices that are present in ALL zones.
    """
    if not zone_dirs:
        return []

    # Get all json files from the first zone
    first_zone = Path(zone_dirs[0])
    available_indices = set()

    for file_path in first_zone.glob("*.json"):
        if file_path.name.replace(".json", "").isdigit():
            available_indices.add(int(file_path.name.replace(".json", "")))

    # Check that all other zones have the same files
    for zone_dir in zone_dirs[1:]:
        zone_path = Path(zone_dir)
        zone_indices = set()

        for file_path in zone_path.glob("*.json"):
            if file_path.name.replace(".json", "").isdigit():
                zone_indices.add(int(file_path.name.replace(".json", "")))

        # Keep only indices that exist in all zones
        available_indices = available_indices.intersection(zone_indices)

    sorted_indices = sorted(list(available_indices))
    logger.info(
        f"Found {len(sorted_indices)} common trace files across all zones: {min(sorted_indices)} to {max(sorted_indices)}"
    )
    return sorted_indices


def run_simulation(strategy: str, env_type: str, trace_paths: List[str]) -> float:
    """
    A general-purpose simulation runner with a human-readable, self-describing caching system.
    """
    # Generate the new, descriptive cache filename instead of an MD5 hash
    cache_filename = generate_human_readable_cache_filename(
        strategy, env_type, trace_paths
    )
    cache_file = CACHE_DIR / cache_filename

    # --- Cache Reading Logic ---
    if cache_file.exists():
        logger.info(
            f"Cache HIT for {strategy} on {trace_paths} using file: {cache_filename}"
        )
        with open(cache_file, "r") as f:
            # We can still just return the cost, but the file now has more info
            return json.load(f)["mean_cost"]

    # --- Cache Miss Logic ---
    logger.info(f"Cache MISS for: {cache_filename}")

    cmd = [
        "python",
        "./main.py",
        f"--strategy={strategy}",
        f"--env={env_type}",
        f"--output-dir={OUTPUT_DIR / 'sim_temp'}",
        f"--task-duration-hours={TASK_DURATION_HOURS}",
        f"--deadline-hours={DEADLINE_HOURS}",
        f"--restart-overhead-hours={RESTART_OVERHEAD_HOURS}",
    ]
    if env_type == "trace":
        cmd.append(f"--trace-file={trace_paths[0]}")
    else:
        cmd.extend(["--trace-files"] + trace_paths)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=900
        )
        output = result.stdout + result.stderr
        match = re.search(r"mean:\s*([\d.]+)", output)
        if not match:
            raise RuntimeError(f"Could not parse cost from output: {' '.join(cmd)}")

        mean_cost = float(match.group(1))

        # --- Cache Writing Logic ---
        with open(cache_file, "w") as f:
            # Create a self-describing cache file
            cache_content = {
                "mean_cost": mean_cost,
                "parameters": {
                    "strategy": strategy,
                    "env": env_type,
                    "traces": sorted(trace_paths),
                },
            }
            json.dump(cache_content, f, indent=2)  # Use indent for readability

        return mean_cost

    except subprocess.CalledProcessError as e:
        logger.error(
            f"FATAL: Simulation failed. Command: {' '.join(e.cmd)}\nStderr:\n{e.stderr}"
        )
        raise e


def get_or_create_mega_union_trace(
    group_name: str, zone_dirs: List[str], trace_index: int
) -> str:
    """
    Create or retrieve a mega union trace for a specific trace file index.
    """
    trace_idx_file = f"{trace_index}.json"
    union_trace_filename = f"mega_union_{group_name}_{trace_index}.json"
    union_trace_path = UNION_TRACES_DIR / union_trace_filename

    if union_trace_path.exists():
        return str(union_trace_path)

    all_data, metadata_template, min_len = [], None, float("inf")

    for zone_dir in zone_dirs:
        trace_file_path = Path(zone_dir) / trace_idx_file
        if not trace_file_path.exists():
            logger.warning(f"Trace file {trace_file_path} does not exist, skipping...")
            continue

        with open(trace_file_path, "r") as f:
            trace = json.load(f)
        all_data.append(trace["data"])
        min_len = min(min_len, len(trace["data"]))
        if metadata_template is None:
            metadata_template = trace["metadata"]

    if not all_data:
        logger.error(f"No valid trace data found for index {trace_index}")
        raise RuntimeError(f"No valid trace data found for index {trace_index}")

    trimmed_data = [data[:min_len] for data in all_data]
    final_union = []

    for i in range(min_len):
        is_all_preempted = all(data_list[i] == 1 for data_list in trimmed_data)
        final_union.append(1 if is_all_preempted else 0)

    union_trace_content = {"metadata": metadata_template, "data": final_union}

    with open(union_trace_path, "w") as f:
        json.dump(union_trace_content, f)
    logger.info(
        f"Created Mega Union Trace for {group_name} (index {trace_index}) at {union_trace_path}"
    )
    return str(union_trace_path)


def plot_all_region_pairs_chart(df: pd.DataFrame, output_dir: Path):
    """Generate and save a bar chart comparing costs across all region pairs."""

    heuristic_color_pairs = [
        ("lightcoral", "firebrick"),
        ("lightgreen", "seagreen"),
        ("plum", "mediumorchid"),
        ("darkkhaki", "olive"),
    ]

    # Pre-process data: clean zone names and calculate statistics
    df_clean = df.copy()
    df_clean["zone_name"] = df_clean["zone_name"].str.replace(r"_t\d+$", "", regex=True)

    stats_df = (
        df_clean.groupby(["group_name", "zone_name", "analysis_type", "strategy"])
        .agg(
            cost_mean=("cost", "mean"),
            cost_std=("cost", "std"),
            samples=("cost", "size"),
        )
        .reset_index()
    )

    stats_df["cost_std"] = stats_df["cost_std"].fillna(0)

    # Get unique instance groups
    groups = stats_df["group_name"].unique()
    n_groups = len(groups)
    if n_groups == 0:
        logger.warning("No data available to plot.")
        return

    # Create subplots
    if n_groups <= 2:
        fig, axes = plt.subplots(1, n_groups, figsize=(8 * n_groups, 7), squeeze=False)
        axes = axes.flatten()
    elif n_groups <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), squeeze=False)
        axes = axes.flatten()
    else:
        ncols = 3
        nrows = (n_groups + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 7 * nrows), squeeze=False)
        axes = axes.flatten()

    for i, group_name in enumerate(groups):
        ax = axes[i]
        group_data = stats_df[stats_df["group_name"] == group_name]

        # --- Data Preparation ---
        optimal_data = group_data[group_data["analysis_type"] == "optimal"].copy()
        all_heuristic_data = group_data[
            group_data["analysis_type"].isin(["single_heuristic", "union_heuristic"])
        ].copy()
        multi_heuristic_data = group_data[
            group_data["analysis_type"] == "multi_heuristic"
        ].copy()

        # Helper function to get the base zone name (e.g., 'us-west-2a' from 'us-west-2a_rc_cr_threshold')
        def get_base_name(row):
            if row["analysis_type"] in ["single_heuristic", "union_heuristic"]:
                strategy_suffix = f"_{row['strategy']}"
                if row["zone_name"].endswith(strategy_suffix):
                    return row["zone_name"][: -len(strategy_suffix)]
            return row["zone_name"]

        # Determine all unique plot categories from all available data, not just optimal
        base_names_df = group_data[
            group_data["analysis_type"] != "multi_heuristic"
        ].copy()
        if base_names_df.empty and multi_heuristic_data.empty:
            logger.warning(f"No data to plot for group {group_name}. Skipping.")
            continue

        base_names_df["base_zone_name"] = base_names_df.apply(get_base_name, axis=1)
        all_base_zones = sorted(base_names_df["base_zone_name"].unique())

        individual_zones_names = [
            z for z in all_base_zones if "All Zones Union" not in z
        ]
        union_zones_names = [z for z in all_base_zones if "All Zones Union" in z]

        # This dataframe now defines the structure of the x-axis
        plot_categories = individual_zones_names + union_zones_names
        plot_data = pd.DataFrame({"zone_name": plot_categories})

        if plot_data.empty and multi_heuristic_data.empty:
            logger.warning(
                f"No single-zone or multi-zone data to plot for group {group_name}. Skipping."
            )
            continue

        # --- Bar Plotting Setup ---
        n_single_heuristics = 0
        if RUN_SINGLE_ZONE_HEURISTICS and not all_heuristic_data.empty:
            n_single_heuristics = all_heuristic_data["strategy"].nunique()

        n_strategies = n_single_heuristics
        if RUN_OPTIMAL_ANALYSIS and not optimal_data.empty:
            n_strategies += 1

        bar_width = 0.8 / n_strategies if n_strategies > 1 else 0.4
        x_positions = np.arange(len(plot_data))

        # --- Plot Optimal Bars (conditionally) ---
        if RUN_OPTIMAL_ANALYSIS and not optimal_data.empty:
            optimal_offset_val = (
                -bar_width * (n_single_heuristics / 2) if n_strategies > 1 else 0
            )
            optimal_colors = ["skyblue"] * len(individual_zones_names) + [
                "orange"
            ] * len(union_zones_names)

            # Map costs and stds to the plot_data order
            costs_map = optimal_data.set_index("zone_name")["cost_mean"]
            stds_map = optimal_data.set_index("zone_name")["cost_std"]
            plot_costs = plot_data["zone_name"].map(costs_map).fillna(0)
            plot_stds = plot_data["zone_name"].map(stds_map).fillna(0)

            bars_optimal = ax.bar(
                x_positions + optimal_offset_val,
                plot_costs,
                bar_width,
                yerr=plot_stds,
                capsize=3,
                color=optimal_colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
            )

            for bar, std_val, cost_val in zip(bars_optimal, plot_stds, plot_costs):
                if cost_val > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + std_val + 2,
                        f"${height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        # --- Plot Heuristic Bars ---
        if RUN_SINGLE_ZONE_HEURISTICS and not all_heuristic_data.empty:
            all_heuristic_data["match_name"] = all_heuristic_data.apply(
                get_base_name, axis=1
            )
            heuristic_colors = ["lightcoral", "darkkhaki", "lightgreen", "plum"]
            present_heuristics = sorted(all_heuristic_data["strategy"].unique())

            # Calculate starting offset for heuristics
            start_offset = -bar_width * ((n_single_heuristics - 1) / 2)
            if RUN_OPTIMAL_ANALYSIS and not optimal_data.empty:
                # Adjust if optimal is also present
                start_offset = bar_width * (-(n_single_heuristics / 2) + 1)

            for i, strategy_name in enumerate(present_heuristics):
                heuristic_offset = start_offset + i * bar_width
                strategy_data = all_heuristic_data[
                    all_heuristic_data["strategy"] == strategy_name
                ]

                for idx, base_zone_name in enumerate(plot_data["zone_name"]):
                    heur_match = strategy_data[
                        strategy_data["match_name"] == base_zone_name
                    ]
                    if not heur_match.empty:
                        cost = heur_match["cost_mean"].iloc[0]
                        std = heur_match["cost_std"].iloc[0]
                        bar_pos = x_positions[idx] + heuristic_offset

                        is_union = "All Zones Union" in base_zone_name
                        color_pair = heuristic_color_pairs[
                            i % len(heuristic_color_pairs)
                        ]
                        color = color_pair[1] if is_union else color_pair[0]

                        ax.bar(
                            bar_pos,
                            cost,
                            bar_width,
                            yerr=std,
                            capsize=3,
                            color=color,
                            alpha=0.8,
                            edgecolor="darkred",
                            linewidth=0.5,
                        )
                        ax.text(
                            bar_pos,
                            cost + std + 2,
                            f"${cost:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                        )

        # --- Plot Multi-Zone Heuristics (Separate) ---
        if RUN_MULTI_ZONE_HEURISTICS and not multi_heuristic_data.empty:
            # Sort by strategy name to ensure consistent color and position assignment
            multi_heuristic_data = multi_heuristic_data.sort_values("strategy")
            present_multi_heuristics = multi_heuristic_data["strategy"].unique()
            multi_zone_colors = ["mediumseagreen", "darkviolet", "teal", "indigo"]

            base_pos = len(plot_data) + 0.5

            # Iterate through the DataFrame rows directly, as it's already aggregated
            # correctly for each strategy within this group.
            for i, row in enumerate(multi_heuristic_data.itertuples()):
                strategy_name = row.strategy
                cost = row.cost_mean
                std = row.cost_std

                # Find the index of the strategy in the unique sorted list for consistent coloring
                strategy_idx = np.where(present_multi_heuristics == strategy_name)[0][0]
                color = multi_zone_colors[strategy_idx % len(multi_zone_colors)]

                # The position is based on the iteration index 'i'
                pos = base_pos + i * (bar_width * 1.6)

                ax.bar(
                    pos,
                    cost,
                    bar_width * 1.5,
                    yerr=std,
                    capsize=3,
                    color=color,
                    alpha=0.8,
                    edgecolor="darkgreen",
                    linewidth=0.5,
                )
                ax.text(
                    pos,
                    cost + std + 2,
                    f"${cost:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            # Add separator line only if there are single-zone/union bars
            if not plot_data.empty:
                ax.axvline(
                    x=len(plot_data) - 0.25,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )

        # --- Customize Subplot ---
        ax.set_title(f"{group_name}", fontweight="bold", fontsize=14)
        ax.set_ylabel("Average Cost ($)", fontweight="bold", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        zone_labels = [
            z.split("_v100")[0] if "All Zones" not in z else "Union"
            for z in plot_data["zone_name"]
        ]
        x_tick_positions = list(x_positions)

        if RUN_MULTI_ZONE_HEURISTICS and not multi_heuristic_data.empty:
            # Sort by strategy name to ensure labels match bar positions and colors
            multi_heuristic_data = multi_heuristic_data.sort_values("strategy")
            present_multi_heuristics = multi_heuristic_data["strategy"].unique()
            multi_heur_labels = [
                STRATEGY_DISPLAY_NAMES.get(name, name.replace("_", " ").title())
                for name in present_multi_heuristics
            ]
            multi_heur_positions = [
                len(plot_data) + 0.5 + i * (bar_width * 1.6)
                for i in range(len(present_multi_heuristics))
            ]

            zone_labels.extend(multi_heur_labels)
            x_tick_positions.extend(multi_heur_positions)

        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(zone_labels, rotation=45, ha="right", fontsize=10)

        if len(individual_zones_names) > 0 and len(union_zones_names) > 0:
            ax.axvline(
                x=len(individual_zones_names) - 0.5,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    # Hide unused subplots and add legend
    for i in range(n_groups, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Cost Comparison: Optimal Strategy vs Heuristics",
        fontweight="bold",
        fontsize=16,
        y=0.98,
    )

    legend_elements = []
    if RUN_OPTIMAL_ANALYSIS:
        legend_elements.append(
            Patch(facecolor="skyblue", label="Individual Zones (Optimal)")
        )
        if RUN_UNION_ANALYSIS:
            legend_elements.append(
                Patch(facecolor="orange", label="Union Zone (Optimal)")
            )

    if RUN_SINGLE_ZONE_HEURISTICS:
        # Get unique heuristic strategies from the dataframe to ensure they were run
        present_heuristics = []
        if not df.empty:
            single_heur_df = df[
                df["analysis_type"].isin(["single_heuristic", "union_heuristic"])
            ]
            if not single_heur_df.empty:
                present_heuristics = sorted(single_heur_df["strategy"].unique())

        for i, name in enumerate(present_heuristics):
            display_name = STRATEGY_DISPLAY_NAMES.get(name, name)
            color_pair = heuristic_color_pairs[i % len(heuristic_color_pairs)]

            # Check if there is data for individual zones for this heuristic
            has_individual_data = not df[
                (df["strategy"] == name) & (df["analysis_type"] == "single_heuristic")
            ].empty
            if has_individual_data:
                legend_elements.append(
                    Patch(facecolor=color_pair[0], label=display_name)
                )

            # Check if there is data for union zones for this heuristic
            has_union_data = not df[
                (df["strategy"] == name) & (df["analysis_type"] == "union_heuristic")
            ].empty
            if RUN_UNION_ANALYSIS and has_union_data:
                legend_elements.append(
                    Patch(facecolor=color_pair[1], label=f"{display_name} (Union)")
                )

    if RUN_MULTI_ZONE_HEURISTICS:
        multi_heur_df = df[df["analysis_type"] == "multi_heuristic"]
        if not multi_heur_df.empty:
            # Dynamically create legend entries for multi-zone heuristics found in the data
            present_multi_heuristics = sorted(multi_heur_df["strategy"].unique())
            multi_zone_colors = ["mediumseagreen", "darkviolet", "teal", "indigo"]
            for i, name in enumerate(present_multi_heuristics):
                display_name = STRATEGY_DISPLAY_NAMES.get(
                    name, name.replace("_", " ").title()
                )
                legend_elements.append(
                    Patch(
                        facecolor=multi_zone_colors[i % len(multi_zone_colors)],
                        label=display_name,
                    )
                )

    fig.legend(
        handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save and close
    plot_path = output_dir / "all_region_pairs_cost_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"✅ All region pairs cost comparison chart saved to: {plot_path}")
    plt.close(fig)

    # Print comprehensive summary statistics with multi-strategy analysis
    logger.info(f"📊 COMPREHENSIVE COST SUMMARY:")
    for group_name in groups:
        group_data = stats_df[stats_df["group_name"] == group_name]

        logger.info(f"\n  === {group_name} ===")

        # Separate by analysis type
        optimal_data = group_data[group_data["analysis_type"] == "optimal"]
        all_heuristic_data = group_data[group_data["analysis_type"] != "optimal"]

        # Process optimal results
        if not optimal_data.empty:
            logger.info(f"    🎯 OPTIMAL STRATEGY RESULTS:")
            union_optimal = optimal_data[
                optimal_data["zone_name"].str.contains("All Zones Union")
            ]

            for _, row in optimal_data.iterrows():
                logger.info(
                    f"      {row['zone_name']}: ${row['cost_mean']:.2f} ± ${row['cost_std']:.2f}"
                )

        # Process heuristic results
        if RUN_SINGLE_ZONE_HEURISTICS and not all_heuristic_data.empty:
            logger.info(f"    🔍 HEURISTIC STRATEGY RESULTS:")
            # Sort by analysis type and then by zone name for consistent ordering
            sorted_heuristics = all_heuristic_data.sort_values(
                by=["analysis_type", "zone_name"]
            )
            for _, row in sorted_heuristics.iterrows():
                logger.info(
                    f"      {row['zone_name']}: ${row['cost_mean']:.2f} ± ${row['cost_std']:.2f}"
                )

        # Cost savings analysis
        if not optimal_data.empty and not all_heuristic_data.empty:
            logger.info(f"    💰 COST SAVINGS ANALYSIS:")
            union_opt_cost = optimal_data[
                optimal_data["zone_name"] == "All Zones Union"
            ]["cost_mean"].iloc[0]
            best_ind_opt_cost = optimal_data[
                ~optimal_data["zone_name"].str.contains("All Zones Union")
            ]["cost_mean"].min()

            logger.info(
                f"      Union (Optimal) vs. Best Single-Zone (Optimal): saves ${best_ind_opt_cost - union_opt_cost:.2f}"
            )

            multi_zone_data = all_heuristic_data[
                all_heuristic_data["analysis_type"] == "multi_heuristic"
            ]
            for _, row in multi_zone_data.iterrows():
                multi_zone_cost = row["cost_mean"]
                strategy_name = row["zone_name"]
                logger.info(
                    f"      Union (Optimal) vs. {strategy_name}: saves ${multi_zone_cost - union_opt_cost:.2f}"
                )


def plot_unified_chart(df: pd.DataFrame, group_name: str, output_dir: Path):
    """Generates a simple bar chart comparing individual zones vs union with error bars."""
    if df.empty:
        return

    # Remove trace index suffix to get clean zone names
    df_clean = df.copy()
    df_clean["zone_name_clean"] = df_clean["zone_name"].str.replace(
        r"_t\d+$", "", regex=True
    )

    # Calculate mean and std for each zone
    stats_df = (
        df_clean.groupby("zone_name_clean")["cost"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats_df.columns = ["zone_name", "cost_mean", "cost_std", "count"]

    # Fill NaN std with 0 (for cases with only one data point)
    stats_df["cost_std"] = stats_df["cost_std"].fillna(0)

    # Separate individual zones from union
    individual_zones = stats_df[
        ~stats_df["zone_name"].str.contains("All Zones Union")
    ].copy()
    union_zones = stats_df[stats_df["zone_name"].str.contains("All Zones Union")].copy()

    # Sort individual zones by name for consistent ordering
    individual_zones = individual_zones.sort_values("zone_name")

    # Combine for plotting, with individual zones first
    plot_data = pd.concat([individual_zones, union_zones], ignore_index=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors: different shades for individual zones, distinct color for union
    colors = ["skyblue"] * len(individual_zones) + ["orange"] * len(union_zones)

    # Create bar plot with error bars
    bars = ax.bar(
        range(len(plot_data)),
        plot_data["cost_mean"],
        yerr=plot_data["cost_std"],
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Customize the plot
    ax.set_title(
        f"Cost Comparison: Individual Zones vs Union - {group_name}",
        fontweight="bold",
        fontsize=16,
    )
    ax.set_ylabel("Average Cost ($)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Availability Zone / Union", fontweight="bold", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Set x-axis labels
    zone_labels = []
    for zone in plot_data["zone_name"]:
        if "All Zones Union" in zone:
            zone_labels.append("Union\n(All Zones)")
        else:
            # Clean up zone name for display
            clean_name = (
                zone.replace("us-west-2", "").replace("us-east-1", "").replace("_", " ")
            )
            zone_labels.append(clean_name)

    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(zone_labels, rotation=45, ha="right")

    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, plot_data.itertuples())):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + row.cost_std + 0.5,
            f"${height:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        # Add sample count below x-axis
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            -ax.get_ylim()[1] * 0.05,
            f"n={int(row.count)}",
            ha="center",
            va="top",
            fontsize=10,
            alpha=0.7,
        )

    # Add separator line between individual zones and union
    if len(individual_zones) > 0 and len(union_zones) > 0:
        separator_pos = len(individual_zones) - 0.5
        ax.axvline(x=separator_pos, color="red", linestyle="--", linewidth=2, alpha=0.7)

        # Add legend
        legend_elements = [
            Patch(facecolor="skyblue", alpha=0.8, label="Individual Zones"),
            Patch(facecolor="orange", alpha=0.8, label="Union (All Zones)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plot_path = output_dir / f"cost_comparison_{group_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"✅ Cost comparison chart saved to: {plot_path}")
    plt.show()

    # Print summary statistics
    logger.info(f"📊 COST SUMMARY for {group_name}:")
    for _, row in plot_data.iterrows():
        zone_type = "Union" if "All Zones Union" in row["zone_name"] else "Individual"
        logger.info(
            f"   {row['zone_name']}: ${row['cost_mean']:.2f} ± ${row['cost_std']:.2f} ({zone_type}, n={int(row['count'])})"
        )

    if len(individual_zones) > 0 and len(union_zones) > 0:
        best_individual = individual_zones.loc[individual_zones["cost_mean"].idxmin()]
        union_cost = union_zones.iloc[0]["cost_mean"]
        savings = best_individual["cost_mean"] - union_cost
        savings_pct = (savings / best_individual["cost_mean"]) * 100
        logger.info(
            f"   💰 Union saves ${savings:.2f} ({savings_pct:.1f}%) vs best individual zone"
        )


def create_simulation_task(
    task_type: str,
    group_name: str,
    zone_name: str,
    trace_index: int,
    trace_paths: List[str],
    strategy: str,
    analysis_type: str = "optimal",
):
    """Create a simulation task dictionary."""
    return {
        "task_type": task_type,  # 'zone', 'union', or 'multi_zone'
        "group_name": group_name,
        "zone_name": zone_name,
        "trace_index": trace_index,
        "trace_paths": trace_paths,
        "strategy": strategy,
        "analysis_type": analysis_type,  # 'optimal', 'single_heuristic', 'multi_heuristic'
    }


def execute_simulation_task(task):
    """Execute a single simulation task and return the result."""
    try:
        start_time = time.time()

        # Determine environment type based on task type and trace paths
        if task["task_type"] == "multi_zone" or len(task["trace_paths"]) > 1:
            env_type = "multi_trace"
        else:
            env_type = "trace"

        cost = run_simulation(task["strategy"], env_type, task["trace_paths"])
        elapsed = time.time() - start_time

        logger.info(
            f"✅ Completed {task['task_type']} {task['analysis_type']} simulation: {task['zone_name']} "
            f"(trace {task['trace_index']}) - Cost: ${cost:.2f} - Time: {elapsed:.1f}s"
        )

        result_name = f"{task['zone_name']}_t{task['trace_index']}"
        return {
            "group_name": task["group_name"],
            "zone_name": result_name,
            "strategy": task["strategy"],
            "analysis_type": task["analysis_type"],
            "cost": cost,
            "trace_index": task["trace_index"],
            "task_type": task["task_type"],
        }
    except Exception as e:
        logger.error(
            f"❌ Failed {task['analysis_type']} simulation: {task['zone_name']} (trace {task['trace_index']}): {e}"
        )
        return None


def run_unified_analysis(max_traces=None, max_workers=MAX_WORKERS):
    """Main evaluation loop that produces a single tidy DataFrame for plotting with parallel execution."""
    instance_groups = find_instance_groups(DATA_PATH)

    all_results_for_plotting = []
    all_tasks = []

    # Step 1: Collect all simulation tasks
    for group_name, zone_dirs in instance_groups.items():
        logger.info(
            f"\n--- Preparing tasks for Unified Group: {group_name} ({len(zone_dirs)} zones) ---"
        )

        # Discover all available trace files
        available_indices = discover_available_trace_files(zone_dirs)

        if not available_indices:
            logger.warning(
                f"No common trace files found for group {group_name}, skipping..."
            )
            continue

        # Limit the number of traces if max_traces is specified
        if max_traces is not None and len(available_indices) > max_traces:
            available_indices = available_indices[:max_traces]
            logger.info(f"Limiting to {max_traces} trace files for group {group_name}")

        # Create tasks for each trace file index
        for trace_index in available_indices:
            # Prepare the list of all individual zone trace paths for this index.
            # This is required for multi-zone heuristics and must be populated
            # regardless of whether optimal analysis is running.
            zone_trace_paths = []
            for zone_dir in zone_dirs:
                trace_file_path = Path(zone_dir) / f"{trace_index}.json"
                if trace_file_path.exists():
                    zone_trace_paths.append(str(trace_file_path))

            # 1. OPTIMAL ANALYSIS TASKS
            if RUN_OPTIMAL_ANALYSIS:
                # Tasks for individual zones - optimal strategy
                for zone_dir in zone_dirs:
                    zone_short_name = Path(zone_dir).name
                    trace_file_path = Path(zone_dir) / f"{trace_index}.json"

                    if not trace_file_path.exists():
                        logger.warning(
                            f"Trace file {trace_file_path} does not exist, skipping zone {zone_short_name}"
                        )
                        continue

                    task = create_simulation_task(
                        "zone",
                        group_name,
                        zone_short_name,
                        trace_index,
                        [str(trace_file_path)],
                        OPTIMAL_STRATEGY_NAME,
                        "optimal",
                    )
                    all_tasks.append(task)

                # Task for union trace (create union trace first) - optimal strategy
                if RUN_UNION_ANALYSIS:
                    try:
                        mega_union_path = get_or_create_mega_union_trace(
                            group_name, zone_dirs, trace_index
                        )
                        union_task = create_simulation_task(
                            "union",
                            group_name,
                            "All Zones Union",
                            trace_index,
                            [mega_union_path],
                            OPTIMAL_STRATEGY_NAME,
                            "optimal",
                        )
                        all_tasks.append(union_task)
                    except Exception as e:
                        logger.error(
                            f"Failed to create union trace for index {trace_index}: {e}"
                        )
                        continue

            # 2. SINGLE ZONE HEURISTIC TASKS
            if RUN_SINGLE_ZONE_HEURISTICS:
                # Tasks for individual zones for each heuristic
                for strategy_name in SINGLE_ZONE_HEURISTICS:
                    for zone_dir in zone_dirs:
                        zone_short_name = Path(zone_dir).name
                        trace_file_path = Path(zone_dir) / f"{trace_index}.json"

                        if not trace_file_path.exists():
                            continue

                        task = create_simulation_task(
                            "zone",
                            group_name,
                            f"{zone_short_name}_{strategy_name}",
                            trace_index,
                            [str(trace_file_path)],
                            strategy_name,
                            "single_heuristic",
                        )
                        all_tasks.append(task)

                    # Task for union trace using the same heuristic strategy
                    if RUN_UNION_ANALYSIS:
                        try:
                            mega_union_path = get_or_create_mega_union_trace(
                                group_name, zone_dirs, trace_index
                            )
                            union_heuristic_task = create_simulation_task(
                                "union",
                                group_name,
                                f"All Zones Union_{strategy_name}",
                                trace_index,
                                [mega_union_path],
                                strategy_name,
                                "union_heuristic",
                            )
                            all_tasks.append(union_heuristic_task)
                        except Exception as e:
                            logger.error(
                                f"Failed to create union heuristic trace for index {trace_index} with strategy {strategy_name}: {e}"
                            )
                            continue

            # 3. MULTI ZONE HEURISTIC TASKS
            if RUN_MULTI_ZONE_HEURISTICS and zone_trace_paths:
                # Create a task for each multi-zone heuristic strategy
                for strategy_name in MULTI_ZONE_HEURISTICS:
                    task = create_simulation_task(
                        "multi_zone",
                        group_name,
                        strategy_name,  # Use the strategy name as the unique zone_name
                        trace_index,
                        zone_trace_paths,
                        strategy_name,
                        "multi_heuristic",
                    )
                    all_tasks.append(task)

    # Step 2: Execute all tasks in parallel
    logger.info(
        f"\n🚀 Starting parallel execution of {len(all_tasks)} simulation tasks with {max_workers} workers..."
    )

    start_time = time.time()
    completed_tasks = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(execute_simulation_task, task): task for task in all_tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            completed_tasks += 1
            result = future.result()
            if result is not None:
                all_results_for_plotting.append(result)

            # Progress logging
            if completed_tasks % 5 == 0 or completed_tasks == len(all_tasks):
                elapsed = time.time() - start_time
                logger.info(
                    f"📊 Progress: {completed_tasks}/{len(all_tasks)} tasks completed "
                    f"({completed_tasks / len(all_tasks) * 100:.1f}%) - Elapsed: {elapsed:.1f}s"
                )

    total_time = time.time() - start_time
    logger.info(
        f"🎯 All parallel simulations completed in {total_time:.1f}s "
        f"(avg {total_time / len(all_tasks):.1f}s per task)"
    )

    # Create one big DataFrame with all results
    df = pd.DataFrame(all_results_for_plotting)
    csv_path = OUTPUT_DIR / "unified_analysis_all_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ All detailed results saved to: {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze unified instance type performance boundaries."
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of trace files to process (for testing)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Maximum number of parallel workers (default: {MAX_WORKERS})",
    )
    args = parser.parse_args()

    logger.info("🔬 Starting Unified Instance Type Boundary Analysis...")
    logger.info(
        f"🔧 Configuration: max_traces={args.max_traces}, max_workers={args.max_workers}"
    )

    # Log enabled analysis types
    enabled_analyses = []
    if RUN_OPTIMAL_ANALYSIS:
        enabled_analyses.append("Optimal Strategy")
    if RUN_SINGLE_ZONE_HEURISTICS:
        enabled_analyses.append("Single-Zone Heuristics")
    if RUN_UNION_ANALYSIS:
        enabled_analyses.append("Union Analysis")
    if RUN_MULTI_ZONE_HEURISTICS:
        enabled_analyses.append("Multi-Zone Heuristics")

    logger.info(f"📋 Enabled Analyses: {', '.join(enabled_analyses)}")
    if not enabled_analyses:
        logger.warning(
            "⚠️ No analyses enabled! Check the control switches at the top of the script."
        )
        exit(1)

    results_df = run_unified_analysis(
        max_traces=args.max_traces, max_workers=args.max_workers
    )

    # dump the df to output directory
    results_df.to_csv(OUTPUT_DIR / "unified_analysis_all_results.csv", index=False)

    # Generate plots for all instance groups
    if not results_df.empty:
        # Generate unified cost comparison chart for all region pairs
        plot_all_region_pairs_chart(results_df, OUTPUT_DIR)
        plot_unified_chart(results_df, "All Zones Union", OUTPUT_DIR)

    logger.info("🎉 Analysis completed!")
