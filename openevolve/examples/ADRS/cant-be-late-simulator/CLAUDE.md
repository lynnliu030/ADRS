# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **SkyPilot Spot Simulator** - a cloud computing simulation framework that models spot instance execution strategies for optimizing cost and reliability when running tasks on preemptible instances across multiple cloud regions. The project is designed for research into multi-region scheduling strategies for long-running tasks with deadlines.

### Historical Context
- **Single-region strategies** (e.g., `rc_cr_threshold`, `rc_cr`, etc.) are the original strategies from the "Can't Be Late" paper, designed for optimizing within a single cloud region.
- **Multi-region strategies** are the newer extension that enables scheduling across multiple regions for better cost optimization.

## Quick Start Guide

### Running Simulations

**Single simulation:**
```bash
# Single-region
python main.py --strategy=rc_cr_threshold --env=trace --trace-file data/real/ping_based/random_start_time/us-west-2a_k80_1/0.json --task-duration-hours=48 --deadline-hours=52 --restart-overhead-hours=0.2

# Multi-region
python main.py --strategy=multi_region_rc_cr_threshold --env=multi_trace --trace-files data/real/ping_based/random_start_time/us-west-2a_k80_1/0.json data/real/ping_based/random_start_time/us-west-2b_k80_1/0.json --task-duration-hours=48 --deadline-hours=52 --restart-overhead-hours=0.2
```

**Strategy development and testing:**
```bash
# Quick test new strategy
python scripts_multi/quick_test_strategy.py my_strategy

# Compare strategies across multiple traces  
python scripts_multi/batch_strategy_comparison.py 10

# Research-grade systematic evaluation
python scripts_multi/benchmark_multi_region_modular.py
```

### Environment Setup

**Note**: The exact environment setup is uncertain and may vary between systems. Some dependencies might have been installed using `uv` or other package managers. When encountering environment issues, diagnose the specific problem and use the appropriate installation method.

Basic requirements typically include:
- Python 3.8+
- NumPy, Pandas, Matplotlib
- Other dependencies as needed (check imports for specifics)

### Don't forget to activate the virtual environment
If you encounter a `ModuleNotFoundError` or errors indicating a missing package, or a SkyPilot error `sky.exceptions.APIVersionMismatchError: Client and local API server version mismatch`, your first step should be to verify that the project's virtual environment is activated. This ensures you are using the project-specific dependencies.

At project root, run:

```fish
source .venv/bin/activate.fish
```

### Logging Configuration

Logging is controlled by the `LOG_LEVEL` environment variable:
```bash
# Set logging level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL=INFO
python main.py ...

# Default is DEBUG if not set
python main.py ...  # Uses DEBUG level
```

## Data Organization

### Primary Data Sources

1. **Original single/two-region data**: `data/real/ping_based/random_start_time/`
   - Contains real trace data organized by region and instance type
   - Format: `{region}_{instance_type}_{count}/0.json`
   - Example: `us-west-2a_k80_1/0.json`

2. **Extended multi-region data (newer)**: `data/converted_multi_region/`
   - Contains traces for more regions
   - Used for comprehensive multi-region analysis

### Trace File Format
```json
{
    "metadata": {
        "gap_seconds": 600,  // Time step between data points
        "start_time": "2022-09-08 07:41:46.331298"  // Random start time
    },
    "data": [
        0, 0, 1, 1, 0, 0, 0, ...  // 1 = preempted, 0 = available
    ]
}
```

### Creating Traces

See `data/README.md` for comprehensive data generation guide. Key points:

1. **Raw data collection**: `scripts/availability/availability_trace.py` (runs on remote servers)
2. **Parsing**: `sky_spot/parsers/parse_availability.py`  
3. **Random start generation**: `data/real/ping_based/random_start_time/parse.py`
4. **Synthetic traces**: Use generators like `two_exp`, `two_gamma`, `poisson`

Example for synthetic traces:
```bash
python ./sky_spot/traces/generate.py --trace-folder data/two_exp \
    --generator two_exp --gap-seconds 600 --length 4320 \
    --num-traces 1 --alive-scale 24.4920 --wait-scale 6.4126984
```

## Architecture: Multi-Region Strategy Design

### Core Design Principle

The multi-region system uses a **yield-based generator pattern** that models real cloud constraints:

```python
# Strategy yields actions to environment
result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)

# Environment returns results
# TryLaunch → LaunchResult(success=bool, region=int, cluster_type=ClusterType)
# Terminate → None
```

### Why This Design?

1. **Information Isolation**: Strategies should only receive information through the proper interface, not by accessing internal environment state. This ensures:
   - Strategies can't bypass billing rules
   - The environment maintains control over resource availability checks
   - Strategies remain portable across different environment implementations

2. **Real-World Constraint**: In actual cloud environments, you can't check spot availability without attempting to launch (which incurs cost if successful).

### Critical Implementation Rules

1. **Multi-Instance Design Space**: 
   - **Current Implementation**: Each region can have at most one instance, but multiple regions can run simultaneously
   - **Progress Tracking**: When multiple instances run, they all contribute the same progress (parallel execution assumption)
   - **Strategic Considerations**:
     - Multiple ON_DEMAND instances never make sense (they don't get preempted, so redundancy is wasteful)
     - Multiple SPOT instances across regions can provide mutual backup, trading higher cost for stability
     - Current strategies choose to run only one instance at a time, but the framework supports multi-instance strategies
2. **No Internal Access**: Never call methods like `_spot_available_in_region()` - this would violate the information isolation principle
3. **Type Assertions**: Always `assert result is not None` after `yield TryLaunch` (Python type system limitation)
4. **Guaranteed Success**: ON_DEMAND launches always succeed

### Core Multi-Region Decision Points

Multi-region strategies face several fundamental decision challenges:

#### 1. Recovery Region Selection - "After preemption, which region to try next?" (Core)
**When current spot is terminated, where should we launch the replacement?**

Decision factors:
- **Duration prediction**: Which region's spot is expected to run longer?
- **Migration costs**: Checkpoint transfer + restart overhead vs potential benefits
- **Launch risk**: `TryLaunch` may fail, but success incurs immediate billing
- **Information gathering vs exploitation trade-off**:
  - Choose known good regions (exploit existing knowledge)
  - Try unexplored regions to gather more information, for availability and duration if launch is successful (explore).
  - "Launch briefly then terminate" strategy to probe region availability could be an option.
  - But if the newer availability information is more effective remains unknown.

#### 2. Redundant Instance Strategy - "Should we run backup instances?"
**Launch multiple spot instances across regions as mutual backup**

Strategic considerations:
- **Cost vs reliability**: Multiple spots cheaper than ON_DEMAND, more reliable than single spot
- **Redundant Execution**: All instances run the identical task in parallel, acting as hot spares for one another.
- **Billing risk**: Pay for all successful launches, not just the one that survives longest
- **Termination timing**: When to terminate redundant instances

#### 3. Active Migration Decision - "When to proactively switch regions?"
**Even with a running instance, should we `TryLaunch` in a potentially better region?**

Decision factors:
- **Cost efficiency**: New region's spot price vs current region
- **Availability prediction**: Expected spot duration in new region vs remaining time in current
- **Migration costs**: Checkpoint transfer + restart overhead vs potential benefits
- **Time constraints**: How much time remains until deadline
- **Strategy philosophy**: Some strategies are conservative (stick with current), others are opportunistic

These decision points are directly informed by our spot duration prediction analysis, which quantifies the value of different prediction approaches across regions.

### Billing Model

- **Immediate Billing**: Instances are billed upon successful launch
- **No Refunds**: Early termination doesn't refund the current tick
- **Same-Tick Restriction**: Cannot terminate in the same tick as launch
- **Minimum Unit**: One tick (gap_seconds)

## Strategy Development Workflow

**For comprehensive strategy development guidance, see `STRATEGY_DEVELOPMENT.md`.**

### Quick Development Tools

**Phase 1 - Rapid Prototyping (30 seconds):**
```bash
python scripts_multi/quick_test_strategy.py my_new_strategy [trace_id]
```

**Phase 2 - Small-scale Comparison (5 minutes):**
```bash
python scripts_multi/batch_strategy_comparison.py 3
```

**Phase 3 - Large-scale Evaluation (30+ minutes):**
```bash
python scripts_multi/batch_strategy_comparison.py 100 --no-viz
```

**Phase 4 - Research-grade Analysis:**
```bash
python scripts_multi/benchmark_multi_region_modular.py --num-traces 200
```

### Tool Comparison

| Tool | Purpose | Speed | Scope | Output |
|------|---------|-------|-------|--------|
| `quick_test_strategy.py` | Single strategy validation | 30s | 1 trace | Timeline viz |
| `batch_strategy_comparison.py` | Strategy comparison | 5min-2h | 1-1000 traces | Cost comparison + viz |
| `benchmark_multi_region_modular.py` | Research analysis | Hours | Multi-parameter | Publication plots |

## Creating New Strategies

### Single-Region Strategy Template
```python
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class MyStrategy(Strategy):
    NAME = 'my_strategy'  # Used in --strategy CLI argument
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Your decision logic here
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        # Add strategy-specific arguments if needed
        return cls(parser.parse_args())
```

### Multi-Region Strategy Template
```python
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, Action, LaunchResult
import typing

class MyMultiRegionStrategy(MultiRegionStrategy):
    NAME = 'my_multi_region_strategy'
    
    def _step_multi(self) -> typing.Generator[Action, typing.Optional[LaunchResult], None]:
        # Try launching in region 0
        result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
        assert result is not None  # Required due to type system
        
        if not result.success:
            # Try another region
            result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
            assert result is not None
    
    @classmethod
    def _from_args(cls, parser):
        return cls(parser.parse_args())
```

### Registration
Add import in `sky_spot/strategies/__init__.py`:
```python
from sky_spot.strategies import my_strategy  # Auto-registers via __init_subclass__
```

## Key Implementation Details

### Restart Overhead Mechanism

1. **Single-Region**: Applied directly in `Strategy.step()` method
   - Tracks `remaining_restart_overhead` 
   - Deducts from available time before calculating progress

2. **Multi-Region**: Uses flag-based tracking
   - `_new_launch_this_tick`: Set when launching new instance
   - `_had_new_launch_last_tick`: Checked next tick to apply overhead
   - Applied in `update_strategy_progress()` method

3. **Migration Overhead**: When switching regions, full restart overhead is reapplied

### Timing Semantics

**Critical Understanding**: There's a one-tick "lag" between decision and progress calculation:

1. **Decision Point**: Made at the **beginning** of each tick
   - `tick=0, observed_tick=-1` initially
   - `observe()` returns the **previous** time period's state

2. **Progress Calculation**: Computes work done in the **previous** tick
   - `progress[0]` = work done during tick -1→0 (always 0)
   - `progress[1]` = work done during tick 0→1 (may be 0 due to restart overhead)

3. **Example Timeline**:
   ```
   Tick 0: Decide to launch → Instance starts running
   Tick 1: Calculate tick 0 progress (0 if overhead > gap_seconds)
   Tick 2: Calculate tick 1 progress (partial if overhead consumed)
   ```

## Current Multi-Region Strategies

1. **`multi_region_rc_cr_threshold`** (Base)
   - Actively seeks available spot instances across all regions
   - Switches to cheapest available option

2. **`multi_region_rc_cr_no_cond2`** (Less Sticky)
   - Removes the sticky condition (_condition2)
   - More willing to switch regions

3. **`multi_region_rc_cr_randomized`** (Load Balanced)
   - Randomly selects among available regions
   - Reduces competition/herding effects

4. **`multi_region_rc_cr_reactive`** (Conservative)
   - Only considers other regions when preempted
   - Prefers to stay in current region

## Common Pitfalls

1. **Wrong Environment Type**: Multi-region strategies require `--env=multi_trace`, not `--env=trace`
2. **File vs Files Parameter**: Single-region uses `--trace-file`, multi-region uses `--trace-files`
3. **Performance**: Full benchmarks are computationally intensive - always limit scope during development
4. **Test Progress Arrays**: Progress arrays in tests show what's calculated at each tick, not what happens during each tick
   - `progress[i]` = progress calculated at tick `i` for work done in tick `i-1` to `i`
   - This is why `progress[0]` is always 0 (no previous tick to calculate)
5. **Long-Running Scripts**: Use background execution (`&`) and poll results to avoid timeouts
6. **Truncated Logs**: Redirect output to files (`> output.log 2>&1`) and check with `head`/`tail`

## Project Configuration

### Cache System
- **Location**: `{OUTPUT_DIR}/cache/`
- **Format**: `{strategy}_{env}_{trace_descriptor}.json`
- **Example**: `quick_optimal_trace_us-west-2a_k80_1_0.json`
- **Refresh**: Delete specific cache files to force recalculation

### Key Settings
Configure in scripts as needed:
- `DATA_PATH`: Input trace location
- `OUTPUT_DIR`: Results and cache location

### Display Name Conventions
For user-facing outputs (charts, reports):
- `rc_cr_threshold` → "Single-Region Uniform Progress" 
- `multi_region_rc_cr_threshold` → "Multi-Region Uniform Progress"
- Add suffixes: "(w/o Cond. 2)", "(Random)", "(Reactive)"

## Project Rules and Development Guidelines

### Core Philosophy for All Repository Work
These general principles apply to all work in this repository to ensure quality and maintainability. Only to break these rules when you have a good reason.

1. **Start Simple, Move Fast**

- **Principle**: Begin with the simplest implementation to validate an idea.

- **Example**: Test a new strategy with `quick_test_strategy.py` on one trace before running a full benchmark.

2. **Fail Quickly and Explicitly**

- **Principle**: Code must surface errors as soon as they occur.

- **Example**: Use an `assert` at the start of an analysis script to validate data rather than letting it crash hours later.

- **Example**: At most cases, you should not use `try-except` blocks in core functions. Any signals of errors should be reported explicitly.

3. **No Complex Fallbacks in Core Functions**

- **Principle**: A function should have a single purpose and report success or failure, not contain internal fallbacks.

- **Example**: A plotting function should error on bad data, not try to generate a different plot as a fallback.

- **Example**: Do not try to be compatible with old logic or the old data format. 

4. **Panic on Unrecoverable State**

- **Principle**: If a state could invalidate the final output, terminate immediately.

- **Example**: An analysis script should panic if it finds inconsistent input data, as a misleading result is worse than no result.

### Performance Warning
**CRITICAL**: Simulations can be very slow. Best practices:
- Always limit test scope during development
- Use caching extensively
- To force refresh: delete specific cache files

### Shell Environment
All commands use **fish** shell syntax. Adjust for your shell:
```bash
# Fish shell
set -x LOG_LEVEL INFO

# Bash/Zsh
export LOG_LEVEL=INFO
```

### In-Place File Modification

**CRITICAL**: When asked to modify or enhance a feature, always edit the existing file directly. This principle is absolute and defines what "modify" means.

The core idea is to **reuse the existing file path as the single source of truth** for a given task, rather than creating a trail of new files.

- **CORRECT**: Read the existing file, understand its contents, and apply the requested changes by overwriting that same file.

- **INCORRECT**: Creating a new file with suffixes like `_enhanced.py`, `_v2.py`, or `_new.py`.

**This rule is especially important in the following scenarios**:

- **When a file has bugs or is rejected**: If you generate a file (e.g., `summary.md`) and I reject it as incorrect or useless, your next attempt to create a correct version must overwrite the original `summary.md`. Do not create `summary_final.md` or `technical_summary.md`. The rejected file should not be left in the codebase.

- **When the purpose evolves**: Even if my feedback changes the purpose or scope of the file (e.g., from a "high-level summary" to a "detailed technical breakdown"), you must still **modify the original file**. The new content should replace the old content at the same file path, because it represents the next iteration of the same task.

In short: **If a file is the successor to a previous attempt, it must replace it by overwriting it.** New files should only be created for genuinely new, distinct modules or components.

## Language and Naming Conventions

**CRITICAL**: All content within this repository must be in English. This is a strict requirement to ensure consistency, maintainability, and accessibility for a global audience.

This rule applies to all forms of text, including but not limited to:
- Code: Variable names, function names, class names, etc.
- Comments: Inline comments and docstrings.
- Documentation: All content in .md files or other documentation formats.
- Commit Messages: Git commit messages must be clear and written in English.
