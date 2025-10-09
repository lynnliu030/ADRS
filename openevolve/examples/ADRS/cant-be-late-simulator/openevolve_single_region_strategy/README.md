# Single-Region Strategy Evolution with OpenEvolve

This directory contains the OpenEvolve setup for evolving single-region cloud scheduling strategies that can outperform the baseline `rc_cr_threshold` strategy.

## Overview

The goal is to evolve strategies that achieve >55% cost savings compared to always using ON_DEMAND instances, beating the current best single-region strategy (`rc_cr_threshold`).

## Key Improvements Made

### 1. Evaluation Methodology (Based on Paper)
- **Comprehensive Testing**: Uses all available traces across multiple AWS zones (up to 9 zones)
- **No Random Sampling**: Evaluates on all traces rather than randomly selecting 2
- **Paper-Aligned Metrics**: Focuses on cost savings vs ON_DEMAND (target: >55%)
- **Proper Scoring**: Score based on beating the baseline's ~55% savings

### 2. Simplified Strategy Registration
- **Direct File Writing**: Strategies are written directly to the strategies directory
- **Dynamic Import**: Uses `importlib` for cleaner strategy registration
- **No Temporary Directories**: Removed unnecessary temp directory creation

### 3. Initial Strategy
- **Simple Greedy Baseline**: Uses a simple greedy approach as starting point
- **Room for Improvement**: Much simpler than `rc_cr_threshold`, giving evolution space to improve

## Running the Evolution

### Quick Test
```bash
# Test the evaluator with initial program
cd openevolve_single_region_strategy
python evaluator.py
```

### Full Evolution
```bash
# Run OpenEvolve (reduced iterations for testing)
./run_evolution.sh
```

### Production Settings
For production runs, edit `config.yaml`:
- Set `max_iterations: 100` (currently 20 for testing)
- Set `population_size: 50` (currently 20 for testing)
- Set `parallel_evaluations: 4` (currently 2 for testing)

## Evaluation Process

### Stage 1: Syntax Check
- Validates Python syntax
- Checks for required Strategy class structure
- Verifies `_step` method exists

### Stage 2: Performance Evaluation
- Runs strategy on all available trace files
- Compares against `rc_cr_threshold` baseline
- Calculates cost savings vs ON_DEMAND
- Scores based on:
  - Success rate (50 points max)
  - Cost savings performance (100+ points for beating baseline)

## Expected Outcomes

Successful strategies should achieve:
- **>55% cost savings** (baseline level)
- **Target: 60-65% savings** (significant improvement)
- **100% task completion rate** (never miss deadlines)

## Files

- `config.yaml`: OpenEvolve configuration
- `evaluator.py`: Two-stage evaluation logic
- `initial_program.py`: Simple greedy starting strategy
- `run_evolution.sh`: Script to run evolution
- `test_initial.sh`: Test baseline performance

## Notes

The evaluator follows the experimental methodology from the "Can't Be Late" paper:
- Uses real traces from multiple AWS zones
- Tests on 48-hour jobs with 52-hour deadlines
- Considers both K80 and V100 instance types where available