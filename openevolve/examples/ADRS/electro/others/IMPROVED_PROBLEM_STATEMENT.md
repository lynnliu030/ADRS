# Improved Problem Statement: Constraint-Based Cost Optimization

## Executive Summary

We have transformed the RAG configuration optimization problem from a generic multi-objective optimization to a more practical **constraint satisfaction + cost minimization** problem:

**Given a target accuracy threshold, find the most cost-effective RAG configuration that meets or exceeds that accuracy level.**

## Problem Transformation

### Before: Generic Multi-Objective Optimization
- **Objective**: Maximize `(accuracy * success_rate) / cost + bonuses`
- **Issue**: Unclear tradeoffs between accuracy and cost
- **Result**: Solutions might sacrifice accuracy for cost savings

### After: Constraint-Based Cost Optimization
- **Hard Constraint**: Configuration must meet minimum accuracy threshold
- **Primary Objective**: Minimize cost per query among valid configurations
- **Secondary Objectives**: Maximize retrieval quality, exceed accuracy target

## Key Improvements

### 1. Clear Business Objective
The new formulation directly addresses a common business question:
> "We need at least 15% accuracy for our RAG system. What's the cheapest way to achieve this?"

### 2. Constraint Satisfaction Framework
- **Accuracy Constraint**: `effective_accuracy >= min_accuracy_threshold`
- **Where**: `effective_accuracy = accuracy * success_rate`
- **Penalty**: Configurations violating constraint get heavily penalized scores

### 3. Cost-First Optimization
Among configurations meeting the accuracy constraint:
- **Primary metric**: Cost per query (lower is better)
- **Secondary metrics**: Retrieval quality, accuracy exceeding target

### 4. Configurable Parameters
```python
TARGET_ACCURACY_CONSTRAINT = {
    'min_accuracy': 0.15,          # Minimum acceptable accuracy
    'accuracy_tolerance': 0.02,    # Small tolerance for constraint satisfaction  
    'penalty_weight': 10.0,        # Heavy penalty for violating constraint
}
```

## Implementation Changes

### 1. Evaluator Scoring (evaluator.py)
**New Scoring Components:**
- `constraint_satisfaction_score` (30%): How well accuracy constraint is met
- `cost_efficiency_score` (50%): Cost optimization for valid configurations
- `search_efficiency_score` (20%): Algorithm efficiency

**Key Function:**
```python
def calculate_constraint_satisfaction_score(metrics: Dict) -> float:
    effective_accuracy = accuracy * success_rate
    
    if effective_accuracy >= min_accuracy - tolerance:
        return 1.0 + bonus_for_exceeding_target
    else:
        violation = (min_accuracy - effective_accuracy) / min_accuracy
        penalty = violation * penalty_weight
        return max(0.0, 1.0 - penalty)
```

### 2. Initial Program Scoring (initial_program.py)
**New Baseline Algorithm:**
```python
def calculate_score(self, metrics: Dict) -> float:
    effective_accuracy = accuracy * success_rate
    
    # Hard constraint check
    if effective_accuracy < MIN_ACCURACY_THRESHOLD - TOLERANCE:
        violation_penalty = (MIN_ACCURACY_THRESHOLD - effective_accuracy) * 1000
        return -violation_penalty
    
    # Cost optimization for valid configurations
    cost_score = (max_reasonable_cost - cost) / max_reasonable_cost * 100
    retrieval_bonus = (precision + recall) * 5
    accuracy_bonus = (effective_accuracy - MIN_ACCURACY_THRESHOLD) * 20
    
    return cost_score + retrieval_bonus + accuracy_bonus
```

## Expected Benefits

### 1. Clearer Optimization Direction
- Algorithms will focus on finding configurations that meet accuracy requirements
- Cost optimization becomes the primary concern for valid configurations
- No more unclear tradeoffs between accuracy and cost

### 2. Business-Relevant Solutions
- Solutions directly answer: "What's the cheapest way to achieve X% accuracy?"
- Guaranteed minimum quality level
- Cost-efficient operations

### 3. Better Algorithm Evolution
OpenEvolve will likely evolve toward:
- **Constraint satisfaction techniques**: Methods to efficiently find valid configurations
- **Cost-aware search**: Sophisticated cost minimization strategies  
- **Multi-stage optimization**: First find valid configs, then optimize cost
- **Adaptive thresholds**: Learning optimal accuracy targets

### 4. Measurable Success Criteria
- **Constraint Satisfaction Rate**: % of solutions meeting accuracy threshold
- **Cost Reduction**: Average cost savings vs naive approaches
- **Search Efficiency**: Time to find cost-optimal valid configurations

## Usage Examples

### Scenario 1: Financial Service RAG
```python
TARGET_ACCURACY_CONSTRAINT = {
    'min_accuracy': 0.20,      # Need 20% accuracy for financial Q&A
    'accuracy_tolerance': 0.02,
    'penalty_weight': 10.0,
}
```

### Scenario 2: Customer Support RAG  
```python
TARGET_ACCURACY_CONSTRAINT = {
    'min_accuracy': 0.12,      # Lower accuracy acceptable for support
    'accuracy_tolerance': 0.01,
    'penalty_weight': 5.0,
}
```

### Scenario 3: High-Stakes RAG
```python
TARGET_ACCURACY_CONSTRAINT = {
    'min_accuracy': 0.30,      # High accuracy required
    'accuracy_tolerance': 0.00, # No tolerance for violations
    'penalty_weight': 20.0,    # Severe penalties
}
```

## Testing the Improved System

### 1. Verify Constraint Enforcement
- Ensure algorithms heavily penalize configurations below accuracy threshold
- Check that valid configurations are ranked by cost efficiency

### 2. Cost Optimization Behavior
- Validate that among valid configurations, cheaper ones score higher
- Confirm retrieval quality bonuses are appropriately weighted

### 3. Algorithm Evolution Direction
- Monitor if evolved algorithms develop constraint satisfaction strategies
- Track improvements in cost-efficiency for valid configurations

## Migration from Current System

### Configuration Updates
1. Set `TARGET_ACCURACY_CONSTRAINT` based on your requirements
2. Adjust evaluation weights to emphasize constraint satisfaction and cost
3. Update any existing algorithms to use constraint-aware scoring

### Testing
1. Run baseline with new scoring system
2. Compare results: old vs new problem formulation
3. Validate business relevance of solutions

The improved system transforms RAG optimization from an academic multi-objective problem into a practical business constraint satisfaction + cost optimization challenge.
