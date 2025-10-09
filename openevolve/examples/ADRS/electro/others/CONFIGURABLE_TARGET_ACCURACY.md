# Configurable Target Accuracy System

## Overview

The system has been redesigned to be **generic** and work with **any** target accuracy threshold, rather than being limited to a hardcoded minimum accuracy value.

## The Problem (Fixed)

**Before:** The `min_accuracy` was hardcoded in `evaluator.py` at 0.15 (15%), creating a contradiction:
- ❌ Goal: "Find most cost-effective config for ANY target accuracy"
- ❌ Reality: Hardcoded to only work with 15% accuracy

**After:** The target accuracy is now **configurable** and can be **any value**:
- ✅ Goal: "Find most cost-effective config for ANY target accuracy" 
- ✅ Reality: Works with any accuracy threshold (0.05, 0.15, 0.30, etc.)

## How It Works

### 1. **Algorithm-Level Configuration**
The target accuracy is set in the optimization algorithm:

```python
def find_optimal_rag_config() -> Dict:
    # Example: Set target accuracy to 0.20 (20%) for this run
    # This can be ANY value the user wants!
    target_accuracy = 0.20
    
    optimizer = RAGConfigurationOptimizer(target_accuracy=target_accuracy)
    # ... optimization logic ...
    
    # Make target accuracy available to evaluator
    global TARGET_ACCURACY
    TARGET_ACCURACY = target_accuracy
    
    return best_config
```

### 2. **Dynamic Constraint Evaluation**
The evaluator reads the target accuracy from the algorithm:

```python
# Get target accuracy from the algorithm if available, otherwise use default
target_accuracy = getattr(program, 'TARGET_ACCURACY', DEFAULT_ACCURACY_CONSTRAINT['min_accuracy'])

constraint_satisfaction_score = calculate_constraint_satisfaction_score(final_metrics, target_accuracy)
cost_efficiency_score = calculate_cost_efficiency_score(final_metrics, target_accuracy)
```

### 3. **Flexible Constraint Functions**
Constraint functions now accept target accuracy as a parameter:

```python
def calculate_constraint_satisfaction_score(metrics: Dict, target_accuracy: float) -> float:
    """
    Calculate how well the configuration satisfies the accuracy constraint.
    
    Args:
        metrics: Configuration metrics
        target_accuracy: The target accuracy threshold (can be any value)
    """
    # ... constraint logic using target_accuracy parameter ...
```

## Usage Examples

### **Low Accuracy Target (5%)**
```python
target_accuracy = 0.05  # Find cheapest config for basic functionality
```
- Use case: Development/testing, basic Q&A
- Result: Very cheap configurations, minimal quality requirements

### **Medium Accuracy Target (15%)**
```python
target_accuracy = 0.15  # Balance cost and quality
```
- Use case: Production systems, general purpose
- Result: Balanced cost-quality tradeoffs

### **High Accuracy Target (30%)**
```python
target_accuracy = 0.30  # Premium quality regardless of cost
```
- Use case: High-stakes applications, premium services
- Result: High-quality configurations, cost secondary

### **Custom Accuracy Target**
```python
target_accuracy = 0.22  # Any specific value needed
```
- Use case: Business-specific requirements
- Result: Tailored optimization for exact needs

## Benefits

### 1. **True Generality**
- Algorithm works with **any** accuracy threshold
- No more hardcoded limitations
- Flexible for different business requirements

### 2. **Business Flexibility**
- Low accuracy: Cost optimization for basic needs
- High accuracy: Quality optimization for premium services
- Custom accuracy: Tailored to specific requirements

### 3. **Better Algorithm Evolution**
- OpenEvolve can explore different accuracy targets
- More diverse optimization strategies
- Better constraint satisfaction techniques

### 4. **Real-World Applicability**
- Different use cases need different accuracy levels
- Cost constraints vary by application
- Quality requirements differ by business domain

## Implementation Details

### **Default Values**
```python
DEFAULT_ACCURACY_CONSTRAINT = {
    'min_accuracy': 0.15,          # Default (can be overridden)
    'accuracy_tolerance': 0.02,    # Small tolerance for constraint satisfaction
    'penalty_weight': 10.0,        # Heavy penalty for violations
}
```

### **Algorithm Interface**
```python
class RAGConfigurationOptimizer:
    def __init__(self, target_accuracy: float = 0.15):
        """
        Initialize optimizer with configurable target accuracy.
        
        Args:
            target_accuracy: The accuracy threshold to meet (can be any value)
        """
        self.target_accuracy = target_accuracy
```

### **Evaluator Integration**
```python
# The evaluator automatically detects and uses the algorithm's target accuracy
target_accuracy = getattr(program, 'TARGET_ACCURACY', DEFAULT_ACCURACY_CONSTRAINT['min_accuracy'])
```

## Testing Results

✅ **Successfully tested with target accuracy = 0.20 (20%)**
- Target accuracy used: 0.2
- Combined score: 1.0000
- Available configs: 216
- Final accuracy: 0.360 (exceeds target)
- Final cost: $0.002963

## Future Enhancements

### 1. **Multi-Target Optimization**
```python
# Optimize for multiple accuracy targets simultaneously
targets = [0.10, 0.20, 0.30]
results = [optimize_for_accuracy(t) for t in targets]
```

### 2. **Dynamic Target Adjustment**
```python
# Algorithm learns optimal accuracy targets
if cost_too_high:
    target_accuracy *= 0.9  # Reduce target to find cheaper configs
```

### 3. **Target Accuracy Discovery**
```python
# Find the "sweet spot" accuracy target
sweet_spot = discover_optimal_accuracy_target(cost_budget, quality_requirements)
```

## Conclusion

The system is now truly **generic** and can handle **any** target accuracy threshold. This makes it much more practical for real-world applications where different use cases require different quality levels and cost constraints.

The algorithm can now answer the question: **"What's the most cost-effective way to achieve X% accuracy?"** for any value of X, not just a hardcoded 15%.
