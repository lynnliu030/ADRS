# OpenEvolve + Electro Integration Design Plan

## Overview

This document outlines a detailed design for using OpenEvolve to optimize RAG configurations in Electro. The system will use Electro's existing experiment data as a simulator to evaluate OpenEvolve-generated configurations without running actual experiments.

## Architecture

```
electro_simulator/
├── initial_program.py    # Starting RAG configuration search algorithm
├── evaluator.py         # Evaluation logic using Electro experiment data
├── config.yaml          # OpenEvolve configuration
├── simulator.py         # Electro data simulator (helper module)
├── data_loader.py       # Load and index Electro experiment results
└── requirements.txt     # Dependencies
```

## Component Design

### 1. Data Loader (`data_loader.py`)

The data loader will index all existing Electro experiment data for fast lookup by scanning all run subdirectories.

```python
class ElectroDataLoader:
    """Loads and indexes Electro experiment results for simulation"""
    
    def __init__(self, grid_search_dirs: List[str]):
        """
        Args:
            grid_search_dirs: List of paths to grid search directories 
                             (e.g., 'electro/outputs/financebench_vrag_llama_gird_search_bm25')
        """
        self.experiments_db = {}  # Configuration -> Full experiment data mapping
        self.configuration_space = {
            'models': set(),
            'embeddings': set(), 
            'retrieval_k': set(),
            'benchmarks': set(),
            'pipelines': set()
        }
        
    def load_experiment_data(self):
        """
        Load all experiment results by scanning run subdirectories.
        For each run, loads:
        - config.json: Full pipeline configuration
        - run_summary.json: Performance metrics and summary
        - results/benchmark_results.json: Detailed query results with timestamps
        """
        for grid_dir in self.grid_search_dirs:
            # Find all run_* subdirectories
            run_dirs = [d for d in Path(grid_dir).iterdir() 
                       if d.is_dir() and d.name.startswith('run_')]
            
            for run_dir in run_dirs:
                experiment_data = self.load_single_experiment(run_dir)
                if experiment_data:
                    config_key = self.make_config_key(experiment_data['config'])
                    self.experiments_db[config_key] = experiment_data
                    self.update_configuration_space(experiment_data['config'])
                    
    def load_single_experiment(self, run_dir: Path) -> Optional[Dict]:
        """Load all data for a single experiment run"""
        config_file = run_dir / 'config.json'
        summary_file = run_dir / 'run_summary.json'  
        results_file = run_dir / 'results' / 'benchmark_results.json'
        
        if not all(f.exists() for f in [config_file, summary_file, results_file]):
            return None
            
        with open(config_file) as f:
            config = json.load(f)
        with open(summary_file) as f:
            summary = json.load(f)
        with open(results_file) as f:
            results = json.load(f)
            
        # Extract configuration parameters
        pipeline_config = config['pipeline_config']['args']
        benchmark_config = config['benchmark_config']
        
        # Calculate experiment duration from timestamps
        query_timestamps = [q['timestamps'] for q in results['queries']]
        start_time = min(ts['start'] for ts in query_timestamps)
        end_time = max(ts['end'] for ts in query_timestamps)
        experiment_duration = end_time - start_time
        
        return {
            'config': {
                'model': self.extract_model_name(pipeline_config['llm']['completion_kwargs']['model']),
                'embedding': pipeline_config['embedding'],
                'retrieval_k': pipeline_config['k'],
                'benchmark': benchmark_config['name'],
                'pipeline': 'vanilla_rag'  # Default for now
            },
            'summary_metrics': summary['summary_metrics'],
            'performance': {
                'duration_seconds': summary['duration_seconds'],
                'experiment_duration': experiment_duration,
                'total_queries': summary['total_queries'],
                'successful_queries': summary['successful_queries'],
                'failed_queries': summary['failed_queries']
            },
            'detailed_results': results,
            'full_config': config,
            'run_directory': str(run_dir)
        }
        
    def extract_model_name(self, full_model_path: str) -> str:
        """Extract simplified model name from full path"""
        # Convert "meta-llama/Meta-Llama-3-8B-Instruct" -> "llama_3_8b"
        model_mappings = {
            'Meta-Llama-3-8B-Instruct': 'llama_3_8b',
            'Meta-Llama-3-70B-Instruct': 'llama_3_70b', 
            'Meta-Llama-3.1-8B-Instruct': 'llama_3_1_8b',
            'Meta-Llama-3.3-70B-Instruct': 'llama_3_3_70b'
        }
        
        for full_name, short_name in model_mappings.items():
            if full_name in full_model_path:
                return short_name
        return full_model_path.split('/')[-1]  # Fallback
        
    def get_experiment_data(self, config: Dict) -> Optional[Dict]:
        """
        Retrieve full experiment data for a specific configuration.
        Returns None if configuration not found in data.
        """
        key = self.make_config_key(config)
        return self.experiments_db.get(key, None)
        
    def get_configuration_bounds(self) -> Dict:
        """Return the bounds of each parameter from available data"""
        return {
            'models': sorted(list(self.configuration_space['models'])),
            'embeddings': sorted(list(self.configuration_space['embeddings'])),
            'retrieval_k': sorted(list(self.configuration_space['retrieval_k'])),
            'benchmarks': sorted(list(self.configuration_space['benchmarks']))
        }
```

### 2. Simulator (`simulator.py`)

The simulator provides a clean interface for OpenEvolve to query experiment results and returns comprehensive data.

```python
class ElectroSimulator:
    """Simulates Electro experiments using historical data"""
    
    def __init__(self, data_loader: ElectroDataLoader):
        self.data_loader = data_loader
        self.cache = {}  # Cache for repeated queries
        
    def run_experiment(self, config: Dict) -> Dict:
        """
        Simulate running an Electro experiment with given configuration.
        
        Args:
            config: Dictionary with keys:
                - model: str (e.g., "llama_3_8b")
                - embedding: str (e.g., "bm25")
                - retrieval_k: int (e.g., 5)
                - benchmark: str (e.g., "financeBench")
                - pipeline: str (e.g., "vanilla_rag")
                
        Returns:
            Dictionary with comprehensive metrics including:
            - All metrics from run_summary.json
            - Calculated experiment duration
            - Cost estimates
            - Full configuration details
            - Success/failure indicators
        """
        cache_key = str(sorted(config.items()))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Get experiment data
        experiment_data = self.data_loader.get_experiment_data(config)
        
        if experiment_data is None:
            return self._get_failure_metrics(config, "Configuration not found in experiment data")
            
        # Extract comprehensive metrics
        summary_metrics = experiment_data['summary_metrics']
        performance = experiment_data['performance']
        
        # Build comprehensive result
        result = {
            # Core accuracy metrics
            'accuracy': summary_metrics.get('llm_judge_accuracy', 0.0),
            'llm_judge_correct_count': summary_metrics.get('llm_judge_correct_count', 0.0),
            
            # Retrieval metrics (allow OpenEvolve to use any of these)
            'retrieval_precision_at_k': summary_metrics.get('retrieval_precision@k', 0.0),
            'retrieval_recall_at_k': summary_metrics.get('retrieval_recall@k', 0.0),
            'retrieval_rp_at_k': summary_metrics.get('retrieval_rp@k', 0.0),
            'retrieval_aic_at_k': summary_metrics.get('retrieval_aic@k', 0.0),
            'retrieval_mrr': summary_metrics.get('retrieval_mrr', 0.0),
            
            # Performance metrics
            'total_queries': performance['total_queries'],
            'successful_queries': performance['successful_queries'],
            'failed_queries': performance['failed_queries'],
            'success_rate': summary_metrics.get('success_rate', 0.0),
            'duration_seconds': performance['duration_seconds'],
            'experiment_duration': performance['experiment_duration'],
            'queries_per_second': performance['total_queries'] / performance['duration_seconds'] if performance['duration_seconds'] > 0 else 0,
            
            # Cost estimation
            'cost_estimate': self.estimate_cost(config, summary_metrics, performance),
            
            # Meta information
            'config': config,
            'run_directory': experiment_data['run_directory'],
            'is_simulated': True,
            'data_source': 'historical_experiment'
        }
        
        self.cache[cache_key] = result
        return result
        
    def estimate_cost(self, config: Dict, metrics: Dict, performance: Dict) -> float:
        """
        Estimate cost per query based on model, duration, and throughput.
        This is the key metric OpenEvolve will optimize.
        """
        # Model inference costs ($ per 1K tokens, rough estimates)
        model_costs = {
            'llama_3_8b': 0.0002,
            'llama_3_70b': 0.002,
            'llama_3_1_8b': 0.0002,
            'llama_3_3_70b': 0.0018,
        }
        
        # Embedding costs ($ per document retrieved)
        embedding_costs = {
            'bm25': 0.00001,  # CPU-based, very cheap
            'financebench_e5_small': 0.00005,
            'financebench_gte_multilingual': 0.0001,
            'financebench_snowflake_s': 0.00008,
        }
        
        # Estimate tokens (simplified)
        avg_input_tokens = 2000  # Context + query
        avg_output_tokens = 200   # Response
        
        # Model inference cost
        model_cost_per_1k = model_costs.get(config['model'], 0.001)
        inference_cost = (avg_input_tokens + avg_output_tokens) / 1000 * model_cost_per_1k
        
        # Embedding cost
        embedding_cost_per_doc = embedding_costs.get(config['embedding'], 0.0001)
        embedding_cost = embedding_cost_per_doc * config['retrieval_k']
        
        # Infrastructure cost (based on experiment duration as proxy for efficiency)
        # Longer experiments suggest less efficient resource usage
        base_infra_cost = 0.001  # Base cost per query
        duration_penalty = performance['duration_seconds'] / performance['total_queries'] / 10.0  # Normalize
        infra_cost = base_infra_cost * (1 + duration_penalty)
        
        total_cost = inference_cost + embedding_cost + infra_cost
        return total_cost
        
    def _get_failure_metrics(self, config: Dict, error_msg: str) -> Dict:
        """Return metrics for failed/unknown configurations"""
        return {
            'accuracy': 0.0,
            'retrieval_precision_at_k': 0.0,
            'retrieval_recall_at_k': 0.0,
            'success_rate': 0.0,
            'cost_estimate': 999.0,  # Very high cost for invalid configs
            'config': config,
            'is_simulated': True,
            'error': error_msg,
            'data_source': 'failure'
        }
        
    def get_available_configurations(self) -> Dict:
        """Return all available configuration options"""
        return self.data_loader.get_configuration_bounds()
```

### 3. Initial Program (`initial_program.py`)

The initial program contains a simple random search baseline that OpenEvolve will improve. The search space is defined by the available configurations from the loaded experiment data.

```python
# EVOLVE-BLOCK-START
"""RAG Configuration Random Search Baseline for Electro"""
import random
from typing import Dict, List, Optional

class RAGConfigurationOptimizer:
    """Simple random search baseline for RAG pipeline configurations"""
    
    def __init__(self):
        # Get available configuration space from the simulator
        # This will be populated by the evaluator when the program runs
        self.available_configs = None
        
    def initialize_search_space(self, available_configs: Dict):
        """Initialize search space with available configurations from experiment data"""
        self.available_configs = available_configs
        
    def random_search(self, budget: int = 50) -> Dict:
        """
        Simple random search baseline.
        OpenEvolve will evolve this into more sophisticated algorithms.
        
        Args:
            budget: Number of configurations to evaluate
            
        Returns:
            Best configuration found based on cost-efficiency
        """
        if not self.available_configs:
            # Fallback to hardcoded values if not initialized
            self.available_configs = {
                'models': ['llama_3_8b', 'llama_3_1_8b', 'llama_3_70b'],
                'embeddings': ['bm25', 'financebench_e5_small'],
                'retrieval_k': [1, 3, 5, 10, 15, 20, 25, 30],
                'benchmarks': ['financeBench']
            }
        
        best_config = None
        best_score = float('-inf')
        evaluated_configs = []
        
        for i in range(budget):
            # Generate random configuration
            config = self.generate_random_config()
            
            # Evaluate configuration (calls simulator)
            metrics = evaluate_configuration(config)
            
            # Calculate combined score (cost-efficiency focus)
            score = self.calculate_score(metrics)
            evaluated_configs.append((config, metrics, score))
            
            # Track best configuration
            if score > best_score:
                best_score = score
                best_config = config
                
        return best_config
        
    def generate_random_config(self) -> Dict:
        """Generate a random configuration from available options"""
        return {
            'model': random.choice(self.available_configs['models']),
            'embedding': random.choice(self.available_configs['embeddings']),
            'retrieval_k': random.choice(self.available_configs['retrieval_k']),
            'benchmark': random.choice(self.available_configs['benchmarks']),
            'pipeline': 'vanilla_rag'
        }
        
    def calculate_score(self, metrics: Dict) -> float:
        """
        Calculate a combined score for configuration ranking.
        Higher score = better configuration.
        
        This simple scoring will be evolved by OpenEvolve into more sophisticated
        multi-objective optimization strategies.
        """
        # Handle failed configurations
        if 'error' in metrics:
            return -1000.0
            
        # Simple cost-efficiency score
        accuracy = metrics.get('accuracy', 0.0)
        cost = metrics.get('cost_estimate', 999.0)
        success_rate = metrics.get('success_rate', 0.0)
        
        # Avoid division by zero
        if cost <= 0:
            cost = 0.001
            
        # Score = accuracy / cost, with success rate bonus
        base_score = (accuracy * success_rate) / cost
        
        # Bonus for retrieval quality
        retrieval_bonus = metrics.get('retrieval_recall_at_k', 0.0) * 0.1
        
        return base_score + retrieval_bonus

# Main entry point for OpenEvolve
def find_optimal_rag_config() -> Dict:
    """
    Main function that OpenEvolve will evolve.
    Starts with simple random search.
    """
    optimizer = RAGConfigurationOptimizer()
    
    # Get available configurations from simulator
    available_configs = get_available_configurations()
    optimizer.initialize_search_space(available_configs)
    
    # Run random search
    best_config = optimizer.random_search(budget=30)
    
    return best_config

# EVOLVE-BLOCK-END

# Fixed helper functions (not evolved)
def evaluate_configuration(config: Dict) -> Dict:
    """
    This will be injected by the evaluator to call the simulator.
    Placeholder for development.
    """
    return {
        'accuracy': 0.1,
        'cost_estimate': 1.0,
        'success_rate': 1.0,
        'retrieval_recall_at_k': 0.1
    }

def get_available_configurations() -> Dict:
    """
    This will be injected by the evaluator to provide available config space.
    Placeholder for development.
    """
    return {
        'models': ['llama_3_8b'],
        'embeddings': ['bm25'],
        'retrieval_k': [1, 5, 10],
        'benchmarks': ['financeBench']
    }
```

### 4. Evaluator (`evaluator.py`)

The evaluator connects OpenEvolve programs to the Electro simulator.

```python
"""
Evaluator for Electro RAG configuration optimization
"""
import importlib.util
import sys
import time
import traceback
import numpy as np
from pathlib import Path

# Add simulator to path
sys.path.append(str(Path(__file__).parent))
from simulator import ElectroSimulator
from data_loader import ElectroDataLoader

# Configurable list of grid search directories
# Add all your experiment output directories here
GRID_SEARCH_DIRS = [
    "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_gird_search_bm25",
    "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_grid_search_dense", 
    "/home/melissa/openevolve/electro/outputs/frames_llama70b_grid_search_poisson_2",
    "/home/melissa/openevolve/electro/outputs/frames_llama70b_grid_search_poisson_3",
    "/home/melissa/openevolve/electro/outputs/frames_llama70b_grid_search_poisson_6",
    "/home/melissa/openevolve/electro/outputs/frames_llama8b_grid_search",
    "/home/melissa/openevolve/electro/outputs/TQA_KILT_llama70b_grid_search_poisson_6",
    "/home/melissa/openevolve/electro/outputs/TQA_KILT_llama8b_grid_search_poisson_6",
    "/home/melissa/openevolve/electro/outputs/rag_eval_fb_grid",
    "/home/melissa/openevolve/electro/outputs/saturation_load_llama8b",
    "/home/melissa/openevolve/electro/outputs/saturation_load_llama70b",
    # Add more directories as needed
]

# Configurable evaluation weights for OpenEvolve scoring
EVALUATION_WEIGHTS = {
    'search_efficiency': 0.3,      # How few configurations needed to find good result
    'config_quality': 0.5,        # Quality of the final configuration found  
    'config_cost': 0.2,           # Cost efficiency of the final configuration
}

# Future: Latency constraints (placeholder)
LATENCY_CONSTRAINTS = {
    'max_p90_latency': None,      # Maximum acceptable P90 latency in seconds
    'max_avg_latency': None,      # Maximum acceptable average latency
    'latency_penalty_weight': 0.0 # Weight for latency constraint violations
}

data_loader = ElectroDataLoader(GRID_SEARCH_DIRS)
data_loader.load_experiment_data()
simulator = ElectroSimulator(data_loader)

def evaluate(program_path):
    """
    Evaluate a RAG configuration search algorithm.
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        
        # Inject the simulator's functions
        program.evaluate_configuration = simulator.run_experiment
        program.get_available_configurations = simulator.get_available_configurations
        
        spec.loader.exec_module(program)
        
        # Check for required function
        if not hasattr(program, 'find_optimal_rag_config'):
            return {
                'cost_efficiency': 0.0,
                'accuracy_score': 0.0,
                'latency_score': 0.0,
                'combined_score': 0.0,
                'error': 'Missing find_optimal_rag_config function'
            }
            
        # Run the configuration search
        start_time = time.time()
        best_config = program.find_optimal_rag_config()
        search_time = time.time() - start_time
        
        if best_config is None:
            return {
                'cost_efficiency': 0.0,
                'accuracy_score': 0.0,
                'latency_score': 0.0,
                'combined_score': 0.0,
                'error': 'No valid configuration found'
            }
            
        # Evaluate the best configuration found
        metrics = simulator.run_experiment(best_config)
        
        # Calculate scores (higher is better)
        # Cost efficiency: Lower cost is better, normalize to 0-1
        max_cost = 1.0  # Estimated max reasonable cost per query
        cost_efficiency = 1.0 - min(metrics['cost_estimate'] / max_cost, 1.0)
        
        # Accuracy score: Direct use
        accuracy_score = metrics['accuracy']
        
        # Success rate score
        success_rate = metrics.get('success_rate', 0.0)
        
        # Retrieval quality score
        retrieval_score = (
            metrics.get('retrieval_precision_at_k', 0.0) + 
            metrics.get('retrieval_recall_at_k', 0.0)
        ) / 2.0
        
        # Search efficiency: How fast did it find a good solution
        search_efficiency = min(1.0 / max(search_time, 0.1), 10.0) / 10.0
        
        # Combined score with emphasis on cost-efficiency and quality
        combined_score = (
            0.35 * cost_efficiency +      # Primary objective: minimize cost
            0.25 * accuracy_score +       # Quality: accuracy
            0.20 * success_rate +         # Reliability 
            0.15 * retrieval_score +      # Retrieval quality
            0.05 * search_efficiency      # Algorithm efficiency
        )
        
        # Configuration diversity for MAP-Elites
        config_hash = f"{best_config['model']}_{best_config['embedding']}_{best_config['retrieval_k']}"
        
        return {
            'cost_efficiency': float(cost_efficiency),
            'accuracy_score': float(accuracy_score),
            'success_rate_score': float(success_rate),
            'retrieval_score': float(retrieval_score),
            'search_efficiency': float(search_efficiency),
            'combined_score': float(combined_score),
            'best_config': str(best_config),
            'total_cost_per_query': float(metrics.get('cost_estimate', 0)),
            'config_diversity': len(config_hash) / 50.0,  # Normalized
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            'cost_efficiency': 0.0,
            'accuracy_score': 0.0,
            'success_rate_score': 0.0,
            'retrieval_score': 0.0,
            'search_efficiency': 0.0,
            'combined_score': 0.0,
            'error': str(e)
        }

def evaluate_stage1(program_path):
    """Quick validation that the program runs"""
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        
        # Inject simulator functions
        program.evaluate_configuration = lambda config: {
            'accuracy': 0.1,
            'cost_estimate': 1.0,
            'success_rate': 1.0,
            'retrieval_precision_at_k': 0.1,
            'retrieval_recall_at_k': 0.1
        }
        program.get_available_configurations = lambda: {
            'models': ['llama_3_8b'],
            'embeddings': ['bm25'],
            'retrieval_k': [1, 5, 10],
            'benchmarks': ['financeBench']
        }
        
        spec.loader.exec_module(program)
        
        if not hasattr(program, 'find_optimal_rag_config'):
            return {'runs_successfully': 0.0}
            
        # Try to run with minimal budget
        optimizer = program.RAGConfigurationOptimizer()
        config = optimizer.random_configuration()
        
        return {'runs_successfully': 1.0}
        
    except Exception as e:
        return {'runs_successfully': 0.0, 'error': str(e)}

def evaluate_stage2(program_path):
    """Full evaluation"""
    return evaluate(program_path)
```

### 5. Configuration (`config.yaml`)

OpenEvolve configuration optimized for this use case.

```yaml
# OpenEvolve configuration for Electro RAG optimization
max_iterations: 500
random_seed: 42
checkpoint_interval: 25

llm:
  models:
    - name: "gpt-4-turbo"
      weight: 0.6
    - name: "claude-3-opus"
      weight: 0.4
  temperature: 0.8
  max_tokens: 2048
  
database:
  population_size: 100
  num_islands: 3
  migration_interval: 20
  # Custom features for RAG optimization
  feature_dimensions: ["cost_efficiency", "accuracy_score", "config_diversity"]
  feature_bins:
    cost_efficiency: 10
    accuracy_score: 10
    config_diversity: 8
    
evaluator:
  enable_artifacts: true
  cascade_evaluation: true
  use_llm_feedback: true
  num_workers: 4
  timeout: 30
  
prompt:
  num_top_programs: 5
  num_diverse_programs: 3
  include_artifacts: true
  
  # Custom templates for RAG optimization
  system_prompt: |
    You are optimizing RAG (Retrieval-Augmented Generation) configurations
    for cost-efficiency while maintaining quality. Consider:
    - Model size vs accuracy tradeoffs
    - Embedding method computational costs
    - Retrieval K impact on both quality and latency
    - Creative search strategies beyond random/grid search
    
output:
  save_all_programs: true
  save_best_every: 10
```

## Key Design Decisions

### 1. Simulator-Based Evaluation
- **Rationale**: Using existing experiment data avoids expensive re-runs
- **Benefits**: Fast evaluation, deterministic results, no GPU requirements
- **Limitations**: Can only evaluate configurations present in historical data

### 2. Cost as Primary Metric
- **Formula**: `cost = (model_cost_per_token * tokens) + (embedding_cost * documents) + (infrastructure_cost * latency)`
- **Optimization**: Multi-objective with cost efficiency as primary goal

### 3. Interpolation for Unseen Configurations
- For configurations not in the dataset, interpolate from nearby points
- Use distance-weighted averaging in parameter space
- Flag interpolated results vs actual data

### 4. Evolution Strategy Focus Areas
OpenEvolve will likely improve:
1. **Search algorithms**: From random to Bayesian optimization, genetic algorithms, etc.
2. **Constraint handling**: Better methods to stay within accuracy/latency bounds
3. **Multi-objective optimization**: Pareto frontier exploration
4. **Adaptive strategies**: Learning from previous evaluations
5. **Configuration encoding**: Better representations of the search space

## Implementation Phases

### Phase 1: Basic Simulator (Week 1)
1. Implement data_loader.py to parse Electro outputs
2. Create simulator.py with exact configuration lookup
3. Test with known configurations

### Phase 2: Initial Evolution (Week 2)
1. Implement initial_program.py with random search
2. Create evaluator.py with basic metrics
3. Configure and run first OpenEvolve experiments

### Phase 3: Advanced Features (Week 3)
1. Add interpolation for unseen configurations
2. Implement multi-objective optimization in evaluator
3. Add sophisticated cost models
4. Create custom prompt templates

### Phase 4: Analysis and Optimization (Week 4)
1. Analyze evolved algorithms
2. Fine-tune OpenEvolve parameters
3. Compare against baseline grid search
4. Document findings and best configurations

## Success Metrics

1. **Cost Reduction**: 50% reduction in cost vs naive configurations
2. **Algorithm Discovery**: Novel search strategies beyond grid/random
3. **Constraint Satisfaction**: 95% of solutions meet accuracy/latency requirements
4. **Search Efficiency**: 10x faster than grid search to find optimal configurations

## Potential Extensions

1. **Online Learning**: Evolve algorithms that adapt during deployment
2. **Multi-Benchmark**: Optimize across multiple benchmarks simultaneously  
3. **Hardware-Aware**: Consider different hardware configurations
4. **Ensemble Methods**: Combine multiple models dynamically
5. **Active Learning**: Suggest which experiments to run next
