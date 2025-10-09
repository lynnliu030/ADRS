"""
Simplified Evaluator for Electro RAG configuration optimization.

This evaluator follows a clean, focused approach:
1. Load evolved program using importlib
2. Generate 4 random target accuracies (one per density quadrant)
3. Evaluate program on each target and average scores
4. Score based on: accuracy threshold, configuration cost, search cost

This file is self-contained and includes all required functionality.
"""
import importlib.util
import time
import traceback
import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

# Configurable list of grid search directories
GRID_SEARCH_DIRS = [
    "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_gird_search_bm25",
    "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_grid_search_dense", 
    "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_grid_search_hybrid"
    # Add more directories as needed
]

# Evaluation configuration
ACCURACY_TOLERANCE = 0.03        # ±0.03 tolerance for accuracy threshold
CONFIG_WEIGHT = 0.8              # Weight for configuration cost
SEARCH_WEIGHT = 0.2              # Weight for search cost

# Cost models (hardcoded for now)
MODEL_COSTS = {
    'llama_3_8b': 0.02,
    'llama_3_70b': 0.18,
    'llama_3_1_8b': 0.02,
    'llama_3_3_70b': 0.18,
    'Llama-3.1-8B-Instruct': 0.02,
    'Llama-3.3-70B-Instruct': 0.18,
}

EMBEDDING_COSTS = {
    'bm25': 0.0001,
    'hybrid': 0.0031,
    'financebench_e5_small': 0.003,
    'financebench_gte_multilingual_base': 0.004,
    'financebench_inf_retriever_v1': 0.008,
    'financebench_inf_retriever_v1_1_5b': 0.005,
    'financebench_snowflake_s': 0.001,
}

K_COST_MULTIPLIER = 0.00001      # Cost per retrieved document

# Default values for missing data
DEFAULT_SEARCH_DURATION = 300.0  # 5 minutes average
DEFAULT_ACCURACY = -0.1          # Negative accuracy for missing configs

# ==============================================================================
# DATA LOADER CLASS - Simplified for evaluator needs
# ==============================================================================

class SimplifiedDataLoader:
    """Simplified data loader for evaluator - only loads what we need"""
    
    def __init__(self):
        self.experiments_db = {}
        self.accuracy_values = []
        self.precomputed_costs = {}  # Configuration key -> precomputed cost mapping
        self.max_search_cost = 0.0   # Maximum possible search cost (sum of all run durations)
        self._load_experiment_data()
        
    def _load_experiment_data(self):
        """Load experiment data and extract accuracy values"""
        total_dirs_checked = 0
        total_experiments_loaded = 0
        total_duration_sum = 0.0  # Track total duration for max search cost calculation
        
        for grid_dir_str in GRID_SEARCH_DIRS:
            grid_path = Path(grid_dir_str)
            if not grid_path.exists():
                print(f"Warning: Grid search directory does not exist: {grid_dir_str}")
                continue
                
            print(f"Loading experiments from: {grid_dir_str}")
            dir_experiments = 0
            
            # Find all run_* subdirectories
            for subdir in grid_path.iterdir():
                if subdir.is_dir() and subdir.name.startswith('run_'):
                    total_dirs_checked += 1
                    experiment_data = self._load_single_experiment(subdir)
                    if experiment_data:
                        config_key = self._make_config_key(experiment_data['config'])
                        self.experiments_db[config_key] = experiment_data
                        dir_experiments += 1
                        total_experiments_loaded += 1
                        
                        # Extract accuracy for quadrant calculation
                        accuracy = experiment_data['summary_metrics'].get('llm_judge_accuracy', 0.0)
                        if accuracy > 0:  # Only valid accuracies
                            self.accuracy_values.append(accuracy)
                        
                        # Add duration to total for max search cost calculation
                        duration = experiment_data['performance']['duration_seconds']
                        total_duration_sum += duration
            
            print(f"  Loaded {dir_experiments} experiments from {grid_dir_str}")
        
        # Set maximum search cost as sum of all run durations
        self.max_search_cost = total_duration_sum
        
        print(f"Total: Checked {total_dirs_checked} directories, loaded {total_experiments_loaded} experiments")
        print(f"Maximum possible search cost: {self.max_search_cost:.1f} seconds ({self.max_search_cost/3600:.1f} hours)")
                            
    def _load_single_experiment(self, run_dir: Path) -> Optional[Dict]:
        """Load single experiment with nested directory support"""
        try:
            # Try direct format first
            config_file = run_dir / 'config.json'
            summary_file = run_dir / 'run_summary.json'
            
            if all(f.exists() for f in [config_file, summary_file]):
                return self._load_format1_experiment(run_dir, config_file, summary_file)
            
            # Try nested format
            for subdir in run_dir.iterdir():
                if subdir.is_dir():
                    nested_config = subdir / 'config.json'
                    nested_summary = subdir / 'run_summary.json'
                    if all(f.exists() for f in [nested_config, nested_summary]):
                        return self._load_format1_experiment(subdir, nested_config, nested_summary)
            
            return None
        except Exception as e:
            print(f"Error loading experiment from {run_dir}: {str(e)}")
            return None
            
    def _load_format1_experiment(self, run_dir: Path, config_file: Path, summary_file: Path) -> Dict:
        """Load Format 1 experiment"""
        with open(config_file) as f:
            config = json.load(f)
        with open(summary_file) as f:
            summary = json.load(f)
            
        pipeline_config = config.get('pipeline_config', {}).get('args', {})
        benchmark_config = config.get('benchmark_config', {})
        
        # Extract model name
        model_name = 'unknown'
        llm_config = pipeline_config.get('llm', {})
        if 'completion_kwargs' in llm_config and 'model' in llm_config['completion_kwargs']:
            model_name = self._extract_model_name(llm_config['completion_kwargs']['model'])
        
        # Extract embedding name
        embedding_name = 'unknown'
        if 'embedding' in pipeline_config:
            embedding_name = pipeline_config['embedding']
        elif 'embedding_function' in pipeline_config:
            emb_func = pipeline_config['embedding_function']
            if isinstance(emb_func, dict) and 'args' in emb_func:
                model_name_emb = emb_func['args'].get('model_name', '')
                embedding_name = self._extract_embedding_name(model_name_emb)
        
        extracted_config = {
            'model': model_name,
            'embedding': embedding_name,
            'retrieval_k': pipeline_config.get('k', 0),
            'benchmark': benchmark_config.get('name', 'unknown'),
            'pipeline': 'vanilla_rag'
        }
        
        summary_metrics = summary.get('summary_metrics', {})
        
        # Load query_stats.json and calculate precomputed cost
        query_stats = self._load_query_stats(run_dir)
        precomputed_cost = self._calculate_precomputed_cost(extracted_config, query_stats)
        
        # Store precomputed cost
        config_key = self._make_config_key(extracted_config)
        self.precomputed_costs[config_key] = precomputed_cost
        
        return {
            'config': extracted_config,
            'summary_metrics': summary_metrics,
            'performance': {
                'duration_seconds': summary.get('duration_seconds', 300),
                'total_queries': summary.get('total_queries', 0),
                'successful_queries': summary.get('successful_queries', 0),
                'failed_queries': summary.get('failed_queries', 0)
            },
            'run_directory': str(run_dir),
            'precomputed_cost': precomputed_cost
        }
        
    def _extract_model_name(self, full_model_path: str) -> str:
        """Extract simplified model name"""
        model_mappings = {
            'Meta-Llama-3-8B-Instruct': 'llama_3_8b',
            'Meta-Llama-3-70B-Instruct': 'llama_3_70b', 
            'Meta-Llama-3.1-8B-Instruct': 'llama_3_1_8b',
            'Meta-Llama-3.3-70B-Instruct': 'llama_3_3_70b'
        }
        
        for full_name, short_name in model_mappings.items():
            if full_name in full_model_path:
                return short_name
        return full_model_path.split('/')[-1]
        
    def _extract_embedding_name(self, embedding_model_path: str) -> str:
        """Extract simplified embedding name"""
        # TODO: Melissa - I don't understand this, why do we need to map the embedding names?
        embedding_mappings = {
            'intfloat/multilingual-e5-small': 'financebench_e5_small',
            'intfloat/e5-small-v2': 'financebench_e5_small',
            'Snowflake/snowflake-arctic-embed-s': 'financebench_snowflake_s'
        }
        
        for full_name, short_name in embedding_mappings.items():
            if full_name in embedding_model_path:
                return short_name
                
        if '/' in embedding_model_path:
            model_name = embedding_model_path.split('/')[-1]
            model_name = model_name.replace('-', '_').replace('.', '_')
            return f'financebench_{model_name.lower()}'
        
        return embedding_model_path
        
    def _make_config_key(self, config: Dict) -> str:
        """Create unique key for configuration"""
        return f"{config['model']}_{config['embedding']}_{config['retrieval_k']}_{config['benchmark']}"
        
    def get_accuracy_range(self) -> tuple:
        """Get min/max accuracy from all experiments"""
        if not self.accuracy_values:
            return (0.0, 1.0)  # Default range
        return (min(self.accuracy_values), max(self.accuracy_values))
        
    def get_experiment_data(self, config: Dict) -> Optional[Dict]:
        """Get experiment data for configuration"""
        key = self._make_config_key(config)
        return self.experiments_db.get(key, None)
        
    def get_configs_by_benchmark(self, benchmark: str) -> List[Dict]:
        """Get all configurations for a specific benchmark"""
        configs = []
        for experiment_data in self.experiments_db.values():
            if experiment_data['config']['benchmark'] == benchmark:
                configs.append(experiment_data)
        return configs
        
    def _load_query_stats(self, run_dir: Path) -> Optional[Dict]:
        """Load query_stats.json from the experiment directory"""
        # Try direct format first (reports/query_stats.json in run_dir)
        query_stats_file = run_dir / 'reports' / 'query_stats.json'
        if query_stats_file.exists():
            try:
                with open(query_stats_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading query_stats.json from {query_stats_file}: {str(e)}")
                return None
        
        # Try nested format (look for reports/query_stats.json in subdirectories)
        for subdir in run_dir.iterdir():
            if subdir.is_dir():
                nested_query_stats = subdir / 'reports' / 'query_stats.json'
                if nested_query_stats.exists():
                    try:
                        with open(nested_query_stats) as f:
                            return json.load(f)
                    except Exception as e:
                        print(f"Error loading query_stats.json from {nested_query_stats}: {str(e)}")
                        return None
        
        print(f"Warning: query_stats.json not found in {run_dir}")
        return None
        
    def _calculate_precomputed_cost(self, config: Dict, query_stats: Optional[Dict]) -> float:
        """
        Calculate precomputed cost using the formula:
        generated_answer_mean * Generation_Model_COST + k * 100 * Generation_Model_COST + k * retriever_cost
        """
        if query_stats is None:
            # Fallback to old cost calculation if query_stats not available
            return self._fallback_cost_calculation(config)
        
        try:
            # Extract values from query_stats and config
            generated_answer_mean = query_stats.get('generated_answer', {}).get('mean', 0.0)
            model_name = config['model']
            embedding_name = config['embedding']
            k = config['retrieval_k']
            
            # Get costs from global dictionaries
            generation_model_cost = MODEL_COSTS.get(model_name, 1.001)
            retriever_cost = EMBEDDING_COSTS.get(embedding_name, 1.0001)
            
            # Calculate cost using the specified formula
            cost = (generated_answer_mean * generation_model_cost + 
                   k * 100 * generation_model_cost + 
                   k * retriever_cost)
            
            return cost
            
        except Exception as e:
            print(f"Error calculating precomputed cost for config {config}: {str(e)}")
            return self._fallback_cost_calculation(config)
    
    def _fallback_cost_calculation(self, config: Dict) -> float:
        """Fallback cost calculation when query_stats is not available"""
        model_cost = MODEL_COSTS.get(config['model'], 0.001)
        embedding_cost = EMBEDDING_COSTS.get(config['embedding'], 0.0001)
        k_cost = config['retrieval_k'] * K_COST_MULTIPLIER
        return model_cost + embedding_cost + k_cost
        
    def get_precomputed_cost(self, config: Dict) -> float:
        """Get precomputed cost for a configuration"""
        config_key = self._make_config_key(config)
        return self.precomputed_costs.get(config_key, self._fallback_cost_calculation(config))

# Initialize data loader
print("Loading experiment data for evaluator...")
data_loader = SimplifiedDataLoader()
print(f"Loaded {len(data_loader.experiments_db)} experiments")
print(f"Accuracy range: {data_loader.get_accuracy_range()}")

# ==============================================================================
# OPTIMAL CONFIGURATION FINDER
# ==============================================================================

# Cache for optimal configs to avoid repeated calculations
_optimal_config_cache = {}

def find_best_config(target_accuracy: float, benchmark: str) -> Optional[Dict]:
    """
    Find the optimal (cheapest) configuration that meets target accuracy within tolerance.
    
    Args:
        target_accuracy: Target accuracy to achieve
        benchmark: Benchmark name to search within
        
    Returns:
        Dictionary with optimal config and cost, or None if no valid config found
    """
    # Check cache first
    cache_key = f"{benchmark}_{target_accuracy:.3f}"
    if cache_key in _optimal_config_cache:
        return _optimal_config_cache[cache_key]
    
    # Get all configurations for the specified benchmark
    benchmark_configs = data_loader.get_configs_by_benchmark(benchmark)
    
    if not benchmark_configs:
        print(f"No configurations found for benchmark: {benchmark}")
        return None
    
    # Find configurations within accuracy tolerance
    valid_configs = []
    for experiment_data in benchmark_configs:
        accuracy = experiment_data['summary_metrics'].get('llm_judge_accuracy', 0.0)
        if abs(accuracy - target_accuracy) <= ACCURACY_TOLERANCE:
            # Use precomputed cost if available, otherwise calculate
            config_cost = experiment_data.get('precomputed_cost')
            if config_cost is None:
                config_cost = calculate_configuration_cost(experiment_data['config'])
            
            valid_configs.append({
                'config': experiment_data['config'],
                'accuracy': accuracy,
                'cost': config_cost,
                'experiment_data': experiment_data
            })
    
    if not valid_configs:
        print(f"No valid configurations found for target accuracy {target_accuracy:.3f} ± {ACCURACY_TOLERANCE:.3f}")
        return None
    
    # Find the cheapest valid configuration
    optimal_config = min(valid_configs, key=lambda x: x['cost'])
    
    print(f"Found optimal config for {benchmark} at accuracy {target_accuracy:.3f}: "
          f"cost=${optimal_config['cost']:.6f}, actual_accuracy={optimal_config['accuracy']:.3f}")
    
    # Cache the result
    _optimal_config_cache[cache_key] = optimal_config
    
    return optimal_config

# ==============================================================================
# CORE EVALUATION FUNCTIONS
# ==============================================================================

def evaluate(program_path):
    """
    Main evaluation function - evaluates evolved program on 4 target accuracies
    
    Args:
        program_path: Path to the evolved program file
        
    Returns:
        Dictionary with final score and detailed metrics
    """
    try:
        # 1. Load evolved program
        program = load_program_with_importlib(program_path)
        
        # 2. Generate 4 random targets (one per density quadrant)
        targets = generate_quadrant_targets()
        
        # 3. Evaluate on each target
        scores = []
        detailed_results = []
        
        for target in targets:
            score, details = evaluate_single_target(program, target)
            scores.append(score)
            detailed_results.append(details)
        
        # 4. Return average score
        final_score = np.mean(scores)
        
        return {
            'final_score': float(final_score),
            'individual_scores': [float(s) for s in scores],
            'targets': [float(t) for t in targets],
            'detailed_results': detailed_results,
            'num_targets': len(targets)
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            'final_score': 0.0,
            'error': str(e)
        }

def load_program_with_importlib(program_path):
    """Load evolved program using importlib"""
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    if not hasattr(program, 'find_optimal_rag_config'):
        raise ValueError("Program missing 'find_optimal_rag_config' function")
    
    return program

def generate_quadrant_targets():
    """Generate 4 targets using uniform quadrant distribution"""
    accuracy_range = data_loader.get_accuracy_range()
    min_acc, max_acc = accuracy_range
    
    # Handle edge case where min_acc == max_acc
    if max_acc <= min_acc:
        print(f"Warning: Invalid accuracy range [{min_acc}, {max_acc}], using default targets")
        return [0.1, 0.2, 0.3, 0.4]
    
    # Create uniform quadrants
    range_size = max_acc - min_acc
    quadrant_size = range_size / 4
    
    targets = []
    for i in range(4):
        quadrant_start = min_acc + i * quadrant_size
        quadrant_end = min_acc + (i + 1) * quadrant_size
        target = random.uniform(quadrant_start, quadrant_end)
        targets.append(target)
    
    print(f"Generated targets: {[f'{t:.3f}' for t in targets]} from range [{min_acc:.3f}, {max_acc:.3f}]")
    return targets

def evaluate_single_target(program, target_accuracy):
    """Evaluate program on single target accuracy"""
    try:
        # Execute evolved program
        result = program.find_optimal_rag_config(target_accuracy)
        
        # Check accuracy threshold
        if not meets_accuracy_threshold(result, target_accuracy):
            return 0.0, {
                'target_accuracy': target_accuracy,
                'final_accuracy': result.get('final_metrics', {}).get('accuracy', 0.0),
                'threshold_met': False,
                'config_cost': 0.0,
                'search_cost': 0.0,
                'score': 0.0
            }
        
        # Calculate costs
        config_cost = calculate_configuration_cost(result['final_config'])
        search_cost = calculate_search_cost(result['search_history'])
        
        # Extract benchmark from result for optimal cost calculation
        benchmark = result.get('final_config', {}).get('benchmark', 'unknown')
        
        # Convert costs to scores (lower cost = higher score)
        config_score = normalize_cost_to_score(config_cost, 'config', target_accuracy, benchmark)
        search_score = normalize_cost_to_score(search_cost, 'search')
        
        # Combine with weights
        final_score = (CONFIG_WEIGHT * config_score + 
                      SEARCH_WEIGHT * search_score)
        
        details = {
            'target_accuracy': target_accuracy,
            'final_accuracy': result.get('final_metrics', {}).get('accuracy', 0.0),
            'threshold_met': True,
            'config_cost': config_cost,
            'search_cost': search_cost,
            'config_score': config_score,
            'search_score': search_score,
            'score': final_score,
            'total_steps': result.get('total_steps', 0)
        }
        
        return final_score, details
        
    except Exception as e:
        print(f"Single target evaluation failed: {str(e)}")
        return 0.0, {
            'target_accuracy': target_accuracy,
            'error': str(e),
            'score': 0.0
        }

def meets_accuracy_threshold(result, target_accuracy):
    """Check if final result meets target within tolerance"""
    final_metrics = result.get('final_metrics', {})
    final_accuracy = final_metrics.get('accuracy', DEFAULT_ACCURACY)
    
    return final_accuracy >= (target_accuracy - ACCURACY_TOLERANCE)

def calculate_configuration_cost(config):
    """Calculate cost of final configuration using precomputed costs"""
    # Try to get precomputed cost first
    precomputed_cost = data_loader.get_precomputed_cost(config)
    if precomputed_cost is not None:
        return precomputed_cost
    
    # Fallback to old calculation method
    model_cost = MODEL_COSTS.get(config['model'], 0.001)
    embedding_cost = EMBEDDING_COSTS.get(config['embedding'], 0.0001)
    k_cost = config['retrieval_k'] * K_COST_MULTIPLIER
    
    return model_cost + embedding_cost + k_cost

def calculate_search_cost(search_history):
    """Calculate cost of search process using simulated durations"""
    total_search_cost = 0.0
    
    for step in search_history:
        config = step['config']
        # Find corresponding experiment data
        experiment_data = data_loader.get_experiment_data(config)
        
        if experiment_data:
            # Use duration from experiment
            duration = experiment_data['performance']['duration_seconds']
            total_search_cost += duration
        else:
            # Default cost if no historical data
            total_search_cost += DEFAULT_SEARCH_DURATION
    
    return total_search_cost

def normalize_cost_to_score(cost, cost_type='config', target_accuracy=None, benchmark=None):
    """Convert cost to score (lower cost = higher score)"""
    if cost_type == 'config' and target_accuracy is not None and benchmark is not None:
        # Find optimal configuration cost for this target accuracy and benchmark
        optimal_config = find_best_config(target_accuracy, benchmark)
        
        if optimal_config is None:
            # Fallback to simple normalization if no optimal config found
            max_reasonable_config_cost = 0.2  # Updated to realistic range
            return max(0, (max_reasonable_config_cost - cost) / max_reasonable_config_cost)
        
        optimal_cost = optimal_config['cost']
        
        # Score based on how close the cost is to optimal
        # If cost equals optimal cost, score = 1.0
        # If cost is 2x optimal cost, score = 0.5
        # If cost is 3x optimal cost, score = 0.33, etc.
        if cost <= 0:
            cost = 0.001  # Prevent division by zero
        
        score = optimal_cost / cost
        return min(1.0, score)  # Cap at 1.0
        
    else:  # search cost
        # Use precomputed maximum search cost (sum of all run durations)
        max_search_cost = data_loader.max_search_cost
        if max_search_cost <= 0:
            # Fallback if max_search_cost not available
            max_search_cost = 3600  # 1 hour fallback
        
        # Normalize: 0 cost = 1.0 score, max_search_cost = 0.0 score
        return max(0, (max_search_cost - cost) / max_search_cost)

# ==============================================================================
# STAGE-BASED EVALUATION (for OpenEvolve compatibility)
# ==============================================================================

def evaluate_stage1(program_path):
    """Stage 1 evaluation with single target"""
    try:
        program = load_program_with_importlib(program_path)
        targets = generate_quadrant_targets()
        
        # Use first target for stage 1
        score, details = evaluate_single_target(program, targets[0])
        
        return {
            'runs_successfully': 1.0 if score > 0 else 0.0,
            'stage1_score': float(score),
            'target_accuracy': float(targets[0])
        }
    except Exception as e:
        return {
            'runs_successfully': 0.0,
            'error': str(e)
        }

def evaluate_stage2(program_path):
    """Stage 2 evaluation with full 4-target evaluation"""
    return evaluate(program_path)
