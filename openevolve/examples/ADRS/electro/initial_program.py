"""RAG Configuration Constraint-Based Optimization Baseline for Electro

This program solves the problem: Given a target accuracy threshold, find the most 
cost-effective RAG configuration that meets or exceeds that accuracy level.

Key objectives:
1. HARD CONSTRAINT: Configuration must meet minimum accuracy threshold
2. PRIMARY OBJECTIVE: Minimize cost per query among valid configurations  
3. SECONDARY OBJECTIVES: Maximize retrieval quality, exceed accuracy target

This approach transforms the problem from generic multi-objective optimization
to a more practical constraint satisfaction + cost minimization problem.

The target accuracy is configurable and can be any value - the algorithm is generic!

This file is self-contained and includes all required functionality.
"""
import random
import json
import os
from typing import Dict, List, Optional
from pathlib import Path

# ==============================================================================
# EMBEDDED DATA LOADER - Self-contained for OpenEvolve compatibility
# ==============================================================================

class EmbeddedElectroDataLoader:
    """Self-contained data loader for OpenEvolve compatibility"""
    
    def __init__(self):
        # Hardcoded paths for self-containment - matching evaluator.py exactly
        self.grid_search_dirs = [
            "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_gird_search_bm25",
            "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_grid_search_dense",
            "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_grid_search_hybrid"
        ]
        self.experiments_db = {}
        self.configuration_space = {
            'models': set(),
            'embeddings': set(), 
            'retrieval_k': set(),
            'benchmarks': set()
        }
        self._load_experiment_data()
        
    def _load_experiment_data(self):
        """Load experiment data from hardcoded paths"""
        total_dirs_checked = 0
        total_experiments_loaded = 0
        
        for grid_dir_str in self.grid_search_dirs:
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
                        self._update_configuration_space(experiment_data['config'])
                        dir_experiments += 1
                        total_experiments_loaded += 1
            
            print(f"  Loaded {dir_experiments} experiments from {grid_dir_str}")
        
        print(f"Total: Checked {total_dirs_checked} directories, loaded {total_experiments_loaded} experiments")
                        
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
        
        # Load query_stats.json and calculate precomputed cost (matching evaluator.py)
        query_stats = self._load_query_stats(run_dir)
        precomputed_cost = self._calculate_precomputed_cost(extracted_config, query_stats)
        
        return {
            'config': extracted_config,
            'summary_metrics': summary_metrics,
            'performance': {
                'duration_seconds': summary.get('duration_seconds', 300),
                'total_queries': summary.get('total_queries', 0),
                'successful_queries': summary.get('successful_queries', 0),
                'failed_queries': summary.get('failed_queries', 0)
            },
            'full_config': config,
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
        
    def _update_configuration_space(self, config: Dict):
        """Update configuration space"""
        self.configuration_space['models'].add(config['model'])
        self.configuration_space['embeddings'].add(config['embedding'])
        self.configuration_space['retrieval_k'].add(config['retrieval_k'])
        self.configuration_space['benchmarks'].add(config['benchmark'])
        
    def get_experiment_data(self, config: Dict) -> Optional[Dict]:
        """Get experiment data for configuration"""
        key = self._make_config_key(config)
        return self.experiments_db.get(key, None)
        
    def get_all_configurations(self) -> List[Dict]:
        """Get all available configurations"""
        return [data['config'] for data in self.experiments_db.values()]
        
    def get_configuration_bounds(self) -> Dict:
        """Return the bounds of each parameter from available data"""
        return {
            'models': sorted(list(self.configuration_space['models'])),
            'embeddings': sorted(list(self.configuration_space['embeddings'])),
            'retrieval_k': sorted(list(self.configuration_space['retrieval_k'])),
            'benchmarks': sorted(list(self.configuration_space['benchmarks']))
        }
        
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
        
        return None  # Don't print warning in initial_program to reduce noise
        
    def _calculate_precomputed_cost(self, config: Dict, query_stats: Optional[Dict]) -> float:
        """
        Calculate precomputed cost using the formula:
        generated_answer_mean * Generation_Model_COST + k * 100 * Generation_Model_COST + k * retriever_cost
        """
        if query_stats is None:
            return 0.0  # Return 0 if no query_stats, will use fallback in _estimate_cost
        
        try:
            # Cost models matching evaluator.py exactly
            model_costs = {
                'llama_3_8b': 0.02,
                'llama_3_70b': 0.18,
                'llama_3_1_8b': 0.02,
                'llama_3_3_70b': 0.18,
                'Llama-3.1-8B-Instruct': 0.02,
                'Llama-3.3-70B-Instruct': 0.18,
            }
            
            embedding_costs = {
                'bm25': 0.0001,
                'hybrid': 0.0031,
                'financebench_e5_small': 0.003,
                'financebench_gte_multilingual_base': 0.004,
                'financebench_inf_retriever_v1': 0.008,
                'financebench_inf_retriever_v1_1_5b': 0.005,
                'financebench_snowflake_s': 0.001,
            }
            
            # Extract values from query_stats and config
            generated_answer_mean = query_stats.get('generated_answer', {}).get('mean', 0.0)
            model_name = config['model']
            embedding_name = config['embedding']
            k = config['retrieval_k']
            
            # Get costs from dictionaries
            generation_model_cost = model_costs.get(model_name, 0.001)
            retriever_cost = embedding_costs.get(embedding_name, 0.0001)
            
            # Calculate cost using the specified formula
            cost = (generated_answer_mean * generation_model_cost + 
                   k * 100 * generation_model_cost + 
                   k * retriever_cost)
            
            return cost
            
        except Exception as e:
            return 0.0  # Return 0 on error, will use fallback in _estimate_cost

# ==============================================================================
# EMBEDDED SIMULATOR - Self-contained for OpenEvolve compatibility  
# ==============================================================================

class EmbeddedElectroSimulator:
    """Self-contained simulator for OpenEvolve compatibility"""
    
    def __init__(self, data_loader: EmbeddedElectroDataLoader):
        self.data_loader = data_loader
        self.cache = {}
        
    def run_experiment(self, config: Dict) -> Dict:
        """Simulate experiment with given configuration"""
        cache_key = str(sorted(config.items()))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        experiment_data = self.data_loader.get_experiment_data(config)
        
        if experiment_data is None:
            return self._get_failure_metrics(config)
            
        summary_metrics = experiment_data['summary_metrics']
        performance = experiment_data['performance']
        
        result = {
            'accuracy': summary_metrics.get('llm_judge_accuracy', 0.0),
            'retrieval_precision_at_k': summary_metrics.get('retrieval_precision@k', 0.0),
            'retrieval_recall_at_k': summary_metrics.get('retrieval_recall@k', 0.0),
            'success_rate': summary_metrics.get('success_rate', 1.0),
            'duration_seconds': performance['duration_seconds'],
            'total_queries': performance['total_queries'],
            'cost_estimate': self._estimate_cost(config, summary_metrics, performance),
            'config': config,
            'run_directory': experiment_data['run_directory'],
            'full_config': experiment_data['full_config']  # Full configuration for access
        }
        
        self.cache[cache_key] = result
        return result
        
    def _estimate_cost(self, config: Dict, metrics: Dict, performance: Dict) -> float:
        """Estimate cost per query - use precomputed cost if available"""
        # Try to get precomputed cost from experiment data first
        experiment_data = self.data_loader.get_experiment_data(config)
        if experiment_data and 'precomputed_cost' in experiment_data:
            return experiment_data['precomputed_cost']
        
        # Fallback to simple cost calculation if precomputed cost not available
        # Cost models matching evaluator.py exactly
        model_costs = {
            'llama_3_8b': 0.02,
            'llama_3_70b': 0.18,
            'llama_3_1_8b': 0.02,
            'llama_3_3_70b': 0.18,
            'Llama-3.1-8B-Instruct': 0.02,
            'Llama-3.3-70B-Instruct': 0.18,
        }
        
        embedding_costs = {
            'bm25': 0.0001,
            'hybrid': 0.0031,
            'financebench_e5_small': 0.003,
            'financebench_gte_multilingual_base': 0.004,
            'financebench_inf_retriever_v1': 0.008,
            'financebench_inf_retriever_v1_1_5b': 0.005,
            'financebench_snowflake_s': 0.001,
        }
        
        model_cost = model_costs.get(config['model'], 0.001)
        embedding_cost = embedding_costs.get(config['embedding'], 0.0001)
        k_cost = config['retrieval_k'] * 0.00001  # K_COST_MULTIPLIER from evaluator.py
        
        return model_cost + embedding_cost + k_cost
        
    def _get_failure_metrics(self, config: Dict) -> Dict:
        """Return metrics for failed configurations"""
        return {
            'accuracy': 0.0,
            'cost_estimate': 999.0,
            'config': config,
            'error': 'Configuration not found'
        }

# Default configuration space for fallback
DEFAULT_AVAILABLE_CONFIGS = {
    'models': ['llama_3_8b', 'llama_3_1_8b', 'llama_3_70b'],
    'embeddings': ['bm25', 'financebench_e5_small'],
    'retrieval_k': [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'benchmarks': ['financeBench', 'frames', 'triviaQA']
}

# ==============================================================================
# GLOBAL DATA LOADER SINGLETON - Fix reinitialization bug
# ==============================================================================

# Global singleton to prevent data reloading
_global_data_loader = None
_global_simulator = None

def get_global_data_loader():
    """Get or create global data loader singleton"""
    global _global_data_loader
    if _global_data_loader is None:
        print("Initializing global data loader (one-time setup)...")
        _global_data_loader = EmbeddedElectroDataLoader()
    return _global_data_loader

def get_global_simulator():
    """Get or create global simulator singleton"""
    global _global_simulator
    if _global_simulator is None:
        data_loader = get_global_data_loader()
        _global_simulator = EmbeddedElectroSimulator(data_loader)
    return _global_simulator

# ==============================================================================
# EVOLVE-BLOCK-START: Core Optimization Algorithm
# ==============================================================================

class RAGConfigurationOptimizer:
    """Constraint-based optimization for RAG configurations
    
    Finds the most cost-effective configuration that meets a target accuracy threshold.
    OpenEvolve will evolve this into more sophisticated constraint satisfaction algorithms.
    """
    
    def __init__(self, target_accuracy: float = 0.15):
        """
        Initialize optimizer with configurable target accuracy.
        
        Args:
            target_accuracy: The accuracy threshold to meet (can be any value)
        """
        # Use global singletons to prevent data reloading
        self.data_loader = get_global_data_loader()
        self.simulator = get_global_simulator()
        
        # Get available configuration space from data loader
        self.available_configs = self.data_loader.get_configuration_bounds()
        self.target_accuracy = target_accuracy
        self.evaluation_history = []  # Track all evaluations with full config access
        
    def random_search(self, budget: int = 30) -> Dict:
        """
        Constraint-aware random search baseline.
        OpenEvolve will evolve this into sophisticated constraint satisfaction algorithms.
        
        Args:
            budget: Number of configurations to evaluate
            
        Returns:
            Most cost-effective configuration that meets accuracy constraint
        """
        if not self.available_configs:
            # Fallback to default values if not initialized
            self.available_configs = DEFAULT_AVAILABLE_CONFIGS
        
        best_config = None
        best_score = float('-inf')
        best_metrics = None
        
        # Get all available configurations from embedded data loader
        all_configs = self.data_loader.get_all_configurations()
        if not all_configs:
            # Fallback to default configs
            all_configs = self._generate_configs_from_defaults()
            
        # Randomly sample configurations
        sampled_configs = random.sample(all_configs, min(budget, len(all_configs)))
        
        for i, config in enumerate(sampled_configs):
            # Evaluate configuration using embedded simulator
            metrics = self.simulator.run_experiment(config)
            
            # Calculate combined score
            score = self.calculate_score(metrics)
            
            # Store evaluation history with full configuration access
            self.evaluation_history.append({
                'config': config,
                'metrics': metrics,
                'score': score,
                'iteration': i,
                'full_config': metrics.get('full_config'),  # Full configuration details
                'run_directory': metrics.get('run_directory')  # Directory for full access
            })
            
            # Track best configuration
            if score > best_score:
                best_score = score
                best_config = config
                best_metrics = metrics
                
        return {
            'best_config': best_config,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'evaluation_history': self.evaluation_history,  # For search cost calculation
            'total_evaluated': len(sampled_configs),
            'full_configuration_access': True  # Indicates full config is available
        }
        
    def search_with_history_tracking(self, budget: int = 30) -> Dict:
        """
        Same as random_search but with explicit history tracking for evaluator
        """
        return self.random_search(budget=budget)
        
    def _generate_configs_from_defaults(self) -> List[Dict]:
        """Generate configurations from default available configs"""
        configs = []
        for model in DEFAULT_AVAILABLE_CONFIGS['models']:
            for embedding in DEFAULT_AVAILABLE_CONFIGS['embeddings']:
                for k in DEFAULT_AVAILABLE_CONFIGS['retrieval_k']:
                    for benchmark in DEFAULT_AVAILABLE_CONFIGS['benchmarks']:
                        configs.append({
                            'model': model,
                            'embedding': embedding,
                            'retrieval_k': k,
                            'benchmark': benchmark,
                            'pipeline': 'vanilla_rag'
                        })
        return configs
        
    def get_full_configuration(self, config: Dict) -> Optional[Dict]:
        """Get full configuration details for a selected configuration"""
        metrics = self.simulator.run_experiment(config)
        return metrics.get('full_config')
        
    def get_run_directory(self, config: Dict) -> Optional[str]:
        """Get run directory for a selected configuration"""
        metrics = self.simulator.run_experiment(config)
        return metrics.get('run_directory')
        
    def calculate_score(self, metrics: Dict) -> float:
        """
        Calculate a combined score for configuration ranking focusing on cost efficiency
        given accuracy constraints. Higher score = better configuration.
        
        This simple scoring will be evolved by OpenEvolve into more sophisticated
        constraint-satisfaction and cost optimization strategies.
        """
        # Handle failed configurations
        if 'error' in metrics:
            return -1000.0
            
        # Configuration parameters - synchronized with evaluator.py
        ACCURACY_TOLERANCE = 0.03      # ±0.03 tolerance matching evaluator.py exactly
        
        accuracy = metrics.get('accuracy', 0.0)
        cost = metrics.get('cost_estimate', 999.0)
        success_rate = metrics.get('success_rate', 0.0)
        
        # Effective accuracy considering success rate
        effective_accuracy = accuracy * success_rate
        
        # Hard constraint: configuration must meet target accuracy threshold
        if effective_accuracy < self.target_accuracy - ACCURACY_TOLERANCE:
            # Heavy penalty for not meeting accuracy constraint
            violation_penalty = (self.target_accuracy - effective_accuracy) * 1000
            return -violation_penalty
        
        # For configurations meeting accuracy constraint, optimize for cost efficiency
        if cost <= 0:
            cost = 0.001
            
        # Primary objective: minimize cost (invert so lower cost = higher score)
        max_reasonable_cost = 0.1  # $0.10 per query
        cost_score = max(0, (max_reasonable_cost - cost) / max_reasonable_cost) * 100
        
        # Secondary objectives (smaller weights)
        retrieval_precision = metrics.get('retrieval_precision_at_k', 0.0)
        retrieval_recall = metrics.get('retrieval_recall_at_k', 0.0)
        retrieval_bonus = (retrieval_precision + retrieval_recall) * 5  # Small bonus
        
        # Bonus for exceeding accuracy target (encouraging better than minimum)
        accuracy_bonus = max(0, (effective_accuracy - self.target_accuracy) * 20)
        
        total_score = cost_score + retrieval_bonus + accuracy_bonus
        
        return total_score

def find_optimal_rag_config(target_accuracy: float = 0.20) -> Dict:
    """
    Main function that OpenEvolve will evolve.
    
    Problem: Given a target accuracy threshold, find the most cost-effective 
    RAG configuration that meets or exceeds that accuracy level.
    
    Args:
        target_accuracy: The target accuracy threshold to achieve
    
    The target accuracy can be ANY value - this makes the algorithm generic!
    Examples:
    - Low accuracy (0.05): Find cheapest config for basic functionality
    - Medium accuracy (0.15): Balance cost and quality
    - High accuracy (0.30): Premium quality regardless of cost
    
    Starts with constraint-aware random search baseline.
    
    This function is completely self-contained and provides full configuration access.
    """
    optimizer = RAGConfigurationOptimizer(target_accuracy=target_accuracy)
    
    # Run search with history tracking for evaluator
    result = optimizer.search_with_history_tracking()
    
    # Return comprehensive result with search history for cost calculation
    return {
        'final_config': result['best_config'],
        'final_metrics': result['best_metrics'],
        'search_history': result['evaluation_history'],  # For search cost calculation
        'total_steps': result['total_evaluated'],
        'target_accuracy': target_accuracy  # For evaluator reference
    }

# EVOLVE-BLOCK-END
