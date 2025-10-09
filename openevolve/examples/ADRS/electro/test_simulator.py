#!/usr/bin/env python3
"""
Test script to verify the Electro simulator implementation works correctly.
Uses the self-contained evaluator for OpenEvolve compatibility.
"""
import sys
from pathlib import Path

# Import from the self-contained evaluator
from evaluator import ElectroDataLoader, ElectroSimulator

def test_data_loading():
    """Test that data loading works with a subset of directories"""
    print("Testing data loading...")
    
    # Test with just one directory first
    test_dirs = [
        "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_gird_search_bm25"
    ]
    
    data_loader = ElectroDataLoader(test_dirs)
    data_loader.load_experiment_data()
    
    stats = data_loader.get_configuration_bounds()
    print(f"Configuration space: {stats}")
    
    all_configs = data_loader.get_all_configurations()
    print(f"Total configurations loaded: {len(all_configs)}")
    
    if len(all_configs) > 0:
        print(f"Sample configuration: {all_configs[0]}")
    
    return data_loader

def test_simulator(data_loader):
    """Test simulator functionality"""
    print("\nTesting simulator...")
    
    simulator = ElectroSimulator(data_loader)
    
    # Get available configurations
    available_configs = simulator.get_available_configurations()
    print(f"Available configurations: {available_configs}")
    
    # Test with a real configuration
    all_configs = data_loader.get_all_configurations()
    if len(all_configs) > 0:
        test_config = all_configs[0]
        print(f"\nTesting configuration: {test_config}")
        
        metrics = simulator.run_experiment(test_config)
        print("Returned metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    # Test with invalid configuration
    print("\nTesting invalid configuration...")
    invalid_config = {
        'model': 'nonexistent_model',
        'embedding': 'nonexistent_embedding',
        'retrieval_k': 999,
        'benchmark': 'nonexistent_benchmark',
        'pipeline': 'vanilla_rag'
    }
    
    invalid_metrics = simulator.run_experiment(invalid_config)
    print(f"Invalid config result: {invalid_metrics.get('error', 'No error field')}")
    
    return simulator

def test_initial_program():
    """Test the initial program works"""
    print("\nTesting initial program...")
    
    # Import and test the initial program
    from initial_program import find_optimal_rag_config, RAGConfigurationOptimizer
    
    # Mock the injected functions
    def mock_evaluate_configuration(config):
        return {
            'accuracy': 0.15,
            'cost_estimate': 0.5,
            'success_rate': 1.0,
            'retrieval_precision_at_k': 0.1,
            'retrieval_recall_at_k': 0.12,
            'latency_p90': 1.2
        }
    
    def mock_get_available_configurations():
        return {
            'models': ['llama_3_8b', 'llama_3_70b'],
            'embeddings': ['bm25', 'financebench_e5_small'],
            'retrieval_k': [1, 5, 10, 20],
            'benchmarks': ['financeBench']
        }
    
    # Inject mock functions
    import initial_program
    initial_program.evaluate_configuration = mock_evaluate_configuration
    initial_program.get_available_configurations = mock_get_available_configurations
    
    # Test the optimizer
    optimizer = RAGConfigurationOptimizer()
    config_space = mock_get_available_configurations()
    optimizer.initialize_search_space(config_space)
    
    # Run with small budget
    best_config = optimizer.random_search(budget=5)
    print(f"Best config found: {best_config}")
    
    # Test main function
    result = find_optimal_rag_config()
    print(f"Main function result: {result}")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Electro Simulator Implementation")
    print("=" * 50)
    
    try:
        # Test data loading
        data_loader = test_data_loading()
        
        # Test simulator
        simulator = test_simulator(data_loader)
        
        # Test initial program
        test_initial_program()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
        # Print summary
        stats = simulator.get_stats()
        print(f"\nSimulator Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

