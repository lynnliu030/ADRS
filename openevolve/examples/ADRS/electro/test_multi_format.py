#!/usr/bin/env python3
"""
Test script to verify the refined data loader works with multiple experiment formats.
Uses the self-contained evaluator for OpenEvolve compatibility.
"""
import sys
from pathlib import Path

# Import from the self-contained evaluator
from evaluator import ElectroDataLoader, ElectroSimulator

def test_multi_format_loading():
    """Test that data loading works with multiple experiment formats"""
    print("Testing multi-format data loading...")
    
    # Test with multiple directories including different formats
    test_dirs = [
        "/home/melissa/openevolve/electro/outputs/financebench_vrag_llama_gird_search_bm25",
        "/home/melissa/openevolve/electro/outputs/frames_llama70b_grid_search_poisson_2",
        "/home/melissa/openevolve/electro/outputs/TQA_KILT_llama8b_grid_search_poisson_6",
    ]
    
    data_loader = ElectroDataLoader(test_dirs)
    data_loader.load_experiment_data()
    
    stats = data_loader.get_configuration_bounds()
    print(f"Configuration space: {stats}")
    
    all_configs = data_loader.get_all_configurations()
    print(f"Total configurations loaded: {len(all_configs)}")
    
    # Group by format/benchmark
    by_benchmark = {}
    for config in all_configs:
        benchmark = config['benchmark']
        if benchmark not in by_benchmark:
            by_benchmark[benchmark] = []
        by_benchmark[benchmark].append(config)
    
    print("\nConfigurations by benchmark:")
    for benchmark, configs in by_benchmark.items():
        print(f"  {benchmark}: {len(configs)} configurations")
        if configs:
            sample = configs[0]
            print(f"    Sample: {sample}")
    
    return data_loader, all_configs

def test_simulator_with_multi_format(data_loader, all_configs):
    """Test simulator with configurations from different formats"""
    print("\nTesting simulator with multi-format data...")
    
    simulator = ElectroSimulator(data_loader)
    
    # Test configurations from different benchmarks
    benchmarks_tested = set()
    
    for config in all_configs:
        benchmark = config['benchmark']
        if benchmark not in benchmarks_tested:
            print(f"\nTesting {benchmark} configuration: {config}")
            
            metrics = simulator.run_experiment(config)
            
            print("Key metrics:")
            key_metrics = ['accuracy', 'cost_estimate', 'retrieval_precision_at_k', 
                          'retrieval_recall_at_k', 'experiment_format']
            for key in key_metrics:
                if key in metrics:
                    print(f"  {key}: {metrics[key]}")
            
            benchmarks_tested.add(benchmark)
            
            # Test only first few benchmarks to keep output manageable
            if len(benchmarks_tested) >= 3:
                break
    
    return simulator

def main():
    """Run all multi-format tests"""
    print("=" * 60)
    print("Testing Multi-Format Electro Data Loading")
    print("=" * 60)
    
    try:
        # Test data loading
        data_loader, all_configs = test_multi_format_loading()
        
        # Test simulator
        simulator = test_simulator_with_multi_format(data_loader, all_configs)
        
        print("\n" + "=" * 60)
        print("Multi-format tests completed successfully!")
        print("=" * 60)
        
        # Print final summary
        stats = simulator.get_stats()
        print(f"\nFinal Statistics:")
        for key, value in stats.items():
            if key != 'available_configs':  # Skip the large config dict
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"\nMulti-format test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

