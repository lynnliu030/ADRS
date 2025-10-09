#!/usr/bin/env python3
"""
Evaluator for openevolve
"""

import os
import sys
import torch
from pathlib import Path

import importlib.util
import inspect
from types import ModuleType
from unittest.mock import patch





def read_micro_metrics(result_dir: str):
    """
    Read micro metrics from JSONL file and compute average density and error.
    
    Args:
        result_dir (str): Path to the results directory containing micro_metrics.jsonl
        
    Returns:
        dict: Dictionary containing computed metrics with keys:
            - average_density: Average attention density across all layers
            - average_error: Average attention output error across all layers
    """
    import json
    from pathlib import Path
    
    # Find the micro_metrics.jsonl file in the result directory
    result_path: Path = Path(result_dir)
    micro_metrics_file: Path | None = None
    for path in result_path.rglob("micro_metrics.jsonl"):
        micro_metrics_file = path
        break
    if micro_metrics_file is None or not micro_metrics_file.exists():
        # Debug: list all files in the directory
        print(f"Debug: Files in {result_dir}:")
        for file_path in result_path.rglob("*"):
            print(f"  {file_path}")
        raise FileNotFoundError(f"micro_metrics.jsonl not found in {result_dir}")
    
    density_values: list[float] = []
    error_values: list[float] = []
    
    # Read the JSONL file line by line
    with open(micro_metrics_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                metric_type = data.get("metric")
                value = data.get("value")
                
                if metric_type == "research_attention_density" and value is not None:
                    density_values.append(float(value))
                elif metric_type == "research_attention_output_error" and value is not None:
                    error_values.append(float(value))
                    
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue
    
    # Compute averages
    average_density = sum(density_values) / len(density_values) if density_values else 0.0
    average_error = sum(error_values) / len(error_values) if error_values else 0.0
    
    metrics = {
        "density": - average_density,
        "error": - average_error,
        "combined_score" : - (average_error + average_density) / 2
    }
    print("combined_score", metrics["combined_score"], flush=True)
    
    return metrics

def run_benchmark_and_collect_metrics(sparse_attention_config):
    """
    Run benchmark using the Benchmark class directly and collect metrics.
    
    Args:
        sparse_attention_config: The sparse attention configuration to test
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    import tempfile
    from pathlib import Path
    from sparse_attention_hub.adapters import ModelAdapterHF
    from benchmark import LongBench
    from benchmark.ruler import Ruler
    from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
    
    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create temporary directory for results with incremental numbering
    base_result_dir = Path("./openevolve_results/")
    base_result_dir.mkdir(exist_ok=True)
    
    # Find the next available directory number
    existing_dirs = []
    for item in base_result_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            existing_dirs.append(int(item.name))
    
    next_dir_num = 1
    if existing_dirs:
        next_dir_num = max(existing_dirs) + 1
    
    result_dir = base_result_dir / str(next_dir_num)
    result_dir.mkdir(exist_ok=True)
    
    print(f"Running benchmarks in {result_dir}")
    
    # Setup metric logger
    logger = MicroMetricLogger()
    logger.configure_logging(
        log_path=str(result_dir),
        enabled_metrics=["research_attention_density", "research_attention_output_error"]
    )
    
    # Load model with sparse attention configuration
    print("  ✓ Loading model...")
    adapter = ModelAdapterHF(
        model_name=model_name,
        sparse_attention_config=sparse_attention_config,
        model_kwargs={
            "torch_dtype": torch.bfloat16, 
            "attn_implementation": "flash_attention_2"
        },
        generate_kwargs={
            "max_new_tokens": 20,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": None,
        },
        device=device
    )
    
    
    # Setup benchmarks (minimal subsets for quick signal)
    benchmarks = [
        ("longbench", LongBench(["passage_retrieval_en"])),
        ("ruler", Ruler(["4096"]))
    ]

    # Run each benchmark into its own subdirectory and collect micro-metrics
    collected: list[dict] = []
    for name, bench in benchmarks:
        bench_dir = result_dir / name
        bench_dir.mkdir(exist_ok=True)
        
        # Configure logger for this specific benchmark
        bench_logger = MicroMetricLogger()
        bench_logger.configure_logging(
            log_path=str(bench_dir),
            enabled_metrics=["research_attention_density", "research_attention_output_error"]
        )
        
        bench.run_benchmark(
            adapter,
            bench_dir,
            request_kwargs={
                "max_requests": 2,
                "max_context_length": 16000
            }
        )
        bench_logger.flush()
        collected.append(read_micro_metrics(str(bench_dir)))

    # Aggregate metrics across benchmarks by simple average
    if collected:
        avg_density = sum(m["density"] for m in collected) / len(collected)
        avg_error = sum(m["error"] for m in collected) / len(collected)
        metrics = {
            "density": avg_density,
            "error": avg_error,
            "combined_score": (avg_density + avg_error) / 2,
        }
    else:
        metrics = {"density": 0.0, "error": 0.0, "combined_score": 0.0}

    print(metrics)
    del adapter.model
    del adapter
    del logger
    torch.cuda.empty_cache()
    return metrics

def evaluate(program_path: str):
    old_path = os.getcwd()
    os.chdir("/workspace/audrey/sparse-attention-hub/")
    sys.path.insert(0, "/workspace/audrey/sparse-attention-hub/")

    if not os.path.isfile(program_path):
        raise FileNotFoundError(f"No such file: {program_path}")
    
    module_name = "new_openevolve_masker"
    


    # Import the masker registry and the original OpenEvolveMaskerConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.base import MaskerRegistry
    from sparse_attention_hub.sparse_attention.research_attention.base import ResearchAttentionConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.openevolve.openevolve_masker import OpenEvolveMaskerConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import LocalMaskerConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import SinkMaskerConfig
    from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import OracleTopKConfig

    spec = importlib.util.spec_from_file_location(module_name, program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Replace the registry entry with the new masker class
    MaskerRegistry._registry[OpenEvolveMaskerConfig] = module.OpenEvolveMasker

    # Use the original config class (which is picklable) but with the new masker class
    sparse_attention_config = ResearchAttentionConfig(masker_configs=[
        LocalMaskerConfig(window_size=0.001),
        SinkMaskerConfig(sink_size=0.001),
        OracleTopKConfig(heavy_size=0.05),
        OpenEvolveMaskerConfig(),
    ])
 
    try:
        metrics = run_benchmark_and_collect_metrics(sparse_attention_config)
        
    except Exception as e:
        print(f"Error running benchmark: {e}", flush=True)
        # release gpu resoruces
        torch.cuda.empty_cache()
        metrics = {
            "density": 100,
            "error": 100,
            "combined_score": -100
        }
    
    os.chdir(old_path)
    return metrics


if __name__ == "__main__":

    import sys
    filename = sys.argv[1]
    evaluate(filename)
