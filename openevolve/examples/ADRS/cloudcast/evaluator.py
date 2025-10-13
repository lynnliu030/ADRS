import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import signal
import json
import os
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from utils import *
from simulator import *
from broadcast import *
from initial_program import search_algorithm


def evaluate(program_path):
    """
    Evaluate the evolved broadcast optimization program across multiple configurations.
    
    Args:
        program_path: Path to the evolved program file
        
    Returns:
        Dictionary with evaluation metrics including required 'combined_score'
    """
    try:
        # Load the evolved program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "search_algorithm"):
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "Missing search_algorithm function"
            }
        
        # Configuration - individual JSON file paths
        config_files = [
            "examples/config/intra_aws.json",
            "examples/config/intra_azure.json", 
            "examples/config/intra_gcp.json",
            "examples/config/inter_agz.json",
            "examples/config/inter_gaz2.json"
        ]
        
        # Filter to only include files that exist
        existing_configs = [f for f in config_files if os.path.exists(f)]
        
        if not existing_configs:
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": f"No configuration files found. Checked: {config_files}"
            }
        
        num_vms = 2
        total_cost = 0.0
        total_transfer_time = 0.0
        successful_configs = 0
        failed_configs = 0
        
        # Process each configuration file
        for jsonfile in existing_configs:
            try:
                print(f"Processing config: {os.path.basename(jsonfile)}")
                
                # Load configuration
                with open(jsonfile, "r") as f:
                    config_name = os.path.basename(jsonfile).split(".")[0]
                    config = json.loads(f.read())

                # Create graph
                G = make_nx_graph(num_vms=int(num_vms))

                # Source and destination nodes
                source_node = config["source_node"]
                terminal_nodes = config["dest_nodes"]

                # Create output directory
                directory = f"paths/{config_name}"
                if not os.path.exists(directory):
                    Path(directory).mkdir(parents=True, exist_ok=True)

                # Run the evolved algorithm
                num_partitions = config["num_partitions"]
                bc_t = program.search_algorithm(source_node, terminal_nodes, G, num_partitions)

                bc_t.set_num_partitions(config["num_partitions"])
                
                # Save the generated paths
                outf = f"{directory}/search_algorithm.json"
                with open(outf, "w") as outfile:
                    outfile.write(
                        json.dumps(
                            {
                                "algo": "search_algorithm",
                                "source_node": bc_t.src,
                                "terminal_nodes": bc_t.dsts,
                                "num_partitions": bc_t.num_partitions,
                                "generated_path": bc_t.paths,
                            }
                        )
                    )

                # Evaluate the generated paths
                input_dir = f"paths/{config_name}"
                output_dir = f"evals/{config_name}"
                if not os.path.exists(output_dir):
                    Path(output_dir).mkdir(parents=True, exist_ok=True)

                # Run simulation
                simulator = BCSimulator(int(num_vms), output_dir)
                transfer_time, cost = simulator.evaluate_path(outf, config)
                
                # Accumulate results
                total_cost += cost
                total_transfer_time += transfer_time
                successful_configs += 1
                
                print(f"Config {config_name}: cost={cost:.2f}, time={transfer_time:.2f}")
                
            except Exception as e:
                print(f"Failed to process {os.path.basename(jsonfile)}: {str(e)}")
                failed_configs += 1
                break
        
        # Check if we have any successful evaluations
        if failed_configs != 0:
            return {
                "combined_score": 0.0,
                "runs_successfully": 0.0,
                "error": "1 or more configuration files failed to process"
            }
        
        # Calculate aggregate metrics
        avg_cost = total_cost / successful_configs
        success_rate = successful_configs / (successful_configs + failed_configs)
        
        print(f"Summary: {successful_configs} successful, {failed_configs} failed")
        print(f"Total cost: {total_cost:.2f}, Max transfer time: {total_transfer_time:.2f}")
        
        # Calculate metrics for OpenEvolve
        # Normalize scores (higher is better)
        time_score = 1.0 / (1.0 + total_transfer_time)  # Lower time = higher score
        cost_score = 1.0 / (1.0 + total_cost)  # Lower cost = higher score
        
        # Combined score considering total cost, max time, and success rate
        combined_score = cost_score
        
        return {
            "combined_score": combined_score,  # Required by OpenEvolve
            "runs_successfully": success_rate,
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "max_transfer_time": total_transfer_time,
            "successful_configs": successful_configs,
            "failed_configs": failed_configs,
            "time_score": time_score,
            "cost_score": cost_score,
            "success_rate": success_rate
        }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
            "combined_score": 0.0,  # Required by OpenEvolve
            "runs_successfully": 0.0,
            "error": str(e)
        }