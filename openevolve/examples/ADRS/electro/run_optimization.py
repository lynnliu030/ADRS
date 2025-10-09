#!/usr/bin/env python3
"""
Run script for Electro RAG configuration optimization using OpenEvolve.
"""
import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run Electro RAG optimization with OpenEvolve')
    parser.add_argument('--iterations', type=int, default=200,
                       help='Number of evolution iterations (default: 200)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='openevolve_output',
                       help='Output directory for results (default: openevolve_output)')
    parser.add_argument('--test-run', action='store_true',
                       help='Run a quick test with 10 iterations')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    initial_program = script_dir / 'initial_program.py'
    evaluator = script_dir / 'evaluator.py'
    config_file = script_dir / args.config
    
    # Check required files exist
    if not initial_program.exists():
        print(f"Error: Initial program not found: {initial_program}")
        sys.exit(1)
    if not evaluator.exists():
        print(f"Error: Evaluator not found: {evaluator}")
        sys.exit(1)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    # Check OpenEvolve is available
    try:
        import openevolve
        print(f"Using OpenEvolve from: {openevolve.__file__}")
    except ImportError:
        print("Error: OpenEvolve not found. Please install it:")
        print("cd /home/melissa/openevolve && pip install -e .")
        sys.exit(1)
    
    # Check API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Please set it for LLM access:")
        print("export OPENAI_API_KEY=your-api-key-here")
        
    # Adjust iterations for test run
    if args.test_run:
        iterations = 10
        print("Running test mode with 10 iterations")
    else:
        iterations = args.iterations
    
    # Build command
    openevolve_run = Path("/home/melissa/openevolve/openevolve-run.py")
    if not openevolve_run.exists():
        print(f"Error: OpenEvolve run script not found: {openevolve_run}")
        sys.exit(1)
    
    cmd_parts = [
        "python", str(openevolve_run),
        str(initial_program),
        str(evaluator),
        "--config", str(config_file),
        "--iterations", str(iterations)
    ]
    
    if args.checkpoint:
        cmd_parts.extend(["--checkpoint", args.checkpoint])
        
    # Set output directory
    output_dir = script_dir / args.output_dir
    os.environ['OPENEVOLVE_OUTPUT_DIR'] = str(output_dir)
    
    print("=" * 60)
    print("Electro RAG Configuration Optimization")
    print("=" * 60)
    print(f"Initial program: {initial_program}")
    print(f"Evaluator: {evaluator}")
    print(f"Config: {config_file}")
    print(f"Iterations: {iterations}")
    print(f"Output directory: {output_dir}")
    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
    print("=" * 60)
    
    # Run the command
    cmd_str = " ".join(cmd_parts)
    print(f"Running: {cmd_str}")
    print()
    
    os.system(cmd_str)

if __name__ == "__main__":
    main()
