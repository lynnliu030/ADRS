"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import importlib.util
import numpy as np
import time
import os
import signal
import subprocess
import tempfile
import traceback
import sys
import pickle


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Function execution timed out")


def run_with_timeout(program_path, timeout_seconds=1200):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        centers, radii, sum_radii tuple from the program
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

# Debugging info
print(f"\033[92mRunning {program_path} in subprocess, Python version: {sys.version}\033[0m")

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the compression function
    print("\033[92mCalling compression()...\033[0m")
    output_dict = program.run_compression()
    print(f"\033[92mcompression() returned successfully: bitrate = {{output_dict['bitrate']}}\033[0m")

    # Save results to a file
    results = output_dict

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {temp_file.name}.results")
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    print(f"Error saved to {temp_file.name}.results")
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            # Always print output for debugging purposes
            print(f"Subprocess stdout: {stdout.decode()}")
            if stderr:
                print(f"Subprocess stderr: {stderr.decode()}")

            # Still raise an error for non-zero exit codes, but only after printing the output
            if exit_code != 0:
                raise RuntimeError(f"Process exited with code {exit_code}")

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error" in results:
                    raise RuntimeError(f"Program execution failed: {results['error']}")

                return results
            else:
                raise RuntimeError("Results file not found")

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            raise TimeoutError(f"Process timed out after {timeout_seconds} seconds")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)

def get_combined_score(bitrate, ptb_ppl, wikitext_ppl):
    """
    Get the combined score for the program
    Prioritizes: lower wikitext ppl > lower ptb ppl > lower bitrate
    Returns 0 if bitrate > 2.6 or wikitext ppl > 40
    """
    # Hard cutoffs
    if bitrate > 2.6 or wikitext_ppl > 10.5 or ptb_ppl > 18:
        return 0.0
    
    # Normalize metrics to 0-1 range (lower is better)
    # Assuming reasonable ranges: wikitext_ppl: 6-10.5, ptb_ppl: 10-18, bitrate: 2.0-2.6
    normalized_wikitext = max(0, 1 - (wikitext_ppl - 6) / 4.5)  # 6=1.0, 10.5=0.0
    normalized_ptb = max(0, 1 - (ptb_ppl - 10) / 8)  # 15=1.0, 18=0.0
    normalized_bitrate = max(0, 1 - (bitrate - 2.0) / 0.6)  # 2.0=1.0, 2.6=0.0

    # Weighted combination with priorities
    # wikitext ppl has highest priority (weight 0.5)
    # ptb ppl has medium priority (weight 0.1)
    # bitrate has lowest priority (weight 0.4)
    combined_score = (0.5 * normalized_wikitext + 
                     0.3 * normalized_ptb + 
                     0.2 * normalized_bitrate)
    
    return combined_score

def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """

    try:
        # For constructor-based approaches, a single evaluation is sufficient
        # since the result is deterministic
        start_time = time.time()

        # Use subprocess to run with timeout
        output_dict = run_with_timeout(
            program_path, timeout_seconds=300  # Single timeout
        )

        end_time = time.time()
        eval_time = end_time - start_time
        
        bitrate = float(output_dict['bitrate'])
        wikitext_ppl = float(output_dict['wikitext_ppl'])
        ptb_ppl = float(output_dict['ptb_ppl'])

        # Combined score - higher is better
        combined_score = get_combined_score(bitrate, ptb_ppl, wikitext_ppl)

        print(
            f"\033[94mEvaluation: bitrate={bitrate:.6f}, ptb_ppl={ptb_ppl:.6f}, wikitext_ppl={wikitext_ppl:.6f}, combined_score={combined_score:.6f}, time={eval_time:.2f}s\033[0m"
        )

        return {
            "bitrate": float(bitrate),
            "ptb_ppl": float(ptb_ppl),
            "wikitext_ppl": float(wikitext_ppl),
            "eval_time": float(eval_time),
            "combined_score": float(combined_score),
        }

    except Exception as e:
        print(f"\033[91mEvaluation failed completely: {str(e)}\033[0m")
        traceback.print_exc()
        return {
            "bitrate": 0.0,
            "ptb_ppl": 0.0,
            "wikitext_ppl": 0.0,
            "eval_time": 0.0,
            "combined_score": 0.0,
        }
