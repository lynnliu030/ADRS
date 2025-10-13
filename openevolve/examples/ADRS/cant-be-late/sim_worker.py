"""
Worker utilities for running a single simulation.

Kept separate from the evaluator so ProcessPoolExecutor can pickle
the callable by module name (avoids evaluation_module import quirks).
"""

import os
import sys
import subprocess
import re
from typing import Dict, List, Any

# Repository root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "simulator")


class SimulationFailure(Exception):
    def __init__(self, error_msg: str, stdout: str, stderr: str):
        self.stdout = stdout
        self.stderr = stderr
        self.error_msg = error_msg
        super().__init__(f"{error_msg}\nSTDOUT: {stdout}\nSTDERR: {stderr}")


def run_single_simulation(program_path: str, trace_file: str, config: dict):
    """Run a single simulation.

    Returns: (success: bool, cost: float, error_msg: str, detailed_info: dict)
    """
    trace_file = os.path.abspath(trace_file)

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "main.py"),
        f"--strategy-file={program_path}",
        "--env=trace",
        f"--trace-file={trace_file}",
        f"--task-duration-hours={config['duration']}",
        f"--deadline-hours={config['deadline']}",
        f"--restart-overhead-hours={config['overhead']}",
        "--silent",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            error_msg = f"Run failed for {os.path.basename(trace_file)}\n"
            detailed_info = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "warnings": _extract_warnings(result.stderr),
                "errors": _extract_errors(result.stderr)
            }
            return False, 0.0, str(SimulationFailure(error_msg, result.stdout, result.stderr)), detailed_info

        if "mean:" not in result.stdout:
            error_msg = f"No 'mean:' found in output for {os.path.basename(trace_file)}\n"
            detailed_info = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "warnings": _extract_warnings(result.stderr),
                "errors": _extract_errors(result.stderr)
            }
            return False, 0.0, str(SimulationFailure(error_msg, result.stdout, result.stderr)), detailed_info

        for line in result.stdout.splitlines():
            if "mean:" in line:
                try:
                    cost_str = line.split("mean:")[1].split(";")[0].strip()
                    cost = float(cost_str)

                    # Extract additional statistics from the output
                    detailed_info = _extract_simulation_details(result.stdout, result.stderr)
                    detailed_info.update({
                        "cost": cost,
                        "trace_file": os.path.basename(trace_file),
                        "config": config
                    })

                    return True, cost, "", detailed_info
                except Exception as e:
                    error_msg = f"Failed to parse cost from line: {line}\nError: {e}"
                    detailed_info = {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "warnings": _extract_warnings(result.stderr),
                        "errors": _extract_errors(result.stderr)
                    }
                    return False, 0.0, str(SimulationFailure(error_msg, result.stdout, result.stderr)), detailed_info

        detailed_info = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "warnings": _extract_warnings(result.stderr),
            "errors": _extract_errors(result.stderr)
        }
        return False, 0.0, str(SimulationFailure("Could not find cost in output", result.stdout, result.stderr)), detailed_info

    except subprocess.TimeoutExpired as e:
        detailed_info = {
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "timeout": True,
            "warnings": [],
            "errors": ["Simulation timeout"]
        }
        return False, 0.0, f"Timeout on trace {os.path.basename(trace_file)}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}", detailed_info
    except Exception as e:
        detailed_info = {
            "stdout": "",
            "stderr": str(e),
            "warnings": [],
            "errors": [str(e)]
        }
        return False, 0.0, f"Error on trace {os.path.basename(trace_file)}: {e}", detailed_info


def _extract_warnings(stderr: str) -> List[str]:
    """Extract warning messages from stderr."""
    warnings = []
    for line in stderr.splitlines():
        if "WARNING" in line.upper() or "WARN" in line.upper():
            warnings.append(line.strip())
        # Check for specific discrete time related warnings
        if any(keyword in line.lower() for keyword in ["discrete", "tick", "gap_seconds", "math.ceil", "math.floor"]):
            warnings.append(f"DISCRETE_TIME_HINT: {line.strip()}")
    return warnings


def _extract_errors(stderr: str) -> List[str]:
    """Extract error messages from stderr."""
    errors = []
    for line in stderr.splitlines():
        if any(keyword in line.upper() for keyword in ["ERROR", "EXCEPTION", "TRACEBACK", "FAILED"]):
            errors.append(line.strip())
        # Check for common strategy implementation errors
        if any(keyword in line.lower() for keyword in ["attributeerror", "syntaxerror", "indentationerror"]):
            errors.append(f"IMPLEMENTATION_ERROR: {line.strip()}")
    return errors


def _extract_simulation_details(stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract detailed simulation information from stdout and stderr."""
    details = {
        "warnings": _extract_warnings(stderr),
        "errors": _extract_errors(stderr),
        "stdout_summary": "",
        "performance_stats": {},
        "spot_availability": None
    }

    stdout_lines = stdout.splitlines()

    # Extract cost statistics if available
    for line in stdout_lines:
        if "mean:" in line:
            details["stdout_summary"] = line.strip()
            # Try to extract std, p99, p90 if available
            try:
                parts = line.split(";")
                for part in parts:
                    part = part.strip()
                    if "std:" in part:
                        std_str = part.split("std:")[1].strip()
                        details["performance_stats"]["std_cost"] = float(std_str)
                    elif "worst 1%:" in part:
                        p99_str = part.split("worst 1%:")[1].strip()
                        details["performance_stats"]["p99_cost"] = float(p99_str)
                    elif "worst 10%:" in part:
                        p90_str = part.split("worst 10%:")[1].strip()
                        details["performance_stats"]["p90_cost"] = float(p90_str)
            except Exception:
                pass  # If parsing fails, just skip

    # Extract SPOT availability statistics from stderr if available
    for line in stderr.splitlines():
        if "Preempted at" in line:
            # Could track preemption events, but for now just note occurrence
            pass

    # Check for discrete time usage indicators
    discrete_indicators = []
    all_text = stdout + stderr
    if "math.ceil" in all_text:
        discrete_indicators.append("uses_math_ceil")
    if "math.floor" in all_text:
        discrete_indicators.append("uses_math_floor")
    if "gap_seconds" in all_text:
        discrete_indicators.append("uses_gap_seconds")
    if "tick" in all_text:
        discrete_indicators.append("mentions_ticks")

    details["discrete_time_indicators"] = discrete_indicators
    details["likely_uses_discrete_time"] = len(discrete_indicators) >= 2

    return details
