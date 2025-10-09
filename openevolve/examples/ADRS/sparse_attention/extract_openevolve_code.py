#!/usr/bin/env python3
"""
Script to extract Python code from OpenEvolve checkpoint metadata.json files.

This script reads the metadata.json files from OpenEvolve checkpoints and extracts
the best program's code, saving it as a Python file.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Parsed JSON data as a dictionary.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {e}")


def extract_code_from_checkpoint(checkpoint_dir: Path, output_dir: Optional[Path] = None) -> list[Path]:
    """Extract all programs' code from a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory.
        output_dir: Directory to save the extracted code. If None, uses the checkpoint directory.
        
    Returns:
        List of paths to the extracted Python files.
        
    Raises:
        FileNotFoundError: If required files are missing.
        KeyError: If the metadata structure is unexpected.
    """
    # Load metadata.json
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = load_json_file(metadata_path)
    
    # Get all program IDs from islands and archive
    all_program_ids = set()
    
    # Add programs from islands
    islands = metadata.get("islands", [])
    for island in islands:
        if isinstance(island, list):
            all_program_ids.update(island)
        elif island:  # Handle case where island might be a single ID
            all_program_ids.add(island)
    
    # Add programs from archive
    archive = metadata.get("archive", [])
    if isinstance(archive, list):
        all_program_ids.update(archive)
    elif archive:  # Handle case where archive might be a single ID
        all_program_ids.add(archive)
    
    # Remove None values
    all_program_ids.discard(None)
    
    if not all_program_ids:
        raise KeyError(f"No program IDs found in {metadata_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = checkpoint_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_files = []
    checkpoint_name = checkpoint_dir.name
    
    # Extract code from each program
    for program_id in sorted(all_program_ids):
        try:
            # Load the program JSON file
            program_file = checkpoint_dir / "programs" / f"{program_id}.json"
            program_data = load_json_file(program_file)
            
            # Extract the code
            code = program_data.get("code")
            if not code:
                print(f"Warning: No 'code' field found in {program_file}")
                continue
            
            # Generate output filename
            output_file = output_dir / f"{checkpoint_name}_{program_id}.py"
            
            # Write the code to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            print(f"Extracted code from {checkpoint_dir} program {program_id} to {output_file}")
            extracted_files.append(output_file)
            
        except Exception as e:
            print(f"Error extracting program {program_id} from {checkpoint_dir}: {e}")
            continue
    
    return extracted_files


def extract_all_checkpoints(openevolve_output_dir: Path, output_dir: Optional[Path] = None) -> list[Path]:
    """Extract code from all checkpoints in the openevolve_output directory.
    
    Args:
        openevolve_output_dir: Path to the openevolve_output directory.
        output_dir: Directory to save extracted code files. If None, saves in each checkpoint directory.
        
    Returns:
        List of paths to extracted Python files.
    """
    checkpoints_dir = openevolve_output_dir / "checkpoints"
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    extracted_files = []
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("_")[1]))  # Sort by checkpoint number
    
    for checkpoint_dir in checkpoint_dirs:
        try:
            extracted_files_list = extract_code_from_checkpoint(checkpoint_dir, output_dir)
            extracted_files.extend(extracted_files_list)
        except Exception as e:
            print(f"Error extracting code from {checkpoint_dir}: {e}")
            continue
    
    return extracted_files


def main():
    """Main function to handle command line arguments and execute the extraction."""
    parser = argparse.ArgumentParser(
        description="Extract Python code from OpenEvolve checkpoint metadata.json files"
    )
    parser.add_argument(
        "openevolve_output_dir",
        type=Path,
        help="Path to the openevolve_output directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save extracted code files (default: save in each checkpoint directory)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Extract from specific checkpoint (e.g., 'checkpoint_1')"
    )
    
    args = parser.parse_args()
    
    try:
        if args.checkpoint:
            # Extract from specific checkpoint
            checkpoint_dir = args.openevolve_output_dir / "checkpoints" / args.checkpoint
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
            extracted_files = extract_code_from_checkpoint(checkpoint_dir, args.output_dir)
            print(f"Successfully extracted {len(extracted_files)} programs from checkpoint {args.checkpoint}")
        else:
            # Extract from all checkpoints
            extracted_files = extract_all_checkpoints(args.openevolve_output_dir, args.output_dir)
            print(f"Successfully extracted {len(extracted_files)} programs from all checkpoints")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 