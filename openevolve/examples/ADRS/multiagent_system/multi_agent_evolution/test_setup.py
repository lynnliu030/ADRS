#!/usr/bin/env python3
"""
Test script to verify the multi-agent evolution setup works correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

def test_initial_program():
    """Test that the initial program runs without errors"""
    print("Testing Initial Program...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import initial_program
        
        print("✓ Initial program imported")
        
        if hasattr(initial_program, 'run_multi_agent_task'):
            print("✓ run_multi_agent_task function found")
        else:
            print("✗ run_multi_agent_task function missing")
            return False
        
        # Test simple run
        async def test_run():
            return await initial_program.run_multi_agent_task(
                idea="Create a hello world function",
                n_rounds=1,
                log_file="/tmp/test_trace.log"
            )
        
        trace = asyncio.run(test_run())
        
        if trace:
            print(f"✓ Program executed, trace length: {len(trace)}")
        
        if os.path.exists("/tmp/test_trace.log"):
            os.unlink("/tmp/test_trace.log")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_evaluator():
    """Test that the evaluator works"""
    print("Testing Evaluator...")
    
    try:
        import evaluator
        
        print("✓ Evaluator imported")
        
        dataset = evaluator.ProgramDevDataset()
        print(f"✓ Dataset loaded with {len(dataset.tasks)} tasks")
        
        # Test stage 1 evaluation
        initial_program_path = Path(__file__).parent / "initial_program.py"
        stage1_result = evaluator.evaluate_stage1(str(initial_program_path))
        print(f"✓ Stage 1 result: {stage1_result}")
        
        if stage1_result.get('runs_successfully', 0) > 0.5:
            print("✓ Initial program passes Stage 1")
            return True
        else:
            print("✗ Initial program fails Stage 1")
            return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run tests"""
    print("Multi-Agent Evolution Setup Test\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not set")
        print("  Set it with: export OPENAI_API_KEY='your-key'")
    
    tests = [
        ("Initial Program", test_initial_program),
        ("Evaluator", test_evaluator),
    ]
    
    passed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✓ {test_name} PASSED\n")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED\n")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}\n")
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Ready to run OpenEvolve!")
        print("   python openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 10")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)