"""
Evaluator for multi-agent system using LLM judge to detect failure modes.
"""

import asyncio
import importlib.util
import logging
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

def chat_completion_request_openai(prompt, client):
    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_response = client.chat.completions.create(
    model='gpt-5-mini',
    messages=messages)
    if chat_response.choices:
        completion_text = chat_response.choices[0].message.content
    else:
        completion_text = None
    return completion_text

definitions = open(Path(Path(__file__).parent.parent.parent / "taxonomy_definitions_examples" / "definitions.txt"), "r").read()
examples = open(Path(Path(__file__).parent.parent.parent / "taxonomy_definitions_examples" / "examples.txt"), "r").read()

def openai_evaluator(trace, definitions=definitions, examples=examples, client=None):
    prompt = (
    "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
    "In the traces, analyze the system behaviour."
    "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
    "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
    "Also tell me whether the task is successfully completed or not, as a binary yes or no."
    "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
    "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
    "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
    "*** begin of things you should answer *** @@"
    "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
    "B. Whether the task is successfully completed or not: <yes or no>"
    "C. Whether you encounter any of the failure modes or inefficiencies:"
    "1.1 Disobey Task Specification: <yes or no>"
    "1.2 Disobey Role Specification: <yes or no>"
    "1.3 Step Repetition: <yes or no>"
    "1.4 Loss of Conversation History: <yes or no>"
    "1.5 Unaware of Termination Conditions: <yes or no>"
    "2.1 Conversation Reset: <yes or no>"
    "2.2 Fail to Ask for Clarification: <yes or no>"
    "2.3 Task Derailment: <yes or no>"
    "2.4 Information Withholding: <yes or no>"
    "2.5 Ignored Other Agent's Input: <yes or no>"
    "2.6 Action-Reasoning Mismatch: <yes or no>"
    "3.1 Premature Termination: <yes or no>"
    "3.2 No or Incorrect Verification: <yes or no>"
    "3.3 Weak Verification: <yes or no>"
    "@@*** end of your answer ***"
    "An example answer is: \n"
    "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
    "B. no \n"
    "C. \n"
    "1.1 no \n"
    "1.2 no \n"
    "1.3 no \n"
    "1.4 no \n"
    "1.5 no \n"
    "1.6 yes \n"
    "2.1 no \n"
    "2.2 no \n"
    "2.3 yes \n"
    "2.4 no \n"
    "2.5 no \n"
    "2.6 yes \n"
    "2.7 no \n"
    "3.1 no \n"
    "3.2 yes \n"
    "3.3 no \n"   
    "Here is the trace: \n"
    f"{trace}"
    "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
    f"{definitions} \n"
    "Here are some examples of the failure modes and inefficiencies: \n"
    f"{examples}"
)
    return chat_completion_request_openai(prompt, client)

class MASTLLMJudge:
    """LLM-as-a-Judge for MAST taxonomy evaluation."""
    
    def __init__(self, api_key: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("OpenAI package is required but not installed")
            
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def evaluate_trace(self, trace: str) -> Dict:
        # """Evaluate a trace and return failure mode analysis."""
        # # Truncate trace if too long
        # if len(trace) > 100000:
        #     trace = trace[:100000]
        
        # prompt = (
        #     "Analyze this multi-agent system trace for failure modes. "
        #     "Respond with yes/no for each failure mode:\n\n"
        #     "1.1 Disobey Task Specification: yes/no\n"
        #     "1.2 Disobey Role Specification: yes/no\n"
        #     "1.3 Step Repetition: yes/no\n"
        #     "1.4 Loss of Conversation History: yes/no\n"
        #     "1.5 Unaware of Termination Conditions: yes/no\n"
        #     "2.1 Conversation Reset: yes/no\n"
        #     "2.2 Fail to Ask for Clarification: yes/no\n"
        #     "2.3 Task Derailment: yes/no\n"
        #     "2.4 Information Withholding: yes/no\n"
        #     "2.5 Ignored Other Agent's Input: yes/no\n"
        #     "2.6 Action-Reasoning Mismatch: yes/no\n"
        #     "3.1 Premature Termination: yes/no\n"
        #     "3.2 No or Incorrect Verification: yes/no\n"
        #     "3.3 Weak Verification: yes/no\n\n"
        #     f"Trace:\n{trace}"
        # )
        
        try:
        #     response = self._call_llm(prompt)
        #     raw_response = response.choices[0].message.content

            raw_response = openai_evaluator(trace, client=self.client)

            failure_modes = self._parse_response_list([raw_response])
            
            return {
                "failure_modes": failure_modes,
                "total_failures": sum(failure_modes.values()),
                "raw_response": raw_response
            }
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Return zero failures on error
            failure_modes = {f'{i}.{j}': 0 for i in range(1,4) for j in range(1,7) if not (i==1 and j>5) and not (i==3 and j>3)}
            return {
                "failure_modes": failure_modes,
                "total_failures": 0,
                "raw_response": str(e)
            }

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM to evaluate the trace."""
        return self.client.chat.completions.create(
            model='o1',
            messages=[
                {"role": "system", "content": "You are an expert at analyzing multi-agent system failures."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
    
    def _parse_response(self, response: str) -> Dict[str, int]:
        """Parse LLM response to extract failure mode scores."""
        failure_modes = {
            '1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0,
            '2.1': 0, '2.2': 0, '2.3': 0, '2.4': 0, '2.5': 0, '2.6': 0,
            '3.1': 0, '3.2': 0, '3.3': 0
        }
        
        for mode in failure_modes.keys():
            pattern = rf"{mode}.*?(yes|no)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                failure_modes[mode] = 1 if match.group(1).lower() == 'yes' else 0
        
        return failure_modes

    def _parse_response_list(self, responses: List[str]) -> Dict[str, int]:
        """
        Parse the LLM responses to extract yes/no answers for each failure mode.
        
        Args:
            responses: List of LLM responses evaluating traces
            
        Returns:
            Dictionary mapping failure mode codes to lists of binary values (0 for no, 1 for yes)
        """
        import re
        
        # Initialize dictionary with empty lists for each failure mode
        failure_modes = {
            '1.1': [], '1.2': [], '1.3': [], '1.4': [], '1.5': [],
            '2.1': [], '2.2': [], '2.3': [], '2.4': [], '2.5': [], '2.6': [],
            '3.1': [], '3.2': [], '3.3': []
        }
        
        for i, response in enumerate(responses):
            try:
                # Clean up the response - remove @@ markers if present
                cleaned_response = response.strip()
                if cleaned_response.startswith('@@'):
                    cleaned_response = cleaned_response[2:]
                if cleaned_response.endswith('@@'):
                    cleaned_response = cleaned_response[:-2]
                
                # Process each failure mode
                for mode in failure_modes.keys():
                    # Various patterns to match different response formats
                    patterns = [
                        # Format with C. prefix and colon
                        rf"C\..*?{mode}.*?(yes|no)",
                        # Format with just C prefix without dot
                        rf"C{mode}\s+(yes|no)",
                        # Format with mode directly (with or without spaces)
                        rf"{mode}\s*[:]\s*(yes|no)",
                        rf"{mode}\s+(yes|no)",
                        # Format with newlines
                        rf"{mode}\s*\n\s*(yes|no)",
                        # Format with C prefix and newlines
                        rf"C\.{mode}\s*\n\s*(yes|no)"
                    ]
                    
                    found = False
                    for pattern in patterns:
                        matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                        if matches:
                            # Use the first match
                            value = 1 if matches[0].lower() == 'yes' else 0
                            failure_modes[mode].append(value)
                            found = True
                            break
                    
                    if not found:
                        # If we still can't find a match, try a more general approach
                        # Look for the mode number followed by any text and then yes/no
                        general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                        match = re.search(general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                        
                        if match:
                            value = 1 if match.group(1).lower() == 'yes' else 0
                            failure_modes[mode].append(value)
                        else:
                            # If all attempts fail, default to 'no'
                            print(f"Warning: Could not find mode {mode} in response {i}")
                            failure_modes[mode].append(0)
                        
            except Exception as e:
                print(f"Error processing response {i}: {e}")
                # If there's an error, default to 'no' for all modes for this response
                for mode in failure_modes:
                    if len(failure_modes[mode]) <= i:  # Only append if we haven't already
                        failure_modes[mode].append(0)
        
        # Ensure all lists have the same length
        max_length = max(len(values) for values in failure_modes.values())
        for mode in failure_modes:
            if len(failure_modes[mode]) < max_length:
                failure_modes[mode].extend([0] * (max_length - len(failure_modes[mode])))
        
        failure_modes_numeric = {
            '1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0,
            '2.1': 0, '2.2': 0, '2.3': 0, '2.4': 0, '2.5': 0, '2.6': 0,
            '3.1': 0, '3.2': 0, '3.3': 0
        }

        if len(responses) == 1:
            for mode in failure_modes_numeric:
                failure_modes_numeric[mode] = sum(failure_modes[mode])
        else:
            for mode in failure_modes_numeric:
                failure_modes_numeric[mode] = sum(failure_modes[mode]) / len(failure_modes[mode])

        return failure_modes_numeric

class ProgramDevDataset:
    """Load and manage the programdev dataset."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "example_mas" / "programdev"
        self.data_dir = Path(data_dir)
        self.tasks = []
        self._load_tasks()
    
    def _load_tasks(self):
        """Load all tasks from the dataset."""
        name_files = sorted(self.data_dir.glob("names_*.txt"))
        
        for name_file in name_files:
            index = name_file.stem.split("_")[1]
            desc_file = self.data_dir / f"descriptions_{index}.txt"
            
            if desc_file.exists():
                with open(name_file, 'r') as f:
                    name = f.read().strip()
                with open(desc_file, 'r') as f:
                    description = f.read().strip()
                
                if name and description:
                    self.tasks.append({
                        'index': index,
                        'name': name,
                        'description': description
                    })
        
        logger.info(f"Loaded {len(self.tasks)} tasks from {self.data_dir}")
    
    def sample_tasks(self, n: int = 3) -> List[Dict]:
        """Sample n random tasks from the dataset."""
        if n > len(self.tasks):
            n = len(self.tasks)
        return random.sample(self.tasks, n)


def evaluate(program_path: str) -> Dict:
    """
    Evaluate the multi-agent program by running it on sample tasks and using LLM judge.
    """
    try:
        # Load the program module
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if required function exists
        if not hasattr(program, 'run_multi_agent_task'):
            return {'combined_score': 0.0, 'error': 'Missing run_multi_agent_task function'}
        
        # Initialize components
        dataset = ProgramDevDataset()
        judge = MASTLLMJudge()
        sample_tasks = dataset.sample_tasks(6)  # Evaluate on 3 random tasks
        
        if not sample_tasks:
            return {'combined_score': 0.0, 'error': 'No tasks in dataset'}
        
        # Run evaluation on each task
        total_failures = 0
        successful_runs = 0
        
        for task in sample_tasks:
            try:
                # Create temporary log file for the trace
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as tmp_file:
                    log_file = tmp_file.name
                
                # Run the multi-agent task with timeout
                async def run_with_timeout():
                    try:
                        return await asyncio.wait_for(
                            program.run_multi_agent_task(
                                idea=task['description'],
                                log_file=log_file
                            ),
                            timeout=60
                        )
                    except asyncio.TimeoutError:
                        return None
                
                trace = asyncio.run(run_with_timeout())
                
                if trace is None:
                    total_failures += 7  # Penalize timeout
                    continue
                
                # Evaluate the trace with LLM judge
                evaluation = judge.evaluate_trace(trace)
                total_failures += evaluation['total_failures']
                successful_runs += 1
                
                logger.info(f"Task {task['name']}: {evaluation['total_failures']} failures")
                
                # Clean up temp file
                try:
                    os.unlink(log_file)
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Error evaluating task {task['name']}: {str(e)}")
                total_failures += 14
        
        # Calculate combined score
        if successful_runs > 0:
            avg_failures_per_task = total_failures / successful_runs
            combined_score = 1.0 / (1.0 + avg_failures_per_task)
        else:
            avg_failures_per_task = 14.0  # Max possible failures
            combined_score = 0.0
        
        return {
            'combined_score': float(combined_score),
            'avg_failures_per_task': float(avg_failures_per_task),
            'total_failures': int(total_failures),
            'successful_runs': int(successful_runs)
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {'combined_score': 0.0, 'error': str(e)}


def evaluate_stage1(program_path: str) -> Dict:
    """Stage 1: Quick validation - check if program runs without errors."""
    try:
        # Load the program module
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if required function exists
        if not hasattr(program, 'run_multi_agent_task'):
            return {
                'runs_successfully': 0.0, 
                'combined_score': 0.0,
                'avg_failures_per_task': 14.0,
                'error': 'Missing run_multi_agent_task function'
            }
        
        # Try to run on a simple test task
        dataset = ProgramDevDataset()
        sample_tasks = dataset.sample_tasks(1)
        
        if not sample_tasks:
            return {
                'runs_successfully': 0.0, 
                'combined_score': 0.0,
                'avg_failures_per_task': 14.0,
                'error': 'No tasks in dataset'
            }
        
        task = sample_tasks[0]
        
        # Create a temporary log file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as tmp_file:
            log_file = tmp_file.name
        
        # Try to run with very short timeout
        async def quick_test():
            try:
                await asyncio.wait_for(
                    program.run_multi_agent_task(
                        idea=task['description'][:100],  # Use truncated description
                        n_rounds=1,
                        log_file=log_file
                    ),
                    timeout=10
                )
                return True
            except Exception:
                return False
        
        success = asyncio.run(quick_test())
        
        # Clean up
        try:
            os.unlink(log_file)
        except:
            pass
        
        if success:
            return {
                'runs_successfully': 1.0, 
                'overall_score': 0.5,
                'combined_score': 0.5,
                'avg_failures_per_task': 7.0
            }
        else:
            return {
                'runs_successfully': 0.5, 
                'overall_score': 0.25,
                'combined_score': 0.1,
                'avg_failures_per_task': 12.0
            }
            
    except Exception as e:
        return {
            'runs_successfully': 0.0, 
            'combined_score': 0.0,
            'avg_failures_per_task': 14.0,
            'error': str(e)
        }


def evaluate_stage2(program_path: str) -> Dict:
    """Stage 2: Full evaluation with LLM judge."""
    return evaluate(program_path)


if __name__ == "__main__":
    import sys
    program_file = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    
    print(f"Evaluating {program_file}...")
    print("Stage 1:", evaluate_stage1(program_file))
    print("Stage 2:", evaluate_stage2(program_file))