# Multi-Agent System Evolution

Use OpenEvolve to automatically improve multi-agent systems by minimizing failure modes detected by an LLM judge.

## What It Does

1. Runs your multi-agent system on programming tasks from `example_mas/programdev`
2. Captures execution traces of agent interactions  
3. Uses LLM judge to detect 14 types of failure modes (MAST taxonomy)
4. Evolves the agent coordination logic to minimize failures

**Optimization Goal**: `combined_score = 1.0 / (1.0 + avg_failures_per_task)`

## Setup

```bash
cd examples/multi_agent_evolution
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

## Run Evolution

```bash
cd /path/to/openevolve
python openevolve-run.py examples/multi_agent_evolution/initial_program.py \
                         examples/multi_agent_evolution/evaluator.py \
                         --config examples/multi_agent_evolution/config.yaml \
                         --iterations 50
```
python openevolve-run.py examples/multi_agent_evolution_v3_gpt5/initial_program.py \
                         examples/multi_agent_evolution_v3_gpt5/evaluator.py \
                         --config examples/multi_agent_evolution_v3_gpt5/config.yaml \
                         --iterations 50

## Design Choices Made

1. **Self-contained evaluator**: MASTLLMJudge is defined inside evaluator.py instead of importing from openevolve/llm_judge.py
2. **Simplified scoring**: Uses simple failure counting (no critical failure weighting)
3. **3 evaluation tasks**: Samples 3 random tasks per evaluation for balance of thoroughness vs speed
4. **2 evaluation rounds**: Uses 2 agent interaction rounds per task for faster iteration

## Monitor Progress

```bash
# View evolution tree
python scripts/visualizer.py --path examples/multi_agent_evolution/openevolve_output/

# Check best program  
cat examples/multi_agent_evolution/openevolve_output/checkpoints/checkpoint_X/best_program.py
```

## Expected Evolution

OpenEvolve should discover improvements like:
- Better role definitions and communication patterns
- Improved error handling and task coordination  
- Stronger verification and validation steps
- Reduced failure modes: task derailment, role confusion, premature termination

## Failure Modes Detected

The LLM judge evaluates 14 failure modes:

**Individual Agent (1.1-1.5)**: Task/role disobedience, repetition, memory loss  
**Inter-Agent (2.1-2.6)**: Communication failures, task derailment, ignored input  
**System-Level (3.1-3.3)**: Premature termination, weak verification