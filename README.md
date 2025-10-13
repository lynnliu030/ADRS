# Barbarians at the Gate: How AI is Upending Systems Research

This repository contains code implementations from the paper **"Barbarians at the Gate: How AI is Upending Systems Research"** ([arXiv:2510.06189](https://arxiv.org/abs/2510.06189)). 

Some examples are still work in progress (WIP) and code may be private — we're actively updating and releasing them. Stay tuned!

## Setup

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
uv sync
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Examples

This repository showcases AI-driven research examples across multiple frameworks and domains:

### OpenEvolve Examples
- **Location**: [`openevolve/examples/ADRS`](openevolve/examples/ADRS) directory
- Collection of AI-driven systems tasks including:
  - [MoE expert placement](openevolve/examples/ADRS/eplb)
  - [Global model scheduling (PRISM)](openevolve/examples/ADRS/prism)
  - [Transaction scheduling](openevolve/examples/ADRS/txn_scheduling)
  - [Telemetry repair](openevolve/examples/ADRS/telemetry_repair)
  - [LLM-SQL optimization](openevolve/examples/ADRS/llm_sql)
  - [Spot instance scheduling for single region](openevolve/examples/ADRS/cant-be-late)
  - [Spot instance scheduling for multi-region](openevolve/examples/ADRS/cant-be-late-multi)
  - [Multi-region data transfer (Cloudcast)](openevolve/examples/ADRS/cloudcast)
  - [Sparse attention design](openevolve/examples/ADRS/sparse_attention)
  - [Multi-agent system design](openevolve/examples/ADRS/multiagent_system)
  - [HP Quantization](openevolve/examples/ADRS/hp_quantization)
  
> **Note**: Check out the README inside each folder for more details about setup and usage. 

#### Test Command 
```
python -m openevolve.cli \
  openevolve/examples/ADRS/<case>/initial_program.py \
  openevolve/examples/ADRS/<case>/evaluator.py \
  --config openevolve/examples/ADRS/<case>/config.yaml \
  --iterations <N> \
  --output openevolve/examples/ADRS/<case>/output
```

  
### Cursor Examples
- **Location**: `cursor/` directory

### GEPA Examples  
- **Location**: `gepa/` directory