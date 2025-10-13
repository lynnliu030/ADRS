# Cant Be Late

This example demonstrates how to use OpenEvolve to optimize the Can't Be Late scheduling problem (NSDI'24).

## Setup

1. Install simulator dependencies and unpack the real traces so the evaluator can find them:

```bash
cd openevolve/examples/ADRS/cant-be-late/simulator
uv sync
mkdir -p data
[ -d data/real ] || tar -xzf real_traces.tar.gz -C data
```

2. Make sure you have the API keys expected by the run (replace with your own values or source a `.env` file).

```bash
source .env # If you have a .env file

echo "$OPENAI_API_KEY"
echo "$GEMINI_API_KEY"
```

## Run the evolution

Change into the example directory so the generated `openevolve_output/` sits alongside the scripts:

```bash
cd openevolve/examples/ADRS/cant-be-late
uv run openevolve-run initial_greedy.py evaluator.py --config config.yaml --output openevolve_output --iterations 100 --log-level INFO
```

The first iteration may show low scores until the evaluator finishes loading the trace data extracted above.
