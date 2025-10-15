# Can't-Be-Late (NSDI'24)

This folder hosts the OpenEvolve reproduction of the “Can’t Be Late” spot/on-demand scheduling problem. It contains the evolution driver, the simulator, and utilities for large-scale evaluation.

---

## Local setup

1. **Install simulator dependencies and unpack the traces.**

   ```bash
   cd openevolve/examples/ADRS/cant-be-late/simulator
   uv sync --active
   mkdir -p data
   [ -d data/real ] || tar -xzf real_traces.tar.gz -C data
   ```

2. **Provide the API keys required by the prompts** (either export them manually or source a `.env`).

   ```bash
   export OPENAI_API_KEY=...
   export GEMINI_API_KEY=...
   ```

3. **(Optional) Run an evolution round.**

   ```bash
   cd openevolve/examples/ADRS/cant-be-late
   uv run openevolve-run initial_greedy.py evaluator.py \
     --config config.yaml \
     --output openevolve_output \
     --iterations 100 \
     --log-level INFO
   ```

   The first iteration may score poorly until the evaluator finishes loading every trace.

---

## Remote full evaluation with SkyPilot

We provide a SkyPilot workflow that launches a 64 vCPU `c6i.16xlarge`, installs the project via `uv`, runs `full_eval.py`, and copies the JSON report back to your workstation.

### Prerequisites

```bash
pip install "skypilot[aws]"
sky check
```

Ensure AWS credentials are available so `sky` can provision the instance.

### Launch the evaluation

```bash
python scripts/run_skypilot_full_eval.py --cluster adrs-eval
```

The script streams logs, compares the target strategy against both `initial_greedy.py` and `referenced_up.py`, downloads `skypilot_artifacts/full_eval_remote.json`, and prints a concise emoji summary (average cost, improvement ranges, overhead = 0.02 slice, etc.).

To tear the node down afterwards:

```bash
python scripts/run_skypilot_full_eval.py --cluster adrs-eval --teardown
# or
sky down adrs-eval -y
```
