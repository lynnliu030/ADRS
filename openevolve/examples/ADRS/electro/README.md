# Electro + OpenEvolve Integration

This directory contains the complete implementation for using OpenEvolve to optimize RAG (Retrieval-Augmented Generation) configurations in Electro.

## Overview

The system uses OpenEvolve's evolutionary coding capabilities to discover optimal RAG configurations that minimize cost while maintaining quality constraints. Instead of running expensive experiments, we use Electro's historical experiment data as a simulator.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /home/melissa/openevolve/electro_simulator
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export OPENAI_API_KEY=your-api-key-here
```

### 3. Test the Implementation
```bash
python test_simulator.py
```

### 4. Run Optimization
```bash
# Quick test run (10 iterations)
python run_optimization.py --test-run

# Full optimization (200 iterations)
python run_optimization.py

# Custom configuration
python run_optimization.py --iterations 100 --config config.yaml
```

### 5. Resume from Checkpoint
```bash
python run_optimization.py --checkpoint openevolve_output/checkpoints/checkpoint_50
```

## 📁 Implementation Files

```
electro_simulator/
├── README.md             # This file - usage instructions
├── DESIGN_PLAN.md       # Detailed design document  
├── PSEUDOCODE.md        # Comprehensive pseudocode
├── data_loader.py       # ✅ Loads Electro experiment data
├── simulator.py         # ✅ Simulates experiments using historical data
├── initial_program.py   # ✅ Random search baseline (evolvable)
├── evaluator.py         # ✅ Evaluates algorithms on 3 metrics
├── config.yaml          # ✅ OpenEvolve configuration
├── requirements.txt     # ✅ Dependencies
├── run_optimization.py  # ✅ Easy run script
└── test_simulator.py    # ✅ Test implementation
```

## 🎯 Key Features

### **Comprehensive Data Loading**
- ✅ Handles multiple experiment formats (financeBench, frames, TQA_KILT)
- ✅ Loads from configurable directories
- ✅ Extracts all available metrics
- ✅ **Tested**: Successfully loads 36 experiments from financeBench data

### **Rich Simulator**
- ✅ Returns comprehensive metrics for evolved algorithms
- ✅ Cost estimation with detailed breakdown
- ✅ Handles missing configurations gracefully
- ✅ **Available metrics**: accuracy, retrieval_precision@k, retrieval_recall@k, cost_estimate, latency_p90, success_rate, and more

### **Configurable Evaluation**
- ✅ Three-metric scoring system:
  - **Search Efficiency** (30%): How few configurations needed
  - **Config Quality** (50%): Accuracy, retrieval quality, success rate  
  - **Config Cost** (20%): Cost per query optimization
- ✅ Easily adjustable weights in `evaluator.py`

### **Evolvable Algorithms**
- ✅ Simple random search baseline
- ✅ Ready for OpenEvolve to evolve into sophisticated algorithms
- ✅ Access to all experiment metrics for optimization

## 📊 Current Data Coverage

**Successfully Tested With:**
- ✅ financeBench: 36 experiments loaded
  - Models: llama_3_8b, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct
  - Embeddings: bm25
  - Retrieval K: 1-50 documents
  - All metrics available: accuracy, retrieval precision/recall, cost estimates

**Configured For:**
- frames experiments (llama70b, llama8b variants)
- TQA_KILT experiments  
- Saturation load tests
- Multiple benchmarks and configurations

## 🔧 Configuration

### Add More Experiment Directories
Edit `evaluator.py`:
```python
GRID_SEARCH_DIRS = [
    "/path/to/your/experiment/output1",
    "/path/to/your/experiment/output2",
    # Add more directories
]
```

### Adjust Evaluation Weights
Edit `evaluator.py`:
```python
EVALUATION_WEIGHTS = {
    'search_efficiency': 0.3,  # Fewer evaluations needed
    'config_quality': 0.5,    # Final config quality
    'config_cost': 0.2,       # Cost optimization
}
```

### Future: Add Latency Constraints
```python
LATENCY_CONSTRAINTS = {
    'max_p90_latency': 2.0,        # Max 2 seconds P90 latency
    'latency_penalty_weight': 0.1   # Penalty weight
}
```

## 📈 Expected Results

Based on the test run, the system:
- ✅ **Loads experiment data**: 36 configurations from financeBench
- ✅ **Provides rich metrics**: 20+ metrics per configuration
- ✅ **Calculates costs**: Realistic cost estimates per query
- ✅ **Runs algorithms**: Random search baseline working
- 🎯 **Ready for evolution**: OpenEvolve can improve the search algorithm

## 🧪 Test Results

```bash
$ python test_simulator.py

==================================================
Testing Electro Simulator Implementation  
==================================================
✅ Data loading: 36/36 experiments loaded successfully
✅ Configuration space: 3 models, 1 embedding, 12 retrieval_k values
✅ Simulator: Returns comprehensive metrics (accuracy: 0.12, cost: $0.003/query)
✅ Initial program: Random search working, finds configurations
✅ All tests passed!
```

## 🚀 Next Steps

1. **Run your first optimization**:
   ```bash
   python run_optimization.py --test-run
   ```

2. **Monitor progress**: Check `openevolve_output/` for checkpoints and results

3. **Analyze results**: Use OpenEvolve's visualization tools:
   ```bash
   cd /home/melissa/openevolve/scripts
   python visualizer.py --path ../electro_simulator/openevolve_output/checkpoints/checkpoint_50
   ```

4. **Scale up**: Add more experiment directories and run longer optimizations

## 📚 Documentation

- **[DESIGN_PLAN.md](DESIGN_PLAN.md)**: Complete system design and architecture
- **[PSEUDOCODE.md](PSEUDOCODE.md)**: Detailed implementation pseudocode  
- **[OpenEvolve Docs](../README.md)**: Main OpenEvolve documentation

## 🎯 Success Metrics

- **Cost Reduction**: Target 50% reduction vs naive configurations
- **Search Efficiency**: 10x faster than grid search
- **Algorithm Discovery**: Novel optimization strategies
- **Quality Maintenance**: 95%+ configurations meet accuracy requirements

The implementation is complete and ready for optimization! 🚀
