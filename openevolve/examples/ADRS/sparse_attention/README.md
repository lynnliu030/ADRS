# OpenEvolve for Sparse Attention Design

## Setup

# Fetch the simulator submodule
git submodule update --init --recursive

1. Replace your `openevolve-run.py` file with the one provided in this directory.
This has 2 line change for handling cuda based processes

2. Install dependencies:
```
conda init
pip install torch 
pip install pyyaml numpy openai scipy transformers matplotlib pandas seaborn datasets rouge nltk bert_score jieba fuzzywuzzy python-Levenshtein
pip install flash-attn --no-build-isolation
```

3. Sample command line
```
python ./openevolve-run.py  ./sparse-attention-hub/sparse_attention_hub/sparse_attention/research_attention/maskers/openevolve/openevolve_masker.py ./evaluator.py --config ./config_3.yaml --iterations 10

```

