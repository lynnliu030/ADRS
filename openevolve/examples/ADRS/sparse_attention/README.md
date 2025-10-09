1. Use openvolve from here:
```
https://github.com/apd10/openevolve.git (branch: old_branch)
```
This has 2 line change for handling cuda based processes

2. Sample command line
```
python ./open_evolve_masker/openevolve/openevolve-run.py  ./sparse_attention_hub/sparse_attention/research_attention/maskers/openevolve/openevolve_masker.py ./open_evolve_masker/evaluator.py --config ./open_evolve_masker/config_phase2.1.yaml --iterations 10

```

Dependencies:
```
/workspace/audrey/miniconda3/bin/conda init
pip install torch 
pip install pyyaml numpy openai scipy transformers matplotlib pandas seaborn datasets rouge nltk bert_score jieba fuzzywuzzy python-Levenshtein
pip install flash-attn --no-build-isolation
```
