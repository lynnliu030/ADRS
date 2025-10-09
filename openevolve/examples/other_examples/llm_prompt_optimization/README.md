# LLM Prompt Optimization with OpenEvolve 🚀

This example demonstrates how to use OpenEvolve to automatically optimize prompts for Large Language Models. The system uses evolutionary search to discover high-performing prompts by testing them against ground truth data from various datasets.

## 🎯 Overview

OpenEvolve automatically:
- Loads datasets from various sources
- Evolves prompts through multiple generations
- Uses cascading evaluation for efficiency
- Finds optimal prompts for your specific task and model

**Key Feature**: The evaluator automatically matches prompt files with dataset configurations using a naming convention (`xxx_prompt.txt` → `xxx_prompt_dataset.yaml`), making it easy to manage multiple benchmark tasks.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd examples/llm_prompt_optimization
pip install -r requirements.txt
```

### 2. Configure Your Model

Update `config.yaml` with your LLM settings:

```yaml
llm:
  api_base: "https://openrouter.ai/api/v1"
  api_key: "your_api_key_here"
  models:
    - name: "google/gemini-2.5-flash"  # Or any OpenAI-compatible model
      weight: 1.0
```

### 3. Set Up Your Dataset and Prompt

This example uses a naming convention to match prompts with their dataset configurations:
- For a prompt file `xxx_prompt.txt`, create a matching `xxx_prompt_dataset.yaml`
- For example: `emotion_prompt.txt` uses `emotion_prompt_dataset.yaml`

Create your dataset configuration file (e.g., `emotion_prompt_dataset.yaml`):

```yaml
# Dataset configuration
dataset_name: "dair-ai/emotion"   # Dataset identifier
input_field: "text"               # Field containing input data
target_field: "label"             # Field containing ground truth
split: "test"                     # Dataset split to use

# Evaluation samples
max_samples: 200   # Number of samples to evaluate
```

Create your initial prompt file (e.g., `emotion_prompt.txt`):

```
Classify the emotion expressed in the following text.

Text: "{input_text}"

Emotion (0-5):
```

### 4. Run OpenEvolve

Use the provided `run_evolution.sh` script to ensure the correct dataset is used:

```bash
# For emotion classification benchmark
./run_evolution.sh emotion_prompt.txt --iterations 50

# For IMDB sentiment analysis
./run_evolution.sh initial_prompt.txt --iterations 50

# With custom iterations and checkpoint
./run_evolution.sh emotion_prompt.txt --iterations 100 --checkpoint-interval 20
```

The script automatically:
- Sets the `OPENEVOLVE_PROMPT` environment variable so the evaluator knows which dataset to use
- Passes all additional arguments to OpenEvolve
- Ensures the correct `_dataset.yaml` file is matched with your prompt

**Note**: If you prefer to run OpenEvolve directly, set the environment variable first:
```bash
export OPENEVOLVE_PROMPT=emotion_prompt.txt
python ../../openevolve-run.py emotion_prompt.txt evaluator.py --config config.yaml --iterations 50
```

## 📊 Supported Datasets

This optimizer works with a wide variety of datasets. Included examples:

- **IMDB Sentiment**: `initial_prompt.txt` + `initial_prompt_dataset.yaml` (binary classification)
- **Emotion**: `emotion_prompt.txt` + `emotion_prompt_dataset.yaml` (6-class, benchmark against DSPy)
- **GSM8K**: `gsm8k_prompt.txt` + `gsm8k_prompt_dataset.yaml` (grade school math, DSPy achieves 97.1%)

### Creating New Tasks

To add a new dataset:
1. Create `yourtask_prompt.txt` with the initial prompt
2. Create `yourtask_prompt_dataset.yaml` with the dataset configuration
3. Run: `./run_evolution.sh yourtask_prompt.txt --iterations 50`

**Note**: If you call OpenEvolve directly without the wrapper script, the evaluator will look for a default `dataset_config.yaml` file.

### Common Dataset Configurations:

### Sentiment Analysis
```yaml
dataset_name: "stanfordnlp/imdb"
input_field: "text"
target_field: "label"  # 0 or 1
```

### Question Answering
```yaml
dataset_name: "squad"
input_field: "question"
target_field: "answers"  # Dict with 'text' field
```

### Text Classification
```yaml
dataset_name: "ag_news"
input_field: "text"
target_field: "label"  # 0-3 for categories
```

### Summarization
```yaml
dataset_name: "xsum"
input_field: "document"
target_field: "summary"
```

## ⚙️ How It Works

### Simple Evaluation

The evaluator uses a straightforward single-stage evaluation:

1. **Load Dataset**: Downloads the specified dataset
2. **Sample Data**: Takes `max_samples` examples from the dataset
3. **Test Prompt**: Sends each example through the LLM with the prompt
4. **Calculate Accuracy**: Compares LLM outputs to ground truth labels

### Evolution Process

1. OpenEvolve starts with your initial prompt
2. The LLM generates variations based on performance feedback
3. Each variant is tested using cascading evaluation
4. Best performers are kept and evolved further
5. Process continues for specified iterations

### 🎭 Custom Templates for Prompt Evolution

By default, OpenEvolve is designed for code evolution. To make it work properly for prompt evolution, this example includes custom templates in the `templates/` directory:

- **`full_rewrite_user.txt`**: Replaces the default code evolution template with prompt-specific language

This ensures the LLM understands it should evolve the prompt text itself, not generate code. The configuration automatically uses these templates via:

```yaml
prompt:
  template_dir: "templates"  # Use custom templates for prompt evolution
```

## 🎯 Configuration Options

### Evaluation Configuration

In `config.yaml`:
```yaml
evaluator:
  parallel_evaluations: 4      # Run 4 evaluations in parallel
  cascade_evaluation: false    # Simple single-stage evaluation
```

### Sample Size

Adjust in `dataset.yaml`:
```yaml
max_samples: 50    # Number of samples to evaluate
```

## 📈 Example Results

Starting prompt:
```
Analyze the sentiment: "{input_text}"
```

Evolved prompt after 100 iterations:
```
Analyze the sentiment of the following text. Determine if the overall emotional tone is positive or negative.

Text: "{input_text}"

Response: Provide only a single digit - either 1 for positive sentiment or 0 for negative sentiment. Do not include any explanation or additional text.
```

Accuracy improvement: 72% → 94%

## 🔧 Advanced Usage

### Custom Evaluation Metrics

The evaluator extracts predictions and compares them to ground truth. For classification tasks, it looks for:
- Exact number matches (0, 1, etc.)
- Keywords (positive/negative, yes/no)
- Custom patterns you define

### Different Task Types

While the default setup is for classification, you can modify the evaluator for:
- **Regression**: Compare numeric outputs
- **Generation**: Use BLEU/ROUGE scores
- **Extraction**: Check if key information is present

## 🐛 Troubleshooting

### Dataset Not Found
- Check the exact dataset name and source
- Some datasets require acceptance of terms

### Low Stage 1 Accuracy
- Your initial prompt may be too vague
- Check if the output format matches expectations
- Verify the dataset fields are correct

### API Errors
- Ensure your API key is valid
- Check rate limits
- Verify the model name is correct

## 🚀 Tips for Best Results

1. **Start Simple**: Begin with a clear, working prompt
2. **Clear Output Format**: Specify exactly what output you expect
3. **Appropriate Samples**: More samples = better evaluation but slower
4. **Multiple Runs**: Evolution has randomness; try multiple runs
5. **Monitor Progress**: Check intermediate best_program.txt files

## 📚 Next Steps

- Try different datasets and benchmarks
- Experiment with different models
- Adjust evolution parameters in config.yaml
- Create task-specific evaluation metrics

Happy prompt evolving! 🧬✨