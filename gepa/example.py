import gepa
import os 

assert os.environ.get("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

# Load AIME dataset
trainset, valset, _ = gepa.examples.aime.init_dataset()
print(f"Size of trainset: {len(trainset)}, size of valset: {len(valset)}")
# just trim to 5 for each train and val set
trainset = trainset[:5]
valset = valset[:5]

seed_prompt = {
    "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
}

# Let's run GEPA optimization process.
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset, valset=valset,
    task_lm="openai/gpt-4.1-mini", # <-- This is the model being optimized
    max_metric_calls=1, # <-- Set a budget
    reflection_lm="openai/gpt-4.1-mini", # <-- Use a strong model to reflect on mistakes and propose better prompts
)

print("GEPA Optimized Prompt:", gepa_result.best_candidate['system_prompt'])