# %%
import json
from transformers import AutoTokenizer

# %%
# Load the tokenizer (same as used in the generation scripts)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
# Load the JSONL file
original_filename = '/disk/u/troitskiid/projects/r1helpers/results/llama8b_r1_math500_steering_layer15_feature188_strength1.5_steer100_seed42.jsonl'
with open(original_filename, 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} entries from the file")

# %%
# Create a summary version of the data with truncated text and removed fields
summary_data = []

for entry in data:
    # Create a copy of the entry
    summary_entry = entry.copy()
    
    # Remove specified fields
    fields_to_remove = ['tokens_before', 'parsed_answer', 'answer']
    for field in fields_to_remove:
        if field in summary_entry:
            del summary_entry[field]
    
    # Truncate text fields to first 500 characters
    if 'original_text_after' in summary_entry:
        summary_entry['original_text_after'] = summary_entry['original_text_after'][:500]
    
    if 'steered_text_after' in summary_entry:
        summary_entry['steered_text_after'] = summary_entry['steered_text_after'][:500]
    
    summary_data.append(summary_entry)

# Save the summary data to a new file
summary_filename = original_filename.replace('.jsonl', '_summary.jsonl')

with open(summary_filename, 'w') as f:
    for entry in summary_data:
        f.write(json.dumps(entry) + '\n')

print(f"Created summary file: {summary_filename}")
print(f"Summary contains {len(summary_data)} entries with truncated text fields")

# %%


# %%
# Compare token counts between original and steered text
original_token_counts = []
steered_token_counts = []
token_differences = []
token_percentage_differences = []

for entry in data:
    original_text = entry.get('original_text_after', '')
    steered_text = entry.get('steered_text_after', '')
    
    orig_tokens = len(tokenizer.encode(original_text)) if original_text else 0
    steered_tokens = len(tokenizer.encode(steered_text)) if steered_text else 0
    
    original_token_counts.append(orig_tokens)
    steered_token_counts.append(steered_tokens)
    token_differences.append(steered_tokens - orig_tokens)
    
    # Calculate percentage difference relative to original
    if orig_tokens > 0:
        percentage_diff = ((steered_tokens - orig_tokens) / orig_tokens) * 100
    else:
        percentage_diff = 0 if steered_tokens == 0 else float('inf')
    token_percentage_differences.append(percentage_diff)

# Calculate statistics
import numpy as np

print("Token Count Comparison Statistics:")
print("=" * 40)
print(f"Original text - Mean tokens: {np.mean(original_token_counts):.2f}")
print(f"Original text - Median tokens: {np.median(original_token_counts):.2f}")
print(f"Original text - Min tokens: {np.min(original_token_counts)}")
print(f"Original text - Max tokens: {np.max(original_token_counts)}")
print()
print(f"Steered text - Mean tokens: {np.mean(steered_token_counts):.2f}")
print(f"Steered text - Median tokens: {np.median(steered_token_counts):.2f}")
print(f"Steered text - Min tokens: {np.min(steered_token_counts)}")
print(f"Steered text - Max tokens: {np.max(steered_token_counts)}")
print()
print(f"Token difference (steered - original):")
print(f"  Mean difference: {np.mean(token_differences):.2f}")
print(f"  Median difference: {np.median(token_differences):.2f}")
print(f"  Min difference: {np.min(token_differences)}")
print(f"  Max difference: {np.max(token_differences)}")
print(f"  Std deviation: {np.std(token_differences):.2f}")
print()
print(f"Token percentage difference ((steered - original) / original * 100):")
print(f"  Mean percentage: {np.mean([p for p in token_percentage_differences if p != float('inf')]):.2f}%")
print(f"  Median percentage: {np.median([p for p in token_percentage_differences if p != float('inf')]):.2f}%")
print(f"  Min percentage: {np.min([p for p in token_percentage_differences if p != float('inf')]):.2f}%")
print(f"  Max percentage: {np.max([p for p in token_percentage_differences if p != float('inf')]):.2f}%")

# Show distribution of token differences
positive_diffs = sum(1 for diff in token_differences if diff > 0)
negative_diffs = sum(1 for diff in token_differences if diff < 0)
zero_diffs = sum(1 for diff in token_differences if diff == 0)

print()
print("Token difference distribution:")
print(f"  Steered text longer: {positive_diffs} ({positive_diffs/len(data)*100:.1f}%)")
print(f"  Steered text shorter: {negative_diffs} ({negative_diffs/len(data)*100:.1f}%)")
print(f"  Same token count: {zero_diffs} ({zero_diffs/len(data)*100:.1f}%)")

# %%