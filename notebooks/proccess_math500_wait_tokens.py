# %%
# Imports
import json
from transformers import AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_FILE = "../results/llama8b_r1_reference_batch6_newtoks5000.jsonl"

# %%
WAIT_TOKEN_IDS = [11748, 14524, 3868, 14144]

# %%
def find_first_wait_token_position(text):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Find first occurrence of any wait token
    for pos, token_id in enumerate(tokens):
        if token_id in WAIT_TOKEN_IDS:
            token_str = tokenizer.decode([token_id])

            # Split tokens into before and after
            tokens_before = tokens[:pos]
            tokens_after = tokens[pos:]

            # Decode to text
            text_before = tokenizer.decode(
                tokens_before) if tokens_before else ""
            text_after = tokenizer.decode(tokens_after)

            return {
                # 'position': pos,
                # 'token_id': token_id,
                # 'token_str': token_str,
                'tokens_before': tokens_before,
                # 'tokens_after': tokens_after,
                'text_before': text_before,
                'text_after': text_after
            }

    # No wait token found
    return {
        # 'position': None,
        # 'token_id': None,
        # 'token_str': None,
        'tokens_before': tokens,
        # 'tokens_after': [],
        'text_before': text,
        'text_after': ""
    }


def process_dataset_for_wait_tokens(file_path=None):
    if file_path is None:
        raise ValueError("file_path is required")

    results = []

    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())

            # Concatenate input_text and generated_text
            input_text = entry.get('input_text', '')
            generated_text = entry.get('generated_text', '')
            full_text = input_text + generated_text

            wait_analysis = find_first_wait_token_position(full_text)

            result = {
                'subject': entry.get('subject', ''),
                'level': entry.get('level', ''),
                'unique_id': entry.get('unique_id', ''),
                'problem': entry.get('problem', ''),
                'solution': entry.get('solution', ''),
                # 'wait_position': wait_analysis['position'],
                # 'wait_token_id': wait_analysis['token_id'],
                # 'wait_token_str': wait_analysis['token_str'],
                'tokens_before': wait_analysis['tokens_before'],
                # 'tokens_after': wait_analysis['tokens_after'],
                'text_before': wait_analysis['text_before'],
                'original_text_after': wait_analysis['text_after'],
                'parsed_answer': entry.get('parsed_answer', ''),
                'answer': entry.get('answer', ''),
                'correct': entry.get('correct', False)
            }
            results.append(result)

    return results

# %%
print("Processing dataset for wait tokens")
results = process_dataset_for_wait_tokens(OUTPUT_FILE)
processed_output_file = OUTPUT_FILE.replace('.jsonl', '_processed.jsonl')
with open(processed_output_file, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')
print(f"Saved {len(results)} processed entries to {processed_output_file}")