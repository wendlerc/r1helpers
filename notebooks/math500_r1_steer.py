# %%
# Imports
import sys
import random
import gc
import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from accelerate import Accelerator
from functools import partial
import re
import os
sys.path.append("/disk/u/troitskiid/projects/r1helpers")
from src.r1helpers.math500.grader import grade_answer

# Set high precision matmul for faster computation on modern GPUs
torch.set_float32_matmul_precision('high')

# %%
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TORCH_DTYPE = torch.bfloat16

DATASET_PATH = "/disk/u/troitskiid/projects/r1helpers/results/llama8b_r1_reference_batch6_newtoks5000_processed.jsonl"
CROSSCODER_BASE_PATH = "/disk/u/troitskiid/data/checkpoints/L1-Crosscoder"

LAYER_IDS = [15]
FEATURE_IDS = [744, 31748, 25929, 188]
STRENGTHS = [1.25]

BATCH_SIZE = 6
N_NEW_TOKS = 7500

DO_SAMPLE = True
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 42

# %%
# Load and sample entries
random.seed(SEED)
results = []
with open(DATASET_PATH, 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        # Drop unwanted fields
        entry.pop('subject', None)
        entry.pop('level', None)
        entry.pop('problem', None)
        entry.pop('solution', None)
        results.append(entry)

correct_entries = [r for r in results if r.get('correct', False)]
incorrect_entries = [r for r in results if not r.get(
    'correct', False) and r.get('parsed_answer', '').strip() != '']
sampled_entries = random.sample(correct_entries, min(81, len(
    correct_entries))) + random.sample(incorrect_entries, min(19, len(incorrect_entries)))
print(f"Sampled {len(sampled_entries)} entries")

# %%


def my_set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_model(model_name, torch_dtype, device="auto"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Fix pad token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer, accelerator


# %%
model, tokenizer, accelerator = load_model(
    MODEL_NAME, TORCH_DTYPE, device="auto")


# %%
def _unwrap_and_get_layers(model):
    base = accelerator.unwrap_model(model)
    # LLaMA-style: base.model.layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base, base.model.layers
    # Fallback if nested
    if hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "layers"):
        return base, base.model.model.layers
    raise AttributeError("Could not locate transformer layers on model.")


@torch.no_grad()
def steer_batch_generation(model, token_sequences, steering_vectors,
                           start_token_idx=-1,
                           steering_mode="all",
                           layer_indices=[],
                           n_new_tokens=10,
                           steering_strength=0.0,
                           threshold=-0.01,
                           do_sample=True,
                           temperature=0.6,
                           top_p=0.95,
                           seed=42):

    assert steering_mode in ["all", "reactive"]

    base, layers = _unwrap_and_get_layers(model)

    # Track reactive stats per-sample
    batch_size = len(token_sequences)
    steering_activation_count = torch.zeros(
        batch_size, dtype=torch.int64, device=model.device)
    generation_steps = 0  # counts S==1 steps

    @torch.no_grad()
    def steering_hook(module, inputs, output, start_token_idx=None, steering_vector=None, strength=None, threshold=None):
        nonlocal steering_activation_count, generation_steps
        hidden_states = output[0] if isinstance(output, tuple) else output
        steering_vector_device = steering_vector.to(
            hidden_states.device, dtype=hidden_states.dtype)

        # Normalize shape to [B, S, D]
        if hidden_states.dim() == 3:
            hidden_states_processed = hidden_states  # [B,S,D]
        else:
            hidden_states_processed = hidden_states.unsqueeze(0)  # [1,S,D]

        current_batch_size, sequence_length, hidden_dimension = hidden_states_processed.shape

        if sequence_length > 1:
            start_position = start_token_idx if (start_token_idx is not None and start_token_idx >= 0) else (
                sequence_length - 1 if start_token_idx == -1 else 0)
            start_position = max(0, min(start_position, sequence_length - 1))
            token_norms = hidden_states_processed[:, start_position:, :].norm(
                dim=-1, keepdim=True)  # [B,S-start,1]

            vector = steering_vector_device
            if vector.dim() == 1:            # [D] -> [1,1,D]
                vector = vector.view(1, 1, -1)
            elif vector.dim() == 2:          # [B,D] -> [B,1,D]
                vector = vector.view(current_batch_size, 1, -1)

            hidden_states_processed[:, start_position:,
                                    :] += token_norms * strength * vector
        else:
            # [B,D]
            current_hidden = hidden_states_processed[:, 0, :]
            current_norm = current_hidden.norm(dim=-1, keepdim=True)   # [B,1]

            if steering_mode == "all":
                vector = steering_vector_device
                if vector.dim() == 1:  # [D] -> [1,D]
                    vector = vector.view(1, -1)
                hidden_states_processed[:, 0,
                                        :] += current_norm * strength * vector
            else:
                # reactive
                vector = steering_vector_device
                if vector.dim() == 1:
                    # [1,D] (broadcast)
                    vector = vector.view(1, -1)
                hidden_normalized = current_hidden / (current_norm + 1e-12)
                vector_normalized = vector / (vector.norm(dim=-1, keepdim=True) +
                                              1e-12)  # [B,D] or [1,D]
                projection = (hidden_normalized *
                              vector_normalized).sum(dim=-1)      # [B]
                activation_mask = projection < threshold
                if activation_mask.any():
                    if vector.dim() == 2 and vector.shape[0] == current_batch_size:
                        hidden_states_processed[activation_mask, 0,
                                                :] += current_norm[activation_mask] * strength * vector[activation_mask]
                    else:
                        hidden_states_processed[activation_mask, 0,
                                                :] += current_norm[activation_mask] * strength * vector
                steering_activation_count += activation_mask.to(
                    steering_activation_count.dtype)

            generation_steps += 1

        return output  # modify in-place

    # Register hooks for each layer
    hooks = [
        layers[layer_idx].register_forward_hook(
            partial(steering_hook, start_token_idx=start_token_idx,
                    steering_vector=steering_vectors[layer_pos], strength=steering_strength, threshold=threshold)
        )
        for layer_pos, layer_idx in enumerate(layer_indices)
    ]

    pad_token_id = tokenizer.pad_token_id
    batch_size = len(token_sequences)
    max_length = max(len(seq) for seq in token_sequences)

    # Create input tensors on model's device
    input_ids = torch.full((batch_size, max_length), pad_token_id,
                           dtype=torch.long, device=model.device)
    for i, seq in enumerate(token_sequences):
        input_ids[i, -len(seq):] = torch.tensor(seq,
                                                dtype=torch.long, device=model.device)
    attention_mask = (input_ids != pad_token_id).long()

    try:
        output_ids = base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=n_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False,
        )
    finally:
        for hook in hooks:
            hook.remove()

    result = {"sequences": output_ids}
    if steering_mode == "reactive" and generation_steps > 0:
        result["steering_activation_fraction"] = (
            100.0 * steering_activation_count.float() / generation_steps).tolist()  # per-sample
    return result


# %%
def decode_batch_generation(output_ids, input_ids, tokenizer):
    """Extract generated text from batched generation with right-aligned padding."""
    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.tolist()
    
    pad_id = tokenizer.pad_token_id
    
    # Skip any padding tokens at the beginning
    lp = 0
    while lp < len(output_ids) and output_ids[lp] == pad_id:
        lp += 1
    
    # For right-aligned batching, the input starts after the padding
    # Extract everything after the input
    generated_tokens = output_ids[lp + len(input_ids):]
    
    # Remove any padding tokens from the end
    while generated_tokens and generated_tokens[-1] == pad_id:
        generated_tokens.pop()
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def parse_answer(generated_text):
    matches = re.search("</think>", generated_text)
    if matches is None:
        return ""
    generated_answer = generated_text[matches.end():]
    # in generated answer select the content of \boxed{}
    matches = re.search("\\\\boxed{", generated_answer)
    if matches is None:
        return ""
    generated_answer = generated_answer[matches.end():]
    # search all the way to the end of the string   }
    reversed_answer = generated_answer[::-1]
    matches = re.search("}", reversed_answer)
    if matches is None:
        return ""
    generated_answer = generated_answer[:len(
        generated_answer) - matches.start()]
    return generated_answer


# %%
# Load the cross-coder decoder matrices
normalize_vectors = True

layer_to_checkpoint_paths = {
    7: "L7R/cc_weights.pt",
    15: "L15R/cc_weights.pt",
    22: "L23R/cc_weights.pt"
}

layer_to_feature_vectors = {}
for layer_idx, checkpoint_path in layer_to_checkpoint_paths.items():
    if layer_idx in LAYER_IDS:
        d = torch.load(os.path.join(
            CROSSCODER_BASE_PATH, checkpoint_path), map_location="cpu")
        layer_to_feature_vectors[layer_idx] = d["decoder.weight"][1] / d["decoder.weight"][1].norm(
            dim=-1, keepdim=True) if normalize_vectors else d["decoder.weight"][1]

# %%
# Main evaluation loop
print("Starting batch processing with steering...")
print(f"Processing {len(sampled_entries)} entries in batches of {BATCH_SIZE}")

# Loop over steering configurations first
for layer_idx in LAYER_IDS:
    if layer_idx not in layer_to_feature_vectors:
        print(f"Layer {layer_idx} not found in layer_to_feature_vectors")
        continue
        
    for feature_id in FEATURE_IDS:
        if feature_id >= layer_to_feature_vectors[layer_idx].shape[0]:
            print(f"Feature {feature_id} not found for layer {layer_idx}")
            continue
            
        for strength in STRENGTHS:
            # Skip strength=0 as it's the same as original
            if strength == 0:
                print(f"Strength {strength} is 0, skipping")
                continue
                
            # Track accuracy for this configuration
            correct_count = 0
            total_count = 0
            
            # Create separate output file for this combination
            output_filename = f"../results/llama8b_r1_math500_steering_layer{layer_idx}_feature{feature_id}_strength{strength}_seed{SEED}.jsonl"
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            # If file exists, create a new one with timestamp or counter
            if os.path.exists(output_filename):
                import time
                timestamp = int(time.time())
                base_name, ext = os.path.splitext(output_filename)
                output_filename = f"{base_name}_{timestamp}{ext}"
                print(f"File exists, creating new file: {output_filename}")
            
            # Process all batches for this steering configuration
            for batch_start in tqdm(range(0, len(sampled_entries), BATCH_SIZE), desc=f"Layer {layer_idx}, Feature {feature_id}, Strength {strength}"):
                batch_end = min(batch_start + BATCH_SIZE, len(sampled_entries))
                batch_entries = sampled_entries[batch_start:batch_end]
                
                # Extract token sequences for this batch
                batch_token_sequences = [entry.get('tokens_before', []) for entry in batch_entries]
                
                # Generate steered batch response for this configuration
                steered_batch_output = steer_batch_generation(
                    model, batch_token_sequences, 
                    steering_vectors=[layer_to_feature_vectors[layer_idx][feature_id]],
                    layer_indices=[layer_idx],
                    steering_mode="all",
                    steering_strength=strength,
                    n_new_tokens=N_NEW_TOKS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    seed=SEED
                )
                
                # Process each item in the batch
                for i, entry in enumerate(batch_entries):
                    token_sequence = batch_token_sequences[i]
                    
                    # Extract the generated text using our batch-aware function
                    output_sequence = steered_batch_output["sequences"][i]
                    steered_generated_text = decode_batch_generation(
                        output_sequence, token_sequence, tokenizer
                    )
                    
                    steered_parsed_answer = parse_answer(steered_generated_text)
                    steered_correct = grade_answer(steered_parsed_answer, entry.get('answer', ''))
                    
                    # Update accuracy tracking
                    if steered_correct:
                        correct_count += 1
                    total_count += 1
                    
                    # Create steered result entry
                    steered_result = {
                        'unique_id': entry.get('unique_id', ''),
                        'tokens_before': token_sequence,
                        'text_before': entry.get('text_before', ''),
                        'original_text_after': entry.get('original_text_after', ''),
                        'steered_text_after': steered_generated_text,
                        'parsed_answer': steered_parsed_answer,
                        'answer': entry.get('answer', ''),
                        'correct': steered_correct
                    }
                    
                    with open(output_filename, 'a') as output_file:
                        output_file.write(json.dumps(steered_result) + '\n')
                
                # Memory cleanup per batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Print final accuracy for this configuration
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            print(f"Final accuracy for Layer {layer_idx}, Feature {feature_id}, Strength {strength}: {correct_count}/{total_count} = {accuracy:.3f}")

# Final cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("Batch processing completed!")
#
# %%
