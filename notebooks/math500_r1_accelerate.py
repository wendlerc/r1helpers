# %%
# Imports
from src.r1helpers.math500.grader import grade_answer
import gc
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from accelerate import Accelerator
import re
import os
import sys
sys.path.append("/disk/u/troitskiid/projects/r1helpers")


# Set high precision matmul for faster computation on modern GPUs
torch.set_float32_matmul_precision('high')

# %%
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TORCH_DTYPE = torch.bfloat16

BATCH_SIZE = 6
N_NEW_TOKS = 5000

DO_SAMPLE = True
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 42

OUTPUT_FILE = f"..results/llama8b_r1_math500_reference_batch{BATCH_SIZE}_newtoks{N_NEW_TOKS}.jsonl"

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
    torch_dtype = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Compile the model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")

    return model, tokenizer, accelerator


# %%
# load math 500 dataset: HuggingFaceH4/MATH-500
dataset = load_dataset("HuggingFaceH4/MATH-500")


model, tokenizer, accelerator = load_model(
    MODEL_NAME, TORCH_DTYPE, device="auto")


# %%
# Text Generation
if "Llama" in MODEL_NAME:
    BOS = 128000
    USER = 128011
    ASSISTANT = 128012
    NEWLINE = 198
    THINK_START = 128013
    THINK_END = 128014
    EOS = 128001
elif "Qwen" in MODEL_NAME:
    BOS = 151646
    USER = 151644
    ASSISTANT = 151645
    NEWLINE = 198
    THINK_START = 151648
    THINK_END = 151649
    EOS = 151643
else:
    raise ValueError(f"Unknown tokens for model {MODEL_NAME}")


def prompt_from_example(example, tokenizer):
    user_message = example['problem']
    math_suffix = " Please reason step by step, and put your final answer within \\boxed{}."
    toks = [BOS] + [USER] + tokenizer.encode(user_message+math_suffix, add_special_tokens=False) + [
        ASSISTANT] + [THINK_START] + [NEWLINE]
    # toks = [BOS] + tokenizer.encode(user_message+math_suffix, add_special_tokens=False) + [THINK_START] + [NEWLINE]
    toks = torch.tensor(toks, dtype=torch.long, device=model.device)
    return toks, tokenizer.decode(toks, skip_special_tokens=False)


# %%
def _unwrap_and_get_layers(m):
    base = accelerator.unwrap_model(m)
    # LLaMA-style: base.model.layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base, base.model.layers
    # Fallback if nested
    if hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "layers"):
        return base, base.model.model.layers
    raise AttributeError("Could not locate transformer layers on model.")


@torch.no_grad()
def gen_batch(lm, toks_batch, n_new_toks=50, temperature=0.6, do_sample=True, top_p=0.95, seed=None):
    if seed is not None:
        my_set_seed(seed)

    base, _ = _unwrap_and_get_layers(lm)

    pad_id = tokenizer.pad_token_id
    B = len(toks_batch)
    max_len = max(len(t) for t in toks_batch)
    # Create on model's device
    input_ids = torch.full((B, max_len), pad_id,
                           dtype=torch.long, device=lm.device)
    for i, t in enumerate(toks_batch):
        input_ids[i, -len(t):] = t.clone().detach().to(device=lm.device,
                                                       dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()

    output_ids = base.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=n_new_toks,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        # Add these for speed
        min_new_tokens=100,  # Minimum tokens to generate
        early_stopping=True,  # Stop early if EOS is generated
        return_dict_in_generate=False,
    )
    return {"sequences": output_ids}

# %%


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
    reversed = generated_answer[::-1]
    matches = re.search("}", reversed)
    if matches is None:
        return ""
    generated_answer = generated_answer[:len(
        generated_answer) - matches.start()]
    return generated_answer


# %%
# evaluate in batches and store results in both dataframe and jsonl file
# Remove old output file if it exists
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

df = pd.DataFrame(columns=['problem', 'answer', 'generated_answer', 'correct'])
n_correct = 0

examples = list(dataset['test'])


# Build indexed token sequences so we can keep original positions
all_indexed = []
for i, example in enumerate(examples):
    toks, _ = prompt_from_example(example, tokenizer)
    all_indexed.append((i, toks))


def make_batches(indexed, batch_size):
    items = list(indexed)
    # length-bucketing from longest to shortest (by toks length)
    items.sort(key=lambda it: len(it[1]), reverse=True)
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]


print("Starting generation")

for batch_idx, idx_toks_batch in enumerate(tqdm(make_batches(all_indexed, BATCH_SIZE), desc="Processing batches")):
    toks_batch = [t for _, t in idx_toks_batch]
    outs = gen_batch(model,
                     toks_batch,
                     n_new_toks=N_NEW_TOKS,
                     do_sample=DO_SAMPLE,
                     temperature=TEMPERATURE,
                     top_p=TOP_P,
                     seed=SEED)

    batch_results = []
    for (orig_idx, toks), out in enumerate(zip(idx_toks_batch, outs["sequences"])):
        (orig_idx, toks), out = out  # unzip tuple-of-tuples from enumerate
        example = examples[orig_idx]

        input_text = tokenizer.decode(toks, skip_special_tokens=True)
        generated_text = tokenizer.decode(
            out[len(toks):], skip_special_tokens=True)

        generated_answer = parse_answer(generated_text)
        example['parsed_answer'] = generated_answer
        example['correct'] = grade_answer(generated_answer, example['answer'])
        example['input_text'] = input_text
        example['generated_text'] = generated_text
        n_correct += int(example['correct'])

        df = pd.concat([df, pd.DataFrame([example])], ignore_index=True)
        batch_results.append(example)

    with open(OUTPUT_FILE, 'a') as f:
        for result in batch_results:
            f.write(json.dumps(result) + '\n')
        f.flush()

    # Clear batch results and force garbage collection
    del batch_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# After the main loop, add final cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()