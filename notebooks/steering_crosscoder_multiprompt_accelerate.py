# %%
# Imports

from collections import defaultdict
from functools import partial
import os
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from transformers import set_seed
import torch
import pandas as pd
from tqdm import tqdm

# Set high precision matmul for faster computation on modern GPUs
torch.set_float32_matmul_precision('high')

# %%
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Fix pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

accelerator = Accelerator()
torch_dtype = torch.bfloat16

# Load model with automatic device distribution
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto",
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)
model.eval()
# Don't use accelerator.prepare() when using device_map="auto"
lm = model  # keep name 'lm' for minimal downstream changes

# %%
# Load crosscoder features to steer with
with open("../assets/l15_examples_backtracking.json", "r") as f:
    l15_examples = json.load(f)
print(l15_examples["explanations"])
layer2featuresidcs = {15: {
    d["feature_id"]: f"backtracking {d['explanation']}" for d in l15_examples["explanations"]}}

# %%
# Load the cross-coder decoder matrices
base_path = "/disk/u/troitskiid/data/checkpoints/L1-Crosscoder"
normalize = True

layer2paths = {7: "L7R/cc_weights.pt",
               15: "L15R/cc_weights.pt", 22: "L23R/cc_weights.pt"}
layer2features = {}
for lidx, path in layer2paths.items():
    if lidx in layer2featuresidcs:
        d = torch.load(os.path.join(base_path, path), map_location="cpu")
        layer2features[lidx] = d["decoder.weight"][1]/d["decoder.weight"][1].norm(
            dim=-1, keepdim=True) if normalize else d["decoder.weight"][1]

# %%


def my_set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def find_wait_token_ids(tokenizer):
    """Finds token IDs for ' wait' and ' Wait'."""
    tokens_to_check = ["wait", "Wait", " wait", " Wait"]
    token_ids = set()
    print("Attempting to encode potential 'wait' tokens:")  # Debug
    for token_str in tokens_to_check:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        print(f"  - '{token_str}' -> IDs: {ids}")  # Debug
        if len(ids) == 1:
            token_ids.add(ids[0])
        elif len(ids) > 1:
            # This warning might be important if 'wait' isn't a single token sometimes
            print(
                f"  - Warning: Token '{token_str}' split into multiple IDs: {ids}. Adding first: {ids[0]}")
            token_ids.add(ids[0])

    if not token_ids:
        raise ValueError("Could not find token IDs for 'wait' or 'Wait'.")
    print(f"Found 'wait'/'Wait' related token IDs: {token_ids}")
    return token_ids


# %%
wait_toks = find_wait_token_ids(tokenizer)
print(wait_toks)

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
def gen(lm, toks, n_new_toks=50, temperature=0.6, do_sample=True, top_p=0.95, seed=42):
    if seed is not None:
        my_set_seed(seed)

    base, _ = _unwrap_and_get_layers(lm)
    # Remove device specification
    input_ids = torch.tensor([toks], dtype=torch.long)
    # The model will handle moving tensors to the right device
    output_ids = base.generate(
        input_ids=input_ids,
        max_new_tokens=n_new_toks,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=False,
        cache_implementation="static"
    )
    return {"sequences": output_ids}


@torch.no_grad()
def steer(lm, toks, vecs,
          from_tok_idx=-1,
          mode="all",
          layer_idcs=[],
          n_new_toks=10,
          alpha=0.0,
          thres=-0.01,
          do_sample=True,
          temperature=0.6,
          top_p=0.95,
          seed=42):
    assert mode in [
        "all", "reactive"], "mode must be either 'all' or 'reactive'"

    base, layers = _unwrap_and_get_layers(lm)

    if from_tok_idx >= 0 and from_tok_idx < len(toks):
        print("WARNING: from_tok_idx is within the input sequence. Is this what you wanted to do?")

    @torch.no_grad()
    def steer_hook(module, input, output, from_tok_idx=None, vec=None, coef=None, thres=None):
        h = output[0] if isinstance(output, tuple) else output
        vec_ = vec.to(h.device, dtype=h.dtype)

        # Normalize to [S, D] (single sample)
        if h.dim() == 3:
            h2 = h[0]
        else:
            h2 = h
        S = h2.shape[0]

        if S > 1:
            start = from_tok_idx if (from_tok_idx is not None and from_tok_idx >= 0) else (
                S - 1 if from_tok_idx == -1 else 0)
            start = max(0, min(start, S - 1))
            tok_norms = h2[start:].norm(dim=-1, keepdim=True)
            h2[start:] += tok_norms * coef * vec_
        else:
            v1 = h2[0]
            v1norm = v1.norm()
            if mode == "all":
                h2[0] += v1norm * coef * vec_
            else:
                # reactive
                v1n = v1 / (v1norm + 1e-12)
                v2n = vec_ / (vec_.norm() + 1e-12)
                proj = torch.dot(v1n, v2n)
                if proj < thres:
                    h2[0] += v1norm * coef * vec_

        return output

    myhooks = [layers[lidx].register_forward_hook(
        partial(steer_hook, from_tok_idx=from_tok_idx, vec=vecs[idx], coef=alpha, thres=thres)) for idx, lidx in enumerate(layer_idcs)]
    try:
        out_ids = base.generate(
            input_ids=torch.tensor(
                [toks], dtype=torch.long, device=accelerator.device),
            max_new_tokens=n_new_toks,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False,
            cache_implementation="static"
        )
    finally:
        for hook in myhooks:
            hook.remove()
    return {"sequences": out_ids}

# %%


@torch.no_grad()
def gen_batch(lm, toks_batch, n_new_toks=50, temperature=0.6, do_sample=True, top_p=0.95, seed=None):
    if seed is not None:
        my_set_seed(seed)

    base, _ = _unwrap_and_get_layers(lm)

    pad_id = tokenizer.pad_token_id
    B = len(toks_batch)
    max_len = max(len(t) for t in toks_batch)
    input_ids = torch.full((B, max_len), pad_id,
                           dtype=torch.long)  # Remove device specification
    for i, t in enumerate(toks_batch):
        input_ids[i, -len(t):] = torch.tensor(t,
                                              dtype=torch.long)  # Remove device specification
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
        return_dict_in_generate=False,
    )
    return {"sequences": output_ids}


@torch.no_grad()
def steer_batch(lm, toks_batch, vecs,
                from_tok_idx=-1,
                mode="all",
                layer_idcs=[],
                n_new_toks=10,
                alpha=0.0,
                thres=-0.01,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                seed=42):

    assert mode in ["all", "reactive"]

    base, layers = _unwrap_and_get_layers(lm)

    # Track reactive stats per-sample
    B = len(toks_batch)
    fire_cnt = torch.zeros(B, dtype=torch.int64, device=accelerator.device)
    gen_steps = 0  # counts S==1 steps

    @torch.no_grad()
    def steer_hook(module, inputs, output, from_tok_idx=None, vec=None, coef=None, thres=None):
        nonlocal fire_cnt, gen_steps
        h = output[0] if isinstance(output, tuple) else output
        vec_ = vec.to(h.device, dtype=h.dtype)

        # Normalize shape to [B, S, D]
        if h.dim() == 3:
            h2 = h  # [B,S,D]
        else:
            h2 = h.unsqueeze(0)  # [1,S,D]

        Bcur, S, D = h2.shape

        if S > 1:
            start = from_tok_idx if (from_tok_idx is not None and from_tok_idx >= 0) else (
                S - 1 if from_tok_idx == -1 else 0)
            start = max(0, min(start, S - 1))
            tok_norms = h2[:, start:, :].norm(
                dim=-1, keepdim=True)  # [B,S-start,1]

            v = vec_
            if v.dim() == 1:            # [D] -> [1,1,D]
                v = v.view(1, 1, -1)
            elif v.dim() == 2:          # [B,D] -> [B,1,D]
                v = v.view(Bcur, 1, -1)

            h2[:, start:, :] += tok_norms * coef * v
        else:
            v1 = h2[:, 0, :]                         # [B,D]
            v1norm = v1.norm(dim=-1, keepdim=True)   # [B,1]

            if mode == "all":
                v = vec_
                if v.dim() == 1:  # [D] -> [1,D]
                    v = v.view(1, -1)
                h2[:, 0, :] += v1norm * coef * v
            else:
                # reactive
                v = vec_
                if v.dim() == 1:
                    v = v.view(1, -1)               # [1,D] (broadcast)
                v1n = v1 / (v1norm + 1e-12)
                v2n = v / (v.norm(dim=-1, keepdim=True) +
                           1e-12)  # [B,D] or [1,D]
                proj = (v1n * v2n).sum(dim=-1)      # [B]
                mask = proj < thres
                if mask.any():
                    if v.dim() == 2 and v.shape[0] == Bcur:
                        h2[mask, 0, :] += v1norm[mask] * coef * v[mask]
                    else:
                        h2[mask, 0, :] += v1norm[mask] * coef * v
                fire_cnt += mask.to(fire_cnt.dtype)

            gen_steps += 1

        return output  # modify in-place

    # One vec per layer_idcs entry, as in your original code
    hooks = [
        layers[lidx].register_forward_hook(
            partial(steer_hook, from_tok_idx=from_tok_idx,
                    vec=vecs[idx], coef=alpha, thres=thres)
        )
        for idx, lidx in enumerate(layer_idcs)
    ]

    pad_id = tokenizer.pad_token_id
    B = len(toks_batch)
    max_len = max(len(t) for t in toks_batch)
    input_ids = torch.full((B, max_len), pad_id,
                           dtype=torch.long)  # Remove device specification
    for i, t in enumerate(toks_batch):
        input_ids[i, -len(t):] = torch.tensor(t,
                                              dtype=torch.long)  # Remove device specification
    attention_mask = (input_ids != pad_id).long()

    try:
        out_ids = base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=n_new_toks,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=False,
        )
    finally:
        for h in hooks:
            h.remove()

    result = {"sequences": out_ids}
    if mode == "reactive" and gen_steps > 0:
        result["fire_fraction"] = (
            100.0 * fire_cnt.float() / gen_steps).tolist()  # per-sample
    return result

# %%
with open("../assets/wait_subsequences_from_outputs.json", "r") as f:
    wait_subsequences = json.load(f)

# %%
len(wait_subsequences)

# %%
mode2strenghts = {
    # "all": [1.5, 1.25, 1, 0.75, -0.75, -1, -1.25, -1.5],
    "all": [0, 1.5],
    # [32, 16, 8] # ,-8, -16, -32]
    "reactive": [5, 4, 3, 2, 1.5, 1, 0.5, -0.5, -1, -1.5, -2, -3, -4, -5],
}
modes = ["all"]  # ["all", "reactive"]

n_new_toks = 1024
n_rollouts_per_prompt = 3

outfile = "../results/multiple_prompts_steering_l1_crosscoder.csv"

# %%
# filter the sequences
dataset = []
counts = defaultdict(int)
seeds = defaultdict(set)
for d in wait_subsequences:
    if d["config"] == "Sampling (DeepSeek recommended)":
        if counts[d["prompt"]] < n_rollouts_per_prompt and d["seed"] not in seeds[d["prompt"]]:
            toks = d["subsequence_tokens"]
            # make sure that the generated output contains "wait" as the first token of the output
            out = gen(lm, toks, n_new_toks=1, seed=d["seed"])
            if "wait" in tokenizer.decode(out["sequences"][0]).lower(): 
                counts[d["prompt"]] += 1
                seeds[d["prompt"]].add(d["seed"])
                dataset.append(d)
print(len(dataset))

# %%
# FIXME test feature list
test_features = [744]  # , 31748, 25929, 188]

# %%
batch_size = 8  # tune based on VRAM
rows = []

# Remove if you want to start fresh
if os.path.exists(outfile):
    os.remove(outfile)


def decode_after_wait(output_ids, input_ids, tokenizer):
    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.tolist()
    pad_id = tokenizer.pad_token_id
    lp = 0
    while lp < len(output_ids) and output_ids[lp] == pad_id:
        lp += 1
    return tokenizer.decode(output_ids[lp + len(input_ids):], skip_special_tokens=True)


def make_batches(dataset, batch_size):
    items = list(dataset)
    items.sort(key=lambda d: len(d["subsequence_tokens"]))  # length-bucketing
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]


for batch in make_batches(dataset, batch_size):
    toks_batch = [d["subsequence_tokens"] for d in batch]
    wait_idx = [d["wait_token_index_in_original"] for d in batch]

    # All should end with wait token per your assert
    assert all(len(t) == w for t, w in zip(toks_batch, wait_idx)
               ), "wait_tok_idx must be last token for all samples"

    # Reference (no steering)
    out_ref = gen_batch(lm,
                        toks_batch,
                        n_new_toks=n_new_toks,
                        do_sample=False,
                        temperature=0.0,
                        top_p=0.95,
                        seed=42
                        )

    # Interventions (batched) - precompute all interventions
    intervention_results = {}
    for mode in modes:
        for layer_idx, vecs in layer2features.items():
            for fidx in layer2featuresidcs[layer_idx].keys():
                if fidx not in test_features:
                    continue
                for strength in mode2strenghts[mode]:
                    out = steer_batch(
                        lm, toks_batch,
                        # one vec per selected layer
                        vecs=vecs[fidx].unsqueeze(0),
                        layer_idcs=[layer_idx],
                        mode=mode,
                        alpha=strength,
                        n_new_toks=n_new_toks,
                        from_tok_idx=-1,
                        do_sample=False,
                        temperature=0.0,
                        top_p=0.95,
                        seed=42
                    )
                    intervention_results[(
                        mode, layer_idx, fidx, strength)] = out

    # Now write rows in the correct order: for each dataset item, write reference then all interventions
    for i, d in enumerate(batch):
        # Reference first
        rows.append({
            "seed": 42,
            "layer_idx": -1,
            "feature_idx": -1,
            "feature_summary": "reference",
            "strength": 0,
            "mode": "gen",
            "text_before_wait": tokenizer.decode(toks_batch[i], skip_special_tokens=True),
            "text_after_wait": decode_after_wait(out_ref["sequences"][i], toks_batch[i], tokenizer),
            "full_response": tokenizer.decode(out_ref["sequences"][i], skip_special_tokens=True),
            "prompt": d["prompt"],
            "steering_fraction": 0,
        })

        # Then all interventions for this sample
        for mode in modes:
            for layer_idx, vecs in layer2features.items():
                for fidx in layer2featuresidcs[layer_idx].keys():
                    if fidx not in test_features:
                        continue
                    for strength in mode2strenghts[mode]:
                        out = intervention_results[(
                            mode, layer_idx, fidx, strength)]
                        if isinstance(out, dict) and "fire_fraction" in out:
                            steering_fraction = out["fire_fraction"][i]
                            print(
                                f"steering fraction: {out['fire_fraction'][i]}")
                        else:
                            steering_fraction = 1
                        rows.append({
                            "seed": 42,
                            "layer_idx": layer_idx,
                            "feature_idx": fidx,
                            "feature_summary": layer2featuresidcs[layer_idx][fidx],
                            "strength": strength,
                            "mode": mode,
                            "text_before_wait": tokenizer.decode(toks_batch[i], skip_special_tokens=True),
                            "text_after_wait": decode_after_wait(out["sequences"][i], toks_batch[i], tokenizer),
                            "full_response": tokenizer.decode(out["sequences"][i], skip_special_tokens=True),
                            "prompt": d["prompt"],
                            "steering_fraction": steering_fraction,
                        })

    # Flush this batch to disk in one go
    pd.DataFrame(rows).to_csv(outfile, mode='a',
                              header=not os.path.exists(outfile), index=False)
    rows.clear()

# %%
