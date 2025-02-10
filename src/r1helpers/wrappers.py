from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os


class R1Wrapper:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                       device="auto", 
                       local_dir="/share/u/models", 
                       dtype=torch.bfloat16,
                       do_sample=True,
                       temperature=0.6,
                       max_length=15000,
                       max_new_tokens=None):
        self.model_name = model_name
        # Text Generation
        if "Llama" in model_name:
            self.BOS = 128000
            self.USER = 128011
            self.ASSISTANT = 128012
            self.NEWLINE = 198
            self.THINK_START = 128013
            self.THINK_END = 128014
            self.EOS = 128001
        elif "Qwen" in model_name:
            self.BOS = 151646
            self.USER = 151644
            self.ASSISTANT = 151645
            self.NEWLINE = 198
            self.THINK_START = 151648
            self.THINK_END = 151649
            self.EOS = 151643
        else:
            raise ValueError(f"Unknown tokens for model {model_name}")

        self.device = device
        self.local_dir = local_dir
        self.dtype = dtype
        # check whether local dir exists and contains model
        if os.path.exists(os.path.join(local_dir, model_name)):
            self.model, self.tokenizer = self.load_model(os.path.join(local_dir, model_name), device=device, dtype=dtype)
        else:
            self.model, self.tokenizer = self.load_model(model_name, device=device, dtype=dtype)
        self.gen = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=do_sample,
            temperature=temperature,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.EOS
        )

    def load_model(self, model_name, device="auto", cache_dir=None, dtype=torch.bfloat16):
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            cache_dir=cache_dir
        )
        return model, tokenizer

    def apply_format(self, user_message, partial_thought=None):
        toks = [self.BOS] + [self.USER] + self.tokenizer.encode(user_message, add_special_tokens=False) + [self.ASSISTANT] + [self.THINK_START] + [self.NEWLINE]
        if partial_thought is not None:
            toks += self.tokenizer.encode(partial_thought, add_special_tokens=False)
        return toks, self.tokenizer.decode(toks, skip_special_tokens=False)
    
    def generate(self, user_message, partial_thought=None):
        toks, text = self.apply_format(user_message, partial_thought)
        # Generate
        with torch.no_grad():
            outputs = self.gen(text)

        return outputs[0]['generated_text'], outputs[0]['generated_text'][len(text):]
    

class R1Prompter:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", local_dir="/share/u/models",):
        self.local_dir = local_dir
        self.model_name = model_name
        # Text Generation
        if "Llama" in model_name:
            self.BOS = 128000
            self.USER = 128011
            self.ASSISTANT = 128012
            self.NEWLINE = 198
            self.THINK_START = 128013
            self.THINK_END = 128014
            self.EOS = 128001
        elif "Qwen" in model_name:
            self.BOS = 151646
            self.USER = 151644
            self.ASSISTANT = 151645
            self.NEWLINE = 198
            self.THINK_START = 151648
            self.THINK_END = 151649
            self.EOS = 151643
        else:
            raise ValueError(f"Unknown tokens for model {model_name}")
        
        if os.path.exists(os.path.join(local_dir, model_name)):
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_dir, model_name))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def apply_format(self, user_message, partial_thought=None):
        toks = [self.BOS] + [self.USER] + self.tokenizer.encode(user_message, add_special_tokens=False) + [self.ASSISTANT] + [self.THINK_START] + [self.NEWLINE]
        if partial_thought is not None:
            toks += self.tokenizer.encode(partial_thought, add_special_tokens=False)
        return toks, self.tokenizer.decode(toks, skip_special_tokens=False)