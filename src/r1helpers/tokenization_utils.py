from transformers import AutoTokenizer
import torch
from typing import Optional, List, Dict

SPECIAL_TOKEN_MAP = {
    "meta-llama": {
        "BOS": 128000,
        "USER": 128011,
        "ASSISTANT": 128012,
        "NEWLINE": 198,
        "THINK_START": 128013,
        "THINK_END": 128014,
        "EOS": 128001,
    },
    "DeepSeek-R1-Distill-Llama": {
        "BOS": 128000,
        "USER": 128011,
        "ASSISTANT": 128012,
        "NEWLINE": 198,
        "THINK_START": 128013,
        "THINK_END": 128014,
        "EOS": 128001,
    },
    "Qwen": {
        "BOS": 151646,
        "USER": 151644,
        "ASSISTANT": 151645,
        "NEWLINE": 198,
        "THINK_START": 151648,
        "THINK_END": 151649,
        "EOS": 151643,
    },
}

def get_special_tokens(model_name: str) -> Dict[str, int]:
    """
    Get the special tokens for the model.
    """
    st = None
    for model_type in SPECIAL_TOKEN_MAP.keys():
        if model_type in model_name:
            st = SPECIAL_TOKEN_MAP[model_type]
            break
    if st is None:
        raise ValueError(
            f"Unknown model: {model_name}. Model name must contain {SPECIAL_TOKEN_MAP.keys()}"
        )
    return st

def custom_encoding(
    model_name: str,
    tokenizer: AutoTokenizer,
    user_message: str,
    thinking_message: str = "",
    user_suffix: str = "",
    assistant_prefill: str = "",
    template: str = "chat",
) -> List[int]:
    """
    Custom encoding for the model.
    """

    # Verify arguments and get special tokens
    st = get_special_tokens(model_name)
    if "meta-llama" in model_name and (template == "chat" or thinking_message != ""):
        raise ValueError(
            "Meta-Llama models do not support chat or thinking tokens. Use 'base' template instead and do not provide a thinking message."
        )

    # Encode user message
    if user_suffix != "":
        user_message = user_message + " " + user_suffix

    user_tokens = tokenizer.encode(user_message, add_special_tokens=False)

    if template == "base":
        # equivalent to calling tokenzer.encode for deepseek-r1-distilled models
        token_ids = [st["BOS"]] + user_tokens
    elif template == "chat":
        # equivalent to calling tokenizer.apply_chat_template for deepseek-r1-distilled models
        token_ids = [st["BOS"]] + [st["USER"]] + user_tokens + [st["ASSISTANT"]]
    else:
        raise ValueError(f"Unknown template: {template}. Choose from 'base' or 'chat'")

    if assistant_prefill != "":
        assert template == "chat", "Assistant prefix is only supported for chat template"
        assistant_prefill_tokens = tokenizer.encode(assistant_prefill, add_special_tokens=False)
        token_ids = token_ids + assistant_prefill_tokens

    # Optionally prefill thinking tokens
    if len(thinking_message) > 0:
        thinking_tokens = tokenizer.encode(thinking_message, add_special_tokens=False)
        token_ids = token_ids + [st["THINK_START"]] + thinking_tokens

    return token_ids


def custom_batch_encoding(
    model_name: str,
    tokenizer: AutoTokenizer,
    user_messages: List[str],
    thinking_message: str = "",
    user_suffix: str = "",
    assistant_prefill: str = "",
    template: str = "chat",
) -> List[int]:
    """
    Custom batch encoding for the model.
    """

    token_ids = [
        custom_encoding(
            model_name=model_name,
            tokenizer=tokenizer,
            user_message=user_message,
            thinking_message=thinking_message,
            user_suffix=user_suffix,
            assistant_prefill=assistant_prefill,
            template=template,
        )
        for user_message in user_messages
    ]
    return token_ids


def custom_decoding(model_name: str, tokenizer: AutoTokenizer, token_ids_BL: torch.Tensor, skip_special_tokens: bool = False) -> List[str]:
    """
    Custom decoding for the model.
    """
    st = get_special_tokens(model_name)
    token_ids = [id for id in token_ids_BL.tolist() if id != st["EOS"]]
    generated_texts = tokenizer.batch_decode(
        token_ids, 
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    return generated_texts

if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Basic test cases
    user_message = "Hello, how are you?"
    thinking_message = "I am thinking"
    user_suffix = "with urgency"
    assistant_prefill = "Let me think about this."

    print("\n=== Basic Encoding Tests ===")

    base_token_ids = custom_encoding(model_name, tokenizer, user_message, template="base")
    base_token_str = tokenizer.decode(base_token_ids)
    print(f"Base template:\n{base_token_str}\n")

    chat_token_ids = custom_encoding(model_name, tokenizer, user_message, template="chat")
    chat_token_str = tokenizer.decode(chat_token_ids)
    print(f"Chat template:\n{chat_token_str}\n")

    # Test with thinking message
    print("=== With Thinking Message ===")
    chat_thinking_ids = custom_encoding(
        model_name, tokenizer, user_message, thinking_message=thinking_message, template="chat"
    )
    chat_thinking_str = tokenizer.decode(chat_thinking_ids)
    print(f"Chat + thinking:\n{chat_thinking_str}\n")

    # Test with user suffix
    print("=== With User Suffix ===")
    chat_suffix_ids = custom_encoding(
        model_name, tokenizer, user_message, user_suffix=user_suffix, template="chat"
    )
    chat_suffix_str = tokenizer.decode(chat_suffix_ids)
    print(f"Chat + user suffix:\n{chat_suffix_str}\n")

    # Test with assistant prefill
    print("=== With Assistant Prefill ===")
    chat_prefill_ids = custom_encoding(
        model_name, tokenizer, user_message, assistant_prefill=assistant_prefill, template="chat"
    )
    chat_prefill_str = tokenizer.decode(chat_prefill_ids)
    print(f"Chat + assistant prefill:\n{chat_prefill_str}\n")

    # Test batch encoding
    print("=== Batch Encoding Tests ===")
    user_messages = ["What is the weather like?", "Tell me a joke", "How does photosynthesis work?"]

    batch_ids = custom_batch_encoding(model_name, tokenizer, user_messages, template="chat")
    print("Batch encoded messages:")
    for i, ids in enumerate(batch_ids):
        print(f"\nMessage {i + 1}:\n{tokenizer.decode(ids)}")

    # Test combining multiple features
    print("\n=== Combined Features Test ===")
    combined_ids = custom_encoding(
        model_name,
        tokenizer,
        user_message,
        thinking_message=thinking_message,
        user_suffix=user_suffix,
        assistant_prefill=assistant_prefill,
        template="chat",
    )
    combined_str = tokenizer.decode(combined_ids)
    print(f"All features combined:\n{combined_str}")
