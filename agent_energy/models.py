from __future__ import annotations

import torch

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_model(name: str, dtype: str, device: torch.device):
    """Load a Hugging Face causal LM, or a tiny random model for name='tiny'."""
    if name == "tiny":
        return tiny_model().to(device)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=DTYPES[dtype], attn_implementation="sdpa"
    )
    return model.to(device).eval()


def tiny_model(seed: int = 0):
    """A 2-layer random Llama for testing the harness on CPU. Its energy means nothing."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
    )
    return LlamaForCausalLM(cfg).eval()
