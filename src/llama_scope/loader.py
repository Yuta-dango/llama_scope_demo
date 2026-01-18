"""Model and SAE loading utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

from .config import Config


def load_model(config: Config):
    """Load the language model and tokenizer.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[config.model.dtype]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        torch_dtype=torch_dtype,
        device_map=config.model.device,
    )
    model.eval()
    
    return model, tokenizer


def load_sae(config: Config, layer: int) -> SAE:
    """Load SAE for a specific layer.
    
    Args:
        config: Configuration object
        layer: Layer number to load SAE for
        
    Returns:
        SAE object for the specified layer
    """
    # Construct SAE ID for the layer (format: l{layer}r_8x)
    sae_id = f"l{layer}r_8x"
    
    # Load SAE
    sae, _, _ = SAE.from_pretrained(
        release=config.model.sae_release,
        sae_id=sae_id,
        device=config.model.device,
    )
    
    return sae
