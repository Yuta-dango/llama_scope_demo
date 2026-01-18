"""SAE encoding utilities."""

from typing import List, Tuple

import torch
from pydantic import BaseModel, Field
from sae_lens import SAE


class TokenResult(BaseModel):
    """Result for a single token's SAE encoding."""
    
    token_pos: int = Field(..., description="Token position in sequence")
    token_str: str = Field(..., description="Token string")
    nnz: int = Field(..., description="Number of non-zero features")
    feat_ids: List[int] = Field(..., description="Active feature IDs")
    feat_vals: List[float] = Field(..., description="Active feature values")


def get_hidden_states(
    sentence: str,
    model,
    tokenizer,
) -> Tuple[torch.Tensor, List[str]]:
    """Get all hidden states from the model for a sentence.
    
    Args:
        sentence: Input sentence to encode
        model: Language model
        tokenizer: Tokenizer
        
    Returns:
        Tuple of (hidden_states, tokens)
        - hidden_states: Tuple of tensors, one per layer (batch, seq_len, hidden_dim)
        - tokens: List of token strings
    """
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    
    # Get all hidden states in one forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    # Get token strings
    tokens = [tokenizer.decode([input_ids[0, i].item()]) for i in range(input_ids.shape[1])]
    
    return outputs.hidden_states, tokens


def encode_hidden_states(
    hidden_states: torch.Tensor,
    tokens: List[str],
    sae: SAE,
    exclude_bos: bool = True,
) -> List[TokenResult]:
    """Encode hidden states using SAE and return token-level features.
    
    Args:
        hidden_states: Hidden states for a specific layer (batch, seq_len, hidden_dim)
        tokens: List of token strings
        sae: SAE for the specified layer
        exclude_bos: Whether to exclude BOS token from results
        
    Returns:
        List of TokenResult objects, one per token
    """
    # Encode with SAE
    sae_output = sae.encode(hidden_states)  # (batch, seq_len, n_features)
    
    # Process each token
    results = []
    seq_len = len(tokens)
    
    for token_pos in range(seq_len):
        token_str = tokens[token_pos]
        
        # Skip BOS token if requested
        if exclude_bos and token_str in ["<|begin_of_text|>", "<s>", "[BOS]"]:
            continue
        
        # Get active features for this token
        features = sae_output[0, token_pos]  # (n_features,)
        
        # Find non-zero features
        nonzero_mask = features != 0
        nonzero_vals = features[nonzero_mask]
        nonzero_ids = torch.where(nonzero_mask)[0]
        
        # Sort by values in descending order
        sorted_indices = torch.argsort(nonzero_vals, descending=True)
        feat_ids = nonzero_ids[sorted_indices].tolist()
        feat_vals = nonzero_vals[sorted_indices].tolist()
        
        result = TokenResult(
            token_pos=token_pos,
            token_str=token_str,
            nnz=len(feat_ids),
            feat_ids=feat_ids,
            feat_vals=feat_vals,
        )
        results.append(result)
    
    return results
