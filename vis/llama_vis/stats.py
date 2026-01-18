"""Statistics calculation utilities."""

from collections import defaultdict
from typing import Dict, List

from .loader import TokenData


def calc_nnz_stats(
    all_data: Dict[int, Dict[str, Dict[int, List[TokenData]]]]
) -> Dict[int, List[int]]:
    """Calculate nnz (number of active features) for each layer.
    
    Args:
        all_data: Nested dict from load_all_data
        
    Returns:
        Dict mapping layer -> list of nnz values (one per token)
    """
    nnz_stats: Dict[int, List[int]] = {}
    
    for layer, categories in all_data.items():
        nnz_list = []
        for category, sentences in categories.items():
            for sentence_id, tokens in sentences.items():
                for token in tokens:
                    nnz_list.append(token.nnz)
        nnz_stats[layer] = nnz_list
    
    return nnz_stats


def calc_global_feat_mean(
    all_data: Dict[int, Dict[str, Dict[int, List[TokenData]]]],
    layer: int,
) -> Dict[int, float]:
    """Calculate global mean of feature values for a layer.
    
    For each feature ID, calculates the mean value across ALL tokens.
    Tokens where the feature is not active contribute 0 to the sum.
    
    Args:
        all_data: Nested dict from load_all_data
        layer: Layer number
        
    Returns:
        Dict mapping feat_id -> mean value
    """
    if layer not in all_data:
        return {}
    
    # Count total tokens and accumulate feature values
    total_tokens = 0
    feat_sums: Dict[int, float] = defaultdict(float)
    
    for category, sentences in all_data[layer].items():
        for sentence_id, tokens in sentences.items():
            for token in tokens:
                total_tokens += 1
                for feat_id, feat_val in zip(token.feat_ids, token.feat_vals):
                    feat_sums[feat_id] += feat_val
    
    if total_tokens == 0:
        return {}
    
    # Calculate means
    feat_means = {
        feat_id: total / total_tokens
        for feat_id, total in feat_sums.items()
    }
    
    return feat_means


def calc_feat_rarity_scores(
    all_data: Dict[int, Dict[str, Dict[int, List[TokenData]]]],
    layer: int,
    max_rank_score: int = 100,
) -> Dict[int, float]:
    """Calculate rarity scores for features based on their ranking across tokens.
    
    Algorithm:
    1. For each token, rank features by value (1st place = max_rank_score, 2nd = max_rank_score-1, ...)
    2. Average these rank scores across all tokens
    3. Higher average = feature appears in top positions more often = more common
    4. Lower average = feature appears rarely or in low positions = rarer
    
    Args:
        all_data: Nested dict from load_all_data
        layer: Layer number
        max_rank_score: Score for 1st place (default 100)
        
    Returns:
        Dict mapping feat_id -> average rank score (0 to max_rank_score)
        Lower score = rarer feature
    """
    if layer not in all_data:
        return {}
    
    # Accumulate rank scores for each feature
    total_tokens = 0
    feat_score_sums: Dict[int, float] = defaultdict(float)
    
    for category, sentences in all_data[layer].items():
        for sentence_id, tokens in sentences.items():
            for token in tokens:
                total_tokens += 1
                
                # Features are already sorted by value (descending)
                # Assign scores: 1st place = max_rank_score, 2nd = max_rank_score-1, ...
                for rank, feat_id in enumerate(token.feat_ids):
                    score = max(0, max_rank_score - rank)
                    feat_score_sums[feat_id] += score
    
    if total_tokens == 0:
        return {}
    
    # Calculate average scores
    feat_rarity_scores = {
        feat_id: total / total_tokens
        for feat_id, total in feat_score_sums.items()
    }
    
    return feat_rarity_scores
