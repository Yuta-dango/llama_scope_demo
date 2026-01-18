"""Plotting utilities for visualization."""

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .loader import TokenData

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'


def plot_nnz_histogram(
    nnz_stats: Dict[int, List[int]],
    output_path: Path,
    ax_cols: int = 3,
    bins: int = 30,
) -> None:
    """Plot histogram of nnz (number of active features) for all layers.
    
    Args:
        nnz_stats: Dict mapping layer -> list of nnz values
        output_path: Path to save the figure
        ax_cols: Number of columns in the figure
        bins: Number of histogram bins
    """
    layers = sorted(nnz_stats.keys())
    n_layers = len(layers)
    
    if n_layers == 0:
        return
    
    ax_rows = math.ceil(n_layers / ax_cols)
    
    fig, axes = plt.subplots(ax_rows, ax_cols, figsize=(5 * ax_cols, 4 * ax_rows))
    
    # Flatten axes for easier indexing
    if n_layers == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for idx, layer in enumerate(layers):
        ax = axes[idx]
        nnz_values = nnz_stats[layer]
        
        if len(nnz_values) == 0:
            ax.set_title(f"Layer {layer} (no data)")
            continue
        
        mean_val = np.mean(nnz_values)
        
        ax.hist(nnz_values, bins=bins, edgecolor="black", alpha=0.7)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Number of Active Features")
        ax.set_ylabel("Count")
        ax.legend()
    
    # Hide unused axes
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_base_color_for_feat(feat_id: int, n_colors: int = 20) -> tuple:
    """Get a consistent base hue for a feature ID.
    
    Uses feat_id to deterministically select a hue from a colormap,
    ensuring the same feat_id always gets the same base color.
    
    Args:
        feat_id: Feature ID
        n_colors: Number of distinct base colors to cycle through
        
    Returns:
        Base HSV hue value (0-1)
    """
    # Use a hash-based approach for consistent color assignment
    hue = (feat_id % n_colors) / n_colors
    return hue


def get_color_for_feat(
    feat_id: int,
    feat_rarity_scores: Dict[int, float],
    max_rank_score: int = 100,
) -> str:
    """Get color for a feature based on its ID (base hue) and rarity score (saturation/brightness).
    
    Color scheme:
    - Each feat_id has a unique base hue (consistent across axes)
    - Lower rarity score (rarer feature) -> more saturated/darker color (stands out)
    - Higher rarity score (common feature) -> lighter/more washed out color
    
    Rarity score algorithm:
    - 1st place in a token = max_rank_score points, 2nd = max_rank_score-1, ...
    - Averaged across all tokens
    - High score = common feature, Low score = rare feature
    
    This allows:
    1. Same feat_id = same hue across different axes (easy to compare)
    2. Rarer features stand out more (higher visual importance)
    """
    import colorsys
    
    # Get base hue for this feat_id
    base_hue = get_base_color_for_feat(feat_id)
    
    # Calculate intensity based on rarity score (lower score = rarer = more intense)
    if feat_id not in feat_rarity_scores or len(feat_rarity_scores) == 0:
        saturation = 0.7
        value = 0.8
    else:
        score = feat_rarity_scores[feat_id]
        
        # Normalize score to 0-1 range (0 = rarest, 1 = most common)
        # Score range is 0 to max_rank_score
        norm_val = min(1.0, score / max_rank_score)
        
        # Lower score (rarer) -> higher saturation (more vivid)
        # Higher score (common) -> lower saturation (outline only)
        # score=0 -> saturation=1.0 (full color)
        # score=100 -> saturation=0.0 (no fill, outline only)
        saturation = 1.0 - norm_val  # 0.0 to 1.0
        value = 0.9                   # Fixed brightness
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(base_hue, saturation, value)
    return mcolors.to_hex(rgb)


def plot_token_bars(
    token_data: TokenData,
    feat_rarity_scores: Dict[int, float],
    top_n: int,
    ax: plt.Axes,
    title: Optional[str] = None,
    max_rank_score: int = 100,
) -> None:
    """Plot bar chart of top N features for a single token.
    
    Args:
        token_data: TokenData object
        feat_rarity_scores: Dict mapping feat_id -> average rank score
        top_n: Number of top features to display
        ax: Matplotlib axes to plot on
        title: Optional title override
        max_rank_score: Maximum rank score for normalization
    """
    # Get top N features (already sorted by value)
    n_feats = min(top_n, len(token_data.feat_ids))
    feat_ids = token_data.feat_ids[:n_feats]
    feat_vals = token_data.feat_vals[:n_feats]
    
    if n_feats == 0:
        ax.set_title(title or f"pos={token_data.token_pos}: '{token_data.token_str}' (no features)")
        return
    
    # Get colors: feat_id determines base hue, rarity score determines intensity
    colors = [get_color_for_feat(fid, feat_rarity_scores, max_rank_score) for fid in feat_ids]
    
    # Create bar chart
    x_pos = range(n_feats)
    bars = ax.bar(x_pos, feat_vals, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add rarity score labels inside each bar
    for i, (bar, fid) in enumerate(zip(bars, feat_ids)):
        score = feat_rarity_scores.get(fid, 0)
        bar_height = bar.get_height()
        # Position text at 50% height of the bar
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar_height * 0.5,
            f"{score:.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    
    # Set x-axis labels as feature IDs
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(fid) for fid in feat_ids], rotation=45, ha="right", fontsize=7)
    
    ax.set_ylabel("Feature Value")
    ax.set_title(title or f"pos={token_data.token_pos}: '{token_data.token_str}'")


def plot_sentence_figure(
    sentence_data: List[TokenData],
    feat_rarity_scores: Dict[int, float],
    top_n: int,
    ax_cols: int,
    output_path: Path,
    max_rank_score: int = 100,
) -> None:
    """Plot all tokens in a sentence as a single figure.
    
    Args:
        sentence_data: List of TokenData objects for one sentence
        feat_rarity_scores: Dict mapping feat_id -> average rank score
        top_n: Number of top features to display
        ax_cols: Number of columns in the figure
        output_path: Path to save the figure
        max_rank_score: Maximum rank score for normalization
    """
    n_tokens = len(sentence_data)
    
    if n_tokens == 0:
        return
    
    ax_rows = math.ceil(n_tokens / ax_cols)
    
    fig, axes = plt.subplots(ax_rows, ax_cols, figsize=(5 * ax_cols, 4 * ax_rows))
    
    # Flatten axes for easier indexing
    if n_tokens == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for idx, token_data in enumerate(sentence_data):
        plot_token_bars(token_data, feat_rarity_scores, top_n, axes[idx], max_rank_score=max_rank_score)
    
    # Hide unused axes
    for idx in range(n_tokens, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_compare_tokens(
    tokens_data: List[tuple],  # List of (TokenData, label)
    feat_rarity_scores: Dict[int, float],
    top_n: int,
    ax_cols: int,
    output_path: Path,
    max_rank_score: int = 100,
) -> None:
    """Plot comparison of specified tokens within a layer.
    
    Args:
        tokens_data: List of (TokenData, label) tuples
        feat_rarity_scores: Dict mapping feat_id -> average rank score
        top_n: Number of top features to display
        ax_cols: Number of columns in the figure
        output_path: Path to save the figure
        max_rank_score: Maximum rank score for normalization
    """
    n_tokens = len(tokens_data)
    
    if n_tokens == 0:
        return
    
    ax_rows = math.ceil(n_tokens / ax_cols)
    
    fig, axes = plt.subplots(ax_rows, ax_cols, figsize=(5 * ax_cols, 4 * ax_rows))
    
    # Flatten axes for easier indexing
    if n_tokens == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for idx, (token_data, label) in enumerate(tokens_data):
        plot_token_bars(token_data, feat_rarity_scores, top_n, axes[idx], title=label, max_rank_score=max_rank_score)
    
    # Hide unused axes
    for idx in range(n_tokens, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_highlighted_title(sentence: str, token_str: str, token_pos: int, max_words_per_line: int = 10) -> str:
    """Create a title with the target token highlighted.
    
    Since matplotlib doesn't support inline formatting easily,
    we'll mark the token position with brackets for visibility.
    If sentence exceeds max_words_per_line, split into 2 lines.
    
    Args:
        sentence: Original sentence
        token_str: Target token string
        token_pos: Token position in sequence
        max_words_per_line: Maximum words before line break
        
    Returns:
        Formatted title string
    """
    # Token strings often have leading space, so we try to highlight the word
    clean_token = token_str.strip()
    
    # Find and highlight the N-th occurrence of the token
    # Count occurrences to find the correct one based on token_pos
    if clean_token in sentence:
        # Find all occurrences
        occurrences = []
        start = 0
        while True:
            idx = sentence.find(clean_token, start)
            if idx == -1:
                break
            occurrences.append(idx)
            start = idx + 1
        
        if len(occurrences) == 1:
            # Only one occurrence, highlight it
            highlighted = sentence.replace(clean_token, f"【{clean_token}】", 1)
        else:
            # Multiple occurrences - append position info to clarify
            highlighted = sentence.replace(clean_token, f"【{clean_token}】", 1)
            # Replace back, then do position-aware replacement
            # For simplicity, just append the position info
            highlighted = f"{sentence} (pos {token_pos}: '{clean_token}')"
    else:
        # Fallback: just append token info
        highlighted = f"{sentence} (pos {token_pos}: '{token_str}')"
    
    # Split into 2 lines if too long
    words = highlighted.split()
    if len(words) > max_words_per_line:
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        return f"{line1}\n{line2}"
    
    return highlighted


def plot_compare_layers(
    tokens_by_layer: Dict[int, List[tuple]],  # layer -> List of (TokenData, sentence, feat_rarity_scores)
    top_n: int,
    output_path: Path,
    max_rank_score: int = 100,
) -> None:
    """Plot comparison across layers.
    
    Args:
        tokens_by_layer: Dict mapping layer -> List of (TokenData, sentence, feat_rarity_scores)
        top_n: Number of top features to display
        output_path: Path to save the figure
        max_rank_score: Maximum rank score for normalization
    """
    layers = sorted(tokens_by_layer.keys())
    
    if len(layers) == 0:
        return
    
    # Get number of tokens from first layer
    n_tokens = len(tokens_by_layer[layers[0]])
    n_layers = len(layers)
    
    if n_tokens == 0:
        return
    
    fig, axes = plt.subplots(n_layers, n_tokens, figsize=(5 * n_tokens, 4 * n_layers))
    
    # Handle single row/column cases
    if n_layers == 1 and n_tokens == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    elif n_tokens == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, layer in enumerate(layers):
        tokens_data = tokens_by_layer[layer]
        
        for col_idx, (token_data, sentence, feat_rarity_scores) in enumerate(tokens_data):
            ax = axes[row_idx, col_idx]
            
            # Create highlighted title showing full sentence with token marked
            highlighted_title = create_highlighted_title(
                sentence, token_data.token_str, token_data.token_pos
            )
            title = f"Layer {layer}: {highlighted_title}"
            
            plot_token_bars(token_data, feat_rarity_scores, top_n, ax, title=title, max_rank_score=max_rank_score)
            
            # Only show y-axis label on first column
            if col_idx == 0:
                ax.set_ylabel("Feature Value", fontsize=9)
            else:
                ax.set_ylabel("")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
