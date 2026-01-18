"""Configuration management for visualization."""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    """Input configuration."""
    
    run_dir: str = Field(..., description="SAE extraction result directory")
    input_csv: str = Field("data/input.csv", description="Original input CSV for sentence text")


class OutputConfig(BaseModel):
    """Output configuration."""
    
    base_dir: str = Field("vis_outputs", description="Base output directory")


class CommonConfig(BaseModel):
    """Common visualization settings."""
    
    ax_cols: int = Field(3, description="Number of columns in figure")


class HistogramConfig(BaseModel):
    """Histogram settings."""
    
    bins: int = Field(30, description="Number of histogram bins")


class TokenComparisonConfig(BaseModel):
    """Token comparison settings."""
    
    top_n: int = Field(10, description="Number of top features to display")
    max_rank_score: int = Field(100, description="Score for 1st place feature in rarity calculation")


class TokenSpec(BaseModel):
    """Token specification for comparison."""
    
    category: str = Field(..., description="Category name")
    sentence_id: int = Field(..., description="Sentence ID")
    token_pos: int = Field(..., description="Token position")


class ComparePattern(BaseModel):
    """A comparison pattern with name and tokens."""
    
    name: str = Field(..., description="Pattern name for output file")
    layers: List[int] = Field(default_factory=list, description="Layers to process (empty = all)")
    tokens: List[TokenSpec] = Field(default_factory=list, description="Tokens to compare")


class CompareTokensConfig(BaseModel):
    """Compare tokens within same layer settings."""
    
    patterns: List[ComparePattern] = Field(default_factory=list, description="List of comparison patterns")


class CompareLayersConfig(BaseModel):
    """Compare across layers settings."""
    
    patterns: List[ComparePattern] = Field(default_factory=list, description="List of comparison patterns")


class ShowTokensConfig(BaseModel):
    """Show tokens command settings."""
    
    output_file: str = Field("tokenized.csv", description="Output CSV filename")


class VisConfig(BaseModel):
    """Main visualization configuration."""
    
    input: InputConfig
    output: OutputConfig
    common: CommonConfig = Field(default_factory=CommonConfig)
    histogram: HistogramConfig = Field(default_factory=HistogramConfig)
    token_comparison: TokenComparisonConfig = Field(default_factory=TokenComparisonConfig)
    compare_tokens: CompareTokensConfig = Field(default_factory=CompareTokensConfig)
    compare_layers: CompareLayersConfig = Field(default_factory=CompareLayersConfig)
    show_tokens: ShowTokensConfig = Field(default_factory=ShowTokensConfig)


def load_vis_config(config_path: str | Path) -> VisConfig:
    """Load visualization configuration from YAML file.
    
    Args:
        config_path: Path to the vis_config.yaml file
        
    Returns:
        Parsed configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return VisConfig(**config_dict)
