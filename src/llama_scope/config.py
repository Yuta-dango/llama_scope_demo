"""Configuration management for Llama Scope SAE extraction."""

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""
    
    model_id: str = Field(..., description="Hugging Face model ID")
    sae_release: str = Field(..., description="SAE release name")
    device: Literal["cpu", "cuda", "mps"] = Field("cuda", description="Device to use")
    dtype: Literal["float32", "float16", "bfloat16"] = Field("bfloat16", description="Data type")


class InputConfig(BaseModel):
    """Input configuration."""
    
    csv_path: str = Field(..., description="CSV file path")
    id_column: str = Field("id", description="ID column name")
    text_column: str = Field("text", description="Text column name")
    category_column: str = Field("category", description="Category column name")
    categories: List[str] = Field(default_factory=list, description="List of categories to process (empty = all)")


class OutputConfig(BaseModel):
    """Output configuration."""
    
    base_dir: str = Field("outputs", description="Base output directory")
    run_id_format: str = Field("run_%Y%m%d_%H%M%S", description="Run ID format string")


class TokenConfig(BaseModel):
    """Token processing configuration."""
    
    exclude_bos: bool = Field(True, description="Exclude BOS token from output")


class Config(BaseModel):
    """Main configuration."""
    
    model: ModelConfig
    layers: List[int] = Field(..., description="List of layers to process")
    input: InputConfig
    output: OutputConfig
    token: TokenConfig


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        Parsed configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If config validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)
