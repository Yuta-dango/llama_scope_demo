"""Data loading utilities for visualization."""

import json
from pathlib import Path
from typing import Dict, List, Any

from pydantic import BaseModel, Field


class TokenData(BaseModel):
    """Token data from JSONL file."""
    
    token_pos: int = Field(..., description="Token position in sequence")
    token_str: str = Field(..., description="Token string")
    nnz: int = Field(..., description="Number of non-zero features")
    feat_ids: List[int] = Field(..., description="Active feature IDs")
    feat_vals: List[float] = Field(..., description="Active feature values")


def load_sentence_data(
    run_dir: Path,
    layer: int,
    category: str,
    sentence_id: int,
) -> List[TokenData]:
    """Load token data from a JSONL file.
    
    Args:
        run_dir: Path to the run directory
        layer: Layer number
        category: Category name
        sentence_id: Sentence ID
        
    Returns:
        List of TokenData objects
    """
    jsonl_path = run_dir / f"layer_{layer:02d}" / category / f"sentence_{sentence_id}.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    tokens = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                tokens.append(TokenData(**data))
    
    return tokens


def load_all_data(run_dir: Path) -> Dict[int, Dict[str, Dict[int, List[TokenData]]]]:
    """Load all data from a run directory.
    
    Returns:
        Nested dict: layer -> category -> sentence_id -> List[TokenData]
    """
    run_dir = Path(run_dir)
    all_data: Dict[int, Dict[str, Dict[int, List[TokenData]]]] = {}
    
    # Find all layer directories
    for layer_dir in sorted(run_dir.glob("layer_*")):
        if not layer_dir.is_dir():
            continue
        
        layer = int(layer_dir.name.split("_")[1])
        all_data[layer] = {}
        
        # Find all category directories
        for category_dir in sorted(layer_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            all_data[layer][category] = {}
            
            # Find all sentence files
            for jsonl_file in sorted(category_dir.glob("sentence_*.jsonl")):
                sentence_id = int(jsonl_file.stem.split("_")[1])
                
                tokens = []
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            tokens.append(TokenData(**data))
                
                all_data[layer][category][sentence_id] = tokens
    
    return all_data


def get_available_layers(run_dir: Path) -> List[int]:
    """Get list of available layers in run directory."""
    run_dir = Path(run_dir)
    layers = []
    
    for layer_dir in run_dir.glob("layer_*"):
        if layer_dir.is_dir():
            layer = int(layer_dir.name.split("_")[1])
            layers.append(layer)
    
    return sorted(layers)


def load_input_csv(csv_path: Path) -> Dict[int, str]:
    """Load input CSV to get original sentences.
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        Dict mapping sentence_id -> original text
    """
    import csv
    
    csv_path = Path(csv_path)
    sentences = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence_id = int(row["id"])
            text = row["text"]
            sentences[sentence_id] = text
    
    return sentences
