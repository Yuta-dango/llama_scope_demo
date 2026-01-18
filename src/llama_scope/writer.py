"""Output writing utilities."""

import json
from pathlib import Path
from typing import List

from .encoder import TokenResult


def write_results(
    results: List[TokenResult],
    output_dir: Path,
    layer: int,
    category: str,
    sentence_id: int,
) -> None:
    """Write SAE encoding results to JSONL file.
    
    Args:
        results: List of TokenResult objects for a sentence
        output_dir: Base output directory (e.g., outputs/run_20231210_120000)
        layer: Layer number
        category: Category name
        sentence_id: Sentence ID from CSV
    """
    # Create layer/category directory
    category_dir = output_dir / f"layer_{layer:02d}" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file path using sentence ID
    output_file = category_dir / f"sentence_{sentence_id}.jsonl"
    
    # Write results as JSONL (one JSON object per line)
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            json_str = result.model_dump_json()
            f.write(json_str + "\n")
