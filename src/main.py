"""CLI entry point for Llama Scope SAE feature extraction."""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from llama_scope.config import load_config
from llama_scope.loader import load_model, load_sae
from llama_scope.encoder import get_hidden_states, encode_hidden_states
from llama_scope.writer import write_results

app = typer.Typer()
console = Console()


@app.command()
def main(
    config_path: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to config.yaml file",
    ),
) -> None:
    """Extract SAE features from input sentences.
    
    Optimized: Each sentence is passed through Llama only ONCE,
    then all layer SAEs are applied to the cached hidden states.
    """
    # Load configuration
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    config = load_config(config_path)
    
    # Load CSV file
    csv_path = Path(config.input.csv_path)
    if not csv_path.exists():
        console.print(f"[red]Error: CSV file not found: {csv_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Loading CSV from: {csv_path}[/cyan]")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = [config.input.id_column, config.input.text_column, config.input.category_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        console.print(f"[red]Error: Missing required columns: {missing_cols}[/red]")
        raise typer.Exit(1)
    
    # Filter by categories if specified
    if config.input.categories:
        df = df[df[config.input.category_column].isin(config.input.categories)]
        console.print(f"[cyan]Filtering to categories: {config.input.categories}[/cyan]")
    
    console.print(f"[green]Loaded {len(df)} sentences[/green]")
    console.print(f"[cyan]Layers to process: {config.layers}[/cyan]")
    
    if len(df) == 0:
        console.print("[red]Error: No sentences to process after filtering[/red]")
        raise typer.Exit(1)
    
    # Create output directory with run ID
    run_id = datetime.now().strftime(config.output.run_id_format)
    output_dir = Path(config.output.base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
    
    # Load model and tokenizer
    console.print(f"[cyan]Loading model: {config.model.model_id}[/cyan]")
    model, tokenizer = load_model(config)
    console.print("[green]Model loaded successfully[/green]")
    
    # Pre-load all SAEs for the layers we need
    console.print(f"[cyan]Loading SAEs for {len(config.layers)} layers...[/cyan]")
    saes: Dict[int, object] = {}
    for layer in track(config.layers, description="Loading SAEs", console=console):
        saes[layer] = load_sae(config, layer)
    console.print("[green]All SAEs loaded successfully[/green]")
    
    # Process each sentence (Llama forward pass only ONCE per sentence)
    console.print(f"\n[cyan]Processing {len(df)} sentences...[/cyan]")
    
    for _, row in track(
        list(df.iterrows()),
        description="Processing sentences",
        console=console,
    ):
        sentence_id = row[config.input.id_column]
        text = row[config.input.text_column]
        category = row[config.input.category_column]
        
        # Get all hidden states in ONE forward pass
        all_hidden_states, tokens = get_hidden_states(
            sentence=text,
            model=model,
            tokenizer=tokenizer,
        )
        
        # Apply each layer's SAE to the corresponding hidden states
        for layer in config.layers:
            # Get hidden states for this layer (+1 because index 0 is embeddings)
            hidden_states = all_hidden_states[layer + 1]
            
            # Encode with SAE
            results = encode_hidden_states(
                hidden_states=hidden_states,
                tokens=tokens,
                sae=saes[layer],
                exclude_bos=config.token.exclude_bos,
            )
            
            # Write results
            write_results(
                results=results,
                output_dir=output_dir,
                layer=layer,
                category=category,
                sentence_id=sentence_id,
            )
    
    console.print(f"\n[green bold]✓ Processing complete![/green bold]")
    console.print(f"[green]Results saved to: {output_dir}[/green]")


if __name__ == "__main__":
    app()
