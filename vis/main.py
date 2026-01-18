"""CLI entry point for Llama Scope visualization."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from llama_vis.config import load_vis_config
from llama_vis.loader import load_all_data, load_sentence_data, get_available_layers
from llama_vis.stats import calc_nnz_stats, calc_feat_rarity_scores
from llama_vis.plotter import (
    plot_nnz_histogram,
    plot_sentence_figure,
    plot_compare_tokens,
    plot_compare_layers,
)

app = typer.Typer()
console = Console()


@app.command("run")
def run(
    config_path: str = typer.Option(
        "vis_config.yaml",
        "--config",
        "-c",
        help="Path to vis_config.yaml file",
    ),
) -> None:
    """Run basic visualization: histogram + all token comparisons."""
    # Load configuration
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    config = load_vis_config(config_path)
    
    run_dir = Path(config.input.run_dir)
    if not run_dir.exists():
        console.print(f"[red]Error: Run directory not found: {run_dir}[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    run_id = run_dir.name
    output_dir = Path(config.output.base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
    
    # Load all data
    console.print(f"[cyan]Loading data from: {run_dir}[/cyan]")
    all_data = load_all_data(run_dir)
    
    layers = sorted(all_data.keys())
    console.print(f"[green]Loaded data for {len(layers)} layers: {layers}[/green]")
    
    # 1. Generate nnz histogram
    console.print("\n[cyan]Generating nnz histogram...[/cyan]")
    nnz_stats = calc_nnz_stats(all_data)
    histogram_path = output_dir / "nnz_histogram.png"
    plot_nnz_histogram(
        nnz_stats,
        histogram_path,
        ax_cols=config.common.ax_cols,
        bins=config.histogram.bins,
    )
    console.print(f"[green]Saved: {histogram_path}[/green]")
    
    # 2. Generate token bar charts for all sentences
    console.print("\n[cyan]Generating token bar charts...[/cyan]")
    max_rank_score = config.token_comparison.max_rank_score
    
    for layer in track(layers, description="Processing layers", console=console):
        # Calculate rarity scores for this layer
        feat_rarity_scores = calc_feat_rarity_scores(all_data, layer, max_rank_score)
        
        categories = all_data[layer]
        for category, sentences in categories.items():
            for sentence_id, tokens in sentences.items():
                output_path = (
                    output_dir / "token_bars" / f"layer_{layer:02d}" / category / f"sentence_{sentence_id}.png"
                )
                plot_sentence_figure(
                    tokens,
                    feat_rarity_scores,
                    config.token_comparison.top_n,
                    config.common.ax_cols,
                    output_path,
                    max_rank_score=max_rank_score,
                )
    
    console.print(f"[green]Token bar charts saved to: {output_dir / 'token_bars'}[/green]")
    console.print(f"\n[green bold]✓ Visualization complete![/green bold]")


@app.command("compare-tokens")
def compare_tokens_cmd(
    config_path: str = typer.Option(
        "vis_config.yaml",
        "--config",
        "-c",
        help="Path to vis_config.yaml file",
    ),
) -> None:
    """Compare specified tokens within same layer."""
    # Load configuration
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    config = load_vis_config(config_path)
    
    run_dir = Path(config.input.run_dir)
    if not run_dir.exists():
        console.print(f"[red]Error: Run directory not found: {run_dir}[/red]")
        raise typer.Exit(1)
    
    # Get patterns from config
    patterns = config.compare_tokens.patterns
    if not patterns:
        console.print("[red]Error: No patterns specified in compare_tokens.patterns[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    run_id = run_dir.name
    output_dir = Path(config.output.base_dir) / run_id / "compare_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    console.print(f"[cyan]Loading data from: {run_dir}[/cyan]")
    all_data = load_all_data(run_dir)
    
    available_layers = sorted(all_data.keys())
    max_rank_score = config.token_comparison.max_rank_score
    
    # Process each pattern
    for pattern in patterns:
        console.print(f"\n[cyan]Processing pattern: {pattern.name}[/cyan]")
        
        token_specs = pattern.tokens
        if not token_specs:
            console.print(f"[yellow]Warning: No tokens in pattern '{pattern.name}', skipping[/yellow]")
            continue
        
        # Determine layers to process
        target_layers = pattern.layers if pattern.layers else available_layers
        
        console.print(f"[cyan]  Layers: {target_layers}[/cyan]")
        console.print(f"[cyan]  Tokens: {len(token_specs)}[/cyan]")
        
        # Create pattern output directory
        pattern_output_dir = output_dir / pattern.name
        pattern_output_dir.mkdir(parents=True, exist_ok=True)
        
        for layer in track(target_layers, description=f"  Processing {pattern.name}", console=console):
            if layer not in all_data:
                console.print(f"[yellow]Warning: Layer {layer} not found, skipping[/yellow]")
                continue
            
            # Calculate rarity scores for this layer
            feat_rarity_scores = calc_feat_rarity_scores(all_data, layer, max_rank_score)
            
            # Collect token data
            tokens_data = []
            for spec in token_specs:
                try:
                    sentence_tokens = all_data[layer][spec.category][spec.sentence_id]
                    # Find token by position
                    token_data = None
                    for t in sentence_tokens:
                        if t.token_pos == spec.token_pos:
                            token_data = t
                            break
                    
                    if token_data is None:
                        console.print(f"[yellow]Warning: Token pos {spec.token_pos} not found in {spec.category}/sentence_{spec.sentence_id}[/yellow]")
                        continue
                    
                    label = f"{spec.category}\nsent_{spec.sentence_id}\n'{token_data.token_str}'"
                    tokens_data.append((token_data, label))
                except KeyError as e:
                    console.print(f"[yellow]Warning: Data not found for {spec}: {e}[/yellow]")
                    continue
            
            if tokens_data:
                output_path = pattern_output_dir / f"layer_{layer:02d}.png"
                plot_compare_tokens(
                    tokens_data,
                    feat_rarity_scores,
                    config.token_comparison.top_n,
                    config.common.ax_cols,
                    output_path,
                    max_rank_score=max_rank_score,
                )
                console.print(f"[green]Saved: {output_path}[/green]")
    
    console.print(f"\n[green bold]✓ Token comparison complete![/green bold]")


@app.command("compare-layers")
def compare_layers_cmd(
    config_path: str = typer.Option(
        "vis_config.yaml",
        "--config",
        "-c",
        help="Path to vis_config.yaml file",
    ),
) -> None:
    """Compare tokens across layers."""
    # Load configuration
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    config = load_vis_config(config_path)
    
    run_dir = Path(config.input.run_dir)
    if not run_dir.exists():
        console.print(f"[red]Error: Run directory not found: {run_dir}[/red]")
        raise typer.Exit(1)
    
    # Get patterns from config
    patterns = config.compare_layers.patterns
    if not patterns:
        console.print("[red]Error: No patterns specified in compare_layers.patterns[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    run_id = run_dir.name
    output_dir = Path(config.output.base_dir) / run_id / "compare_layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    console.print(f"[cyan]Loading data from: {run_dir}[/cyan]")
    all_data = load_all_data(run_dir)
    
    # Load original sentences from input CSV
    from llama_vis.loader import load_input_csv
    input_csv_path = Path(config.input.input_csv)
    sentences = load_input_csv(input_csv_path)
    console.print(f"[cyan]Loaded {len(sentences)} sentences from: {input_csv_path}[/cyan]")
    
    available_layers = sorted(all_data.keys())
    max_rank_score = config.token_comparison.max_rank_score
    
    # Process each pattern
    for pattern in patterns:
        console.print(f"\n[cyan]Processing pattern: {pattern.name}[/cyan]")
        
        token_specs = pattern.tokens
        if not token_specs:
            console.print(f"[yellow]Warning: No tokens in pattern '{pattern.name}', skipping[/yellow]")
            continue
        
        # Determine layers to process
        target_layers = pattern.layers if pattern.layers else available_layers
        
        console.print(f"[cyan]  Layers: {target_layers}[/cyan]")
        console.print(f"[cyan]  Tokens: {len(token_specs)}[/cyan]")
        
        # Build data structure for plotting
        tokens_by_layer = {}
        
        for layer in target_layers:
            if layer not in all_data:
                console.print(f"[yellow]Warning: Layer {layer} not found, skipping[/yellow]")
                continue
            
            # Calculate rarity scores for this layer
            feat_rarity_scores = calc_feat_rarity_scores(all_data, layer, max_rank_score)
            
            tokens_data = []
            for spec in token_specs:
                try:
                    sentence_tokens = all_data[layer][spec.category][spec.sentence_id]
                    # Find token by position
                    token_data = None
                    for t in sentence_tokens:
                        if t.token_pos == spec.token_pos:
                            token_data = t
                            break
                    
                    if token_data is None:
                        console.print(f"[yellow]Warning: Token pos {spec.token_pos} not found[/yellow]")
                        continue
                    
                    # Get original sentence text
                    sentence_text = sentences.get(spec.sentence_id, f"Sentence {spec.sentence_id}")
                    tokens_data.append((token_data, sentence_text, feat_rarity_scores))
                except KeyError as e:
                    console.print(f"[yellow]Warning: Data not found: {e}[/yellow]")
                    continue
            
            if tokens_data:
                tokens_by_layer[layer] = tokens_data
        
        if tokens_by_layer:
            output_path = output_dir / f"{pattern.name}.png"
            plot_compare_layers(
                tokens_by_layer,
                config.token_comparison.top_n,
                output_path,
                max_rank_score=max_rank_score,
            )
            console.print(f"[green]Saved: {output_path}[/green]")
    
    console.print(f"\n[green bold]✓ Layer comparison complete![/green bold]")


@app.command("show-tokens")
def show_tokens_cmd(
    config_path: str = typer.Option(
        "vis_config.yaml",
        "--config",
        "-c",
        help="Path to vis_config.yaml file",
    ),
) -> None:
    """Show tokenization results as CSV for reference."""
    import csv
    
    # Load configuration
    console.print(f"[cyan]Loading configuration from: {config_path}[/cyan]")
    config = load_vis_config(config_path)
    
    run_dir = Path(config.input.run_dir)
    if not run_dir.exists():
        console.print(f"[red]Error: Run directory not found: {run_dir}[/red]")
        raise typer.Exit(1)
    
    # Load original sentences from input CSV
    from llama_vis.loader import load_input_csv
    input_csv_path = Path(config.input.input_csv)
    sentences = load_input_csv(input_csv_path)
    console.print(f"[cyan]Loaded {len(sentences)} sentences from: {input_csv_path}[/cyan]")
    
    # Load tokenization data from any layer (layer 0 is typical)
    all_data = load_all_data(run_dir)
    available_layers = sorted(all_data.keys())
    
    if not available_layers:
        console.print("[red]Error: No layers found in run directory[/red]")
        raise typer.Exit(1)
    
    # Use first available layer to get tokenization
    first_layer = available_layers[0]
    layer_data = all_data[first_layer]
    
    # Collect all sentence tokenizations
    tokenized_data = []
    max_tokens = 0
    
    for category, category_data in layer_data.items():
        for sentence_id, tokens in category_data.items():
            # Sort tokens by position
            sorted_tokens = sorted(tokens, key=lambda t: t.token_pos)
            token_strs = [t.token_str for t in sorted_tokens]
            
            original_text = sentences.get(sentence_id, "")
            
            tokenized_data.append({
                "id": sentence_id,
                "text": original_text,
                "tokens": token_strs,
            })
            
            max_tokens = max(max_tokens, len(token_strs))
    
    # Sort by sentence ID
    tokenized_data.sort(key=lambda x: x["id"])
    
    # Create output directory
    run_id = run_dir.name
    output_dir = Path(config.output.base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / config.show_tokens.output_file
    
    # Write CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        # Create header: id, text, 1, 2, 3, ...
        fieldnames = ["id", "text"] + [str(i) for i in range(1, max_tokens + 1)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in tokenized_data:
            row = {"id": item["id"], "text": item["text"]}
            for i, token_str in enumerate(item["tokens"], start=1):
                row[str(i)] = token_str
            writer.writerow(row)
    
    console.print(f"[green]Saved: {output_path}[/green]")
    console.print(f"[cyan]Total sentences: {len(tokenized_data)}[/cyan]")
    console.print(f"[cyan]Max tokens per sentence: {max_tokens}[/cyan]")
    console.print(f"\n[green bold]✓ Tokenization CSV complete![/green bold]")


if __name__ == "__main__":
    app()
