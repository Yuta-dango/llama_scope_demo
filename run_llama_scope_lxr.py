"""
Llama Scope の学習済みSAE（LXR）を、Llama-3.1-8B の hidden states に当てて
特徴（スパース表現）を取り出すデモ。

- HF の meta-llama は gated のため HF_TOKEN が必要。
- 複数のプロンプトに対してループ処理で実行可能。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_prompt(
    prompt: str,
    model,
    tok,
    sae,
    device: str,
    layer: int,
    output_dir: Path,
    prompt_idx: int,
) -> dict:
    """1つのプロンプトに対してSAE分析を実行し、結果を保存する。"""
    
    inputs = tok(prompt, return_tensors="pt").to(device)
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
    
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer + 1]  # [batch, seq, d_model]
        z = sae.encode(h)  # [batch, seq, d_sae]
    
    # === プロンプト用ディレクトリ作成 ===
    prompt_dir = output_dir / f"prompt_{prompt_idx:02d}"
    prompt_dir.mkdir(exist_ok=True)
    
    # === 各トークン位置の分析 ===
    token_analyses = []
    for pos in range(len(tokens)):
        token_z = z[0, pos]
        nonzero_mask = token_z > 0
        nonzero_vals = token_z[nonzero_mask]
        nonzero_indices = torch.where(nonzero_mask)[0]
        
        # Top-10
        if len(nonzero_vals) > 0:
            k = min(10, len(nonzero_vals))
            top_vals, top_order = torch.topk(nonzero_vals, k)
            top_indices = nonzero_indices[top_order]
            top10 = [(int(idx), float(val)) for idx, val in zip(top_indices.tolist(), top_vals.tolist())]
            
            # 全非ゼロ特徴（降順）
            sorted_vals, sorted_order = torch.sort(nonzero_vals, descending=True)
            sorted_indices = nonzero_indices[sorted_order]
            all_nonzero = [(int(idx), float(val)) for idx, val in zip(sorted_indices.tolist(), sorted_vals.tolist())]
        else:
            top10 = []
            all_nonzero = []
        
        token_analyses.append({
            "position": pos,
            "token": tokens[pos],
            "num_nonzero": int(nonzero_mask.sum().item()),
            "max_value": float(token_z.max().item()),
            "mean_nonzero": float(nonzero_vals.mean().item()) if len(nonzero_vals) > 0 else 0.0,
            "top10": top10,
            "all_nonzero": all_nonzero,
        })
    
    # === レポート出力 ===
    report_path = prompt_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SAE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Layer: {layer}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        # --- 基本情報 ---
        f.write("-" * 70 + "\n")
        f.write("BASIC INFO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Hidden states shape (h): {list(h.shape)}  [batch, seq, d_model]\n")
        f.write(f"SAE output shape (z):    {list(z.shape)}  [batch, seq, d_sae]\n")
        f.write(f"Number of tokens: {len(tokens)}\n")
        f.write(f"Tokens: {tokens}\n\n")
        
        # --- 全トークンのTop-10サマリー ---
        f.write("-" * 70 + "\n")
        f.write("TOP-10 FEATURES SUMMARY (ALL TOKENS)\n")
        f.write("-" * 70 + "\n\n")
        
        for ta in token_analyses:
            f.write(f"[{ta['position']}] '{ta['token']}' | {ta['num_nonzero']} active | max={ta['max_value']:.4f}\n")
            if ta['top10']:
                for rank, (idx, val) in enumerate(ta['top10'], 1):
                    f.write(f"    {rank:2d}. feature {idx:6d} = {val:.4f}\n")
            else:
                f.write("    (no active features)\n")
            f.write("\n")
        
        # --- 各トークンの全非ゼロ特徴 ---
        f.write("-" * 70 + "\n")
        f.write("ALL NON-ZERO FEATURES (PER TOKEN)\n")
        f.write("-" * 70 + "\n\n")
        
        for ta in token_analyses:
            f.write(f"[{ta['position']}] '{ta['token']}' - {ta['num_nonzero']} features\n")
            if ta['all_nonzero']:
                for idx, val in ta['all_nonzero']:
                    f.write(f"    {idx:6d}: {val:.6f}\n")
            f.write("\n")
    
    logger.info(f"  Report saved: {report_path}")
    
    # === JSON出力 ===
    json_path = prompt_dir / "analysis.json"
    json_data = {
        "prompt": prompt,
        "layer": layer,
        "tokens": tokens,
        "h_shape": list(h.shape),
        "z_shape": list(z.shape),
        "token_analyses": token_analyses,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  JSON saved: {json_path}")
    
    # === Hidden states保存 ===
    h_path = prompt_dir / "tensors.pt"
    torch.save({"h": h.cpu(), "z": z.cpu()}, h_path)
    logger.info(f"  Tensors saved: {h_path}")
    
    # === プロット作成 ===
    create_plots(token_analyses, tokens, prompt, layer, prompt_dir)
    
    return json_data


def create_plots(token_analyses: list, tokens: list, prompt: str, layer: int, output_dir: Path):
    """全トークン位置のプロットを作成する。"""
    
    num_tokens = len(tokens)
    
    # --- Plot 1: サマリーグラフ ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"SAE Analysis Summary\nPrompt: '{prompt}' | Layer: {layer}", fontsize=11)
    
    positions = [ta['position'] for ta in token_analyses]
    
    # 左: 各トークンの非ゼロ特徴数
    ax1 = axes[0]
    nonzero_counts = [ta['num_nonzero'] for ta in token_analyses]
    ax1.bar(positions, nonzero_counts, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Non-zero Feature Count')
    ax1.set_title('Active Features per Token')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"[{p}]\n{t}" for p, t in zip(positions, tokens)], fontsize=8)
    
    # 右: 各トークンのmax値
    ax2 = axes[1]
    max_vals = [ta['max_value'] for ta in token_analyses]
    ax2.bar(positions, max_vals, color='coral', edgecolor='black')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Max Feature Value')
    ax2.set_title('Max Feature Value per Token')
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"[{p}]\n{t}" for p, t in zip(positions, tokens)], fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150)
    plt.close()
    
    # --- Plot 2: 全トークンの非ゼロ特徴分布 ---
    cols = min(3, num_tokens)
    rows = (num_tokens + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f"Non-zero Feature Distribution per Token\nPrompt: '{prompt}'", fontsize=12)
    
    if num_tokens == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, ta in enumerate(token_analyses):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        
        if ta['all_nonzero']:
            vals = [v for _, v in ta['all_nonzero']]
            ax.hist(vals, bins=30, color='purple', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No active features', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f"[{ta['position']}] '{ta['token']}' (n={ta['num_nonzero']})", fontsize=10)
    
    for i in range(num_tokens, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "distributions.png", dpi=150)
    plt.close()
    
    # --- Plot 3: 全トークンのランクプロット ---
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f"Feature Values (Sorted Descending) per Token\nPrompt: '{prompt}'", fontsize=12)
    
    if num_tokens == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, ta in enumerate(token_analyses):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        
        if ta['all_nonzero']:
            vals = [v for _, v in ta['all_nonzero']]
            ax.plot(vals, color='darkgreen', linewidth=1.0)
            ax.set_xlabel('Rank')
            ax.set_ylabel('Feature Value')
            ax.set_xlim(0, len(vals))
        else:
            ax.text(0.5, 0.5, 'No active features', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f"[{ta['position']}] '{ta['token']}' (n={ta['num_nonzero']})", fontsize=10)
    
    for i in range(num_tokens, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "rank_plots.png", dpi=150)
    plt.close()
    
    logger.info(f"  Plots saved: summary.png, distributions.png, rank_plots.png")


def main() -> None:
    """複数プロンプトに対してSAE分析を実行する。"""
    load_dotenv()

    # ============================================================
    # CONFIG 
    # ============================================================
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16
    
    model_id = "meta-llama/Llama-3.1-8B"
    
    # 分析対象のレイヤー（0-31）
    layer = 0
    
    # SAE設定
    sae_release = "llama_scope_lxr_8x"
    sae_id = f"l{layer}r_8x"  # layerに合わせて変更する必要がある場合あり
    
    # 分析対象のプロンプト一覧
    prompts = [
        "The capital of France is",
        "Tokyo is the capital of",
        "1 + 1 =",
    ]
    
    # ============================================================
    # END CONFIG
    # ============================================================

    # === 出力ディレクトリ準備 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: model={model_id}, layer={layer}, sae={sae_release}/{sae_id}")

    # === モデル・SAEロード（一度だけ） ===
    logger.info("Loading SAE...")
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    logger.info("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # === 各プロンプトを処理 ===
    all_results = []
    for idx, prompt in enumerate(prompts):
        logger.info(f"\n[{idx + 1}/{len(prompts)}] Analyzing: '{prompt}'")
        result = analyze_prompt(
            prompt=prompt,
            model=model,
            tok=tok,
            sae=sae,
            device=device,
            layer=layer,
            output_dir=output_dir,
            prompt_idx=idx,
        )
        all_results.append(result)

    # === 全体サマリー保存 ===
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model_id": model_id,
            "layer": layer,
            "sae_release": sae_release,
            "sae_id": sae_id,
            "num_prompts": len(prompts),
            "prompts": prompts,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nAll done! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
