# Llama Scope Demo

Llama-3.1-8B モデルの hidden states に対して、学習済み Sparse Autoencoder (SAE) を適用し、解釈可能な特徴（スパース表現）を抽出するデモプロジェクトです。

## 概要

このプロジェクトは、**Llama Scope** の学習済み SAE (LXR) を使用して、大規模言語モデル (LLM) の内部表現を解釈可能な特徴に分解します。

### Sparse Autoencoder (SAE) とは？

SAE は、ニューラルネットワークの hidden states を「スパース」な（ほとんどの値がゼロに近い）特徴ベクトルに変換する手法です。これにより：

- **解釈可能性**: 各特徴が特定の概念やパターンに対応する可能性がある
- **メカニズム理解**: LLM が内部でどのような表現を学習しているか分析できる
- **特徴可視化**: 活性化している特徴から、モデルが「何を考えているか」を推測できる

### Llama Scope とは？

Llama Scope は、Llama シリーズのモデル向けに事前学習された SAE の集合体です。LXR (Llama eXtractor) は、その中でも効率的な 8x 拡張版です。

## コードの動作説明

`run_llama_scope_lxr.py` は以下のステップで動作します：

### 1. 環境設定
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16
```
- Apple Silicon Mac では MPS (Metal Performance Shaders) を使用
- メモリ効率のため float16 精度を使用

### 2. SAE のロード
```python
sae, _, _ = SAE.from_pretrained(
    release="llama_scope_lxr_8x",
    sae_id="l31r_8x",
    device=device,
)
```
- `llama_scope_lxr_8x`: Llama Scope の 8x 拡張版 SAE
- `l31r_8x`: Layer 31 の Residual Stream 用 SAE
- SAELens ライブラリ経由で Hugging Face Hub から自動ダウンロード

### 3. Llama-3.1-8B モデルのロード
```python
tok = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    token=True,
    low_cpu_mem_usage=True,
).to(device)
```
- `meta-llama/Llama-3.1-8B` ベースモデルをロード
- `token=True` で Hugging Face 認証トークンを使用（gated モデルのため必要）

### 4. Hidden States の取得
```python
out = model(**inputs, output_hidden_states=True, use_cache=False)
h = out.hidden_states[layer + 1]  # shape: [batch, seq, d_model]
```
- `output_hidden_states=True` で全レイヤーの hidden states を出力
- `hidden_states` は `[embedding, layer0, layer1, ..., layer31]` の順
- Layer 31 の出力は `index=32` (layer + 1) に格納

### 5. SAE による特徴抽出
```python
z = sae.encode(h)  # shape: [batch, seq, d_sae]
last = z[0, -1]    # 最終トークン位置の特徴
vals, idx = torch.topk(last, k=10)
```
- `sae.encode()`: hidden states をスパース特徴に変換
- `d_sae` = `d_model × 8` (8x 拡張のため)
- 最終トークン位置の top-10 特徴を抽出

### 6. 結果の表示
```
top features (index, value):
 12345  0.8234
 67890  0.7123
 ...
```
- 活性化値が高い上位10個の特徴インデックスと値を表示

## データフロー図

```
┌─────────────────────────────────────────────────────────────┐
│  Input: "The capital of France is"                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Llama-3.1-8B Tokenizer                                     │
│  → Token IDs: [The, capital, of, France, is]               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Llama-3.1-8B Model (32 Transformer Layers)                │
│                                                             │
│  Layer 0  → hidden_states[1]                               │
│  Layer 1  → hidden_states[2]                               │
│    ...                                                      │
│  Layer 31 → hidden_states[32]  ← これを使用                │
│                                                             │
│  各 hidden state の shape: [batch=1, seq=5, d_model=4096]  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  SAE Encoder (l31r_8x)                                      │
│                                                             │
│  Input:  h ∈ ℝ^4096 (hidden state)                         │
│  Output: z ∈ ℝ^32768 (sparse features, 8x expansion)       │
│                                                             │
│  z = ReLU(W_enc @ (h - b_dec) + b_enc)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Top-K Feature Selection (k=10)                             │
│                                                             │
│  最終トークン "is" の位置で最も活性化した特徴を抽出        │
│  → 各特徴は特定の概念（例：首都、国名など）に対応する可能性│
└─────────────────────────────────────────────────────────────┘
```

## セットアップ

### 前提条件

- Python 3.12+
- Hugging Face アカウント（Llama-3.1-8B へのアクセス権限が必要）
- 十分なメモリ（8B モデルには 16GB+ RAM 推奨）

### インストール

```bash
# 依存関係のインストール
uv sync

# または pip を使用
pip install -e .
```

### 環境変数の設定

`.env` ファイルを作成し、Hugging Face トークンを設定：

```
HF_TOKEN=your_huggingface_token_here
```

トークンは [Hugging Face Settings](https://huggingface.co/settings/tokens) で取得できます。

## 実行

```bash
python run_llama_scope_lxr.py
```

## 出力例

```
INFO:__main__:top features (index, value):
INFO:__main__: 28456  1.2345
INFO:__main__: 15234  0.9876
INFO:__main__:  8901  0.8765
...
```

各行は以下を表します：
- **index**: SAE 特徴のインデックス（0〜32767）
- **value**: 活性化値（高いほどその特徴が強く反応）

## 注意事項

- ⚠️ Mac (MPS) で 8B モデルを動かすのは遅い場合があります
- ⚠️ Llama-3.1-8B は gated モデルのため、HF での利用許諾が必要です
- ⚠️ 初回実行時はモデルと SAE のダウンロードに時間がかかります

## 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| torch | >=2.9.1 | PyTorch (テンソル演算) |
| transformers | >=4.57.3 | Hugging Face Transformers |
| sae-lens | >=6.22.3 | SAE のロード・適用 |
| accelerate | >=1.12.0 | モデルの効率的なロード |
| python-dotenv | >=1.2.1 | 環境変数の管理 |

## 参考リンク

- [Llama Scope Paper](https://arxiv.org/abs/2410.20526)
- [SAELens Documentation](https://jbloomaus.github.io/SAELens/)
- [Llama 3.1 on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B)

## ライセンス

このデモコードは MIT ライセンスで提供されます。ただし、使用するモデル (Llama-3.1-8B) には Meta の利用規約が適用されます。
