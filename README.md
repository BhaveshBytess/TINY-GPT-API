# TinyGPT

A transformer language model built from scratch in PyTorch.
Character-level, trained on TinyShakespeare (~1MB).

## What I Built

- Scaled dot-product attention with causal masking
- Multi-head attention with configurable head count
- Full transformer block (pre-norm, residual connections, FFN)
- Complete GPT model with training and text generation
- Ablation experiments demonstrating why each component matters

## Architecture

Default training config (from train.py): d_model=128, num_heads=4, num_layers=4, d_ff=512, seq_len=128, max_seq_len=256, dropout=0.1, batch_size=64, lr=3e-4, weight_decay=0.01, num_steps=5000. The model uses character-level vocabulary from TinyShakespeare (vocab size 65) and has ~842,817 parameters in this configuration.

## What I Learned (Ablation Experiments)

### Removing √dₖ scaling
When the √d_k scaling is removed, attention logits become too large as head dimension grows, which makes softmax saturate. In the experiment, unscaled attention entropy drops quickly (and can even become numerically unstable), while max attention weight approaches 1.0. With scaling, entropy stays much more stable, so attention remains trainable instead of collapsing into almost one-hot choices.

### Removing causal mask
Without a causal mask, each token can attend to future tokens during training, which causes information leakage. That means the model can partially "peek" at answers instead of learning true next-token prediction. Training may look easier, but behavior becomes invalid for autoregressive generation because future tokens are not available at inference time.

### 1 head vs 4 vs 8 heads
Using multiple heads improves representation power because each head can focus on different token relationships in parallel. In simple tasks, the gap may be small, but generally 4 or 8 heads converge better than 1 head when patterns are diverse. Multi-head attention is especially useful when the model must track several dependency types at once.

### Removing residual connections
Residual connections are critical for stable deep training. In the no-residual experiment, deeper models show weaker gradient flow and worse optimization, while residual versions maintain healthier gradients and lower loss. Residual paths let information and gradients pass across layers without being repeatedly distorted.

### Removing LayerNorm
LayerNorm keeps activations in a stable range across depth. When removed, training is much more fragile: loss can spike, plateau at worse values, or become NaN in deeper stacks. With LayerNorm, optimization is more predictable and the model trains faster and more reliably.

### Removing positional embeddings
Without positional embeddings, the model still learns token statistics but loses strong order information. Output may contain familiar words or local patterns, yet sentence structure and grammar degrade because the model cannot reliably distinguish where tokens occur in sequence. Position signals are essential for coherent ordering.

### Depth comparison (1 vs 4 vs 8 layers)
Increasing depth improves the model’s ability to represent longer-range dependencies and richer structure. In practice, 1 layer learns shallow local patterns, 4 layers gives noticeably stronger text structure, and 8 layers can perform best but costs more compute and trains more slowly. Depth is a quality/cost trade-off.

## How to Run

Install dependency: `pip install torch`

Train model: `python train.py`

Run experiments:
- `python experiments/no_scaling.py`
- `python experiments/no_mask.py`
- `python experiments/single_vs_multi_head.py`
- `python experiments/no_residuals.py`
- `python experiments/no_layernorm.py`
- `python experiments/no_pos_encoding.py`
- `python experiments/depth_comparison.py`
- `python experiments/head_inspection.py`

## Sample Output

Early training (high loss):
`First Citizen:\nTh th the, and to to, I.`

Mid training (improving structure):
`First Citizen:\nWhat say you, sir? the people call for bread.`

Later training (better coherence):
`First Citizen:\nBefore we proceed any further, hear me speak;\nwe are resolved to answer with our voices.`