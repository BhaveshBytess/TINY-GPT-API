import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.attention import scaled_dot_product_attention, create_causal_mask


def attention_no_scaling(Q, K, V, mask=None):
    """Same as scaled_dot_product_attention but WITHOUT the √d_k division."""
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # NOTE: No scaling here!
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


def compare_scaling():
    """Show what happens when you remove the scaling factor."""
    batch, seq_len = 1, 8
    
    for d_k in [8, 64, 256, 1024]:
        Q = torch.randn(batch, seq_len, d_k)
        K = torch.randn(batch, seq_len, d_k)
        V = torch.randn(batch, seq_len, d_k)
        
        _, weights_scaled = scaled_dot_product_attention(Q, K, V)
        _, weights_no_scale = attention_no_scaling(Q, K, V)
        
        # Entropy: high entropy = spread attention, low entropy = concentrated on one token
        # Entropy of a uniform distribution over 8 tokens = log(8) ≈ 2.08
        entropy_scaled = -(weights_scaled * weights_scaled.log()).sum(dim=-1).mean().item()
        entropy_no_scale = -(weights_no_scale * weights_no_scale.log()).sum(dim=-1).mean().item()
        
        # Max weight: how peaked is the distribution?
        max_scaled = weights_scaled.max(dim=-1).values.mean().item()
        max_no_scale = weights_no_scale.max(dim=-1).values.mean().item()
        
        print(f"d_k = {d_k:4d} | "
              f"Scaled entropy: {entropy_scaled:.3f}, max weight: {max_scaled:.3f} | "
              f"Unscaled entropy: {entropy_no_scale:.3f}, max weight: {max_no_scale:.3f}")
    
    print(f"\n(Uniform distribution entropy for {seq_len} tokens: {torch.tensor(1/seq_len).log().item() * -seq_len * (1/seq_len):.3f})")
    print("\nWhat you should see:")
    print("- As d_k grows, unscaled entropy drops toward 0 (one-hot)")
    print("- Scaled entropy stays relatively stable")
    print("- Unscaled max weight approaches 1.0 (all attention on one token)")


if __name__ == "__main__":
    compare_scaling()