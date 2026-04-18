import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.attention import scaled_dot_product_attention, create_causal_mask


def demonstrate_information_leakage():
    """Show that without a mask, position 0 can 'see' all future tokens."""
    batch, seq_len, d_k = 1, 6, 16
    
    torch.manual_seed(42)
    Q = torch.randn(batch, seq_len, d_k)
    K = torch.randn(batch, seq_len, d_k)
    V = torch.randn(batch, seq_len, d_k)
    
    # WITHOUT mask
    _, weights_no_mask = scaled_dot_product_attention(Q, K, V)
    
    # WITH causal mask
    mask = create_causal_mask(seq_len)
    _, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print("=== Without Causal Mask ===")
    print(f"Position 0 attention weights: {weights_no_mask[0, 0].tolist()}")
    print(f"  -> Position 0 can see ALL positions (information leakage!)")
    print(f"  -> Sum of weights on future positions: {weights_no_mask[0, 0, 1:].sum():.4f}")
    
    print(f"\n=== With Causal Mask ===")
    print(f"Position 0 attention weights: {weights_masked[0, 0].tolist()}")
    print(f"  -> Position 0 can ONLY see itself")
    print(f"  -> Sum of weights on future positions: {weights_masked[0, 0, 1:].sum():.6f}")
    
    print(f"\n=== Position 3 comparison ===")
    print(f"Without mask: attends to all 6 positions: {weights_no_mask[0, 3].tolist()}")
    print(f"With mask:    attends to positions 0-3 only: {weights_masked[0, 3].tolist()}")
    
    print("\nWhy this matters:")
    print("In language modeling, you predict the NEXT token.")
    print("If the model can see future tokens during training, it just copies instead of learning.")
    print("At inference time, future tokens don't exist — so a model trained without masking")
    print("would have learned a 'cheating strategy' that doesn't work in practice.")


if __name__ == "__main__":
    demonstrate_information_leakage()