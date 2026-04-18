import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.attention import MultiHeadAttention


def inspect_head_patterns():
    """
    Feed a structured input and visualize what each head attends to.
    This shows that different heads learn different patterns.
    """
    d_model, num_heads, seq_len = 64, 4, 8
    
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Create structured input: 8 tokens with distinct patterns
    # Even positions have similar embeddings, odd positions have similar embeddings
    x = torch.randn(1, seq_len, d_model)
    
    mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    _, weights = mha(x, mask=mask)
    
    print("Attention patterns per head (rows=query position, cols=key position)")
    print("Higher values = stronger attention\n")
    
    for h in range(num_heads):
        print(f"--- Head {h} ---")
        w = weights[0, h]  # (seq_len, seq_len)
        
        # Print as a grid with 2 decimal places
        header = "     " + "".join([f"  k{j}  " for j in range(seq_len)])
        print(header)
        for i in range(seq_len):
            row = f"q{i}:  " + "".join([f"{w[i,j]:.3f} " for j in range(seq_len)])
            print(row)
        
        # Summarize: which positions does each query attend to most?
        top_k = torch.topk(w, k=min(3, seq_len), dim=-1)
        for i in range(seq_len):
            tops = [(idx.item(), val.item()) for idx, val in zip(top_k.indices[i], top_k.values[i])]
            tops_str = ", ".join([f"pos {idx} ({val:.2f})" for idx, val in tops])
            # Only show first and last query for brevity
            if i == 0 or i == seq_len - 1:
                print(f"  q{i} attends to: {tops_str}")
        print()
    
    print("Key observation:")
    print("Even with random (untrained) weights, different heads already produce")
    print("different attention patterns due to their independent projection matrices.")
    print("After training, these differences become meaningful specializations.")
    print("\nIn trained models, researchers have found heads that specialize in:")
    print("  - Attending to the previous token (positional)")
    print("  - Attending to the first token (anchor)")
    print("  - Attending to syntactically related words (structural)")
    print("  - Attending to semantically similar words (meaning)")


if __name__ == "__main__":
    inspect_head_patterns()