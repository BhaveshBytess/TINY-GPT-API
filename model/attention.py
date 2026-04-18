import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query tensor of shape (batch, seq_len, d_k)
        K: Key tensor of shape (batch, seq_len, d_k)
        V: Value tensor of shape (batch, seq_len, d_k)
        mask: Optional mask tensor of shape (1, seq_len, seq_len) or (batch, seq_len, seq_len)
              Contains 0 for positions to attend to and -inf for positions to block.
    
    Returns:
        output: Attention output of shape (batch, seq_len, d_k)
        weights: Attention weights of shape (batch, seq_len, seq_len)
    """
    d_k = Q.size(-1)
    
    # Step 1: Compute raw attention scores
    # Q @ K^T gives (batch, seq_len, seq_len) — a score for every pair of positions
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 2: Scale by √d_k
    scores = scores / (d_k ** 0.5)
    
    # Step 3: Apply mask (if provided)
    if mask is not None:
        scores = scores + mask  # mask has -inf where we want to block
    
    # Step 4: Softmax to get attention weights (probabilities that sum to 1)
    weights = F.softmax(scores, dim=-1)
    
    # Step 5: Weighted sum of values
    output = torch.matmul(weights, V)
    
    return output, weights


def test_attention_basic():
    """Verify shapes and basic behavior."""
    batch, seq_len, d_k = 2, 4, 8
    Q = torch.randn(batch, seq_len, d_k)
    K = torch.randn(batch, seq_len, d_k)
    V = torch.randn(batch, seq_len, d_k)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    # Check output shape
    assert output.shape == (batch, seq_len, d_k), f"Expected {(batch, seq_len, d_k)}, got {output.shape}"
    
    # Check weights shape
    assert weights.shape == (batch, seq_len, seq_len), f"Expected {(batch, seq_len, seq_len)}, got {weights.shape}"
    
    # Check weights sum to 1 along the last dimension (they're probabilities)
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), \
        f"Weights don't sum to 1: {weight_sums}"
    
    print("Basic test passed!")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Sample attention weights (batch 0):\n{weights[0]}")


def create_causal_mask(seq_len):
    """Create a causal (triangular) mask for decoder attention."""
    # Upper triangle (above diagonal) = -inf, rest = 0
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return mask.unsqueeze(0)  # Add batch dimension: (1, seq_len, seq_len)


def test_attention_with_causal_mask():
    """Verify causal mask prevents attending to future positions."""
    batch, seq_len, d_k = 1, 4, 8
    Q = torch.randn(batch, seq_len, d_k)
    K = torch.randn(batch, seq_len, d_k)
    V = torch.randn(batch, seq_len, d_k)
    
    mask = create_causal_mask(seq_len)
    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    # Check that future positions have zero weight
    # For position 0: should only attend to position 0
    # For position 1: should attend to positions 0 and 1
    # etc.
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert weights[0, i, j].item() < 1e-6, \
                f"Position {i} attending to future position {j} with weight {weights[0, i, j]}"
    
    print("Causal mask test passed!")
    print(f"Attention weights with causal mask:\n{weights[0]}")


if __name__ == "__main__":
    test_attention_basic()
    print()
    test_attention_with_causal_mask()





import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Splits the model dimension into multiple heads, runs attention
    independently on each head, then concatenates and projects back.
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # The four projection matrices
        self.W_Q = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # Output projection
    
    def split_heads(self, x):
        """
        Reshape (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        
        This doesn't copy data — it's a view/reshape operation.
        We transpose seq_len and num_heads so attention operates on (seq_len, d_k) per head.
        """
        batch, seq_len, d_model = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
    
    def combine_heads(self, x):
        """
        Reshape (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        
        Reverse of split_heads. Concatenates all heads back together.
        """
        batch, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        return x.contiguous().view(batch, seq_len, self.d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional causal mask of shape (1, 1, seq_len, seq_len)
                  Note the extra dimension for broadcasting across heads.
        
        Returns:
            output: (batch, seq_len, d_model)
            weights: (batch, num_heads, seq_len, seq_len) — attention weights per head
        """
        # Step 1: Project input into Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Step 2: Split into multiple heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Step 3: Scaled dot-product attention (per head, in parallel via broadcasting)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        # scores shape: (batch, num_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores + mask
        
        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, V)  # (batch, num_heads, seq_len, d_k)
        
        # Step 4: Concatenate heads
        concat = self.combine_heads(attn_output)  # (batch, seq_len, d_model)
        
        # Step 5: Final projection
        output = self.W_O(concat)  # (batch, seq_len, d_model)
        
        return output, weights
    

def test_multi_head_attention():
    """Verify shapes and basic behavior of MultiHeadAttention."""
    batch, seq_len, d_model, num_heads = 2, 8, 128, 4
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq_len, d_model)
    
    # Without mask
    output, weights = mha(x)
    
    assert output.shape == (batch, seq_len, d_model), \
        f"Output shape mismatch: {output.shape}"
    assert weights.shape == (batch, num_heads, seq_len, seq_len), \
        f"Weights shape mismatch: {weights.shape}"
    
    # Weights should sum to 1 along last dim for each head
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Attention weights don't sum to 1"
    
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"d_k per head: {mha.d_k}")
    print(f"Total parameters: {sum(p.numel() for p in mha.parameters()):,}")
    print("  W_Q: {:,}  W_K: {:,}  W_V: {:,}  W_O: {:,}".format(
        mha.W_Q.weight.numel() + mha.W_Q.bias.numel(),
        mha.W_K.weight.numel() + mha.W_K.bias.numel(),
        mha.W_V.weight.numel() + mha.W_V.bias.numel(),
        mha.W_O.weight.numel() + mha.W_O.bias.numel(),
    ))
    
    # With causal mask (note the extra dimension for heads)
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len) — broadcasts over batch and heads
    
    output_masked, weights_masked = mha(x, mask=mask)
    
    # Check causal property: no head should attend to future positions
    for h in range(num_heads):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert weights_masked[0, h, i, j].item() < 1e-6, \
                    f"Head {h}, position {i} attending to future position {j}"
    
    print("Causal mask test passed for all heads!")


def test_split_combine_roundtrip():
    """Verify that split_heads → combine_heads is lossless."""
    batch, seq_len, d_model, num_heads = 2, 8, 128, 4
    mha = MultiHeadAttention(d_model, num_heads)
    
    x = torch.randn(batch, seq_len, d_model)
    split = mha.split_heads(x)
    recombined = mha.combine_heads(split)
    
    assert torch.allclose(x, recombined, atol=1e-6), "Split-combine roundtrip failed!"
    print(f"Roundtrip test passed!")
    print(f"  Original:    {x.shape}")
    print(f"  Split:       {split.shape}")
    print(f"  Recombined:  {recombined.shape}")


if __name__ == "__main__":
    test_multi_head_attention()
    print()
    test_split_combine_roundtrip()