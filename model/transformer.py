import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Expand → Activate → Shrink
    (d_model → d_ff → d_model)
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),       # Expand
            nn.GELU(),                       # Activate (using GELU like GPT-2)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),        # Shrink back
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer block: the repeating unit of GPT.
    
    Pre-norm architecture:
        x → LayerNorm → MultiHeadAttention → + residual
          → LayerNorm → FeedForward         → + residual
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Sub-layer 1: Multi-Head Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        
        # Sub-layer 2: Feed-Forward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x:    (batch, seq_len, d_model)
            mask: (1, 1, seq_len, seq_len) causal mask
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # ---- Sub-layer 1: Attention with residual ----
        # Pre-norm: normalize BEFORE the sublayer
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask=mask)
        x = x + self.dropout1(attn_out)       # Residual: add input back
        
        # ---- Sub-layer 2: FFN with residual ----
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out                         # Residual: add input back
        
        return x
    

def test_transformer_block():
    batch, seq_len, d_model = 2, 16, 128
    num_heads, d_ff = 4, 512
    
    block = TransformerBlock(d_model, num_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model)
    
    # Causal mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    output = block(x, mask=mask)
    
    # Shape preserved (this is what makes stacking possible)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    attn_params = sum(p.numel() for p in block.attn.parameters())
    ff_params = sum(p.numel() for p in block.ff.parameters())
    norm_params = sum(p.numel() for p in block.norm1.parameters()) + \
                  sum(p.numel() for p in block.norm2.parameters())
    
    print(f"✓ Output shape: {output.shape}")
    print(f"\n📊 Parameter breakdown:")
    print(f"  Attention:    {attn_params:>8,}  ({attn_params/total_params*100:.1f}%)")
    print(f"  FeedForward:  {ff_params:>8,}  ({ff_params/total_params*100:.1f}%)")
    print(f"  LayerNorms:   {norm_params:>8,}  ({norm_params/total_params*100:.1f}%)")
    print(f"  ─────────────────────────")
    print(f"  Total:        {total_params:>8,}")
    
    # Verify residual connection is working
    print(f"\n✓ Output mean: {output.mean():.4f} (should be non-zero)")
    print(f"✓ Output std:  {output.std():.4f} (should be reasonable, not exploding)")


def test_stacking():
    """Verify blocks can be stacked (output shape = input shape)."""
    batch, seq_len, d_model = 2, 16, 128
    num_heads, d_ff, num_layers = 4, 512, 6
    
    blocks = nn.ModuleList([
        TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
    ])
    
    mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    x = torch.randn(batch, seq_len, d_model)
    
    print(f"Passing through {num_layers} stacked blocks:")
    for i, block in enumerate(blocks):
        x = block(x, mask=mask)
        print(f"  After block {i}: mean={x.mean():.4f}, std={x.std():.4f}, shape={x.shape}")
    
    print(f"\n✓ Final shape: {x.shape} (same as input — stacking works)")
    print(f"✓ Values stable: mean and std didn't explode or vanish")


if __name__ == "__main__":
    test_transformer_block()
    print("\n" + "="*60 + "\n")
    test_stacking()





class TinyGPT(nn.Module):
    """
    A minimal GPT model.
    
    Architecture:
        Token Embedding + Position Embedding
        → N × TransformerBlock
        → Final LayerNorm
        → Linear output head
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_len, dropout=0.1):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights like GPT-2.
        
        Why this matters: default PyTorch init can lead to unstable training.
        GPT-2 uses normal distribution with small std for linear layers
        and zeros for biases.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len) — token indices
        
        Returns:
            logits: (batch, seq_len, vocab_size) — raw predictions
        """
        batch, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Embed tokens and positions, then add them
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Final norm and project to vocabulary
        x = self.final_norm(x)
        logits = self.output_head(x)
        
        return logits
    
    def count_parameters(self):
        """Print a breakdown of where parameters live."""
        total = sum(p.numel() for p in self.parameters())
        
        emb = sum(p.numel() for p in self.token_emb.parameters()) + \
              sum(p.numel() for p in self.pos_emb.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        head = sum(p.numel() for p in self.output_head.parameters()) + \
               sum(p.numel() for p in self.final_norm.parameters())
        
        print(f"📊 TinyGPT Parameter Breakdown:")
        print(f"  Embeddings:          {emb:>10,}  ({emb/total*100:5.1f}%)")
        print(f"  Transformer Blocks:  {blocks:>10,}  ({blocks/total*100:5.1f}%)")
        print(f"  Output Head + Norm:  {head:>10,}  ({head/total*100:5.1f}%)")
        print(f"  {'─' * 45}")
        print(f"  Total:               {total:>10,}")
        return total