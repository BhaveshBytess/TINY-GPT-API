import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.attention import MultiHeadAttention


class TransformerBlockNoResidual(nn.Module):
    """Transformer block WITHOUT residual connections."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x, mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, mask=mask)
        x = attn_out                    # ❌ NO residual: x = attn_out (not x + attn_out)
        
        normed = self.norm2(x)
        x = self.ff(normed)             # ❌ NO residual
        return x


class TransformerBlockWithResidual(nn.Module):
    """Normal Transformer block WITH residual connections."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x, mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, mask=mask)
        x = x + attn_out                # ✅ Residual connection
        
        normed = self.norm2(x)
        x = x + self.ff(normed)          # ✅ Residual connection
        return x


class TinyModel(nn.Module):
    """Minimal model for testing: embed → N blocks → predict."""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, use_residual=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(128, d_model)
        
        BlockClass = TransformerBlockWithResidual if use_residual else TransformerBlockNoResidual
        self.blocks = nn.ModuleList([
            BlockClass(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, mask=None):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x, mask=mask)
        
        return self.head(x)


def run_experiment():
    """Compare training with and without residuals at different depths."""
    
    vocab_size, d_model, num_heads, d_ff = 64, 64, 4, 256
    seq_len, num_steps = 32, 300
    
    # Generate simple data
    data = torch.randint(0, vocab_size, (500, seq_len + 1))
    inputs, targets = data[:, :-1], data[:, 1:]
    
    mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    print("=" * 70)
    print("  EXPERIMENT: Training with vs without residual connections")
    print("=" * 70)
    
    for num_layers in [1, 4, 8]:
        print(f"\n{'─' * 70}")
        print(f"  {num_layers} layers")
        print(f"{'─' * 70}")
        
        for use_residual in [True, False]:
            label = "WITH residual" if use_residual else "  NO residual"
            
            torch.manual_seed(42)
            model = TinyModel(vocab_size, d_model, num_heads, d_ff, num_layers, use_residual)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            losses = []
            grad_norms = []
            
            for step in range(num_steps):
                idx = torch.randint(0, 500, (32,))
                batch_in, batch_tgt = inputs[idx], targets[idx]
                
                logits = model(batch_in, mask=mask)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_tgt.reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                
                # Track gradient norm of the FIRST layer's parameters
                first_block_params = list(model.blocks[0].parameters())
                grad_norm = torch.cat([p.grad.flatten() for p in first_block_params if p.grad is not None]).norm().item()
                
                optimizer.step()
                
                if step % 100 == 0:
                    losses.append(loss.item())
                    grad_norms.append(grad_norm)
            
            loss_str = " → ".join([f"{l:.3f}" for l in losses])
            grad_str = " → ".join([f"{g:.4f}" for g in grad_norms])
            
            print(f"  {label} | Loss: {loss_str}")
            print(f"  {' ' * len(label)} | Grad: {grad_str}")
    
    print(f"\n{'=' * 70}")
    print("  📌 What to look for:")
    print("  • At 1 layer: difference may be small (gradient only travels 1 layer)")
    print("  • At 4 layers: no-residual starts struggling, gradients shrink")
    print("  • At 8 layers: no-residual likely fails — gradient norms → 0")
    print("  • With residuals: gradients stay healthy at ALL depths")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()