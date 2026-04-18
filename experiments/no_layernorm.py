import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.attention import MultiHeadAttention


class BlockWithNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)    # ✅ Has LayerNorm
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)    # ✅ Has LayerNorm
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attn(self.norm1(x), mask=mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class BlockWithoutNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    
    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, mask=mask)     # ❌ No normalization before attention
        x = x + attn_out
        x = x + self.ff(x)                         # ❌ No normalization before FFN
        return x


def run_experiment():
    vocab_size, d_model, num_heads, d_ff, num_layers = 64, 64, 4, 256, 6
    seq_len, num_steps = 32, 500
    
    data = torch.randint(0, vocab_size, (500, seq_len + 1))
    inputs, targets = data[:, :-1], data[:, 1:]
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).unsqueeze(0).unsqueeze(0)
    
    print("=" * 70)
    print("  EXPERIMENT: Training with vs without LayerNorm (6 layers)")
    print("=" * 70)
    
    for label, BlockClass in [("WITH LayerNorm", BlockWithNorm), ("  NO LayerNorm", BlockWithoutNorm)]:
        torch.manual_seed(42)
        
        tok_emb = nn.Embedding(vocab_size, d_model)
        pos_emb = nn.Embedding(128, d_model)
        blocks = nn.ModuleList([BlockClass(d_model, num_heads, d_ff) for _ in range(num_layers)])
        head = nn.Linear(d_model, vocab_size)
        
        all_params = list(tok_emb.parameters()) + list(pos_emb.parameters()) + \
                     list(blocks.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(all_params, lr=3e-4)
        
        losses = []
        nan_detected = False
        
        for step in range(num_steps):
            idx = torch.randint(0, 500, (32,))
            batch_in, batch_tgt = inputs[idx], targets[idx]
            
            pos = torch.arange(seq_len).unsqueeze(0)
            x = tok_emb(batch_in) + pos_emb(pos)
            for block in blocks:
                x = block(x, mask=mask)
            logits = head(x)
            
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_tgt.reshape(-1))
            
            if torch.isnan(loss):
                print(f"  {label} | 💥 NaN loss at step {step}!")
                nan_detected = True
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                losses.append(loss.item())
        
        if not nan_detected:
            loss_str = " → ".join([f"{l:.3f}" for l in losses])
            print(f"  {label} | Loss: {loss_str}")
    
    print(f"\n  📌 Without LayerNorm, you'll likely see:")
    print(f"     • Loss spikes or NaN within a few hundred steps")
    print(f"     • Or loss that stalls much higher than the normalized version")
    print(f"     • The deeper the model, the worse this gets")


if __name__ == "__main__":
    run_experiment()