import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.attention import MultiHeadAttention


class TinyAttentionModel(nn.Module):
    """Minimal model: embedding → multi-head attention → predict next token."""
    
    def __init__(self, vocab_size, d_model, num_heads, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len
    
    def forward(self, input_ids, mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x, weights = self.attn(x, mask=mask)
        logits = self.output_head(x)
        return logits, weights


def generate_copy_data(num_samples, seq_len, vocab_size):
    """
    Simple task: input is a sequence of random tokens,
    target is the same sequence shifted by 1 (next token prediction).
    This forces the model to learn positional/copying patterns.
    """
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    inputs = data[:, :-1]
    targets = data[:, 1:]
    return inputs, targets


def train_and_compare():
    vocab_size = 32
    d_model = 64
    max_seq_len = 32
    seq_len = 17  # 16 input + 1 for shift
    num_samples = 1000
    num_steps = 500
    
    inputs, targets = generate_copy_data(num_samples, seq_len, vocab_size)
    
    mask = torch.triu(
        torch.ones(inputs.size(1), inputs.size(1)) * float('-inf'), diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    configs = [
        ("1 head  (d_k=64)", 1),
        ("4 heads (d_k=16)", 4),
        ("8 heads (d_k=8) ", 8),
    ]
    
    for name, num_heads in configs:
        torch.manual_seed(42)  # Same init for fair comparison
        model = TinyAttentionModel(vocab_size, d_model, num_heads, max_seq_len)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        losses = []
        for step in range(num_steps):
            # Random mini-batch
            idx = torch.randint(0, num_samples, (64,))
            batch_in, batch_tgt = inputs[idx], targets[idx]
            
            logits, _ = model(batch_in, mask=mask)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_tgt.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                losses.append(loss.item())
        
        loss_str = " → ".join([f"{l:.3f}" for l in losses])
        print(f"{name} | Loss trajectory (every 100 steps): {loss_str} | Final: {losses[-1]:.4f}")
    
    print("\nWhat to look for:")
    print("- More heads generally converge faster or to a lower loss")
    print("- 1 head has limited capacity to capture multiple patterns simultaneously")
    print("- The gap is more visible on tasks requiring multiple types of relationships")
    print("- On very simple tasks, the difference may be small — that's also a valid lesson")
    print("  (multi-head shines when there ARE multiple patterns to learn)")


if __name__ == "__main__":
    train_and_compare()