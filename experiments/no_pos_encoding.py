import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.transformer import TinyGPT
from model.dataset import load_shakespeare


class TinyGPTNoPos(TinyGPT):
    """TinyGPT with positional embeddings ZEROED OUT."""
    
    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        
        # Only token embedding — NO position embedding
        x = self.token_emb(input_ids)  # ❌ Missing: + self.pos_emb(positions)
        x = self.dropout(x)
        
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.final_norm(x)
        return self.output_head(x)


def compare():
    train_dataset, _ = load_shakespeare(seq_len=128)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    device = 'cpu'
    
    print("=" * 70)
    print("  EXPERIMENT: With vs Without Positional Embeddings")
    print("=" * 70)
    
    for label, ModelClass in [("WITH positions", TinyGPT), ("  NO positions", TinyGPTNoPos)]:
        torch.manual_seed(42)
        model = ModelClass(
            vocab_size=train_dataset.vocab_size,
            d_model=64, num_heads=4, num_layers=4, d_ff=256,
            max_seq_len=256
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        train_iter = iter(train_loader)
        
        for step in range(1000):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, train_dataset.vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 200 == 0:
                # Generate sample
                from model.train import generate
                sample = generate(model, train_dataset, "The ", max_tokens=80, device=device)
                sample_preview = sample[:80].replace('\n', '\\n')
                print(f"  {label} | Step {step:>4d} | Loss: {loss.item():.3f} | '{sample_preview}'")
        
        print()
    
    print("  📌 What to look for:")
    print("  • Both models reduce loss (they can still learn character frequencies)")
    print("  • WITHOUT positions: generated text has real words but WRONG ORDER")
    print("  • Word order, grammar structure, and sentence flow break down")
    print("  • The model can learn WHAT tokens appear together but not WHERE")


if __name__ == "__main__":
    compare()