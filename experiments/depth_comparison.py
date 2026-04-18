import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.transformer import TinyGPT
from model.dataset import load_shakespeare


def compare_depths():
    train_dataset, _ = load_shakespeare(seq_len=128)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    device = 'cpu'
    
    print("=" * 70)
    print("  EXPERIMENT: Effect of Model Depth (1 vs 4 vs 8 layers)")
    print("=" * 70)
    
    for num_layers in [1, 4, 8]:
        torch.manual_seed(42)
        model = TinyGPT(
            vocab_size=train_dataset.vocab_size,
            d_model=64, num_heads=4, num_layers=num_layers, d_ff=256,
            max_seq_len=256
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        train_iter = iter(train_loader)
        
        print(f"\n  {'─' * 60}")
        print(f"  {num_layers} layer(s) | {n_params:,} parameters")
        print(f"  {'─' * 60}")
        
        for step in range(2000):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, train_dataset.vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 500 == 0:
                from model.train import generate
                sample = generate(model, train_dataset, "The ", max_tokens=60, device=device)
                sample_preview = sample[:60].replace('\n', '\\n')
                print(f"  Step {step:>4d} | Loss: {loss.item():.3f} | '{sample_preview}'")
    
    print(f"\n  📌 What to look for:")
    print(f"  • 1 layer: learns character patterns but weak grammar/coherence")
    print(f"  • 4 layers: much better structure, starts forming real phrases")
    print(f"  • 8 layers: best quality but more parameters and slower")
    print(f"  • Depth = capacity to model LONGER RANGE dependencies")


if __name__ == "__main__":
    compare_depths()