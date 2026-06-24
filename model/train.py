import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from pathlib import Path

from model.transformer import TinyGPT
from model.dataset import load_shakespeare


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "model" / "weights" / "tiny_gpt.pt"


def train():
    # ─────────────────────────────────────────
    #  Config
    # ─────────────────────────────────────────
    config = {
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 4,
        'd_ff': 512,
        'max_seq_len': 256,
        'dropout': 0.1,
        'batch_size': 64,
        'seq_len': 128,
        'lr': 3e-4,
        'weight_decay': 0.01,
        'num_steps': 5000,
        'eval_interval': 500,
        'eval_steps': 50,
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}\n")
    
    # ─────────────────────────────────────────
    #  Data
    # ─────────────────────────────────────────
    train_dataset, val_dataset = load_shakespeare(seq_len=config['seq_len'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    
    # ─────────────────────────────────────────
    #  Model
    # ─────────────────────────────────────────
    model = TinyGPT(
        vocab_size=train_dataset.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
    ).to(device)
    
    model.count_parameters()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    
    # ─────────────────────────────────────────
    #  Training Loop
    # ─────────────────────────────────────────
    print(f"\n🚀 Training for {config['num_steps']} steps...\n")
    
    model.train()
    train_iter = iter(train_loader)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for step in range(config['num_steps']):
        # Get batch (restart iterator if exhausted)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        
        # Loss: reshape to (batch * seq_len, vocab_size) vs (batch * seq_len)
        loss = F.cross_entropy(
            logits.view(-1, train_dataset.vocab_size),
            y.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # ── Logging ──
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step:>5d} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
        
        # ── Evaluation ──
        if step > 0 and step % config['eval_interval'] == 0:
            val_loss = evaluate(model, val_dataset, config, device)
            
            print(f"\n  {'─' * 50}")
            print(f"  📊 Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            
            # Generate sample
            sample = generate(model, train_dataset, "First Citizen:\n", 
                            max_tokens=200, device=device)
            print(f"  📝 Sample:\n  {'─' * 50}")
            for line in sample.split('\n')[:8]:
                print(f"  {line}")
            print(f"  {'─' * 50}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), WEIGHTS_PATH)
                print(
                    f"  💾 Saved best model to {WEIGHTS_PATH} "
                    f"(val_loss={val_loss:.4f})\n"
                )
            
            model.train()
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time:.1f}s")
    print(f"   Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate(model, val_dataset, config, device):
    """Compute average loss over validation set."""
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=True, drop_last=True)
    
    total_loss = 0
    count = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, val_dataset.vocab_size), y.view(-1)
        )
        total_loss += loss.item()
        count += 1
        if count >= config['eval_steps']:
            break
    
    return total_loss / count


@torch.no_grad()
def generate(model, dataset, prompt, max_tokens=200, temperature=0.8, 
             top_k=40, device='cpu'):
    """
    Generate text from the model.
    
    Three sampling strategies demonstrated:
    
    GREEDY (temperature → 0):
        Always pick the highest-probability token.
        Deterministic but repetitive.
    
    TEMPERATURE SAMPLING:
        Divide logits by temperature before softmax.
        temperature < 1 → more focused (conservative)
        temperature > 1 → more random (creative)
        temperature = 1 → raw probabilities
    
    TOP-K SAMPLING:
        Only consider the top K most probable tokens.
        Prevents sampling very unlikely tokens (reduces garbage).
    """
    model.eval()
    
    # Encode prompt
    input_ids = dataset.encode(prompt).unsqueeze(0).to(device)
    
    for _ in range(max_tokens):
        # Truncate if beyond max_seq_len
        input_ids_cond = input_ids[:, -model.max_seq_len:]
        
        # Get predictions
        logits = model(input_ids_cond)
        
        # Only care about the LAST position's prediction
        logits = logits[:, -1, :]  # (batch, vocab_size)
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-k: zero out everything except top k tokens
        if top_k is not None:
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits[logits < min_top_k] = float('-inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return dataset.decode(input_ids[0].tolist())


if __name__ == "__main__":
    train()
