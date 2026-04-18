import torch
from torch.utils.data import Dataset, DataLoader


class CharDataset(Dataset):
    """
    Character-level dataset for language modeling.
    
    Takes raw text, builds a character vocabulary, and returns
    (input, target) pairs where target is input shifted by 1.
    """
    
    def __init__(self, text, seq_len, chars=None):
        self.seq_len = seq_len
        
        # Build or reuse a shared vocabulary.
        # Reusing the same vocab for train/val keeps model output size consistent.
        self.chars = sorted(list(set(text))) if chars is None else list(chars)
        self.vocab_size = len(self.chars)
        
        # Character ↔ index mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Encode entire text as indices
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        
        print(f"📖 Dataset loaded:")
        print(f"   Characters: {len(text):,}")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Vocab: {''.join(self.chars[:30])}{'...' if len(self.chars) > 30 else ''}")
        print(f"   Sequences:  {len(self):,}")
    
    def __len__(self):
        # How many (seq_len+1) chunks fit in the text
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Grab seq_len+1 tokens: first seq_len are input, last seq_len are target
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]   # Input:  positions 0 to seq_len-1
        y = chunk[1:]     # Target: positions 1 to seq_len (shifted by 1)
        return x, y
    
    def decode(self, indices):
        """Convert list of indices back to string."""
        return ''.join([self.idx_to_char[i.item() if torch.is_tensor(i) else i] for i in indices])
    
    def encode(self, text):
        """Convert string to tensor of indices."""
        return torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)


def load_shakespeare(seq_len=128):
    """
    Load TinyShakespeare dataset.
    
    If the file doesn't exist, creates a small sample.
    Download the real one from:
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    """
    import os
    
    filepath = "data/tinyshakespeare.txt"
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            text = f.read()
    else:
        # Fallback: create the data directory and remind user to download
        os.makedirs("data", exist_ok=True)
        print("⚠️  TinyShakespeare not found! Downloading...")
        
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, filepath)
            with open(filepath, 'r') as f:
                text = f.read()
            print(f"✓ Downloaded {len(text):,} characters")
        except Exception:
            print("Could not download. Using built-in sample text.")
            text = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.
""" * 200  # Repeat to get enough data
    
    # Split into train (90%) and val (10%)
    split = int(0.9 * len(text))
    train_text = text[:split]
    val_text = text[split:]
    
    # Use one vocabulary for both splits to avoid train/val vocab mismatch.
    shared_chars = sorted(list(set(text)))
    train_dataset = CharDataset(train_text, seq_len, chars=shared_chars)
    val_dataset = CharDataset(val_text, seq_len, chars=shared_chars)
    
    return train_dataset, val_dataset