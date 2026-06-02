import torch
import sys
from pathlib import Path

# Add project root to path so package imports like model.* resolve reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TinyGPT
from model.dataset import CharDataset


class ModelInference:
    """
    Wraps the TinyGPT model for use by the API.
    
    Why a class?
    - Holds the model and dataset (vocab) in memory
    - Load once at startup, reuse across all requests
    - Clean interface: the API calls generate(), never touches tensors
    """
    
    def __init__(self):
        self.model = None
        self.dataset = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load(self, weights_path: str, data_path: str):
        """
        Load model weights and vocabulary.
        
        Called ONCE at server startup, not per request.
        Loading a model takes ~1-2 seconds.
        Doing that per request would mean 1-2 second latency
        BEFORE generation even starts. Unacceptable.
        """
        # Load the dataset to get the vocabulary (char_to_idx, idx_to_char)
        with open(data_path, 'r') as f:
            text = f.read()
        self.dataset = CharDataset(text, seq_len=128)
        
        # Initialize model with same config as training
        self.model = TinyGPT(
            vocab_size=self.dataset.vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=4,
            d_ff=512,
            max_seq_len=256,
            dropout=0.0,        # No dropout during inference
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()       # Set to evaluation mode (disables dropout)
        
        print(f"✓ Model loaded on {self.device}")
        print(f"  Vocab size: {self.dataset.vocab_size}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    @torch.no_grad()            # No gradient computation during inference
    def generate(self, prompt: str, max_tokens: int = 50,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        """
        Generate text from a prompt.
        
        This is the ONLY function the API needs to know about.
        It takes strings in, returns strings out.
        No tensors, no model details leak to the API layer.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Encode prompt to token IDs
        input_ids = self.dataset.encode(prompt).unsqueeze(0).to(self.device)
        
        for _ in range(max_tokens):
            # Truncate to max context length
            input_ids_cond = input_ids[:, -self.model.max_seq_len:]
            
            # Get model predictions
            logits = self.model(input_ids_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                top_k_values, _ = torch.topk(logits, top_k)
                min_val = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < min_val] = float('-inf')
            
            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode back to string
        generated = self.dataset.decode(input_ids[0].tolist())
        return generated


    @torch.no_grad()
    def generate_stream(self, prompt: str, max_tokens: int = 50,
                        temperature: float = 0.8, top_k: int = 40):
        """
        Generator version of generate(). Yields one character at a time
        as it's produced, instead of returning the full string at the end.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        input_ids = self.dataset.encode(prompt).unsqueeze(0).to(self.device)

        for _ in range(max_tokens):
            input_ids_cond = input_ids[:, -self.model.max_seq_len:]
            logits = self.model(input_ids_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k_values, _ = torch.topk(logits, top_k)
                min_val = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < min_val] = float('-inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode just this one token and yield it immediately
            char = self.dataset.idx_to_char[next_token.item()]
            yield char



# Singleton instance — created once, shared across all requests
model_inference = ModelInference()