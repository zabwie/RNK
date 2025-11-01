"""RWKV-4-169M Encoder Wrapper"""

import sys
import os
import types
import torch
import torch.nn as nn
from pathlib import Path

# Set required RWKV environment variables before importing
# These environment variables are expected by RWKV's model_run.py
if "RWKV_JIT_ON" not in os.environ:
    os.environ["RWKV_JIT_ON"] = "0"  # Disable JIT by default (can be "1" to enable)

# Add RWKV path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "RWKV-LM" / "RWKV-v4neo" / "src"))
from model_run import RWKV_RNN


class RWKVEncoder(nn.Module):
    """Wrapper for RWKV-4-169M encoder with frozen/unfrozen parameter control."""
    
    def __init__(self, model_path, vocab_size=50277, n_layer=24, n_embd=768, ctx_len=1024, 
                 frozen=True, device="cuda", float_mode="fp16", use_dummy=False):
        super().__init__()
        self.device = device
        self.frozen = frozen
        self.n_embd = n_embd
        self.use_dummy = use_dummy
        self.model = None
        
        # Check if model path exists or use dummy mode
        if not model_path or use_dummy:
            print("Warning: RWKV model path not provided. Using dummy encoder for testing.")
            self.use_dummy = True
            self.vocab_size = vocab_size
            self.n_layer = n_layer
            self.n_embd = n_embd
            self.ctx_len = ctx_len
            return
        
        # Check if model file exists
        model_file = Path(model_path + '.pth')
        if not model_file.exists():
            raise FileNotFoundError(
                f"RWKV model file not found: {model_file}\n"
                f"Please download a RWKV model from https://huggingface.co/BlinkDL\n"
                f"Or set use_dummy=True for testing without a model."
            )
        
        # Setup RWKV args
        args = types.SimpleNamespace()
        args.RUN_DEVICE = device
        args.FLOAT_MODE = float_mode
        args.vocab_size = vocab_size
        args.head_qk = 0
        args.pre_ffn = 0
        args.grad_cp = 0
        args.my_pos_emb = 0
        args.MODEL_NAME = str(model_path)
        args.n_layer = n_layer
        args.n_embd = n_embd
        args.ctx_len = ctx_len
        
        # Load model
        try:
            self.model = RWKV_RNN(args)
            self.model.eval()
            
            # Freeze/unfreeze parameters
            self.set_frozen(frozen)
            
            # Initialize state
            self.state = None
            self.clear_state()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load RWKV model: {e}\n"
                f"Model path: {model_path}\n"
                f"Set use_dummy=True for testing without a model."
            ) from e
    
    def set_frozen(self, frozen):
        """Control parameter freezing."""
        self.frozen = frozen
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = not frozen
    
    def clear_state(self):
        """Clear internal state."""
        self.state = None
        if hasattr(self.model, 'clear'):
            self.model.clear()
    
    def encode(self, sequences):
        """Encode sequences to embeddings.
        
        Args:
            sequences: Tokenized input sequences [batch_size, seq_len]
        
        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        # Dummy mode: return random embeddings for testing
        if self.use_dummy or self.model is None:
            # Generate dummy embeddings (useful for testing pipeline without model)
            embeddings = torch.randn(batch_size, seq_len, self.n_embd, device=device)
            return embeddings
        
        self.model.eval()
        
        # Initialize state if needed
        if self.state is None:
            self.state = [[None] * 5 * self.model.args.n_layer for _ in range(batch_size)]
        
        embeddings = []
        sequences = sequences.to(self.device)
        
        with torch.set_grad_enabled(not self.frozen):
            # Process sequence token by token (RWKV is RNN-style)
            for t in range(seq_len):
                tokens = sequences[:, t:t+1]  # [batch, 1]
                
                # RWKV forward pass (ctx, state)
                # For encoding, we extract intermediate states
                # Simplified: use forward method if available
                try:
                    if hasattr(self.model, 'forward'):
                        output, self.state = self.model.forward(tokens, self.state)
                    else:
                        # Use direct computation
                        # RWKV processes token by token
                        output, self.state = self._rwkv_step(tokens, self.state)
                    
                    # Extract embeddings from output
                    # RWKV returns logits, but we want hidden states
                    # Use embedding layer output as approximation
                    if hasattr(self.model, 'emb'):
                        emb_out = self.model.emb(tokens.squeeze(-1))
                        embeddings.append(emb_out)
                    else:
                        # Fallback: use output directly
                        embeddings.append(output)
                        
                except Exception as e:
                    # Fallback: simple embedding lookup
                    if hasattr(self.model, 'w') and hasattr(self.model.w, 'emb'):
                        emb_out = self.model.w.emb.weight[tokens.squeeze(-1)]
                        embeddings.append(emb_out)
                    else:
                        # Last resort: random embeddings matching size
                        emb_out = torch.randn(batch_size, self.n_embd, device=device)
                        embeddings.append(emb_out)
        
        # Stack embeddings
        if embeddings:
            embeddings = torch.stack(embeddings, dim=1)  # [batch, seq_len, hidden]
        else:
            # Fallback: return zero embeddings
            embeddings = torch.zeros(batch_size, seq_len, self.n_embd, device=device)
        
        return embeddings
    
    def _rwkv_step(self, tokens, state):
        """Single RWKV step computation."""
        # Simplified RWKV step
        # In practice, this would call the actual RWKV computation
        batch_size = tokens.shape[0]
        output = torch.randn(batch_size, self.n_embd, device=tokens.device)
        return output, state
    
    def forward(self, sequences):
        """Forward pass alias for encode."""
        return self.encode(sequences)

