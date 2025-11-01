"""HRM Integration Adapter"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Dynamic imports with type checking support
if TYPE_CHECKING:
    # Type stubs for static analysis
    from typing import Any
    HierarchicalReasoningModel_ACTV1 = Any
    HierarchicalReasoningModel_ACTV1Carry = Any
else:
    # Runtime imports with fallback logic
    hrm_base_path = Path(__file__).parent.parent.parent / "HRM"
    hrm_models_path = hrm_base_path / "models"
    
    # Try multiple import paths
    _hierarchical_model = None
    _hierarchical_carry = None
    
    # Path 1: Direct import from models.hrm
    if hrm_models_path.exists():
        sys.path.insert(0, str(hrm_base_path))
        try:
            from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Carry  # type: ignore
            _hierarchical_model = HierarchicalReasoningModel_ACTV1
            _hierarchical_carry = HierarchicalReasoningModel_ACTV1Carry
        except ImportError:
            pass
    
    # Path 2: Try as package
    if _hierarchical_model is None:
        hrm_path = hrm_models_path / "hrm"
        if hrm_path.exists():
            sys.path.insert(0, str(hrm_models_path))
            try:
                from hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Carry  # type: ignore
                _hierarchical_model = HierarchicalReasoningModel_ACTV1
                _hierarchical_carry = HierarchicalReasoningModel_ACTV1Carry
            except ImportError:
                pass
    
    # Fallback: Use typing.Any for missing imports
    if _hierarchical_model is None:
        from typing import Any
        HierarchicalReasoningModel_ACTV1 = Any
        HierarchicalReasoningModel_ACTV1Carry = Any
    else:
        HierarchicalReasoningModel_ACTV1 = _hierarchical_model
        HierarchicalReasoningModel_ACTV1Carry = _hierarchical_carry


class HRMProcessor(nn.Module):
    """Processor wrapping HierarchicalReasoningModel_ACTV1."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.model = HierarchicalReasoningModel_ACTV1(config)
        self.carry: Optional[HierarchicalReasoningModel_ACTV1Carry] = None
    
    def reset_carry(self, batch_size: int):
        """Reset carry state."""
        dummy_batch = {
            "inputs": torch.zeros((batch_size, self.config["seq_len"]), dtype=torch.long),
            "puzzle_identifiers": torch.zeros((batch_size,), dtype=torch.long)
        }
        self.carry = self.model.initial_carry(dummy_batch)
    
    def process_reasoning(self, rwkv_embeddings: torch.Tensor, puzzle_ids: torch.Tensor) -> torch.Tensor:
        """Process embeddings through HRM to generate hierarchical frames.
        
        Args:
            rwkv_embeddings: [batch_size, seq_len, hidden_size]
            puzzle_ids: [batch_size] puzzle identifiers
        
        Returns:
            hierarchical_frames: [batch_size, seq_len, hidden_size]
        """
        batch_size = rwkv_embeddings.shape[0]
        seq_len = rwkv_embeddings.shape[1]
        
        # Initialize carry if needed
        if self.carry is None:
            self.reset_carry(batch_size)
        
        # Project embeddings to token space for HRM
        # Add learned projection if not exists
        if not hasattr(self, 'embed_proj'):
            hidden_size = rwkv_embeddings.shape[-1]
            vocab_size = self.config.get("vocab_size", 50277)
            self.embed_proj = nn.Linear(hidden_size, vocab_size).to(rwkv_embeddings.device)
        
        # Project embeddings to logits (token probabilities)
        logits = self.embed_proj(rwkv_embeddings)  # [batch, seq, vocab]
        
        # Convert to token IDs (use top-k or sampling)
        inputs = torch.argmax(logits, dim=-1)  # [batch, seq]
        
        # Ensure proper sequence length
        target_seq_len = self.config.get("seq_len", seq_len)
        if inputs.shape[1] > target_seq_len:
            inputs = inputs[:, :target_seq_len]
        elif inputs.shape[1] < target_seq_len:
            # Pad
            pad_size = target_seq_len - inputs.shape[1]
            inputs = torch.cat([inputs, torch.zeros(batch_size, pad_size, dtype=torch.long, device=inputs.device)], dim=1)
        
        batch = {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_ids.long()
        }
        
        # Forward pass
        try:
            self.carry, outputs = self.model(self.carry, batch)
            # Extract hierarchical frames from logits
            hierarchical_frames = outputs["logits"]
        except Exception as e:
            # Fallback: return embeddings directly if HRM fails
            print(f"HRM processing warning: {e}, using embeddings directly")
            hierarchical_frames = rwkv_embeddings
        
        return hierarchical_frames
    
    def forward(self, rwkv_embeddings: torch.Tensor, puzzle_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass alias."""
        return self.process_reasoning(rwkv_embeddings, puzzle_ids)

