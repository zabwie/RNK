"""Tensor-LNN Evaluation Interface"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Tuple, Optional

# Add TensorLNN path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "TensorLNN" / "src"))
from tensorlnn import NeuralNet
import util


class TensorLNNEvaluator(nn.Module):
    """Evaluator wrapping NeuralNet from TensorLNN."""
    
    def __init__(self, num_variables: int, gpu_device: bool = True, 
                 nepochs: int = 100, lr: float = 1e-4, optimizer: str = 'AdamW'):
        super().__init__()
        self.num_variables = num_variables
        self.model = NeuralNet(num_variables, gpu_device, nepochs, lr, optimizer)
        self.gpu_device = gpu_device
    
    def frame_to_variables(self, hierarchical_frames: torch.Tensor) -> torch.Tensor:
        """Convert hierarchical frames to logical variables.
        
        Args:
            hierarchical_frames: [batch_size, seq_len, hidden_size]
        
        Returns:
            variables: [batch_size, num_variables] binary/continuous variables
        """
        batch_size = hierarchical_frames.shape[0]
        
        # Pool frames to fixed-size representation
        # Mean pooling across sequence length
        pooled = hierarchical_frames.mean(dim=1)  # [batch, hidden]
        
        # Project to num_variables dimensions
        if not hasattr(self, 'projection'):
            self.projection = nn.Linear(pooled.shape[-1], self.num_variables).to(pooled.device)
        
        variables = self.projection(pooled)  # [batch, num_variables]
        
        # Normalize to [0, 1] range for logical evaluation
        variables = torch.sigmoid(variables)
        
        return variables
    
    def evaluate_logic(self, hierarchical_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Evaluate logic on hierarchical frames.
        
        Args:
            hierarchical_frames: [batch_size, seq_len, hidden_size]
        
        Returns:
            L: Lower bounds [batch_size, 1]
            U: Upper bounds [batch_size, 1]
            inference_time: Time taken in seconds
        """
        # Convert frames to variables
        variables = self.frame_to_variables(hierarchical_frames)
        
        # Move to CPU for TensorLNN (adjust based on implementation)
        variables_cpu = variables.detach().cpu()
        
        # Inference
        L, U, inference_time = self.model.infer(variables_cpu)
        
        # Move back to original device
        device = hierarchical_frames.device
        L = L.to(device)
        U = U.to(device)
        
        return L, U, inference_time
    
    def forward(self, hierarchical_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning bounds."""
        L, U, _ = self.evaluate_logic(hierarchical_frames)
        return L, U

