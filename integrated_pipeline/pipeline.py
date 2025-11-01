"""Main pipeline orchestrator for RWKV → HRM → Tensor-LNN flow."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .components import RWKVEncoder, HRMProcessor, TensorLNNEvaluator


@dataclass
class PipelineOutput:
    """Output from pipeline forward pass."""
    rwkv_embeddings: torch.Tensor
    hierarchical_frames: torch.Tensor
    logic_bounds: Tuple[torch.Tensor, torch.Tensor]  # (L, U)
    inference_time: float
    iteration: int


class IntegratedPipeline(nn.Module):
    """Main pipeline orchestrating RWKV → HRM → Tensor-LNN flow with recursive reasoning."""
    
    def __init__(self, rwkv_encoder: RWKVEncoder, hrm_processor: HRMProcessor, 
                 tensorlnn_evaluator: TensorLNNEvaluator, max_iterations: int = 5,
                 convergence_threshold: float = 0.01):
        super().__init__()
        self.rwkv_encoder = rwkv_encoder
        self.hrm_processor = hrm_processor
        self.tensorlnn_evaluator = tensorlnn_evaluator
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def generate_feedback(self, logic_bounds: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Generate feedback signal from logic evaluation results.
        
        Args:
            logic_bounds: (L, U) tuple of lower and upper bounds
        
        Returns:
            feedback: Feedback tensor for RWKV refinement
        """
        L, U = logic_bounds
        
        # Compute confidence and gap
        confidence = (L + U) / 2  # Mean confidence
        gap = U - L  # Uncertainty gap
        
        # Feedback: encourage convergence (small gap) and high confidence
        feedback = confidence * (1 - gap)
        
        # Expand to match embedding dimensions
        batch_size = L.shape[0]
        hidden_size = self.rwkv_encoder.n_embd
        
        # Project feedback to embedding space
        if not hasattr(self, 'feedback_proj'):
            self.feedback_proj = nn.Linear(1, hidden_size).to(L.device)
        
        feedback_expanded = self.feedback_proj(feedback.unsqueeze(-1))
        
        return feedback_expanded
    
    def forward(self, sequences: torch.Tensor, puzzle_ids: torch.Tensor, 
                recursive: bool = True) -> PipelineOutput:
        """Forward pass through complete pipeline.
        
        Args:
            sequences: Tokenized input sequences [batch_size, seq_len]
            puzzle_ids: Puzzle identifiers [batch_size]
            recursive: Whether to perform recursive reasoning
        
        Returns:
            PipelineOutput with all intermediate and final results
        """
        batch_size = sequences.shape[0]
        device = sequences.device
        
        # Initial encoding
        rwkv_embeddings = self.rwkv_encoder(sequences)
        
        best_bounds = None
        best_iteration = 0
        
        # Recursive reasoning loop
        for iteration in range(self.max_iterations if recursive else 1):
            # HRM processing
            hierarchical_frames = self.hrm_processor(rwkv_embeddings, puzzle_ids)
            
            # Tensor-LNN evaluation
            L, U, inference_time = self.tensorlnn_evaluator.evaluate_logic(hierarchical_frames)
            
            # Check convergence
            gap = (U - L).mean().item()
            
            if best_bounds is None or gap < (best_bounds[1] - best_bounds[0]).mean().item():
                best_bounds = (L.clone(), U.clone())
                best_iteration = iteration
            
            # Generate feedback for next iteration
            if recursive and iteration < self.max_iterations - 1:
                feedback = self.generate_feedback((L, U))
                
                # Apply feedback to embeddings (simple additive update)
                rwkv_embeddings = rwkv_embeddings + 0.1 * feedback.unsqueeze(1)
                
                # Re-encode with feedback (optional: can skip re-encoding for efficiency)
                # For efficiency, we apply feedback directly to embeddings
            
            # Check convergence
            if gap < self.convergence_threshold:
                break
        
        return PipelineOutput(
            rwkv_embeddings=rwkv_embeddings,
            hierarchical_frames=hierarchical_frames,
            logic_bounds=best_bounds or (L, U),
            inference_time=inference_time,
            iteration=best_iteration
        )
    
    def inference(self, sequences: torch.Tensor, puzzle_ids: torch.Tensor) -> Dict:
        """Inference-only mode."""
        self.eval()
        with torch.no_grad():
            output = self.forward(sequences, puzzle_ids, recursive=True)
        
        L, U = output.logic_bounds
        confidence = (L + U) / 2
        
        return {
            'bounds': (L.item(), U.item()),
            'confidence': confidence.item(),
            'iteration': output.iteration,
            'inference_time': output.inference_time
        }

