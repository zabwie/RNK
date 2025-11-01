"""Example usage of the integrated pipeline."""

import torch
from .config import PipelineConfig
from .components import RWKVEncoder, HRMProcessor, TensorLNNEvaluator
from .pipeline import IntegratedPipeline

def example_basic_usage():
    """Basic pipeline usage example."""
    print("Creating pipeline components...")
    
    # Configuration
    config = PipelineConfig.get_default()
    
    # Create components
    rwkv_encoder = RWKVEncoder(
        model_path=config.rwkv.model_path,
        vocab_size=config.rwkv.vocab_size,
        n_layer=config.rwkv.n_layer,
        n_embd=config.rwkv.n_embd,
        frozen=True
    )
    
    hrm_processor = HRMProcessor(config.hrm.__dict__)
    
    tensorlnn_evaluator = TensorLNNEvaluator(
        num_variables=config.tensorlnn.num_variables,
        gpu_device=config.tensorlnn.gpu_device
    )
    
    # Create pipeline
    pipeline = IntegratedPipeline(
        rwkv_encoder=rwkv_encoder,
        hrm_processor=hrm_processor,
        tensorlnn_evaluator=tensorlnn_evaluator
    )
    
    # Example input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.rwkv.vocab_size, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 10000, (batch_size,))
    
    print("Running forward pass...")
    output = pipeline(input_ids, puzzle_ids, recursive=True)
    
    print(f"Output shapes:")
    print(f"  RWKV embeddings: {output.rwkv_embeddings.shape}")
    print(f"  Hierarchical frames: {output.hierarchical_frames.shape}")
    print(f"  Logic bounds L: {output.logic_bounds[0].shape}")
    print(f"  Logic bounds U: {output.logic_bounds[1].shape}")
    print(f"  Iterations: {output.iteration}")
    print(f"  Inference time: {output.inference_time:.4f}s")


if __name__ == "__main__":
    example_basic_usage()

