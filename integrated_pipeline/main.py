"""Main entry point for training and inference."""

import argparse
import torch
from pathlib import Path

from .config import PipelineConfig
from .components import RWKVEncoder, HRMProcessor, TensorLNNEvaluator
from .pipeline import IntegratedPipeline
from .training import TrainingOrchestrator
from .data import CoTDataset, HRDocDataset, GSM8KDataset, collate_fn
from torch.utils.data import DataLoader


def create_pipeline(config: PipelineConfig) -> IntegratedPipeline:
    """Create the integrated pipeline from configuration."""
    # RWKV Encoder
    # Use dummy mode if model path is not provided
    use_dummy = not config.rwkv.model_path or not Path(config.rwkv.model_path + '.pth').exists()
    if use_dummy:
        print("Warning: RWKV model not found. Using dummy encoder mode for testing.")
        print("To use a real model, download from https://huggingface.co/BlinkDL and set model_path in config.")
    
    rwkv_encoder = RWKVEncoder(
        model_path=config.rwkv.model_path,
        vocab_size=config.rwkv.vocab_size,
        n_layer=config.rwkv.n_layer,
        n_embd=config.rwkv.n_embd,
        ctx_len=config.rwkv.ctx_len,
        frozen=config.rwkv.frozen,
        device=config.rwkv.device,
        float_mode=config.rwkv.float_mode,
        use_dummy=use_dummy
    )
    
    # HRM Processor
    hrm_config = config.hrm.__dict__
    hrm_processor = HRMProcessor(hrm_config)
    
    # Tensor-LNN Evaluator
    tensorlnn_evaluator = TensorLNNEvaluator(
        num_variables=config.tensorlnn.num_variables,
        gpu_device=config.tensorlnn.gpu_device,
        nepochs=config.tensorlnn.nepochs,
        lr=config.tensorlnn.lr,
        optimizer=config.tensorlnn.optimizer
    )
    
    # Integrated Pipeline
    pipeline = IntegratedPipeline(
        rwkv_encoder=rwkv_encoder,
        hrm_processor=hrm_processor,
        tensorlnn_evaluator=tensorlnn_evaluator,
        max_iterations=config.training.max_iterations,
        convergence_threshold=config.training.convergence_threshold
    )
    
    return pipeline


def train(config: PipelineConfig):
    """Train the integrated pipeline."""
    print("Initializing pipeline...")
    pipeline = create_pipeline(config)
    
    print("Loading datasets...")
    # CoT Dataset
    cot_dataset = CoTDataset(
        config.data.cot_path,
        max_examples=config.training.cot_max_examples
    )
    cot_loader = DataLoader(
        cot_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers
    )
    
    # HRDoc Dataset
    hrdoc_dataset = HRDocDataset(
        config.data.hrdoc_path,
        max_examples=config.training.hrdoc_max_examples
    )
    hrdoc_loader = DataLoader(
        hrdoc_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers
    )
    
    # GSM8K Dataset
    gsm8k_dataset = GSM8KDataset(config.data.gsm8k_path, split='train')
    gsm8k_loader = DataLoader(
        gsm8k_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers
    )
    
    print("Setting up training orchestrator...")
    orchestrator = TrainingOrchestrator(pipeline, config.training.__dict__)
    
    print("\n" + "="*60)
    print("Phase 1: Frozen Training")
    print("="*60)
    orchestrator.train_phase_frozen(
        cot_loader, hrdoc_loader, gsm8k_loader,
        epochs=config.training.frozen_epochs
    )
    
    # Save checkpoint after Phase 1
    checkpoint_path = Path(config.training.checkpoint_dir) / "phase1_checkpoint.pt"
    orchestrator.save_checkpoint(str(checkpoint_path))
    
    print("\n" + "="*60)
    print("Phase 2: End-to-End Fine-tuning")
    print("="*60)
    orchestrator.train_phase_end_to_end(
        cot_loader, hrdoc_loader, gsm8k_loader,
        epochs=config.training.e2e_epochs
    )
    
    # Save final checkpoint
    final_checkpoint = Path(config.training.checkpoint_dir) / "final_checkpoint.pt"
    orchestrator.save_checkpoint(str(final_checkpoint))
    
    print("\nTraining complete!")


def evaluate(config: PipelineConfig, checkpoint_path: str):
    """Evaluate the trained pipeline."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    pipeline = create_pipeline(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.training.device)
    pipeline.load_state_dict(checkpoint['pipeline_state'])
    pipeline.eval()
    
    print("Running evaluation...")
    # Example evaluation
    test_input = torch.randint(0, 50277, (1, 128)).to(config.training.device)
    test_puzzle_ids = torch.tensor([0]).to(config.training.device)
    
    result = pipeline.inference(test_input, test_puzzle_ids)
    print(f"Evaluation result: {result}")


def infer(config: PipelineConfig, checkpoint_path: str, input_text: str):
    """Run inference on input text."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    pipeline = create_pipeline(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.training.device)
    pipeline.load_state_dict(checkpoint['pipeline_state'])
    pipeline.eval()
    
    # Simple tokenization (replace with proper tokenizer)
    input_ids = torch.tensor([hash(word) % 50277 for word in input_text.split()[:512]])
    input_ids = input_ids.unsqueeze(0).to(config.training.device)
    puzzle_ids = torch.tensor([0]).to(config.training.device)
    
    result = pipeline.inference(input_ids, puzzle_ids)
    print(f"Inference result: {result}")


def main():
    parser = argparse.ArgumentParser(description="RWKV-HRM-TensorLNN Integrated Pipeline")
    parser.add_argument('--mode', choices=['train', 'eval', 'infer'], required=True,
                       help='Mode: train, eval, or infer')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (for eval/infer)')
    parser.add_argument('--input', type=str, default=None,
                       help='Input text for inference')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig.get_default()
    
    # Run mode
    if args.mode == 'train':
        train(config)
    elif args.mode == 'eval':
        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval mode")
        evaluate(config, args.checkpoint)
    elif args.mode == 'infer':
        if not args.checkpoint or not args.input:
            raise ValueError("--checkpoint and --input required for infer mode")
        infer(config, args.checkpoint, args.input)


if __name__ == "__main__":
    main()

