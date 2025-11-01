"""Train RWKV model on Wikipedia dataset."""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import json

# Add RWKV paths
sys.path.insert(0, str(Path(__file__).parent.parent / "RWKV-LM" / "RWKV-v4neo" / "src"))

# Set environment variables
os.environ["RWKV_JIT_ON"] = "0"

from utils import TOKENIZER


def process_wikipedia_to_tokenized(wikipedia_dir: str, output_file: str, max_files: int = None):
    """Process Wikipedia parquet files and convert to tokenized numpy array."""
    wikipedia_path = Path(wikipedia_dir)
    parquet_files = sorted(wikipedia_path.glob("train-*.parquet"))
    
    if max_files:
        parquet_files = parquet_files[:max_files]
    
    print(f"Processing {len(parquet_files)} Wikipedia files...")
    
    # Initialize tokenizer
    WORD_NAME = ["20B_tokenizer.json", "20B_tokenizer.json"]
    tokenizer_path = Path(__file__).parent.parent / "RWKV-LM" / "RWKV-v4neo" / "20B_tokenizer.json"
    
    tokenizer = None
    if tokenizer_path.exists():
        try:
            # RWKV tokenizer expects a list with same file for both
            tokenizer = TOKENIZER([str(tokenizer_path), str(tokenizer_path)], UNKNOWN_CHAR=None)
            print(f"Using RWKV tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            tokenizer = None
    
    if tokenizer is None:
        print("Warning: Using simple character-level tokenization fallback")
        print("For best results, ensure 20B_tokenizer.json is available")
    
    all_tokens = []
    total_text_length = 0
    
    for idx, parquet_file in enumerate(parquet_files):
        print(f"Processing file {idx + 1}/{len(parquet_files)}: {parquet_file.name}")
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Combine title and text
            for _, row in df.iterrows():
                text = f"{row.get('title', '')}\n\n{row.get('text', '')}"
                
                if tokenizer:
                    # Tokenize using RWKV tokenizer
                    try:
                        # Refine context first
                        refined = tokenizer.refine_context(text)
                        # Encode - tokenizer can be transformers or word-level
                        if hasattr(tokenizer, 'tokenizer'):
                            # Using transformers PreTrainedTokenizerFast
                            tokens = tokenizer.tokenizer.encode(refined, add_special_tokens=False)
                        elif hasattr(tokenizer, 'stoi'):
                            # Using word-level tokenizer
                            refined_words = refined.split()
                            tokens = []
                            for word in refined_words:
                                if word in tokenizer.stoi:
                                    tokens.append(tokenizer.stoi[word])
                                else:
                                    tokens.append(tokenizer.UNKNOWN_CHAR)
                        else:
                            # Fallback
                            tokens = [ord(c) % 50277 for c in refined[:50000]]
                        
                        # Limit token length to prevent memory issues
                        tokens = tokens[:50000]
                    except Exception as e:
                        print(f"Tokenization error for text length {len(text)}: {e}")
                        tokens = [ord(c) % 50277 for c in text[:50000]]
                else:
                    # Simple character-level fallback
                    tokens = [ord(c) % 50277 for c in text[:50000]]  # Limit length
                
                if len(tokens) > 0:
                    all_tokens.extend(tokens)
                    total_text_length += len(text)
        
        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue
        
        # Progress update
        if (idx + 1) % 10 == 0:
            print(f"  Processed {len(all_tokens)} tokens so far...")
    
    print(f"\nTotal tokens: {len(all_tokens)}")
    print(f"Total text length: {total_text_length:,} characters")
    
    # Save as numpy array
    token_array = np.array(all_tokens, dtype=np.int32)
    np.save(output_file, token_array)
    print(f"Saved tokenized data to {output_file}")
    
    return len(all_tokens)


def train_rwkv_wikipedia(
    data_file: str,
    model_name: str = "RWKV-4-Pile-169M",
    ctx_len: int = 1024,
    n_layer: int = 24,
    n_embd: int = 768,
    vocab_size: int = 50277,
    epochs: int = 10,
    batch_size: int = 8,
    lr_init: float = 6e-4,
    lr_final: float = 1e-5,
    proj_dir: str = "rwkv_wikipedia_out",
    device: str = "cuda"
):
    """Train RWKV model on Wikipedia data."""
    
    # Convert paths to absolute paths to avoid issues when changing directories
    data_file = str(Path(data_file).resolve())
    proj_dir = str(Path(proj_dir).resolve())
    
    # Check if we need to download a pretrained model
    model_path = None
    base_path = Path(__file__).parent.parent / "RWKV-LM"
    
    # Verify data file exists
    if not Path(data_file).exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Make sure the file exists or run preprocessing first."
        )
    
    # Look for existing models
    potential_models = [
        base_path / "RWKV-v4neo" / "math_demo" / "rwkv-200.pth",
        # Add more potential paths
    ]
    
    for pm in potential_models:
        if pm.exists():
            model_path = str(pm.resolve())
            print(f"Found existing model: {model_path}")
            break
    
    # If no model found, we'll train from scratch
    if not model_path:
        print("No pretrained model found. Training from scratch.")
        model_path = ""
    
    # Prepare training command
    train_script = base_path / "RWKV-v4neo" / "train.py"
    
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    
    # Check if deepspeed is available (optional)
    try:
        import deepspeed
        use_deepspeed = True
    except ImportError:
        use_deepspeed = False
        print("Note: DeepSpeed not available, using standard PyTorch Lightning strategy")
    
    # Build command
    cmd_parts = [
        "python", str(train_script),
        "--load_model", model_path,
        "--wandb", "",  # Disable wandb
        "--proj_dir", proj_dir,
        "--data_file", data_file,
        "--data_type", "numpy",
        "--vocab_size", str(vocab_size),
        "--ctx_len", str(ctx_len),
        "--epoch_steps", "1000",
        "--epoch_count", str(epochs),
        "--epoch_begin", "0",
        "--epoch_save", "5",
        "--micro_bsz", str(batch_size),
        "--n_layer", str(n_layer),
        "--n_embd", str(n_embd),
        "--pre_ffn", "0",
        "--head_qk", "0",
        "--lr_init", str(lr_init),
        "--lr_final", str(lr_final),
        "--warmup_steps", "0",
        "--beta1", "0.9",
        "--beta2", "0.99",
        "--adam_eps", "1e-8",
        "--grad_cp", "0",
    ]
    
    # Add device/precision/strategy arguments based on PyTorch Lightning version
    if device == "cuda":
        cmd_parts.extend([
            "--accelerator", "gpu",
            "--devices", "1",
            "--precision", "bf16",
        ])
    else:
        cmd_parts.extend([
            "--accelerator", "cpu",
            "--devices", "1",
            "--precision", "fp32",
        ])
    
    # Strategy: use deepspeed only if available, otherwise use auto
    if use_deepspeed and device == "cuda":
        cmd_parts.extend(["--strategy", "deepspeed_stage_2"])
    else:
        cmd_parts.extend(["--strategy", "auto"])
    
    print("\n" + "="*60)
    print("Starting RWKV Training on Wikipedia")
    print("="*60)
    print(f"Data file: {data_file}")
    print(f"Model: {model_name}")
    print(f"Layers: {n_layer}, Embedding: {n_embd}")
    print(f"Context length: {ctx_len}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr_init} -> {lr_final}")
    print(f"Output directory: {proj_dir}")
    print("="*60 + "\n")
    
    # Execute training
    import subprocess
    
    print("Executing training command...")
    print("Command:", " ".join(cmd_parts[:10]) + "...")
    print()
    
    result = subprocess.run(cmd_parts, cwd=str(base_path / "RWKV-v4neo"))
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"Check output in: {proj_dir}")
        
        # Find the final model
        proj_path = Path(proj_dir)
        final_model = proj_path / "rwkv-final.pth"
        if final_model.exists():
            print(f"\nTrained model saved to: {final_model}")
            print(f"\nTo use this model, update your config:")
            print(f"  config.rwkv.model_path = '{final_model.stem}'  # Without .pth")
    else:
        print("\n" + "="*60)
        print("Training failed. Check errors above.")
        print("="*60)
        print("\nCommon issues:")
        print("1. Missing dependencies: pip install pytorch-lightning==1.9.5")
        print("2. DeepSpeed (optional): Not required, but helps with multi-GPU. On Windows, skip it.")
        print("3. CUDA/GPU issues: Check CUDA installation and GPU availability")
        print("4. Out of memory: Reduce --batch_size or --ctx_len")
        print("5. Windows compatibility: Use 'auto' strategy (already set)")
        return False
    
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import pytorch_lightning
        print(f"✓ pytorch-lightning {pytorch_lightning.__version__} installed")
    except ImportError:
        missing.append("pytorch-lightning==1.9.5")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
    except ImportError:
        missing.append("torch")
    
    try:
        import deepspeed
        print(f"✓ deepspeed installed")
    except ImportError:
        print("⚠ deepspeed not installed (optional, but recommended for multi-GPU)")
    
    if missing:
        print("\n❌ Missing required dependencies:")
        for dep in missing:
            print(f"  pip install {dep}")
        return False
    
    return True


def main():
    """Main entry point."""
    import argparse
    
    # Check dependencies first
    print("Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return
    
    print()
    
    parser = argparse.ArgumentParser(description="Train RWKV on Wikipedia dataset")
    parser.add_argument("--wikipedia_dir", type=str, default="Data/wikipedia",
                       help="Directory containing Wikipedia parquet files")
    parser.add_argument("--output_data", type=str, default="rwkv_wikipedia_tokens.npy",
                       help="Output path for tokenized numpy file")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of parquet files to process (None = all)")
    parser.add_argument("--skip_preprocess", action="store_true",
                       help="Skip preprocessing if data file already exists")
    parser.add_argument("--model_name", type=str, default="RWKV-4-Pile-169M",
                       help="Model name for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--ctx_len", type=int, default=1024,
                       help="Context length")
    parser.add_argument("--n_layer", type=int, default=24,
                       help="Number of layers")
    parser.add_argument("--n_embd", type=int, default=768,
                       help="Embedding dimension")
    
    args = parser.parse_args()
    
    # Step 1: Preprocess Wikipedia data
    if not args.skip_preprocess or not Path(args.output_data).exists():
        print("Step 1: Preprocessing Wikipedia dataset...")
        process_wikipedia_to_tokenized(
            args.wikipedia_dir,
            args.output_data,
            max_files=args.max_files
        )
    else:
        print(f"Step 1: Using existing tokenized data: {args.output_data}")
    
    # Step 2: Train RWKV
    print("\nStep 2: Training RWKV model...")
    success = train_rwkv_wikipedia(
        data_file=args.output_data,
        model_name=args.model_name,
        ctx_len=args.ctx_len,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    if success:
        print("\n" + "="*60)
        print("Training pipeline completed!")
        print("="*60)
        print(f"Tokenized data: {args.output_data}")
        print(f"Trained model: rwkv_wikipedia_out/")
        print("\nTo use the trained model, update config:")
        print(f"  config.rwkv.model_path = 'rwkv_wikipedia_out/rwkv-final.pth'")
    else:
        print("\nTraining failed. Please check the errors above.")


if __name__ == "__main__":
    main()

