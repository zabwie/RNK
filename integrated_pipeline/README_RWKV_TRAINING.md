# RWKV Training on Wikipedia

This script trains an RWKV model on the Wikipedia dataset.

## Quick Start

```bash
# Process Wikipedia data and train RWKV
python -m integrated_pipeline.train_rwkv_wikipedia --epochs 10 --batch_size 8

# Use only first 10 files for faster testing
python -m integrated_pipeline.train_rwkv_wikipedia --max_files 10 --epochs 2

# Skip preprocessing if data already processed
python -m integrated_pipeline.train_rwkv_wikipedia --skip_preprocess
```

## Arguments

- `--wikipedia_dir`: Directory with Wikipedia parquet files (default: `Data/wikipedia`)
- `--output_data`: Output path for tokenized numpy file (default: `rwkv_wikipedia_tokens.npy`)
- `--max_files`: Maximum parquet files to process (None = all 41 files)
- `--skip_preprocess`: Skip preprocessing if data file exists
- `--model_name`: Model name (default: `RWKV-4-Pile-169M`)
- `--epochs`: Training epochs (default: 10)
- `--batch_size`: Batch size (default: 8)
- `--ctx_len`: Context length (default: 1024)
- `--n_layer`: Number of layers (default: 24)
- `--n_embd`: Embedding dimension (default: 768)

## Requirements

- PyTorch with CUDA support
- PyTorch Lightning 1.9.5
- pandas, numpy
- RWKV tokenizer (20B_tokenizer.json)

## Model Configuration

The script uses RWKV-4neo architecture which is stable and efficient:
- **Architecture**: RWKV-4neo (latest stable version)
- **Default size**: 24 layers, 768 embedding (169M parameters)
- **Context**: 1024 tokens
- **Precision**: BFloat16 for GPU, FP32 for CPU

## Output

After training, you'll find:
- `rwkv_wikipedia_tokens.npy`: Tokenized Wikipedia data
- `rwkv_wikipedia_out/`: Training output directory
  - `rwkv-{epoch}.pth`: Checkpoints every 5 epochs
  - `rwkv-final.pth`: Final trained model

## Using the Trained Model

Update your config to use the trained model:

```python
from integrated_pipeline.config import PipelineConfig
config = PipelineConfig.get_default()
config.rwkv.model_path = "rwkv_wikipedia_out/rwkv-final"  # Without .pth extension
```

## Performance Notes

- Processing all 41 Wikipedia files may take time
- Use `--max_files` to limit for testing
- Training requires GPU for reasonable speed
- Adjust batch size based on available VRAM

