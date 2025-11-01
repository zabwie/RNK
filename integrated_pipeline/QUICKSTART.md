# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Train the Pipeline

```bash
python -m integrated_pipeline.main --mode train
```

This will:
- Load CoT, HRDoc, and GSM8K datasets
- Run Phase 1: Frozen training (HRM on CoT/HRDoc, Tensor-LNN on GSM8K)
- Run Phase 2: End-to-end fine-tuning
- Save checkpoints to `checkpoints/`

### 2. Evaluate

```bash
python -m integrated_pipeline.main --mode eval --checkpoint checkpoints/final_checkpoint.pt
```

### 3. Inference

```bash
python -m integrated_pipeline.main --mode infer --checkpoint checkpoints/final_checkpoint.pt --input "Your question here"
```

## Programmatic Usage

```python
from integrated_pipeline import IntegratedPipeline, PipelineConfig
from integrated_pipeline.components import RWKVEncoder, HRMProcessor, TensorLNNEvaluator

# Load config
config = PipelineConfig.get_default()

# Create pipeline
rwkv = RWKVEncoder(...)
hrm = HRMProcessor(config.hrm.__dict__)
tensorlnn = TensorLNNEvaluator(...)

pipeline = IntegratedPipeline(rwkv, hrm, tensorlnn)

# Inference
result = pipeline.inference(input_ids, puzzle_ids)
```

## Configuration

Edit `config.py` or create a YAML file:

```yaml
rwkv:
  model_path: "path/to/rwkv-model"
  frozen: true

hrm:
  hidden_size: 512
  H_cycles: 2
  L_cycles: 2

training:
  frozen_epochs: 3
  batch_size: 8
```

## Architecture

```
Input → RWKV Encoder → HRM Processor → Tensor-LNN Evaluator
                            ↑              ↓
                            └── Feedback ──┘
```

## Key Features

- **Frozen Training**: Stabilize embeddings before joint optimization
- **Recursive Reasoning**: Feedback loop for iterative refinement  
- **Hierarchical Frames**: Multi-level reasoning representations
- **Logic Evaluation**: Symbolic reasoning with confidence bounds

