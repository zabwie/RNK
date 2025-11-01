# RWKV-HRM-TensorLNN Integrated Pipeline

Concise, powerful integration of three neural architectures for multi-step reasoning:
- **RWKV-4-169M**: Sequence encoding
- **HRM**: Hierarchical reasoning
- **Tensor-LNN**: Symbolic logical reasoning

## Quick Start

### Training

```bash
python -m integrated_pipeline.main --mode train
```

### Evaluation

```bash
python -m integrated_pipeline.main --mode eval --checkpoint checkpoints/final_checkpoint.pt
```

### Inference

```bash
python -m integrated_pipeline.main --mode infer --checkpoint checkpoints/final_checkpoint.pt --input "Your question here"
```

## Structure

- `components/`: Wrappers for RWKV, HRM, Tensor-LNN
- `data/`: Dataset loaders (CoT, HRDoc, GSM8K)
- `pipeline.py`: Main orchestrator with recursive reasoning
- `training.py`: Training system (frozen + end-to-end phases)
- `config.py`: Configuration management
- `main.py`: Entry point

## Configuration

See `config.py` for configuration options. Create a YAML file or use defaults:

```python
from integrated_pipeline.config import PipelineConfig
config = PipelineConfig.get_default()
```

## Training Phases

1. **Phase 1 (Frozen)**: RWKV frozen, HRM trains on CoT/HRDoc, Tensor-LNN on GSM8K
2. **Phase 2 (End-to-End)**: All components jointly optimized

