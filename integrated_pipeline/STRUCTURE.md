# Project Structure

## Overview

Complete integration of RWKV-4-169M, HRM, and Tensor-LNN for multi-step reasoning.

## Directory Structure

```
integrated_pipeline/
├── __init__.py              # Package initialization
├── README.md               # Main documentation
├── QUICKSTART.md          # Quick start guide
├── STRUCTURE.md           # This file
├── requirements.txt       # Python dependencies
│
├── components/            # Core component wrappers
│   ├── __init__.py
│   ├── rwkv_encoder.py    # RWKV-4-169M encoder wrapper
│   ├── hrm_processor.py    # HRM integration adapter
│   └── tensorlnn_evaluator.py  # Tensor-LNN evaluator
│
├── data/                  # Dataset loaders
│   ├── __init__.py
│   └── datasets.py        # CoT, HRDoc, GSM8K processors
│
├── config.py             # Configuration management
├── pipeline.py           # Main pipeline orchestrator
├── training.py           # Training system (frozen + e2e)
├── main.py              # Entry point (train/eval/infer)
├── utils.py             # Utility functions
└── example_usage.py     # Usage examples
```

## Component Details

### 1. RWKV Encoder (`components/rwkv_encoder.py`)
- Loads pretrained RWKV-4-169M model
- Supports frozen/unfrozen parameter control
- Extracts embeddings from sequences
- Handles RNN-style token-by-token processing

### 2. HRM Processor (`components/hrm_processor.py`)
- Wraps HierarchicalReasoningModel_ACTV1
- Converts embeddings to hierarchical frames
- Manages carry state for sequential reasoning
- Learns projection from embeddings to token space

### 3. Tensor-LNN Evaluator (`components/tensorlnn_evaluator.py`)
- Wraps NeuralNet from TensorLNN
- Converts hierarchical frames to logical variables
- Performs logic evaluation with confidence bounds
- Returns lower and upper bounds (L, U)

### 4. Main Pipeline (`pipeline.py`)
- Orchestrates RWKV → HRM → Tensor-LNN flow
- Implements recursive reasoning with feedback
- Convergence detection and iteration limits
- Supports both training and inference modes

### 5. Training System (`training.py`)
- Phase 1: Frozen training (RWKV frozen)
- Phase 2: End-to-end fine-tuning
- Separate optimizers per component
- Checkpoint saving/loading

### 6. Data Loaders (`data/datasets.py`)
- CoT: Chain-of-Thought reasoning (50k examples)
- HRDoc: Hierarchical documents (50k examples)
- GSM8K: Mathematical problems (full dataset)
- Unified batch collation

## Key Features

✅ Complete component integration  
✅ Recursive reasoning with feedback loops  
✅ Two-phase training (frozen → end-to-end)  
✅ Comprehensive configuration system  
✅ Dataset processing for all three datasets  
✅ Checkpoint management  
✅ Inference and evaluation modes  
✅ Error handling and fallbacks  

## Data Flow

```
Text Input
  ↓
RWKV Encoder (frozen/unfrozen)
  ↓
Embeddings [batch, seq, hidden]
  ↓
HRM Processor
  ↓
Hierarchical Frames [batch, seq, hidden]
  ↓
Tensor-LNN Evaluator
  ↓
Logic Bounds (L, U)
  ↓
Feedback Signal (recursive)
  ↑
  └──────────────┘
```

## Training Phases

**Phase 1: Frozen Training (3 epochs)**
- RWKV: Frozen
- HRM: Trains on CoT + HRDoc (50k each)
- Tensor-LNN: Trains on GSM8K

**Phase 2: End-to-End Fine-tuning (1 epoch)**
- All components: Joint optimization
- RWKV: Unfrozen with low learning rate
- Maintains learned representations

## Usage

See `QUICKSTART.md` for detailed usage instructions.

