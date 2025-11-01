"""Configuration management for the integrated pipeline."""

from dataclasses import dataclass, asdict
from typing import Dict, Optional
import yaml
from pathlib import Path


@dataclass
class RWKVConfig:
    """RWKV encoder configuration."""
    model_path: str = ""  # Path to RWKV model .pth file (without .pth extension)
    # Note: If model_path is empty, a dummy encoder will be used for testing
    # Download models from: https://huggingface.co/BlinkDL
    vocab_size: int = 50277
    n_layer: int = 24
    n_embd: int = 768
    ctx_len: int = 1024
    frozen: bool = True
    device: str = "cuda"
    float_mode: str = "fp16"


@dataclass
class HRMConfig:
    """HRM processor configuration."""
    batch_size: int = 8
    seq_len: int = 512
    puzzle_emb_ndim: int = 512
    num_puzzle_identifiers: int = 10000
    vocab_size: int = 50277
    H_cycles: int = 2
    L_cycles: int = 2
    H_layers: int = 4
    L_layers: int = 4
    hidden_size: int = 512
    expansion: float = 4.0
    num_heads: int = 8
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    forward_dtype: str = "bfloat16"


@dataclass
class TensorLNNConfig:
    """Tensor-LNN evaluator configuration."""
    num_variables: int = 100
    gpu_device: bool = True
    nepochs: int = 100
    lr: float = 1e-4
    optimizer: str = "AdamW"


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Phase 1: Frozen training
    frozen_epochs: int = 3
    cot_max_examples: int = 50000
    hrdoc_max_examples: int = 50000
    
    # Phase 2: End-to-end
    e2e_epochs: int = 1
    
    # Optimizers
    hrm_lr: float = 1e-4
    tensorlnn_lr: float = 1e-4
    rwkv_lr: float = 1e-5
    weight_decay: float = 0.01
    
    # Training settings
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda"
    
    # Recursive reasoning
    max_iterations: int = 5
    convergence_threshold: float = 0.01
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000


@dataclass
class DataConfig:
    """Data configuration."""
    cot_path: str = "Data/CoT/CoT_collection_en.json"
    hrdoc_path: str = "Data/HRDoc/train"
    gsm8k_path: str = "Data/GSM8K/main"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    rwkv: RWKVConfig
    hrm: HRMConfig
    tensorlnn: TensorLNNConfig
    training: TrainingConfig
    data: DataConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary."""
        return cls(
            rwkv=RWKVConfig(**config_dict.get('rwkv', {})),
            hrm=HRMConfig(**config_dict.get('hrm', {})),
            tensorlnn=TensorLNNConfig(**config_dict.get('tensorlnn', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {}))
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def get_default() -> 'PipelineConfig':
        """Get default configuration."""
        return PipelineConfig(
            rwkv=RWKVConfig(),
            hrm=HRMConfig(),
            tensorlnn=TensorLNNConfig(),
            training=TrainingConfig(),
            data=DataConfig()
        )

