"""Training system with frozen and end-to-end phases."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os
from pathlib import Path

from .pipeline import IntegratedPipeline
from .data import CoTDataset, HRDocDataset, GSM8KDataset, collate_fn


class TrainingOrchestrator:
    """Orchestrates training across frozen and end-to-end phases."""
    
    def __init__(self, pipeline: IntegratedPipeline, config: Dict):
        self.pipeline = pipeline
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Separate optimizers for each component
        self.optimizer_hrm = optim.AdamW(
            self.pipeline.hrm_processor.parameters(),
            lr=config.get('hrm_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.optimizer_tensorlnn = optim.AdamW(
            self.pipeline.tensorlnn_evaluator.parameters(),
            lr=config.get('tensorlnn_lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # RWKV optimizer (for end-to-end phase)
        self.optimizer_rwkv = None
        
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def setup_end_to_end(self):
        """Setup for end-to-end fine-tuning phase."""
        # Unfreeze RWKV
        self.pipeline.rwkv_encoder.set_frozen(False)
        
        # Create RWKV optimizer
        self.optimizer_rwkv = optim.AdamW(
            self.pipeline.rwkv_encoder.parameters(),
            lr=self.config.get('rwkv_lr', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
    
    def train_phase_frozen(self, cot_loader: DataLoader, hrdoc_loader: DataLoader,
                          gsm8k_loader: DataLoader, epochs: int = 3):
        """Phase 1: Frozen training where RWKV parameters remain fixed."""
        print("Starting Phase 1: Frozen Training")
        
        self.pipeline.rwkv_encoder.set_frozen(True)
        self.pipeline.train()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train HRM on CoT and HRDoc
            self._train_hrm_epoch(cot_loader, name="CoT")
            self._train_hrm_epoch(hrdoc_loader, name="HRDoc")
            
            # Train Tensor-LNN on GSM8K
            self._train_tensorlnn_epoch(gsm8k_loader)
    
    def train_phase_end_to_end(self, cot_loader: DataLoader, hrdoc_loader: DataLoader,
                               gsm8k_loader: DataLoader, epochs: int = 1):
        """Phase 2: End-to-end fine-tuning."""
        print("\nStarting Phase 2: End-to-End Fine-tuning")
        
        self.setup_end_to_end()
        self.pipeline.train()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Joint training
            self._train_joint_epoch(cot_loader, hrdoc_loader, gsm8k_loader)
    
    def _train_hrm_epoch(self, loader: DataLoader, name: str = ""):
        """Train HRM for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(self.device)
            puzzle_ids = batch['puzzle_ids'].to(self.device)
            
            self.optimizer_hrm.zero_grad()
            
            # Forward pass
            output = self.pipeline(input_ids, puzzle_ids, recursive=False)
            
            # Loss: encourage hierarchical frame quality
            # Simplified loss: minimize variance in frames
            frames = output.hierarchical_frames
            loss = frames.std().mean()
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pipeline.hrm_processor.parameters(), 1.0)
            self.optimizer_hrm.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  {name} batch {batch_idx}, loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  {name} epoch average loss: {avg_loss:.4f}")
    
    def _train_tensorlnn_epoch(self, loader: DataLoader):
        """Train Tensor-LNN for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(self.device)
            puzzle_ids = batch['puzzle_ids'].to(self.device)
            
            self.optimizer_tensorlnn.zero_grad()
            
            # Forward pass
            output = self.pipeline(input_ids, puzzle_ids, recursive=False)
            
            # Loss: minimize uncertainty gap (U - L)
            L, U = output.logic_bounds
            gap = (U - L).mean()
            loss = gap
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pipeline.tensorlnn_evaluator.parameters(), 1.0)
            self.optimizer_tensorlnn.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  GSM8K batch {batch_idx}, loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  GSM8K epoch average loss: {avg_loss:.4f}")
    
    def _train_joint_epoch(self, cot_loader: DataLoader, hrdoc_loader: DataLoader,
                          gsm8k_loader: DataLoader):
        """Joint training across all components."""
        # Combine loaders (simplified: alternate between them)
        loaders = [
            (cot_loader, 'CoT'),
            (hrdoc_loader, 'HRDoc'),
            (gsm8k_loader, 'GSM8K')
        ]
        
        for loader, name in loaders:
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                puzzle_ids = batch['puzzle_ids'].to(self.device)
                
                # Zero all gradients
                self.optimizer_hrm.zero_grad()
                self.optimizer_tensorlnn.zero_grad()
                if self.optimizer_rwkv:
                    self.optimizer_rwkv.zero_grad()
                
                # Forward pass
                output = self.pipeline(input_ids, puzzle_ids, recursive=True)
                
                # Combined loss
                frames = output.hierarchical_frames
                L, U = output.logic_bounds
                
                loss = frames.std().mean() + (U - L).mean()
                
                # Backward
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.pipeline.hrm_processor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.pipeline.tensorlnn_evaluator.parameters(), 1.0)
                if self.optimizer_rwkv:
                    torch.nn.utils.clip_grad_norm_(self.pipeline.rwkv_encoder.parameters(), 1.0)
                
                # Step optimizers
                self.optimizer_hrm.step()
                self.optimizer_tensorlnn.step()
                if self.optimizer_rwkv:
                    self.optimizer_rwkv.step()
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'pipeline_state': self.pipeline.state_dict(),
            'optimizer_hrm': self.optimizer_hrm.state_dict(),
            'optimizer_tensorlnn': self.optimizer_tensorlnn.state_dict(),
            'epoch': self.current_epoch,
            'config': self.config
        }
        
        if self.optimizer_rwkv:
            checkpoint['optimizer_rwkv'] = self.optimizer_rwkv.state_dict()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.pipeline.load_state_dict(checkpoint['pipeline_state'])
        self.optimizer_hrm.load_state_dict(checkpoint['optimizer_hrm'])
        self.optimizer_tensorlnn.load_state_dict(checkpoint['optimizer_tensorlnn'])
        
        if 'optimizer_rwkv' in checkpoint and self.optimizer_rwkv:
            self.optimizer_rwkv.load_state_dict(checkpoint['optimizer_rwkv'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        print(f"Checkpoint loaded from {path}")

