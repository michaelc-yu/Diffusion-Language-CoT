
# Training loop, supports VLB + diffusion


"""
training/train.py
Main training loop for diffusion CoT models with support for PLAID, SEDD, and LADI-R
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import time
import wandb
import yaml

from models.base_diffusion_adapter import DiffusionTransformer
from training.diffusion_loss import VLBLoss, NoiseSchedule
from training.corruptions import CorruptionScheduler


class DiffusionCoTTrainer:
    """
    Trainer for diffusion models on CoT reasoning tasks.
    Supports fine-tuning PLAID, SEDD, and LADI-R architectures.
    """
    
    def __init__(
        self,
        model: DiffusionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        model_type: str = 'base'  # 'base', 'plaid', 'sedd', 'ladir'
    ):
        """
        Args:
            model: DiffusionTransformer model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration dict
            device: Device to train on
            model_type: Type of diffusion model architecture
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.model_type = model_type
        
        # Training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Diffusion components
        self.noise_schedule = NoiseSchedule(
            num_timesteps=config.get('num_timesteps', 1000),
            schedule_type=config.get('schedule_type', 'cosine')
        )
        
        self.vlb_loss = VLBLoss(
            noise_schedule=self.noise_schedule,
            vocab_size=model.vocab_size,
            parameterization=config.get('parameterization', 'x0'),
            loss_type=config.get('loss_type', 'mse')
        )
        
        # Corruption scheduler
        self.corruption_scheduler = self._setup_corruption_scheduler()
        
        # Tracking metrics
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.best_val_accuracy = 0.0
        
        # Gradient accumulation
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('random_name', 'diffusion-cot'),
                config=config,
                name=config.get('run_name', None),
                tags=[model_type, config.get('experiment_name', 'default')]
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'results/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_counter = 0
        
        # Print hyperparameters
        print(f"\n{'='*70}")
        print(f"Trainer initialized: {model_type.upper()}")
        print(f"{'='*70}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Gradient accumulation: {self.accumulation_steps} steps")
        print(f"Effective batch size: {train_loader.batch_size * self.accumulation_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Corruption enabled: {config.get('use_corruption', True)}")
        print(f"{'='*70}\n")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer: # TODO: check this
        """Setup optimizer with proper weight decay and parameter groups."""
        # Separate parameters into decay and no-decay groups
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.get('weight_decay', 0.01)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        self.config['learning_rate'] = float(self.config['learning_rate'])
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get('learning_rate', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        warmup_steps = self.config.get('warmup_steps', 1000)
        total_steps = len(self.train_loader) * self.config.get('num_epochs', 100)
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Main scheduler
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        # Combine
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    def _setup_corruption_scheduler(self) -> CorruptionScheduler:
        """Setup corruption scheduler from config."""
        corruption_config = self.config.get('corruption', {})
        
        return CorruptionScheduler(
            mask_prob=corruption_config.get('mask_prob', 0.3),
            shuffle_prob=corruption_config.get('shuffle_prob', 0.2),
            drop_prob=corruption_config.get('drop_prob', 0.15),
            noise_std=corruption_config.get('noise_std', 0.0),
            curriculum=corruption_config.get('curriculum', False),
            mask_token_id=self.model.vocab_size - 1,  # Use last token as mask
            warmup_epochs=corruption_config.get('warmup_epochs', 10)
        )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            step: Current global step
            
        Returns:
            Dictionary of metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Apply corruptions if enabled
        if self.config.get('use_corruption', True):
            corruption_types = self.config.get('corruption_types', ['mask', 'shuffle', 'drop'])
            corrupted_batch = self.corruption_scheduler(
                batch, 
                corruption_types=corruption_types,
                vocab_size=self.model.vocab_size
            )
        else:
            corrupted_batch = batch
        
        # Get embeddings
        with torch.no_grad():
            # Corrupted embeddings for input
            corrupted_ids = corrupted_batch['input_ids']
            corrupted_embeddings = self.model.token_embeddings(corrupted_ids)
            
            # Clean embeddings for target
            clean_ids = batch['input_ids']
            clean_embeddings = self.model.token_embeddings(clean_ids)
        
        # Mixed precision training
        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss, metrics = self.vlb_loss(
                    self.model,
                    x_start=clean_ids,
                    embeddings=corrupted_embeddings
                )
                loss = loss / self.accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
        else:
            # Standard training
            loss, metrics = self.vlb_loss(
                self.model,
                x_start=clean_ids,
                embeddings=corrupted_embeddings
            )
            loss = loss / self.accumulation_steps
            loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            'loss': loss.item() * self.accumulation_steps,
            **metrics
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.corruption_scheduler.update_epoch(self.current_epoch)

        epoch_losses = []
        metrics_buffer = {k: [] for k in ['loss_t_mean', 'loss_t_std']}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", dynamic_ncols=True)
        start = time.time()

        for batch in pbar:
            metrics = self.train_step(batch, self.global_step)
            epoch_losses.append(metrics['loss'])

            # Track extra metrics
            for k in metrics_buffer:
                if k in metrics:
                    metrics_buffer[k].append(metrics[k])

            # Progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'step': self.global_step
            })

            # WandB logging
            if self.use_wandb and self.global_step % self.config.get('log_interval', 10) == 0:
                wandb.log({
                    'train/loss': metrics['loss'],
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.current_epoch,
                    'train/step': self.global_step,
                    **{f'train/{k}': v for k, v in metrics.items() if k != 'loss'},
                    **{f'corruption/{k}': v for k, v in self.corruption_scheduler.get_corruption_stats().items()}
                }, step=self.global_step)

            self.global_step += 1

        elapsed = time.time() - start

        return {
            'loss': np.mean(epoch_losses),
            'loss_std': np.std(epoch_losses),
            'time': elapsed,
            'samples_per_sec': len(self.train_loader.dataset) / elapsed,
            **{f'{k}_mean': np.mean(v) for k, v in metrics_buffer.items() if v}
        }


    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        losses = []
        metrics_buffer = {k: [] for k in ['loss_t_mean', 'loss_t_std']}

        pbar = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True)

        for batch in pbar:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            x_start = batch['input_ids']
            embeddings = self.model.token_embeddings(x_start)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, metrics = self.vlb_loss(self.model, x_start=x_start, embeddings=embeddings)
            else:
                loss, metrics = self.vlb_loss(self.model, x_start=x_start, embeddings=embeddings)

            losses.append(loss.item())
            for k in metrics_buffer:
                if k in metrics:
                    metrics_buffer[k].append(metrics[k])

            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        return {
            'loss': np.mean(losses),
            'loss_std': np.std(losses),
            **{f'{k}_mean': np.mean(v) for k, v in metrics_buffer.items() if v}
        }
    
    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
        additional_info: Optional[Dict] = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
        
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Checkpoint loaded: {path}")
        print(f"   Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 100)
        
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*70}\n")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            
            # Print epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Summary:")
            print(f"{'='*70}")
            print(f"Train Loss: {train_metrics['loss']:.4f} ± {train_metrics['loss_std']:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f} ± {val_metrics['loss_std']:.4f}")
            print(f"Time:       {train_metrics['time']:.2f}s ({train_metrics['samples_per_sec']:.2f} samples/s)")
            print(f"LR:         {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*70}\n")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_loss_std': train_metrics['loss_std'],
                    'val/loss': val_metrics['loss'],
                    'val/loss_std': val_metrics['loss_std'],
                    'time/epoch_time': train_metrics['time'],
                    'time/samples_per_sec': train_metrics['samples_per_sec']
                }, step=self.global_step)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            if epoch % self.config.get('save_interval', 5) == 0 or is_best:
                self.save_checkpoint(
                    f'checkpoint_epoch_{epoch}.pt',
                    is_best=is_best,
                    additional_info={
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss']
                    }
                )
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {self.early_stopping_patience} epochs without improvement")
                break
        
        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total steps: {self.global_step}")
        print(f"{'='*70}\n")
        
        if self.use_wandb:
            wandb.finish()


# Factory function for different model types
def create_trainer(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str = 'cuda'
) -> DiffusionCoTTrainer:
    """
    Factory function to create trainer for different diffusion model types.
    
    Args:
        model_type: 'base', 'plaid', 'sedd', or 'ladir'
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        DiffusionCoTTrainer instance
    """
    if model_type == 'base':
        from models.base_diffusion_adapter import DiffusionTransformer
        model = DiffusionTransformer(
            backbone=config['model']['backbone'],
            hidden_dim=config['model']['hidden_dim'],
            vocab_size=config['model']['vocab_size'],
            max_seq_length=config['model']['max_seq_length'],
            parameterization=config['model'].get('parameterization', 'x0')
        )
    
    elif model_type == 'plaid':
        from models.plaid_adapter import PLAIDDiffusion
        model = PLAIDDiffusion(
            backbone=config['model']['backbone'],
            hidden_dim=config['model']['hidden_dim'],
            vocab_size=config['model']['vocab_size']
        )
    
    elif model_type == 'sedd':
        from models.sedd_adapter import SEDDDiffusion
        model = SEDDDiffusion(
            backbone=config['model']['backbone'],
            hidden_dim=config['model']['hidden_dim'],
            vocab_size=config['model']['vocab_size']
        )
    
    elif model_type == 'ladir':
        from models.ladir_adapter import LADIRDiffusion
        model = LADIRDiffusion(
            backbone=config['model']['backbone'],
            hidden_dim=config['model']['hidden_dim'],
            vocab_size=config['model']['vocab_size'],
            latent_dim=config['model'].get('latent_dim', 256)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return DiffusionCoTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu',
        model_type=model_type
    )


# For testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from data.loaders import get_dataloaders
    
    # Configuration
    config = {
        'model': {
            'backbone': 'gpt2',
            'hidden_dim': 768,
            'vocab_size': 50257,
            'max_seq_length': 512,
            'parameterization': 'x0'
        },
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 50,
        'num_timesteps': 1000,
        'schedule_type': 'cosine',
        'loss_type': 'mse',
        'batch_size': 16,
        'gradient_accumulation_steps': 2,
        'grad_clip': 1.0,
        'warmup_steps': 1000,
        'use_amp': True,
        'use_wandb': False,
        'use_corruption': True,
        'corruption': {
            'mask_prob': 0.3,
            'shuffle_prob': 0.2,
            'drop_prob': 0.15,
            'curriculum': True,
            'warmup_epochs': 10
        },
        'corruption_types': ['mask', 'shuffle', 'drop'],
        'save_interval': 5,
        'log_interval': 10,
        'early_stopping_patience': 10,
        'checkpoint_dir': 'results/checkpoints/test',
        'dataset_name': 'gsm8k',
        'train_path': 'data/gsm8k/train.jsonl',
        'val_path': 'data/gsm8k/test.jsonl',
        'batch_size': 8,
        'corruptions': {
            'masking': 0.3,
            'shuffle': 0.2,
            'drop_steps': 0.1
        }
    }
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = get_dataloaders(config, tokenizer)
    
    # Create trainer
    trainer = create_trainer(
        model_type='base',  # or 'plaid', 'sedd', 'ladir'
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train
    trainer.train()

