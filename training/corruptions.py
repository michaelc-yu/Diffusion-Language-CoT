
"""
Training-time corruption recipes (apply to target tokens; optionally to prior steps in MP)

These produce on-the-fly noisy views before injecting Gaussian/noise per diffusion step. (Think “discrete corruptions” layered with continuous diffusion noise.)

Token masking: replace p% of target tokens with a special <MASK> token (or embed dropout).
Span masking: Poisson-length spans → <MASK> (encourages reconstruction of phrases).

Local shuffling: within a small window (e.g., 3-5 tokens) to test order robustness.

Numeric jitter: randomly perturb digits in numbers with small probability.

Rationale dropout: drop an entire intermediate sentence (forces model to bridge gaps).

Step corruption (MP only): when generating step k, inject small corruption (mask/jitter) into cached steps 1..k-1 so the model learns to recover from imperfect earlier steps (scheduled sampling analogue for diffusion).
"""

"""
training/corruptions.py
Implements structural corruptions for CoT reasoning chains
"""

import torch
import random
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
import copy


class CorruptionScheduler:
    """Manages different types of corruptions applied during training."""
    
    def __init__(
        self,
        mask_prob: float = 0.3,
        shuffle_prob: float = 0.2,
        drop_prob: float = 0.15,
        noise_std: float = 0.0,
        curriculum: bool = False,
        mask_token_id: Optional[int] = None,
        warmup_epochs: int = 10
    ):
        """
        Args:
            mask_prob: Probability of masking a reasoning step
            shuffle_prob: Probability of shuffling step order
            drop_prob: Probability of dropping a step entirely
            noise_std: Standard deviation for token-level noise
            curriculum: If True, gradually increase corruption over time
            mask_token_id: Token ID for <|mask|> token
            warmup_epochs: Number of epochs to reach full corruption intensity
        """
        self.mask_prob = mask_prob
        self.shuffle_prob = shuffle_prob
        self.drop_prob = drop_prob
        self.noise_std = noise_std
        self.curriculum = curriculum
        self.mask_token_id = mask_token_id
        self.warmup_epochs = warmup_epochs
        
        self.current_epoch = 0
        self.current_mask_prob = 0.0 if curriculum else mask_prob
        self.current_shuffle_prob = 0.0 if curriculum else shuffle_prob
        self.current_drop_prob = 0.0 if curriculum else drop_prob
        
    def update_epoch(self, epoch: int):
        """Update corruption intensity for curriculum learning."""
        self.current_epoch = epoch
        if self.curriculum:
            # Gradually increase corruption intensity
            scale = min(1.0, epoch / self.warmup_epochs)
            self.current_mask_prob = self.mask_prob * scale
            self.current_shuffle_prob = self.shuffle_prob * scale
            self.current_drop_prob = self.drop_prob * scale
        else:
            self.current_mask_prob = self.mask_prob
            self.current_shuffle_prob = self.shuffle_prob
            self.current_drop_prob = self.drop_prob
    
    def get_step_segments(
        self,
        input_ids: torch.Tensor,
        step_boundaries: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Extract step segments from boundaries.
        
        Args:
            input_ids: [seq_len] token indices
            step_boundaries: [num_steps] indices of step starts
            
        Returns:
            List of (start, end) tuples for each step
        """
        if len(step_boundaries) == 0:
            return [(0, len(input_ids))]
        
        segments = []
        boundaries = step_boundaries.tolist()
        
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(input_ids)
            segments.append((start, end))
        
        return segments
    
    def mask_steps(
        self,
        input_ids: torch.Tensor,
        step_boundaries: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        Mask entire reasoning steps with special token.
        
        Strategy: Randomly select steps and replace all tokens in those steps
        with the mask token. We preserve the first step (question) and last step
        (final answer) to maintain task structure.
        
        Args:
            input_ids: [seq_len] token indices
            step_boundaries: [num_steps] indices of step starts
            mask_ratio: Override default mask probability
            
        Returns:
            Masked input_ids [seq_len]
        """
        if self.mask_token_id is None or len(step_boundaries) == 0:
            return input_ids
        
        mask_prob = mask_ratio if mask_ratio is not None else self.current_mask_prob
        
        if mask_prob == 0:
            return input_ids
        
        masked_ids = input_ids.clone()
        segments = self.get_step_segments(input_ids, step_boundaries)
        
        # Don't mask the question (first segment) or final answer (last segment)
        # Only mask intermediate reasoning steps
        maskable_segments = segments[1:-1] if len(segments) > 2 else []
        
        for start, end in maskable_segments:
            if random.random() < mask_prob:
                # Replace entire step with mask token
                masked_ids[start:end] = self.mask_token_id
        
        return masked_ids
    
    def shuffle_steps(
        self,
        input_ids: torch.Tensor,
        step_boundaries: torch.Tensor,
        shuffle_prob: Optional[float] = None
    ) -> torch.Tensor:
        """
        Shuffle the order of reasoning steps.
        
        Strategy: Extract intermediate reasoning steps and randomly permute their order.
        Keep question and final answer in their original positions.
        
        Args:
            input_ids: [seq_len] token indices
            step_boundaries: [num_steps] indices of step starts
            shuffle_prob: Override default shuffle probability
            
        Returns:
            Shuffled input_ids [seq_len]
        """
        shuffle_p = shuffle_prob if shuffle_prob is not None else self.current_shuffle_prob
        
        if random.random() > shuffle_p or len(step_boundaries) < 3:
            return input_ids
        
        shuffled_ids = input_ids.clone()
        segments = self.get_step_segments(input_ids, step_boundaries)
        
        if len(segments) <= 2:
            return input_ids
        
        # Extract question, reasoning steps, and answer
        question_segment = segments[0]
        answer_segment = segments[-1]
        reasoning_segments = segments[1:-1]
        
        # Extract reasoning step tokens
        reasoning_steps = []
        for start, end in reasoning_segments:
            step_tokens = input_ids[start:end].clone()
            reasoning_steps.append(step_tokens)
        
        # Shuffle reasoning steps
        random.shuffle(reasoning_steps)
        
        # Reconstruct sequence
        current_pos = question_segment[1]  # After question
        
        for step_tokens in reasoning_steps:
            step_len = len(step_tokens)
            if current_pos + step_len <= answer_segment[0]:
                shuffled_ids[current_pos:current_pos + step_len] = step_tokens
                current_pos += step_len
        
        return shuffled_ids
    
    def drop_steps(
        self,
        input_ids: torch.Tensor,
        step_boundaries: torch.Tensor,
        attention_mask: torch.Tensor,
        drop_prob: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly drop (remove) reasoning steps.
        
        Strategy: Set attention mask to 0 for dropped steps, effectively
        removing them from the model's view. The tokens remain but are ignored.
        
        Args:
            input_ids: [seq_len] token indices
            step_boundaries: [num_steps] indices of step starts
            attention_mask: [seq_len] attention mask
            drop_prob: Override default drop probability
            
        Returns:
            Tuple of (input_ids, attention_mask) with steps dropped
        """
        drop_p = drop_prob if drop_prob is not None else self.current_drop_prob
        
        if drop_p == 0 or len(step_boundaries) < 3:
            return input_ids, attention_mask
        
        dropped_ids = input_ids.clone()
        dropped_mask = attention_mask.clone()
        segments = self.get_step_segments(input_ids, step_boundaries)
        
        # Only drop intermediate reasoning steps
        droppable_segments = segments[1:-1] if len(segments) > 2 else []
        
        for start, end in droppable_segments:
            if random.random() < drop_p:
                # Zero out attention mask for this step
                dropped_mask[start:end] = 0
        
        return dropped_ids, dropped_mask
    
    def inject_noise(
        self,
        input_ids: torch.Tensor,
        vocab_size: int,
        noise_std: Optional[float] = None
    ) -> torch.Tensor:
        """
        Add random token-level noise.
        
        Strategy: Replace random tokens with other tokens from vocabulary.
        This simulates typos, OCR errors, or other input noise.
        
        Args:
            input_ids: [seq_len] token indices
            vocab_size: Size of vocabulary
            noise_std: Override default noise level (treated as probability)
            
        Returns:
            Noisy input_ids [seq_len]
        """
        noise_level = noise_std if noise_std is not None else self.noise_std
        
        if noise_level == 0:
            return input_ids
        
        noisy_ids = input_ids.clone()
        
        # Create noise mask: which tokens to corrupt
        noise_mask = torch.rand_like(input_ids.float()) < noise_level
        
        # Replace masked positions with random tokens
        random_tokens = torch.randint(
            0, vocab_size, 
            input_ids.shape, 
            device=input_ids.device
        )
        noisy_ids[noise_mask] = random_tokens[noise_mask]
        
        return noisy_ids
    
    def apply_mixed_corruption(
        self,
        input_ids: torch.Tensor,
        step_boundaries: torch.Tensor,
        attention_mask: torch.Tensor,
        vocab_size: int,
        corruption_types: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multiple corruption types sequentially.
        
        Args:
            input_ids: [seq_len] token indices
            step_boundaries: [num_steps] indices of step starts
            attention_mask: [seq_len] attention mask
            vocab_size: Size of vocabulary
            corruption_types: List of corruption types to apply
            
        Returns:
            Tuple of (corrupted_input_ids, corrupted_attention_mask)
        """
        corrupted_ids = input_ids.clone()
        corrupted_mask = attention_mask.clone()
        
        # Apply corruptions in order
        for corruption_type in corruption_types:
            if corruption_type == 'mask':
                corrupted_ids = self.mask_steps(corrupted_ids, step_boundaries)
            
            elif corruption_type == 'shuffle':
                corrupted_ids = self.shuffle_steps(corrupted_ids, step_boundaries)
            
            elif corruption_type == 'drop':
                corrupted_ids, corrupted_mask = self.drop_steps(
                    corrupted_ids, step_boundaries, corrupted_mask
                )
            
            elif corruption_type == 'noise':
                corrupted_ids = self.inject_noise(corrupted_ids, vocab_size)
        
        return corrupted_ids, corrupted_mask
    
    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
        corruption_types: List[str] = ['mask', 'shuffle', 'drop'],
        vocab_size: int = 50257
    ) -> Dict[str, torch.Tensor]:
        """
        Apply selected corruptions to a batch.
        
        This method processes an entire batch, applying corruptions to each
        sample independently.
        
        Args:
            batch: Dictionary with keys:
                - 'input: [batch_size, seq_len]
                - 'attention_mask': [batch_size, seq_len]
                - 'step_boundaries': List of tensors, one per sample
            corruption_types: Which corruptions to apply
            vocab_size: Size of vocabulary (for noise corruption)
            
        Returns:
            Corrupted batch with same structure as input
        """
        corrupted_batch = {}
        
        # Clone all tensors
        for k, v in batch.items():
            if torch.is_tensor(v):
                corrupted_batch[k] = v.clone()
            else:
                corrupted_batch[k] = copy.deepcopy(v)
        
        batch_size = batch['input_ids'].shape[0]
        
        # Process each sample in the batch
        for i in range(batch_size):
            input_ids = corrupted_batch['input_ids'][i]
            attention_mask = corrupted_batch['attention_mask'][i]
            
            # Get step boundaries for this sample
            if 'step_boundaries' in batch:
                if isinstance(batch['step_boundaries'], list):
                    step_boundaries = batch['step_boundaries'][i]
                else:
                    # If it's a tensor, need to extract boundaries for this sample
                    step_boundaries = torch.tensor([])
            else:
                step_boundaries = torch.tensor([])
            
            # Apply corruptions
            corrupted_ids, corrupted_mask = self.apply_mixed_corruption(
                input_ids,
                step_boundaries,
                attention_mask,
                vocab_size,
                corruption_types
            )
            
            corrupted_batch['input_ids'][i] = corrupted_ids
            corrupted_batch['attention_mask'][i] = corrupted_mask
        
        return corrupted_batch
    
    def get_corruption_stats(self) -> Dict[str, float]:
        """Get current corruption probabilities."""
        return {
            'mask_prob': self.current_mask_prob,
            'shuffle_prob': self.current_shuffle_prob,
            'drop_prob': self.current_drop_prob,
            'noise_std': self.noise_std,
            'epoch': self.current_epoch
        }


# Predefined corruption presets
CORRUPTION_PRESETS = {
    'none': {
        'mask_prob': 0.0,
        'shuffle_prob': 0.0,
        'drop_prob': 0.0,
        'noise_std': 0.0
    },
    'mild': {
        'mask_prob': 0.15,
        'shuffle_prob': 0.1,
        'drop_prob': 0.05,
        'noise_std': 0.01
    },
    'moderate': {
        'mask_prob': 0.3,
        'shuffle_prob': 0.2,
        'drop_prob': 0.15,
        'noise_std': 0.02
    },
    'aggressive': {
        'mask_prob': 0.5,
        'shuffle_prob': 0.3,
        'drop_prob': 0.25,
        'noise_std': 0.05
    },
    'mask_only': {
        'mask_prob': 0.4,
        'shuffle_prob': 0.0,
        'drop_prob': 0.0,
        'noise_std': 0.0
    },
    'shuffle_only': {
        'mask_prob': 0.0,
        'shuffle_prob': 0.4,
        'drop_prob': 0.0,
        'noise_std': 0.0
    },
    'drop_only': {
        'mask_prob': 0.0,
        'shuffle_prob': 0.0,
        'drop_prob': 0.3,
        'noise_std': 0.0
    }
}


def get_corruption_scheduler(preset: str = 'moderate', **kwargs) -> CorruptionScheduler:
    """
    Factory function to create corruption scheduler from preset.
    
    Args:
        preset: Name of preset ('none', 'mild', 'moderate', 'aggressive', etc.)
        **kwargs: Additional parameters to override preset values
        
    Returns:
        CorruptionScheduler instance
    """
    params = CORRUPTION_PRESETS.get(preset, CORRUPTION_PRESETS['moderate']).copy()
    params.update(kwargs)
    return CorruptionScheduler(**params)


# Example usage and testing
if __name__ == "__main__":
    print("Testing CorruptionScheduler...")
    
    # Create sample batch
    batch_size = 4
    seq_len = 128
    vocab_size = 50257
    
    batch = {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'step_boundaries': [
            torch.tensor([0, 20, 40, 60, 80, 100]),  # Sample 0
            torch.tensor([0, 15, 35, 55, 75, 95]),   # Sample 1
            torch.tensor([0, 25, 50, 75, 100]),      # Sample 2
            torch.tensor([0, 10, 30, 50, 70, 90, 110])  # Sample 3
        ]
    }
    
    print(f"Original batch shape: {batch['input_ids'].shape}")
    print(f"Original first sample (first 20 tokens): {batch['input_ids'][0][:20]}")
    
    # Test different presets
    for preset_name in ['mild', 'moderate', 'aggressive', 'mask_only']:
        print(f"\n{'='*60}")
        print(f"Testing preset: {preset_name}")
        print(f"{'='*60}")
        
        corruptor = get_corruption_scheduler(preset_name, mask_token_id=50256)
        
        # Apply corruptions
        corrupted = corruptor(
            batch,
            corruption_types=['mask', 'shuffle', 'drop'],
            vocab_size=vocab_size
        )
        
        print(f"Corrupted first sample (first 20 tokens): {corrupted['input_ids'][0][:20]}")
        
        # Check how many tokens were masked
        num_masked = (corrupted['input_ids'][0] == 50256).sum().item()
        print(f"Tokens masked: {num_masked}")
        
        # Check attention mask changes
        num_dropped = (batch['attention_mask'][0] - corrupted['attention_mask'][0]).sum().item()
        print(f"Tokens dropped (attention=0): {num_dropped}")
        
        # Show stats
        stats = corruptor.get_corruption_stats()
        print(f"Corruption stats: {stats}")
    
    # Test curriculum learning
    print(f"\n{'='*60}")
    print("Testing curriculum learning")
    print(f"{'='*60}")
    
    corruptor = get_corruption_scheduler(
        'moderate',
        curriculum=True,
        warmup_epochs=5,
        mask_token_id=50256
    )
    
    for epoch in range(7):
        corruptor.update_epoch(epoch)
        stats = corruptor.get_corruption_stats()
        print(f"Epoch {epoch}: mask_prob={stats['mask_prob']:.3f}, "
              f"shuffle_prob={stats['shuffle_prob']:.3f}, "
              f"drop_prob={stats['drop_prob']:.3f}")
    
    print("\nAll tests completed successfully!")


