from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    """Training-related arguments."""

    save_interval: Optional[int] = 1000
    """Number of optimizer steps between checkpoint saves."""
    log_interval: int = 1
    """Number of iterations between log lines."""
    global_batch_size: int = 64
    """Total samples per optimizer step across all data-parallel ranks."""
    micro_batch_size: int = 4
    """Samples per data-parallel rank per forward/backward pass."""
    lr_warmup_steps: Optional[int] = 100
    """Number of warmup iterations (linear ramp from 0 to max_lr)."""
    epochs: Optional[int] = None
    """Number of epochs to train."""
    max_steps: Optional[int] = None
    """Hard cap on optimizer steps (overrides epochs if reached first)."""
    max_seq_length: Optional[int] = None
    """Truncate samples longer than this."""

    def gradient_accumulation_iters(self, devices: int, num_nodes: int = 1) -> int:
        n = self.batch_size(devices, num_nodes) // self.micro_batch_size
        assert n > 0
        return n

    def batch_size(self, devices: int, num_nodes: int = 1) -> int:
        n = self.global_batch_size // (devices * num_nodes)
        assert n > 0
        return n


@dataclass
class EvalArgs:
    """Evaluation-related arguments."""

    interval: int = 600
    """Number of optimizer steps between validation runs."""
    max_new_tokens: Optional[int] = None
    """Generation length cap (only used by validation-time generation paths)."""
    max_iters: int = 100
    """Max number of validation batches per run."""
    initial_validation: bool = False
    """Run one validation pass before the first training step."""
    final_validation: bool = True
    """Run one validation pass after training completes."""
