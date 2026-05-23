import argparse
from pathlib import Path

from src.miniomni3.generate.base import run_inference
import torch

CHECKPOINT_DIR = "/Users/yansc-xzf/Desktop/工作/Mini-Omni3/github/omni3/Mini-Omni3/checkpoints"
AUDIO_PATH = "/Users/yansc-xzf/Desktop/工作/Mini-Omni3/github/omni3/Mini-Omni3/assets/what_can_you_do.m4a"



def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")





if __name__ == "__main__":
    p = argparse.ArgumentParser(description="One-shot offline inference.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    args = p.parse_args()

    run_inference(
        checkpoint_dir=str(CHECKPOINT_DIR),
        audio_paths=[AUDIO_PATH],
        rounds=1,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        device=get_best_device()
    )
