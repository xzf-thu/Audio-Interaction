import argparse
from pathlib import Path
import torch
from src.miniomni3.generate.base import run_inference


CHECKPOINT_DIR = "./checkpoints"

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Streaming inference entry point.")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    args = p.parse_args()

    run_inference(
        checkpoint_dir=str(CHECKPOINT_DIR),
        rounds=args.rounds,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        device=get_best_device()
    )
