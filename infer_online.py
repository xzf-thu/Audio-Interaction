import argparse
from pathlib import Path

from src.miniomni3.generate.base import run_inference


CHECKPOINT_DIR = ""


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
    )
