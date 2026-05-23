import argparse
from pathlib import Path

from src.miniomni3.generate.base import run_inference


CHECKPOINT_DIR = ""
AUDIO_PATH = ""


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
    )
