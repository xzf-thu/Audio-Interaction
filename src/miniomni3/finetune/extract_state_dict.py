"""Extract a model state-dict from a Lightning training checkpoint.

Training writes `<out_dir>/step-NNNNNN/lit_model.pth`, which bundles the model,
optimizer, scheduler, and counters. Inference only needs the model weights —
this script drops the rest so the file is smaller and faster to load.

Usage:
    python mini_omni3/finetune/extract_state_dict.py \\
        <input lit_model.pth> <output state_dict.pt>
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pth_path", help="input Lightning checkpoint (lit_model.pth)")
    parser.add_argument("save_path", help="output state-dict file (.pt)")
    args = parser.parse_args()

    # Lightning checkpoints contain non-tensor scalars (iter_num, optimizer
    # config, ...) so we must allow the full pickle path here.
    ckpt = torch.load(args.pth_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Drop the audio encoder (it lives in a separate checkpoint).
    state_dict = {k: v for k, v in state_dict.items() if "audio_encoder" not in k}

    torch.save(state_dict, args.save_path)
    print(f"Saved {len(state_dict)} tensors to {args.save_path}")


if __name__ == "__main__":
    main()
