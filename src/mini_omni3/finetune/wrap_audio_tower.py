"""Bake the trained audio_adapter into Qwen2.5-Omni audio_tower's proj layer.

Convention:
  - The first-stage audio_tower checkpoint contains the audio encoder ONLY
    (conv1, conv2, layers, ln_post, audio_bos_eos_token). It must NOT carry
    a `proj.*` (the projection from audio-encoder dim to LM hidden dim).
  - The `proj` layer is finetuned as part of the main training run and lives
    in the total training checkpoint under the name `audio_adapter.*`.

This script combines the two so the result is a stock Qwen2.5-Omni audio_tower
state-dict — `proj.weight/bias` already populated with the trained values. After
this, downstream code can use the audio_tower without commenting out the
`self.proj(each_audio_states)` line in `modeling_qwen2_5_omni.py`.

Usage:
    python mini_omni3/finetune/wrap_audio_tower.py \
        <trained_ckpt.pt> <stage1_audio_tower.pth> \
        --out_audio_tower wrapped_audio_tower.pth \
        --out_gpt        gpt_only.pt              # optional
"""

import argparse
from collections import OrderedDict

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trained_ckpt",
                   help="path to the trained state-dict (contains audio_adapter.* + GPT body)")
    p.add_argument("stage1_audio_tower",
                   help="path to the stage-1 audio_tower checkpoint (.pth)")
    p.add_argument("--out_audio_tower", required=True,
                   help="output path for the wrapped audio_tower ckpt (proj=adapter weights)")
    p.add_argument("--out_gpt", default=None,
                   help="optional output for an adapter-free GPT state-dict (audio_adapter.* removed)")
    args = p.parse_args()

    trained = torch.load(args.trained_ckpt, map_location="cpu", weights_only=False)
    stage1 = torch.load(args.stage1_audio_tower, map_location="cpu", weights_only=False)

    if "audio_adapter.weight" not in trained or "audio_adapter.bias" not in trained:
        raise SystemExit("Trained ckpt has no audio_adapter.* — nothing to wrap.")

    adapter_w = trained["audio_adapter.weight"]
    adapter_b = trained["audio_adapter.bias"]
    print(f"Found audio_adapter: weight={tuple(adapter_w.shape)}, bias={tuple(adapter_b.shape)}")

    # Per project rule, stage-1 audio_tower must not carry proj.*; strip if present.
    stale = sorted(k for k in stage1 if k.startswith("proj."))
    if stale:
        for k in stale:
            del stage1[k]
        print(f"Stripped {stale} from stage-1 (project rule: proj lives in the trained ckpt).")

    # Bake the trained adapter into the audio_tower's proj slot.
    stage1["proj.weight"] = adapter_w
    stage1["proj.bias"] = adapter_b
    torch.save(stage1, args.out_audio_tower)
    print(f"Wrote wrapped audio_tower to {args.out_audio_tower} ({len(stage1)} tensors).")

    if args.out_gpt:
        gpt_only = OrderedDict((k, v) for k, v in trained.items() if not k.startswith("audio_adapter"))
        torch.save(gpt_only, args.out_gpt)
        print(f"Wrote adapter-free GPT ckpt to {args.out_gpt} ({len(gpt_only)} tensors).")


if __name__ == "__main__":
    main()
