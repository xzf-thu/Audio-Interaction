import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

from src.miniomni3.generate.base import AUDIO_TOKENS_PER_CHUNK  # noqa: F401  (re-export for callers)
from src.miniomni3.model import GPT, Config
from src.miniomni3.utils import load_checkpoint


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import json
from pathlib import Path

from safetensors.torch import load_file


def load_model(fabric, model_config_dir, checkpoint_dir):
    """Load a GPT from a local sharded safetensors directory.

    `checkpoint_dir` must contain:
        model.safetensors.index.json
        model-00001-of-0000N.safetensors
        ...
    """
    config = Config.from_file(Path(model_config_dir) / "model_config.yaml")
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    model = fabric.setup(model)

    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"No model.safetensors.index.json under {checkpoint_dir}. "
            f"Expected a sharded safetensors directory."
        )

    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    state_dict = {}
    for shard in shard_files:
        state_dict.update(load_file(str(checkpoint_dir / shard), device="cpu"))

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"[load_model] missing={missing[:3]}… unexpected={unexpected[:3]}…")
    return model

def load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device):
    print(qwen_omni_ckpt)
    cfg = AutoConfig.from_pretrained(qwen_omni_ckpt)
    # Omni 的 config 是嵌套的：thinker_config.audio_config 才是 audio encoder 的配置
    audio_cfg = cfg.thinker_config.audio_config
    encoder = Qwen2_5OmniAudioEncoder._from_config(audio_cfg)
    state_dict = torch.load(audio_tower_ckpt, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device).requires_grad_(False).eval()
    return encoder


def resolve_checkpoint_paths(checkpoint_dir: str):
    """Map a single checkpoint root → (model_config_dir, trained_checkpoint,
    qwen_omni_ckpt, audio_tower_ckpt). The release layout is:

        <checkpoint_dir>/
            model_config.yaml + tokenizer.json + ...   ← model_config_dir = root
            MiniOmni3_LM.pt
            MiniOmni3_ChunkwisedEncoder.pth
            qwen_2_5_omni_config/
    """

    ckpt = Path(checkpoint_dir)
    return (
        str(ckpt),
        str(ckpt),
        str(ckpt / "qwen25OmniConfig"),
        str(ckpt / "MiniOmni3_ChunkwisedEncoder.pth"),
    )

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

