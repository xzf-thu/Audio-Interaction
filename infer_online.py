from typing import List, Optional

import lightning as L
import torch

from src.miniomni3.dataset.TOKENS import ENGLISH, ONLINE, SYSTEM, TEXT_BEGIN, TEXT_END
from src.miniomni3.generate.base import streaming_generate
from utils import (
    load_audio_encoder, load_model, resolve_checkpoint_paths, set_seed,
)
from src.miniomni3.tokenizer import Tokenizer
from src.miniomni3.utils import get_default_supported_precision


SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)


def run_inference(
    *,
    checkpoint_dir: str,
    rounds: int = 10,
    audio_paths: Optional[List[str]] = None,
    seed: int = 1337,
    max_new_tokens: int = 4096,
    device: str = "cuda:0",
):
    """End-to-end: build fabric, load model + audio encoder, run streaming_generate.

    If `audio_paths` is given, runs one round per path non-interactively
    (offline mode). Otherwise prompts stdin each round (online mode).
    """
    if not checkpoint_dir:
        raise RuntimeError("`checkpoint_dir` is empty — set it before calling run_inference().")
    model_config_dir, trained_checkpoint, qwen_omni_ckpt, audio_tower_ckpt = \
        resolve_checkpoint_paths(checkpoint_dir)

    set_seed(seed)
    fabric = L.Fabric(
        devices=1, num_nodes=1, strategy="auto",
        precision=get_default_supported_precision(training=False),
        loggers="tensorboard",
    )
    model = load_model(fabric, model_config_dir, trained_checkpoint).to(device)
    audio_encoder = load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
    tokenizer = Tokenizer(model_config_dir)

    system_ids = tokenizer.encode(SYSTEM_PROMPT).cpu().tolist()
    prefix_ids = torch.LongTensor(
        [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
    ).to(model.device)

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    model.eval()
    try:
        with torch.inference_mode():
            return streaming_generate(
                model, audio_encoder, tokenizer, prefix_ids,
                rounds=rounds, audio_paths=audio_paths,
                max_returned_tokens=max_new_tokens,
                temperature=0.0, top_p=0.0,  # greedy/argmax → deterministic output
            )
    finally:
        model.clear_kv_cache()