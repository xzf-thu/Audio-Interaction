"""Encode raw audio samples with Qwen2.5-Omni's audio tower; save to AudioFeat.pt.

The encoder is loaded once on first call (lazy, cached). Output features are
projected to the LM hidden dim by audio_tower.proj so the rest of the pipeline
can use them directly.
"""

import os

import numpy as np
import torch
import whisper
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration


# ============================================================
# Fill in these paths before running.
# ============================================================
QWEN_OMNI_CKPT   = ""  # path to the Qwen2.5-Omni model directory
AUDIO_TOWER_CKPT = ""  # path to the finetuned audio_tower .pth file
# ============================================================

_encoder = None  # populated on first extract_audio_features() call


def _load_encoder(device):
    if not QWEN_OMNI_CKPT or not AUDIO_TOWER_CKPT:
        raise RuntimeError(
            "Set QWEN_OMNI_CKPT and AUDIO_TOWER_CKPT at the top of extract_audio_features.py."
        )
    cfg = AutoConfig.from_pretrained(QWEN_OMNI_CKPT)
    enc = Qwen2_5OmniForConditionalGeneration._from_config(cfg).thinker.audio_tower
    enc.load_state_dict(torch.load(AUDIO_TOWER_CKPT, map_location=device))
    return enc.to(device).requires_grad_(False).eval()


def _split_into_chunks(n, chunk_size):
    chunks = [chunk_size] * (n // chunk_size)
    if n % chunk_size:
        chunks.append(n % chunk_size)
    return chunks


def extract_audio_features(audio_samples, save_dir, device="cuda"):
    """Encode raw audio samples; save the feature tensor to `<save_dir>/AudioFeat.pt`."""
    global _encoder
    if _encoder is None:
        _encoder = _load_encoder(device)

    mel = whisper.log_mel_spectrogram(np.array(audio_samples, dtype=np.float32), n_mels=128)
    len_feature = mel.shape[1]

    with torch.no_grad():
        feat = _encoder(
            torch.tensor(mel).to(device),
            torch.tensor(_split_into_chunks(len_feature, 40)).to(device),
            torch.tensor((len_feature - 1) // 2 + 1).to(device),
        ).last_hidden_state

    os.makedirs(save_dir, exist_ok=True)
    torch.save(feat.detach().cpu(), os.path.join(save_dir, "AudioFeat.pt"))
