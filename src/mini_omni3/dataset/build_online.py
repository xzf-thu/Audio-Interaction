"""Build online (streaming) SFT samples.

Every `chunk_size` audio frames the model gets an [AUDIO_BEGIN, PAD*C, ASSISTANT, ...]
block; most blocks just emit KEEP_SILENCE, the last block of each turn emits the reply.

Input:  {"conversation": [{"audio_path", "assistant", "emotion"}, ...]} or single-turn
        {"merge_path", "assistant", "emotion"}.
Output: {"tasks": "online", "idx", "input_ids", "labels", "audio_pos", "pt_path_dir"}.
Run again to resume — already-written idx are skipped.
"""

import argparse
import json
import os
import random

from tqdm import tqdm

from mini_omni3.dataset.audio_io import SAMPLES_PER_FRAME, _load_mel
from mini_omni3.dataset.extract_audio_features import extract_audio_features
from mini_omni3.dataset.tokens import (
    ASSISTANT, AUDIO_BEGIN, EMOTION_TO_ID, ENGLISH, KEEP_SILENCE, MASK,
    NORMAL, ONLINE, PAD, SYSTEM, TEXT_BEGIN, TEXT_END,
)
from mini_omni3.tokenizer import Tokenizer


# ============================================================
# Fill in this path before running.
# ============================================================
QWEN_OMNI_CKPT = ""  # path to the Qwen2.5-Omni model directory (used for the tokenizer)
# ============================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)

# Markers that mean "the model should keep silent for this turn"
NO_RESPONSE_MARKERS = {"<no need to response>", "no need to response"}


def _load_audio_aligned(audio_path):
    """Load audio @ 16 kHz, pad/crop to (output_len * SAMPLES_PER_FRAME) samples.
    Returns (audio_samples, output_len) where output_len is in encoder-frame units.
    """
    audio, _, _, _, output_len = _load_mel(audio_path)
    target_samples = output_len * SAMPLES_PER_FRAME
    if len(audio) < target_samples:
        audio = audio + [0] * (target_samples - len(audio))
    elif len(audio) > target_samples:
        max_start = (len(audio) - target_samples) // SAMPLES_PER_FRAME
        start = random.randint(0, max_start) * SAMPLES_PER_FRAME
        audio = audio[start: start + target_samples]
    return audio, output_len


def _silence_chunk(chunk_size):
    """Mid-turn or tail chunk: model should keep waiting."""
    ids = [AUDIO_BEGIN] + [PAD] * chunk_size + [ASSISTANT, KEEP_SILENCE]
    labels = [MASK] + [MASK] * chunk_size + [MASK, KEEP_SILENCE]
    return ids, labels


def _response_chunk(assistant_text, emotion, tokenizer, chunk_size):
    """Last chunk of a turn: actual reply, or silence if marked as no-response."""
    if isinstance(assistant_text, str) and assistant_text.strip().lower() in NO_RESPONSE_MARKERS:
        return _silence_chunk(chunk_size)
    emotion_tok = EMOTION_TO_ID.get(emotion.lower(), NORMAL) if isinstance(emotion, str) else NORMAL
    assistant_ids = tokenizer.encode(assistant_text).cpu().tolist()
    ids = [AUDIO_BEGIN] + [PAD] * chunk_size + [ASSISTANT, TEXT_BEGIN, emotion_tok] + assistant_ids + [TEXT_END]
    labels = [MASK] + [MASK] * chunk_size + [MASK, TEXT_BEGIN, emotion_tok] + assistant_ids + [TEXT_END]
    return ids, labels


# === Sample builder ===

def build_online_sample(
    data_item,
    idx,
    *,
    tokenizer,
    system_ids,
    feature_dir,
    chunk_size=10,
    min_noise_len=20,
    max_noise_len=60,
):
    """Build one online streaming SFT sample.

    Layout: prefix + repeated (silence | audio + per-chunk emissions) + tail silence.
    Every `chunk_size` encoder-output frames triggers an
    `[AUDIO_BEGIN, PAD*chunk_size, ASSISTANT, ...]` block in input_ids;
    most blocks emit KEEP_SILENCE, the last block of each turn emits the
    actual assistant reply (or KEEP_SILENCE if marked no-response).

    Returns: dict with keys input_ids, labels, audio_pos, pt_path_dir.
    """
    audio_list, emotion_list, assistant_list = _extract_turns(data_item)

    input_ids = [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
    labels = [MASK] * len(input_ids)
    audio_pos = []

    audio_samples = []
    total_frames = 0       # cumulative count in encoder-output-frame units

    for i, (audio_path, emotion, assistant_text) in enumerate(
        zip(audio_list, emotion_list, assistant_list)
    ):
        noise_frames = random.randint(min_noise_len, max_noise_len)
        audio_samples += [0] * (noise_frames * SAMPLES_PER_FRAME)

        audio_seg, audio_frames = _load_audio_aligned(audio_path)
        audio_samples += audio_seg

        # New chunks added this turn: floor((total + noise + audio) / C) - floor(total / C).
        new_chunks = (
            (total_frames + noise_frames + audio_frames) // chunk_size
            - total_frames // chunk_size
        )
        total_frames += noise_frames + audio_frames

        # First turn gets one extra trailing chunk.
        if i == 0:
            new_chunks += 1

        for j in range(new_chunks):
            pos_start = len(input_ids) + 1
            audio_pos.append((pos_start, pos_start + chunk_size))

            if j == new_chunks - 1:
                chunk_ids, chunk_labels = _response_chunk(assistant_text, emotion, tokenizer, chunk_size)
            else:
                chunk_ids, chunk_labels = _silence_chunk(chunk_size)
            input_ids += chunk_ids
            labels += chunk_labels

    # Tail silence: round up to a chunk boundary, then emit (cycle - 1) silence chunks.
    tail_noise = random.randint(min_noise_len, max_noise_len)
    tail_noise -= (total_frames + tail_noise) % chunk_size
    audio_samples += [0] * (tail_noise * SAMPLES_PER_FRAME)

    tail_chunks = (total_frames + tail_noise) // chunk_size - total_frames // chunk_size - 1
    for _ in range(tail_chunks):
        pos_start = len(input_ids) + 1
        audio_pos.append((pos_start, pos_start + chunk_size))
        chunk_ids, chunk_labels = _silence_chunk(chunk_size)
        input_ids += chunk_ids
        labels += chunk_labels

    pt_path_dir = os.path.join(feature_dir, str(idx))
    extract_audio_features(audio_samples, pt_path_dir)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "audio_pos": audio_pos,
        "pt_path_dir": pt_path_dir,
    }


# === Input parsing ===

def _extract_turns(data_item):
    """Return (audio_paths, emotions, assistants) parallel lists.

    Accepts multi-turn `conversation` list or single-turn `merge_path` form.
    """
    convs = data_item.get("conversation", [])
    if convs:
        audio_paths = [c["audio_path"] for c in convs]
        emotions = [c.get("emotion") or "normal" for c in convs]
        assistants = [c["assistant"] for c in convs]
        return audio_paths, emotions, assistants
    if "merge_path" in data_item and "assistant" in data_item:
        return ([data_item["merge_path"]],
                [data_item.get("emotion", "normal")],
                [data_item["assistant"]])
    raise ValueError("missing 'conversation' or single-turn fields")


# === Main driver ===

def parse_args():
    parser = argparse.ArgumentParser(description="Build online (streaming) SFT samples.")
    parser.add_argument("data_file", type=str, help="input jsonl file")
    parser.add_argument("output_path", type=str, help="output jsonl file")
    parser.add_argument("error_path", type=str, help="error log file")
    parser.add_argument("feature_dir", type=str, help="dir to save audio feature .pt files")
    parser.add_argument("--checkpoint_dir", type=str, default=QWEN_OMNI_CKPT,
                        help="tokenizer checkpoint dir (defaults to QWEN_OMNI_CKPT at the top of this file)")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="encoder-output frames per audio chunk")
    parser.add_argument("--min_noise_len", type=int, default=20,
                        help="min silence frames between turns")
    parser.add_argument("--max_noise_len", type=int, default=60,
                        help="max silence frames between turns")
    return parser.parse_args()


def _load_processed_indices(output_path):
    """For resume: collect idx of all valid records already in output_path."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as fr:
        for line in fr:
            try:
                done.add(json.loads(line)["idx"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    args = parse_args()
    os.makedirs(args.feature_dir, exist_ok=True)

    with open(args.data_file, "r", encoding="utf-8") as f:
        data_lines = f.readlines()

    processed = _load_processed_indices(args.output_path)
    todo = [i for i in range(len(data_lines)) if i not in processed]
    print(f"Total {len(data_lines)} | already done {len(processed)} | to process {len(todo)}")

    tokenizer = Tokenizer(args.checkpoint_dir)
    system_ids = tokenizer.encode(DEFAULT_SYSTEM_PROMPT).cpu().tolist()

    with open(args.output_path, "a", encoding="utf-8", buffering=1) as fout, \
         open(args.error_path, "a", encoding="utf-8", buffering=1) as ferr:
        for idx in tqdm(todo):
            try:
                data_item = json.loads(data_lines[idx])
                sample = build_online_sample(
                    data_item, idx,
                    tokenizer=tokenizer, system_ids=system_ids,
                    feature_dir=args.feature_dir,
                    chunk_size=args.chunk_size,
                    min_noise_len=args.min_noise_len,
                    max_noise_len=args.max_noise_len,
                )
                fout.write(json.dumps(
                    {"tasks": "online", "idx": idx, **sample},
                    ensure_ascii=False,
                ) + "\n")
            except Exception as e:
                ferr.write(f"[idx {idx}] {type(e).__name__}: {e}\n")
                ferr.write(data_lines[idx])


if __name__ == "__main__":
    main()
