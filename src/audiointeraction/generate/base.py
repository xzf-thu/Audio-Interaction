"""Core streaming generation primitives for the audio-enhanced GPT.

The model alternates between two states inside `streaming_generate`:
  - LISTENING: each step consumes one encoder-output chunk of audio. The model
    emits either KEEP_SILENCE (keep listening) or TEXT_BEGIN (start replying).
  - SPEAKING:  autoregressive text generation until TEXT_END, then back to
    LISTENING for the next audio chunk.

Public surface:
    AUDIO_TOKENS_PER_CHUNK
    sample, encode_audio_chunks, streaming_generate
"""
SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)



import os
import sys
from typing import List, Optional

import numpy as np
import torch
import whisper

from src.audiointeraction.dataset.TOKENS import (
    ASSISTANT, AUDIO_BEGIN, KEEP_SILENCE, PAD, TEXT_BEGIN, TEXT_END,
    HAPPY, SAD, ANGRY, SURPRISE, NORMAL, URGENT,
)
from src.audiointeraction.model import GPT
from src.audiointeraction.tokenizer import Tokenizer


# Encoder-output frames per [AUDIO_BEGIN, PAD*N, ASSISTANT, ...] block.
AUDIO_TOKENS_PER_CHUNK = 10


# 情绪 token -> 颜文字（终端/GitHub 都能正常显示）
EMOTION_KAOMOJI = {
    HAPPY:    "(◕‿◕)",
    SAD:      "(╥﹏╥)",
    ANGRY:    "(╬ಠ益ಠ)",
    SURPRISE: "(⊙o⊙)",
    NORMAL:   "(・_・)",
    URGENT:   "(°□°;)",
}


# === Sampling ===

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=False)
    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumprobs <= (1 - top_p)
    remove[-1:] = 0  # always keep the most probable token
    return logits.masked_fill(remove.scatter(0, sorted_idx, remove), float("-inf"))


def sample(logits: torch.Tensor, *, temperature=1.0, top_k=None, top_p=1.0) -> torch.Tensor:
    """Sample one token id from the last position of `logits` ([1, T, V])."""
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    if temperature <= 0.0 and top_p <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature > 0.0:
        logits = logits / temperature
    if top_p < 1.0:
        logits = _top_p_filter(logits, top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# === Audio feature extraction ===

def _split_into_chunks(n: int, chunk_size: int) -> List[int]:
    chunks = [chunk_size] * (n // chunk_size)
    if n % chunk_size:
        chunks.append(n % chunk_size)
    return chunks


def _encode_audio_samples(audio: List[float], audio_encoder: torch.nn.Module, device) -> List[torch.Tensor]:
    """Run the audio_tower on raw 16 kHz samples and split the output into AUDIO_TOKENS_PER_CHUNK chunks."""
    audio = list(audio)
    # Pad to a 0.4-s boundary (6400 samples @ 16 kHz).
    if len(audio) % 6400 != 0:
        audio += [0] * (6400 - len(audio) % 6400)
    mel = whisper.log_mel_spectrogram(np.array(audio, dtype=np.float32), n_mels=128)
    len_feature = mel.shape[1]

    with torch.no_grad():
        feat = audio_encoder(
            torch.tensor(mel).to(device),
            torch.tensor(_split_into_chunks(len_feature, 40)).to(device),
            torch.tensor((len_feature - 1) // 2 + 1).to(device),
        ).last_hidden_state

    # Drop any trailing partial chunk so each chunk is exactly AUDIO_TOKENS_PER_CHUNK frames.
    keep = feat.shape[0] - feat.shape[0] % AUDIO_TOKENS_PER_CHUNK
    return [feat[i: i + AUDIO_TOKENS_PER_CHUNK] for i in range(0, keep, AUDIO_TOKENS_PER_CHUNK)]


def encode_audio_chunks(audio_path: str, audio_encoder: torch.nn.Module, device) -> List[torch.Tensor]:
    """Run the audio_tower on `audio_path` and split the output into AUDIO_TOKENS_PER_CHUNK chunks."""
    audio = whisper.load_audio(audio_path, sr=16000).tolist()
    return _encode_audio_samples(audio, audio_encoder, device)


def encode_silence_chunks(seconds: float, audio_encoder: torch.nn.Module, device) -> List[torch.Tensor]:
    return _encode_audio_samples([0.0] * int(seconds * 16000), audio_encoder, device)


# === Streaming generation ===

def _forward(model, tokens, input_pos, *, input_pos_maxp1, audio_feat):
    return model(
        tokens, None, 1, audio_feat, input_pos,
        input_pos_maxp1=input_pos_maxp1,
        audio_tokens_per_chunk=AUDIO_TOKENS_PER_CHUNK,
    )


def _init_input_pos_maxp1(model, prompt_size, device):
    # input_pos_maxp1 introduces data-dependent shapes; skip if a Thunder module is involved.
    if any(m.__class__.__name__ == "ThunderModule" for m in model.modules()):
        return None
    return torch.tensor(prompt_size, device=device)


def _append_listening_block(token, input_pos, input_pos_maxp1, device):
    """Append `[AUDIO_BEGIN, PAD*N, ASSISTANT]` to the running context."""
    new_tokens = torch.LongTensor([AUDIO_BEGIN] + [PAD] * AUDIO_TOKENS_PER_CHUNK + [ASSISTANT]).to(device)
    new_positions = input_pos[-1] + torch.arange(1, len(new_tokens) + 1, device=device)
    token = torch.cat([token, new_tokens])
    input_pos = torch.cat([input_pos, new_positions])
    if input_pos_maxp1 is not None:
        input_pos_maxp1.add_(len(new_tokens))
    return token, input_pos, input_pos_maxp1


def _advance_one(input_pos, input_pos_maxp1):
    """Move input_pos forward by one (for the next single-token call)."""
    new_pos = input_pos[-1].unsqueeze(0).add_(1)
    if input_pos_maxp1 is not None:
        input_pos_maxp1.add_(1)
    return new_pos, input_pos_maxp1


# === Pretty printing helpers ===

class _Pretty:
    """Tiny renderer for the streaming UI. Auto-disables color on non-TTY."""

    # ANSI codes; emptied below when not a TTY.
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    GREY = "\033[90m"
    CLEAR_LINE = "\033[2K\r"

    BAR_WIDTH = 24

    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self.use_color = stream.isatty()
        self.use_cr = stream.isatty()  # only do in-place updates on a real terminal
        if not self.use_color:
            for name in ("DIM", "BOLD", "RESET", "CYAN", "GREEN",
                         "YELLOW", "MAGENTA", "GREY"):
                setattr(self, name, "")
            self.CLEAR_LINE = ""
        # Track whether we're currently sitting on a transient status line
        # (so the next permanent write knows to clear it first).
        self._status_active = False

    # --- low-level ---
    def _write(self, s: str) -> None:
        self.stream.write(s)
        self.stream.flush()

    def _clear_status(self) -> None:
        if self._status_active and self.use_cr:
            self._write(self.CLEAR_LINE)
        self._status_active = False

    # --- public ---
    def header(self, round_idx: int, n_rounds: int, audio_path: str, n_chunks: int) -> None:
        self._clear_status()
        name = os.path.basename(audio_path)
        bar = "━" * 60
        self._write(
            f"\n{self.CYAN}{bar}{self.RESET}\n"
            f"{self.BOLD}{self.CYAN}▶ Round {round_idx}/{n_rounds}{self.RESET}  "
            f"{self.DIM}{name}{self.RESET}  "
            f"{self.GREY}[{n_chunks} chunks]{self.RESET}\n"
            f"{self.CYAN}{bar}{self.RESET}\n"
        )

    def status(self, chunk_idx: int, n_chunks: int, silent_run: int, replied: int) -> None:
        """Transient one-line progress; updates in place on a TTY."""
        if not self.use_cr:
            return  # don't spam a logfile with thousands of status lines
        filled = int(self.BAR_WIDTH * chunk_idx / max(n_chunks, 1))
        bar = (
            f"{self.GREEN}{'█' * filled}{self.RESET}"
            f"{self.GREY}{'·' * (self.BAR_WIDTH - filled)}{self.RESET}"
        )
        dots = f"{self.DIM}{'·' * min(silent_run, 40)}{self.RESET}" if silent_run else ""
        replied_tag = (
            f"  {self.GREEN}✓ {replied} reply{'ies' if replied != 1 else ''}{self.RESET}"
            if replied else ""
        )
        line = (
            f"{self.CLEAR_LINE}"
            f"{self.DIM}listening{self.RESET} {bar} "
            f"{self.BOLD}{chunk_idx:>3}/{n_chunks}{self.RESET}"
            f"{replied_tag}  {dots}"
        )
        self._write(line)
        self._status_active = True

    def reply_begin(self) -> None:
        """Promote to a permanent line: clear status, drop to a new line, print prefix."""
        self._clear_status()
        self._write(f"  {self.MAGENTA}▸{self.RESET} ")

    def reply_token(self, text: str) -> None:
        self._write(text)

    def reply_emotion(self, token_id: int) -> None:
        """Print the emotion kaomoji inline at the start of a reply."""
        kao = EMOTION_KAOMOJI.get(token_id, f"[emo:{token_id}]")
        self._write(f"{self.YELLOW}{kao}{self.RESET} ")

    def reply_end(self) -> None:
        self._write("\n")

    def round_summary(self, replied: int, silent: int, total: int) -> None:
        self._clear_status()
        self._write(
            f"  {self.GREY}└─ {replied} reply chunk(s), "
            f"{silent} silent, {total} total{self.RESET}\n"
        )

    def finish(self) -> None:
        self._clear_status()


def streaming_generate(
    model: GPT,
    audio_encoder: torch.nn.Module,
    tokenizer: Tokenizer,
    prefix_ids: torch.Tensor,
    *,
    rounds: int = 10,
    audio_paths: Optional[List[str]] = None,
    max_returned_tokens: int = 4096,
    temperature: float = 0,
    top_k: Optional[int] = 1,
    top_p: float = 1.0,
):
    """Stream audio→text. If `audio_paths` is given, run one round per path
    non-interactively (offline); otherwise prompt stdin each round (online)."""
    device = prefix_ids.device
    ui = _Pretty()
    ui._write(f"{ui.DIM}device: {device}{ui.RESET}\n")

    token = prefix_ids
    input_pos = torch.arange(0, prefix_ids.size(0), device=device, dtype=torch.int64)
    input_pos_maxp1 = _init_input_pos_maxp1(model, prefix_ids.size(0), device)

    turns: List[List[int]] = []  # one inner list per assistant turn (across all rounds)

    if audio_paths is not None:
        steps = []
        for ap in audio_paths:
            steps.append((ap, False))
            steps.append((None, True))
        n_real = len(audio_paths)
    else:
        steps = None
        n_real = rounds

    n_steps = len(steps) if steps is not None else rounds
    real_idx = 0
    acc_replied = 0
    acc_silent = 0

    for step_idx in range(n_steps):
        if steps is not None:
            audio_path, is_silence = steps[step_idx]
        else:
            is_silence = False
            ui._clear_status()
            audio_path = input(
                f"{ui.BOLD}Round {real_idx + 1}/{n_real}{ui.RESET} — enter audio path: "
            ).strip()

        if is_silence:
            audio_chunks = encode_silence_chunks(1.0, audio_encoder, device)
        else:
            audio_chunks = encode_audio_chunks(audio_path, audio_encoder, device)
            real_idx += 1
            acc_replied = 0
            acc_silent = 0
            ui.header(real_idx, n_real, audio_path, len(audio_chunks))

        # Per-round counters for the progress line / summary.
        replied_chunks = 0
        silent_chunks = 0
        silent_run = 0  # consecutive silent chunks since the last reply (for the dots)

        # Buffer of decoded ids for the *current* assistant turn — emitted token-by-token.
        # turn[0] is TEXT_BEGIN. turn[1] is *usually* an emotion tag, but if not we
        # treat it as regular text.
        current_turn: List[int] = []
        text_started = False  # have we opened a reply line yet?

        listening, audio_idx = True, -1
        ui.status(0, len(audio_chunks), silent_run, replied_chunks)

        for _ in range(max_returned_tokens - input_pos.numel()):
            if listening:
                audio_idx += 1
                if audio_idx >= len(audio_chunks):
                    break
                token, input_pos, input_pos_maxp1 = _append_listening_block(
                    token, input_pos, input_pos_maxp1, device
                )
                logits = _forward(
                    model, token.view(1, -1), input_pos,
                    input_pos_maxp1=input_pos_maxp1,
                    audio_feat=audio_chunks[audio_idx].to(device),
                )
            else:
                logits = _forward(
                    model, token.view(1, -1), input_pos,
                    input_pos_maxp1=input_pos_maxp1,
                    audio_feat=None,
                )

            token = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p).to(torch.int64)
            int_token = token.item()
            input_pos, input_pos_maxp1 = _advance_one(input_pos, input_pos_maxp1)

            if listening:
                if int_token == TEXT_BEGIN:
                    listening = False
                    current_turn = [int_token]
                    text_started = False
                    # Don't print yet — wait until we know whether the next token is an emotion.
                elif int_token == KEEP_SILENCE:
                    turns.append([int_token])
                    silent_chunks += 1
                    silent_run += 1
                    ui.status(audio_idx + 1, len(audio_chunks), silent_run, replied_chunks)
                else:
                    raise ValueError(f"Unexpected token {int_token} while listening")
            else:
                current_turn.append(int_token)

                if int_token == TEXT_END:
                    if text_started:
                        ui.reply_end()
                    turns.append(current_turn)
                    current_turn = []
                    text_started = False
                    replied_chunks += 1
                    silent_run = 0
                    listening = True
                    ui.status(audio_idx + 1, len(audio_chunks), silent_run, replied_chunks)
                else:
                    # Layout: [TEXT_BEGIN, (optional EMOTION), t0, t1, ..., TEXT_END]
                    # current_turn already has TEXT_BEGIN + this token in it.
                    n = len(current_turn)
                    if n == 2:
                        # First token after TEXT_BEGIN — could be an emotion tag or
                        # just regular text if the model skipped the emotion.
                        ui.reply_begin()
                        text_started = True
                        if int_token in EMOTION_KAOMOJI:
                            ui.reply_emotion(int_token)
                        else:
                            piece = tokenizer.decode(torch.tensor([int_token]))
                            if piece:
                                ui.reply_token(piece)
                    elif n >= 3:
                        # Stream this token's surface form immediately.
                        piece = tokenizer.decode(torch.tensor([int_token]))
                        if piece:
                            ui.reply_token(piece)

        if not listening and text_started:
            ui.reply_end()
            text_started = False

        acc_replied += replied_chunks
        acc_silent += silent_chunks
        if steps is None or is_silence:
            ui.round_summary(acc_replied, acc_silent, acc_replied + acc_silent)

    ui.finish()
    return turns