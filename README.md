# Mini-Omni3: Towards Unified Audio Interaction Model

<p align="center">
  <img src="assets/figures/top.png" alt="Mini-Omni3 Logo" width="100%">
</p>

Today's Large Audio Language Models (LALMs) are stuck in an offline paradigm: you hand them a complete audio clip, wait, and get a reply. Streaming audio models exist, but each one only handles a single, isolated task. There has never been a general streaming audio language model. We formalize that missing capability as a new concept **the Audio Interaction Model** and build the first one.
Mini-Omni3 is a unified Audio Interaction Model that:

✅ Runs conventional offline audio tasks (ASR, S2TT, AQA...)

✅ Runs streaming audio tasks in real time (Voice chatting...)

✅ Achieves general streaming audio instruction following on a live stream

✅ Does all of the above inside a single, all-in-one model, and be always-on and proactive


<p align="center">
  <a href="https://arxiv.org/abs/2605.XXXXX">Technical Report 📖</a> /
  <a href="https://huggingface.co/datasets/mini-omni3/SoundFlow-260K">StreamAudio-2M 🤗</a> /
  <a href="https://huggingface.co/mini-omni3/Mini-Omni3">Mini-Omni3 Model 🤗</a> /
  <a href="https://github.com/mini-omni3/Streaming-Audio-Bench">Streaming-Audio-Bench 🏆</a>
</p>

<p align="center">
  <a href="https://github.com/mini-omni3/Mini-Omni3/raw/main/assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>&nbsp;<a href="https://mini-omni3.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>&nbsp;<a href="https://x.com/"><img src="https://img.shields.io/badge/X-@MiniOmni3-black?logo=x&logoColor=white" alt="X"></a>
</p>


<p align="center">
  <a href="https://www.youtube.com/watch?v=r1S4xiUBg9s">
    <img src="https://img.youtube.com/vi/r1S4xiUBg9s/maxresdefault.jpg" alt="Watch Mini-Omni3 running live" width="95%">
  </a>
</p>
<p align="center"><em>▶ Click to watch Mini-Omni3 listen, decide, and speak — live (YouTube)</em></p>



## 🔥 News

- [Coming]: We will release the full dataset and data curation pipeline.
- [Coming]: The full training configs and pipeline.


- **May 20, 2026**: 🔥 We release **StreamAudio-2M**.
- **May 20, 2026**: 🔥 We release the **Mini-Omni3 Inference and Training Codebase**.
- **May 19, 2026**: 🔥 **Mini-Omni3** model weights are now available on Hugging Face.
- **May 19, 2026**: 🔥 We release the **Mini-Omni3 Technical Report**.


## Contents

* **[Quick Start](#quick-start)**
* **[Demos](#demos)** 
* **[SoundFlow: Train your own Audio Interaction Model](#how-it-works)**
* **[StreamAudio-2M dataset](#datasets)**
* **[Evaluation results](#evaluation)**
* **[License, Citation & Stars](#citation)**


## <a id="quick-start"></a>⚡ Quick Start

Mini-Omni3 is an always-on model: it keeps listening to incoming audio frames and **decides for itself when to speak**. By default it stays in a `⟨Silent⟩` state and only emits output when the task or the acoustic context warrants it — so you can open a single session, stream audio into it continuously, and watch every capability take turns on its own.

**Installation**
```bash
git clone https://github.com/mini-omni3/Mini-Omni3.git
cd Mini-Omni3

conda create -n mini-omni3 python=3.12 -y
conda activate mini-omni3
# please check if you are using torch-cuda
pip install -r requirements.txt
# install ffmpeg
conda install -c conda-forge ffmpeg
```

**Download Weights**
```bash
# download model weights from huggingface
export PYTHONPATH=./
python download.py
```

**WebUI real-time demo**
```bash
# download model weights from huggingface
export PYTHONPATH=./
python web/server.py

# then goto localhost:5001
```

**Inference**
```bash
# infer with default audio. make sure add ./ to the project base dir.
# infer_online stimulate continous audio input.
export PYTHONPATH=./
python infer_online.py

# Use your own audio for offline testing:
python infer_offline.py
```


## <a id="demos"></a>🎬 Demos

Most audio models do one job and wait to be asked. Mini-Omni3's defining trait is that **all of its abilities live in the same continuous stream**, and the model itself decides which one is needed at each moment. The demo below is **one unbroken session, one model, no mode switches, no prompts** — transcription, understanding, conversation, and proactive intervention simply happen as the soundscape changes.

<div align="center">
  <video src="assets/demo/all_in_one_session.mp4" controls width="320"></video>
</div>


#### Capability 1 — Online audio understanding

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">Continuous ambient audio: footsteps, a door opening, distant traffic.</td>
    <td valign="top">❌ Record-then-infer: waits for the clip to end, then returns one summary — no incremental narration.</td>
    <td valign="top">⚠️ Speech-centric: lumps non-speech into "background noise" and misses individual events.</td>
    <td valign="top">⚠️ Buffers a fixed window first, so narration lags several seconds behind the sound.</td>
    <td valign="top">✅ Detects each event incrementally and narrates the scene in real time, without waiting for the clip to end.</td>
  </tr>
</table>

<details>
<summary><strong>Capabilities 2 – 4 (transcription &amp; translation · full-spectrum chat · proactive intervention)</strong></summary>

<br>

#### Capability 2 — Real-time transcription &amp; translation

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">A speaker talking continuously while the model listens.</td>
    <td valign="top">⚠️ Clean transcript, but only after the utterance finishes — no mid-sentence partials.</td>
    <td valign="top">⚠️ Streams ASR well, but translation is turn-based and only fires at sentence boundaries.</td>
    <td valign="top">⚠️ Emits chunks but re-decodes aggressively, causing flicker and unstable partials.</td>
    <td valign="top">✅ Emits partial transcripts and translations chunk by chunk with low latency, correcting incrementally as context arrives.</td>
  </tr>
</table>

#### Capability 3 — Voice chat beyond speech

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">A user asks about a song playing in the background while talking.</td>
    <td valign="top">⚠️ Hears the speech but ignores the music — answers as if no song were playing.</td>
    <td valign="top">❌ Treats the music as noise to suppress; can't reason about it.</td>
    <td valign="top">⚠️ Can ID the song in isolation, but can't fuse it with the ongoing conversation.</td>
    <td valign="top">✅ Jointly perceives speech, music, and general audio, and responds in a context-aware, full-spectrum conversation.</td>
  </tr>
</table>

#### Capability 4 — Proactive intervention

<table>
  <tr>
    <th valign="top">Input (streaming)</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">Mini-Omni3 (Ours)</th>
  </tr>
  <tr>
    <td valign="top">A smoke alarm starts beeping while the user is silent.</td>
    <td valign="top">❌ Stays silent — only responds when prompted; no self-initiated speech.</td>
    <td valign="top">❌ Waits for a wake word / user turn; never volunteers a warning.</td>
    <td valign="top">❌ No notion of <em>when</em> to speak; requires an explicit query.</td>
    <td valign="top">✅ Holds <code>⟨Silent⟩</code> until the acoustic cue appears, then switches to <code>⟨Speak⟩</code> and warns the user — no prompt required.</td>
  </tr>
</table>

</details>



## <a id="how-it-works"></a>⚙️ SoundFlow: Train your own Audio Interaction Model
Offline audio models answer a finished clip, but real audio needs a model that listens continuously and decides, moment to moment, whether to speak. SoundFlow trains a single model that at every chunk chooses between `⟨Speak⟩` and `⟨Silent⟩`, so recognition, translation, and dialogue become instructions inside one always-on perceive–decide–respond loop — a Large Audio Interaction Model (LAIM) — instead of separate per-task models. The framework covers the whole pipeline: stitching short clips into long interactions for data, chunk-level decision training with history review and comprehension-aware silence, and asynchronous FIFO inference that cuts first-frame latency by 4.5×.

<p align="center">
  <img src="./assets/figures/soundflow.png" alt="SoundFlow framework" width="92%">
</p>

&nbsp;

## <a id="finetuning"></a>🔧 Finetuning ** data samples are in /src/miniomni3/dataset/examples

You can fine-tune Mini-Omni3 on your own streaming data, and you can also use this repository to train standard offline audio language models. There are two steps: build the training data, then train.

### 1. Prepare training data

Edit the path constants at the top of each script first:

| File | Constants to fill in |
|---|---|
| `src/mini_omni3/dataset/get_feat.py` | `QWEN_OMNI_CKPT`, `AUDIO_TOWER_CKPT` |
| `src/mini_omni3/dataset/get_dataset_online.py` | `QWEN_OMNI_CKPT` |
| `src/mini_omni3/dataset/get_dataset_offline.py` | `QWEN_OMNI_CKPT`, `AUDIO_TOWER_CKPT` |

#### Input JSONL format

**Online** (streaming, multi-turn audio). One JSON object per line:

```json
{"conversation": [
    {"audio_path": "/path/to/turn1.wav", "assistant": "reply 1", "emotion": "normal"},
    {"audio_path": "/path/to/turn2.wav", "assistant": "reply 2", "emotion": "happy"}
]}
```

- `audio_path` and `assistant` are required on every turn.
- `emotion` is optional and defaults to `"normal"`. Allowed values: `happy`, `sad`, `angry`, `surprise`, `normal`, `urgent`.
- To make the model stay silent on a turn, set `assistant` to `"<no need to response>"`.

A single-turn shorthand is also accepted:

```json
{"merge_path": "/path/to/audio.wav", "assistant": "reply", "emotion": "normal"}
```

**Offline** (single-turn). One JSON object per line, either the flat form:

```json
{"user": "user text", "assistant": "reply", "audio_path": "/path/to/audio.wav"}
```

or the online-style multi-turn shape, in which case only the **first** turn is used:

```json
{"conversation": [{"user": "...", "assistant": "...", "audio_path": "..."}, ...]}
```

`assistant` is always required. The task variant is decided by which other fields are present:

| Has `audio_path`? | Has `user`? | Task |
|:---:|:---:|---|
| ✓ | ✓ | `A_T_T` — audio + user text → assistant |
| ✓ |   | `A_T` — audio → assistant |
|   | ✓ | `T_T` — user text → assistant |

#### Data process

```bash
# Online: <input.jsonl> <output.jsonl> <error.log> <feature_dir>
CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/get_dataset_online.py \
    <input.jsonl> <output.jsonl> <error.log> <feature_dir>
# Example:
# CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/get_dataset_online.py \
#     data/online_raw.jsonl data/online.jsonl logs/online.err features/online

# Offline: <input.jsonl> <output.jsonl> <error.log> <feature_dir>
CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/get_dataset_offline.py \
    <input.jsonl> <output.jsonl> <error.log> <feature_dir>
# Example:
# CUDA_VISIBLE_DEVICES=0 python src/mini_omni3/dataset/get_dataset_offline.py \
#     data/offline_raw.jsonl data/offline.jsonl logs/offline.err features/offline
```

Both scripts are resumable: re-running picks up where the previous run stopped, skipping any `idx` that was already written. For a parallel multi-GPU template, see `src/mini_omni3/dataset/process_get_feature.sh`.

### 2. Train

```bash
# 1. Set the two data roots referenced by config.yaml
export DATA_ROOT=/path/to/your/jsonl/data
export CHECKPOINT_ROOT=/path/to/your/checkpoints
# Example:
# export DATA_ROOT=/data/mini_omni3/jsonl
# export CHECKPOINT_ROOT=/data/mini_omni3/ckpts

# 2. Edit hyperparameters / data sources in src/mini_omni3/finetune/config.yaml

# 3. Launch
python src/mini_omni3/finetune/full.py --config src/mini_omni3/finetune/config.yaml
# Example:
# python src/mini_omni3/finetune/full.py --config src/mini_omni3/finetune/config.yaml
```

## <a id="datasets"></a> 🎊 StreamAudio-2M: a large-scale stream audio instruction following corpus
<p align="center">
  <img src="./assets/figures/dataset.png" alt="SoundFlow framework" width="92%">
</p>

StreamAudio-2M is a ~2.6M-item streaming instruction-following corpus (7.4M rounds, 66.7K hours) covering seven capabilities — audio understanding, real-time ASR, speech translation, voice chatting, proactive response, and environment-aware agent — built by collecting clips from real-world datasets (AudioSet, CommonVoice, CoVoST2, MOSS, …), synthesizing text into speech with CosyVoice, then concatenating them into streaming sequences with environmental noise and token-level annotation.

### Sample structure

Each line is one streaming sequence made of multiple turns:

```json
{
  "id": "voice_chatting_000123",
  "stream_scene_type": "Home Smart",
  "num_turns": 2,
  "turns": [
    {
      "user": "Turn the living room lights down a bit.",
      "assistant": "Sure, dimming them to 40%.",
      "emotion": "normal",
      "scene_type": "Home Smart",
      "audio_path": "voice_chatting/000123/turn_0.wav"
    },
    {
      "user": "Thanks. What's the temperature in here?",
      "assistant": "It's 22.5 degrees in the living room.",
      "emotion": "normal",
      "scene_type": "Home Smart",
      "audio_path": "voice_chatting/000123/turn_1.wav"
    }
  ]
}
```

Set `assistant` to `"<no need to response>"` for a turn where the model should stay silent.

## Acknowledgements

We sincerely thank the creators, maintainers, and contributors of the public datasets and resources used in this work. We also thank the broader large audio language model community for laying the groundwork that made streaming audio modeling possible.

In particular, this project builds on the following open-source repositories:

- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) — the audio encoder and language model backbone behind Mini-Omni3.
- [LitGPT](https://github.com/Lightning-AI/litgpt) — the training framework our finetuning code is built on.
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — the text-to-speech model used to synthesize speech during data construction.


## <a id="citation"></a>License, Citation & Stars

This project will be released under the **Apache-2.0 License**. You can do everything with Mini-Omni3 🎉

**Citation**: You can cite Mini-Omni3 using the following BibTeX entry. Thank you for your kindness 🙂

```bibtex
@misc{miniomni3,
      title={Mini-Omni3: An Always-On Streaming Audio Language Model for the Real World},
      author={Mini-Omni3 Team},
      year={2026},
      eprint={2605.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2605.XXXXX},
}
```

<a href="https://www.star-history.com/?repos=xzf-thu%2FMini-Omni3&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=xzf-thu/Mini-Omni3&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=xzf-thu/Mini-Omni3&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=xzf-thu/Mini-Omni3&type=date&legend=top-left" />
 </picture>
</a>