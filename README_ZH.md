# 音频交互模型（Audio Interaction Model）

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  <img src="assets/figures/top.png" alt="AudioInteraction Logo" width="100%">
</p>

如今的大型音频语言模型（LALMs）仍停留在离线范式：你需要把一段完整的音频交给它，等待一段时间，然后得到一个回复。流式音频模型虽然存在，但每一个都只能处理单一、孤立的任务。从未出现过一个**通用的流式音频语言模型**。我们将这一缺失的能力形式化为一个全新的概念——**音频交互模型（Audio Interaction Model）**，并构建了第一个这样的模型。
AudioInteraction 是一个统一的音频交互模型，它能够：

✅ 运行常规的离线音频任务（ASR、S2TT、AQA……）

✅ 实时运行流式音频任务（语音对话……）

✅ 在实时音频流上实现通用的流式音频指令跟随

✅ 在单一的一体化模型中完成上述所有任务，并且始终在线、主动响应


<p align="center">
  <a href="https://arxiv.org/pdf/2606.05121">技术报告 📖</a> /
  <a href="https://huggingface.co/datasets/zhifeixie/StreamAudio-2M">StreamAudio-2M 🤗</a> /
  <a href="https://huggingface.co/zhifeixie/AudioInteraction">AudioInteraction 模型 🤗</a> /
  <a href="https://github.com/masaz14/Proactive-Sound-Effect-Benchmark">Streaming-Audio-Bench 🏆</a>
</p>

<p align="center">
  <a href="https://github.com/AudioInteraction/AudioInteraction/raw/main/assets/wechat.jpg"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat"></a>&nbsp;<a href="https://xzf-thu.github.io/Audio-Interaction"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>&nbsp;<a href="https://x.com/XieZhifei14110"><img src="https://img.shields.io/badge/X-@audiointeraction-black?logo=x&logoColor=white" alt="X"></a>
</p>


<p align="center">
  <a href="https://www.youtube.com/watch?v=4YuBkMm1cmU">
    <img src="https://img.youtube.com/vi/4YuBkMm1cmU/maxresdefault.jpg" alt="Watch AudioInteraction running live" width="95%">
  </a>
</p>
<p align="center"><em>▶ 点击观看 AudioInteraction 实时聆听、决策与发声（YouTube）</em></p>



## 🔥 新闻动态

- [即将发布]：我们将发布完整数据集及数据构建流程。
- [即将发布]：完整的训练配置与流程。


- **2026 年 5 月 20 日**：🔥 我们发布 **StreamAudio-2M**。
- **2026 年 5 月 20 日**：🔥 我们发布 **AudioInteraction 推理与训练代码库**。
- **2026 年 5 月 19 日**：🔥 **AudioInteraction** 模型权重已在 Hugging Face 上线。
- **2026 年 5 月 19 日**：🔥 我们发布 **AudioInteraction 技术报告**。


## 目录

* **[快速开始](#quick-start)**
* **[演示](#demos)**
* **[SoundFlow：训练你自己的音频交互模型](#how-it-works)**
* **[StreamAudio-2M 数据集](#datasets)**
* **[评测结果](#evaluation)**
* **[许可证、引用与 Star](#citation)**


## <a id="quick-start"></a>⚡ 快速开始

AudioInteraction 是一个始终在线的模型：它持续监听传入的音频帧，并**自行决定何时发声**。默认情况下，它保持在 `⟨Silent⟩`（静默）状态，仅在任务需要或声学场景需要时才输出。因此，你可以打开单个会话，持续不断地把音频流送入其中，看着每一项能力自动轮流登场。

**安装**
```bash
git clone https://github.com/AudioInteraction/AudioInteraction.git
cd AudioInteraction

conda create -n AudioInteraction python=3.12 -y
conda activate AudioInteraction
# 请检查你使用的是否为 torch-cuda
pip install -r requirements.txt
# 安装 ffmpeg
conda install -c conda-forge ffmpeg
```

**下载权重**
```bash
# 从 huggingface 下载模型权重
export PYTHONPATH=./
python download.py
```

## 推理与 WebUI

先运行推理，再启动 WebUI 演示。

```bash
# 将项目根目录加入 PYTHONPATH
export PYTHONPATH=./

# 1. 离线推理
python infer_offline.py

# 若要测试自带样例，请将 infer_offline.py 中的 input_path 设置为以下之一：
# sample/01_count_bark/sequence.json
# sample/02_translate/sequence.json
# sample/03_cough_music/sequence.json

# 2. 实时推理
python infer_online.py
````

### WebUI 实时演示

```bash
# 请先从 Hugging Face 下载模型权重
python web/server.py

# 在浏览器中打开：
# http://localhost:5001
```



## <a id="demos"></a>🎬 演示

大多数音频模型只做一件事，并且要等被询问才会响应。AudioInteraction 的核心特征在于：**它的所有能力都存在于同一条连续的音频流中**，模型自身在每一时刻决定需要哪种能力。下面的演示是**一段不间断的会话、一个模型、无模式切换、无提示词**——转写、理解、对话与主动干预，会随着声音环境的变化自然而然地发生。

<div align="center">
  <video src="assets/demo/all_in_one_session.mp4" controls width="320"></video>
</div>


#### 能力 1 —— 在线音频理解

<table>
  <tr>
    <th valign="top">输入（流式）</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">AudioInteraction（本文方法）</th>
  </tr>
  <tr>
    <td valign="top">连续的环境音：脚步声、开门声、远处的车流声。</td>
    <td valign="top">❌ 录制后再推理：等待音频片段结束后才返回一段总结——没有增量式的叙述。</td>
    <td valign="top">⚠️ 以语音为中心：把非语音内容笼统归为"背景噪声"，遗漏单个声音事件。</td>
    <td valign="top">⚠️ 先缓存一个固定窗口，导致叙述比声音滞后数秒。</td>
    <td valign="top">✅ 增量式地检测每个事件，实时描述场景，无需等待片段结束。</td>
  </tr>
</table>

<details>
<summary><strong>能力 2 – 4（转写与翻译 · 全频谱对话 · 主动干预）</strong></summary>

<br>

#### 能力 2 —— 实时转写与翻译

<table>
  <tr>
    <th valign="top">输入（流式）</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">AudioInteraction（本文方法）</th>
  </tr>
  <tr>
    <td valign="top">说话者持续讲话，模型同时聆听。</td>
    <td valign="top">⚠️ 转写干净，但只在整句说完后才输出——没有句中部分结果。</td>
    <td valign="top">⚠️ ASR 流式表现不错，但翻译是按轮次进行的，只在句子边界触发。</td>
    <td valign="top">⚠️ 会输出分块结果，但激进地重新解码，导致闪烁和不稳定的部分结果。</td>
    <td valign="top">✅ 以低延迟逐块输出部分转写与翻译，并随上下文到来增量式地修正。</td>
  </tr>
</table>

#### 能力 3 —— 超越语音的语音对话

<table>
  <tr>
    <th valign="top">输入（流式）</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">AudioInteraction（本文方法）</th>
  </tr>
  <tr>
    <td valign="top">用户在说话的同时，询问背景里正在播放的一首歌。</td>
    <td valign="top">⚠️ 听到了语音却忽略了音乐——回答时仿佛没有任何歌曲在播放。</td>
    <td valign="top">❌ 把音乐当作需要抑制的噪声；无法对其进行推理。</td>
    <td valign="top">⚠️ 能在孤立情况下识别歌曲，但无法将其与正在进行的对话融合。</td>
    <td valign="top">✅ 联合感知语音、音乐与一般音频，在情境感知的全频谱对话中作出回应。</td>
  </tr>
</table>

#### 能力 4 —— 主动干预

<table>
  <tr>
    <th valign="top">输入（流式）</th>
    <th valign="top">gpt-audio</th>
    <th valign="top">doubao-voicechat</th>
    <th valign="top">gemini-omni</th>
    <th valign="top">AudioInteraction（本文方法）</th>
  </tr>
  <tr>
    <td valign="top">用户保持沉默时，烟雾报警器开始鸣响。</td>
    <td valign="top">❌ 保持沉默——只在被提示时才响应；不会自发说话。</td>
    <td valign="top">❌ 等待唤醒词／用户轮次；从不主动发出警告。</td>
    <td valign="top">❌ 没有"何时该说话"的概念；需要明确的查询。</td>
    <td valign="top">✅ 在声学线索出现前保持 <code>⟨Silent⟩</code>，随后切换到 <code>⟨Speak⟩</code> 并向用户发出警告——无需任何提示。</td>
  </tr>
</table>

</details>



## <a id="how-it-works"></a>⚙️ SoundFlow：训练你自己的音频交互模型
离线音频模型针对一段已完成的片段作答，但真实的音频需要一个能够持续聆听、并在每时每刻决定是否发声的模型。SoundFlow 训练出单一模型，使其在每个音频块都在 `⟨Speak⟩` 与 `⟨Silent⟩` 之间作出选择，从而让识别、翻译和对话成为同一个始终在线的"感知—决策—响应"循环中的指令——即一个大型音频交互模型（Large Audio Interaction Model, LAIM）——而非彼此独立的逐任务模型。该框架覆盖整条流程：将短片段拼接为长交互以构建数据；带历史回顾与理解感知静默的块级决策训练；以及将首帧延迟降低 4.5 倍的异步 FIFO 推理。

<p align="center">
  <img src="./assets/figures/soundflow.png" alt="SoundFlow framework" width="92%">
</p>

&nbsp;

## <a id="finetuning"></a>🔧 微调 ** 数据样例位于 /src/audiointeraction/dataset/examples

你可以在自己的流式数据上微调 AudioInteraction，也可以使用本仓库训练标准的离线音频语言模型。共分两步：构建训练数据，然后训练。

### 1. 准备训练数据

请先填写每个脚本顶部的路径常量：

| 文件 | 需要填写的常量 |
|---|---|
| `src/audiointeraction/dataset/get_feat.py` | `QWEN_OMNI_CKPT`、`AUDIO_TOWER_CKPT` |
| `src/audiointeraction/dataset/get_dataset_online.py` | `QWEN_OMNI_CKPT` |
| `src/audiointeraction/dataset/get_dataset_offline.py` | `QWEN_OMNI_CKPT`、`AUDIO_TOWER_CKPT` |

#### 输入 JSONL 格式

**在线**（流式、多轮音频）。每行一个 JSON 对象：

```json
{"conversation": [
    {"audio_path": "/path/to/turn1.wav", "assistant": "reply 1", "emotion": "normal"},
    {"audio_path": "/path/to/turn2.wav", "assistant": "reply 2", "emotion": "happy"}
]}
```

- 每一轮都必须包含 `audio_path` 和 `assistant`。
- `emotion` 是可选项，默认为 `"normal"`。允许的取值：`happy`、`sad`、`angry`、`surprise`、`normal`、`urgent`。
- 若要让模型在某一轮保持沉默，请将 `assistant` 设为 `"<no need to response>"`。

也接受单轮简写形式：

```json
{"merge_path": "/path/to/audio.wav", "assistant": "reply", "emotion": "normal"}
```

**离线**（单轮）。每行一个 JSON 对象，可为扁平形式：

```json
{"user": "user text", "assistant": "reply", "audio_path": "/path/to/audio.wav"}
```

或在线风格的多轮形式，此时仅使用**第一**轮：

```json
{"conversation": [{"user": "...", "assistant": "...", "audio_path": "..."}, ...]}
```

`assistant` 始终为必填项。任务类型由存在哪些其它字段决定：

| 是否有 `audio_path`？ | 是否有 `user`？ | 任务 |
|:---:|:---:|---|
| ✓ | ✓ | `A_T_T` —— 音频 + 用户文本 → assistant |
| ✓ |   | `A_T` —— 音频 → assistant |
|   | ✓ | `T_T` —— 用户文本 → assistant |

#### 数据处理

```bash
# 在线：<input.jsonl> <output.jsonl> <error.log> <feature_dir>
python src/audiointeraction/dataset/get_dataset_online.py \
    <input.jsonl> <output.jsonl> <error.log> <feature_dir>
# 示例：
# python src/audiointeraction/dataset/get_dataset_online.py \
#     data/online_raw.jsonl data/online.jsonl logs/online.err features/online

# 离线：<input.jsonl> <output.jsonl> <error.log> <feature_dir>
python src/audiointeraction/dataset/get_dataset_offline.py \
    <input.jsonl> <output.jsonl> <error.log> <feature_dir>
# 示例：
python src/audiointeraction/dataset/get_dataset_offline.py \
#     data/offline_raw.jsonl data/offline.jsonl logs/offline.err features/offline
```

两个脚本均支持断点续跑：重新运行时会从上次停止处继续，跳过已写入的任意 `idx`。如需多 GPU 并行处理模板，请参见 `src/audiointeraction/dataset/process_get_feature.sh`。

### 2. 训练

```bash
# 1. 设置 config.yaml 引用的两个数据根目录
export DATA_ROOT=/path/to/your/jsonl/data
export CHECKPOINT_ROOT=/path/to/your/checkpoints
# 示例：
# export DATA_ROOT=/data/audiointeraction/jsonl
# export CHECKPOINT_ROOT=/data/audiointeraction/ckpts

# 2. 在 src/audiointeraction/finetune/config.yaml 中编辑超参数／数据源

# 3. 启动
python src/audiointeraction/finetune/full.py --config src/audiointeraction/finetune/config.yaml
# 示例：
# python src/audiointeraction/finetune/full.py --config src/audiointeraction/finetune/config.yaml
```

## <a id="datasets"></a> 🎊 StreamAudio-2M：大规模流式音频指令跟随语料库
<p align="center">
  <img src="./assets/figures/dataset.png" alt="SoundFlow framework" width="92%">
</p>

StreamAudio-2M 是一个约 260 万条目的流式指令跟随语料库（740 万轮、6.67 万小时），涵盖七项能力——音频理解、实时 ASR、语音翻译、语音对话、主动响应和环境感知智能体。其构建方式为：从真实世界数据集（AudioSet、CommonVoice、CoVoST2、MOSS……）中收集音频片段，使用 CosyVoice 将文本合成为语音，再将它们拼接成带环境噪声和 token 级标注的流式序列。

### 样本结构

每一行是一条由多轮组成的流式序列：

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

对于需要模型保持沉默的轮次，请将 `assistant` 设为 `"<no need to response>"`。

## <a id="evaluation"></a>📊 Audio-Interaction 的实验结果

### 表 1：MMAU 基准测试结果

| 模型 | 规模 | 流式 | 多轮 | Text Sound | Text Music | Text Speech | Text Avg. | Audio Sound | Audio Music | Audio Speech | Audio Avg. |
|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **_大型音频语言模型_** |  |  |  |  |  |  |  |  |  |  |  |
| Audio Flamingo 2 | 3B | ✗ | ✗ | **71.47** | **70.96** | 44.74 | 62.40 | 1.50 | 1.49 | 0.35 | 1.16 |
| Qwen2-Audio-Instruct | 8.4B | ✗ | ✓ | 54.95 | 50.98 | 42.04 | 49.20 | 22.32 | 19.16 | 16.31 | 19.41 |
| Voxtral-Mini | 3B | ✗ | ✓ | 58.56 | 49.70 | 43.53 | 50.60 | 46.08 | 34.13 | 30.50 | 37.24 |
| Audio-Reasoner | 8.4B | ✗ | ✗ | 60.06 | 64.30 | **60.70** | 61.71 | 20.48 | 26.65 | 13.48 | 20.57 |
| **_全模态语言模型_** |  |  |  |  |  |  |  |  |  |  |  |
| Qwen2.5-Omni | 3B | ✗ | ✓ | 65.36 | 48.94 | 57.78 | 57.81 | 51.81 | 44.01 | 29.79 | 42.51 |
| Qwen2.5-Omni | 7B | ✗ | ✓ | <u>67.87</u> | <u>69.16</u> | <u>59.76</u> | **65.60** | 60.54 | <u>50.90</u> | <u>35.11</u> | <u>49.58</u> |
| Phi-4-multimodal | 7B | ✗ | ✓ | 60.97 | 52.87 | 52.83 | 55.56 | 44.65 | 27.84 | 21.99 | 31.75 |
| Baichuan-Omni-1.5 | 11B | ✗ | ✓ | 65.47 | 58.98 | 55.26 | 59.90 | 57.53 | 36.53 | 24.82 | 40.40 |
| **_流式音频语言模型_** |  |  |  |  |  |  |  |  |  |  |  |
| **Audio-Interaction** | **3B** | **✓** | **✓** | 64.12 | 47.80 | 55.13 | 55.68 | **65.63** | **57.93** | **39.68** | **58.15** |

### 表 2：口语对话基准测试性能

| 模型 | 规模 | SpokenQA LLa. Q. | SpokenQA Web Q. | Voicebench Alpa. | Voicebench SD-QA |
|---|---:|---:|---:|---:|---:|
| **_专用模型_** |  |  |  |  |  |
| Moshi | 7B | 62.20 | 26.30 | 2.01 | 15.01 |
| Freeze-Omni | 7B | 72.00 | 44.73 | 4.14 | 50.16 |
| **_全模态与音频语言模型_** |  |  |  |  |  |
| Baichuan-Omni-1.5 | 7B | **78.50** | <u>59.10</u> | **4.50** | 43.40 |
| Qwen2-Audio | 7B | 69.67 | 45.20 | 3.74 | 35.71 |
| Qwen2.5-Omni | 3B | 66.00 | 27.95 | 4.32 | 49.37 |
| Qwen2.5-Omni | 7B | 75.33 | **62.80** | <u>4.49</u> | **55.71** |
| Phi-4-multimodal | 7B | 60.2 | 26.6 | 3.81 | 39.78 |
| **_流式音频语言模型_** |  |  |  |  |  |
| **Audio-Interaction** | **3B** | 67.31 | 54.34 | 4.28 | <u>52.14</u> |

### 表 3：LibriSpeech 与 CoVoST2 上的 ASR WER 与 S2TT BLEU

| 模型 | 规模 | ASR Clean ↓ | ASR Other ↓ | S2TT en-zh ↑ | S2TT zh-en ↑ |
|---|---:|---:|---:|---:|---:|
| **_专用模型_** |  |  |  |  |  |
| Canary | 1B | **1.48** | **2.93** | - | - |
| Canary-Qwen | 2.5B | 1.49 | <u>3.10</u> | - | - |
| **_全模态与音频语言模型_** |  |  |  |  |  |
| Baichuan-Omni-1.5 | 7B | 5.71 | 10.09 | - | - |
| Qwen2-Audio | 7B | 1.60 | 3.60 | 45.20 | 24.40 |
| Qwen2.5-Omni | 3B | 2.87 | 5.90 | 39.50 | 18.17 |
| Qwen2.5-Omni | 7B | <u>1.80</u> | 3.40 | 41.40 | <u>29.40</u> |
| Phi-4-multimodal | 5.6B | 1.69 | 3.82 | <u>46.30</u> | 22.39 |
| **_流式音频语言模型_** |  |  |  |  |  |
| **Audio-Interaction** | **3B** | 3.17 | 6.04 | **55.22** | **35.21** |


## 致谢

我们由衷感谢本工作中所用公开数据集与资源的创建者、维护者和贡献者。我们也感谢更广泛的大型音频语言模型社区，正是他们奠定的基础使流式音频建模成为可能。

特别地，本项目建立在以下开源仓库之上：

- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) —— AudioInteraction 背后的音频编码器与语言模型主干。
- [LitGPT](https://github.com/Lightning-AI/litgpt) —— 我们微调代码所基于的训练框架。
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) —— 数据构建过程中用于合成语音的文本转语音模型。


## <a id="citation"></a>许可证、引用与 Star

本项目将在 **Apache-2.0 许可证**下发布。你可以用 AudioInteraction 做任何事 🎉

**引用**：你可以使用以下 BibTeX 条目引用 AudioInteraction。感谢你的支持 🙂

```bibtex
@misc{xie2026audiointeractionmodel,
      title={Audio Interaction Model}, 
      author={Zhifei Xie and Zihang Liu and Ze An and Xiaobin Hu and Yue Liao and Ziyang Ma and Dongchao Yang and Mingbao Lin and Deheng Ye and Shuicheng Yan and Chunyan Miao},
      year={2026},
      eprint={2606.05121},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2606.05121}, 
}
```

<a href="https://www.star-history.com/?repos=xzf-thu%2FAudioInteraction&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=xzf-thu/Audio-Interaction&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=xzf-thu/Audio-Interaction&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=xzf-thu/Audio-Interaction&type=date&legend=top-left" />
 </picture>
</a>