"""Training / inference helpers."""

import inspect
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.loggers import WandbLogger
from typing_extensions import Self


# === Output dir / checkpoint resume ===

def init_out_dir(out_dir: Path) -> Path:
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    if not out_dir.is_absolute() and "LIGHTNING_ARTIFACTS_DIR" in os.environ:
        return Path(os.getenv("LIGHTNING_ARTIFACTS_DIR")) / out_dir
    return out_dir


def find_resume_path(resume: Union[bool, Literal["auto"], Path], out_dir: Path) -> Optional[Path]:
    if not resume or isinstance(resume, Path):
        return resume or None
    # `resume` is True or "auto"; find the newest lit_model.pth under out_dir.
    candidates = list(out_dir.rglob("lit_model.pth"))
    if not candidates:
        if resume == "auto":
            return None
        raise FileNotFoundError(
            f"resume=True but no `lit_model.pth` under {out_dir}; set resume='auto' to skip or False to start fresh."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# === Misc math / introspection ===

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            total += p.numel()
    return total


# === Cross-entropy ===

def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Memory-friendly cross-entropy: optionally chunk along the seq axis to bound the activation peak."""
    if isinstance(logits, list):
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1).reshape(-1, logits[0].size(-1))
            return torch.nn.functional.cross_entropy(logits, targets.reshape(-1), ignore_index=ignore_index)
        logit_chunks = [c.reshape(-1, c.size(-1)) for c in logits]
        target_chunks = [c.reshape(-1) for c in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(lc, tc, ignore_index=ignore_index, reduction="none")
            for lc, tc in zip(logit_chunks, target_chunks)
        ]
        non_masked = (targets != ignore_index).sum()
        return torch.cat(loss_chunks).sum() / non_masked.maximum(torch.ones_like(non_masked))

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(lc, tc, ignore_index=ignore_index, reduction="none")
        for lc, tc in zip(logit_chunks, target_chunks)
    ]
    non_masked = (targets != ignore_index).sum()
    return torch.cat(loss_chunks).sum() / non_masked.maximum(torch.ones_like(non_masked))


# === Precision / checkpoint IO ===

def get_default_supported_precision(training: bool) -> str:
    """Return bf16 if the GPU supports it, otherwise 16-bit. Picks `-mixed` for training, `-true` for inference."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed" if training else "bf16-true"
    return "16-mixed" if training else "16-true"


def load_checkpoint(fabric: L.Fabric, model: nn.Module, checkpoint_path: Path, strict: bool = True) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)


# === Iteration ===

class CycleIterator:
    """Indefinitely cycles through an iterable, exposing the current epoch count."""

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


# === Setup helpers (out dir, logger, devices, optimizer) ===

def copy_config_files(source_dir: Path, out_dir: Path) -> None:
    """Copy model_config.yaml and tokenizer files (when present) into the output directory."""
    for name in ("config.json", "generation_config.json", "model_config.yaml",
                 "tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        src = source_dir / name
        if src.exists():
            shutil.copy(src, out_dir)


def parse_devices(devices: Union[str, int]) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    resume: Optional[bool] = None,
    **kwargs: Any,
):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project=name, resume=resume, **kwargs)
    raise ValueError(f"`--logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'.")


def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    """Resolve `optimizer` (str name like 'AdamW' or a dict with class_path/init_args) to an actual optimizer."""
    if isinstance(optimizer, str):
        class_module, class_name = optimizer.rsplit(".", 1) if "." in optimizer else ("torch.optim", optimizer)
        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)
        valid = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return optimizer_cls(model_parameters, **kwargs)
    if isinstance(optimizer, dict):
        optimizer = dict(optimizer)
        class_module, class_name = optimizer["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        valid = set(inspect.signature(getattr(module, class_name)).parameters)
        optimizer["init_args"].update({k: v for k, v in kwargs.items() if k in valid})
        return instantiate_class(model_parameters, optimizer)
    raise ValueError(f'Unrecognized "optimizer" value: {optimizer}')


# === GPU topology (nvlink / xgmi) ===

def check_nvlink_connectivity(fabric=None):
    """Best-effort check that the multi-GPU interconnect is fast. Prints a warning if not."""
    custom_print = fabric.print if fabric is not None else print
    if os.getenv("RANK", "0") != "0":
        return
    try:
        if not torch.cuda.is_available():
            custom_print("No GPUs available")
            return
        gpu_name = torch.cuda.get_device_properties(0).name.lower()
        if "nvidia" in gpu_name:
            _check_nvidia_connectivity(custom_print)
        elif "advanced micro devices" in gpu_name or "amd" in gpu_name:
            _check_amd_connectivity(custom_print)
        else:
            custom_print(f"Unrecognized GPU vendor: {gpu_name}")
    except Exception as e:
        custom_print(f"An error occurred while checking GPU connectivity: {e}")


def _check_nvidia_connectivity(custom_print):
    result = subprocess.run(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("Failed to run nvidia-smi"); return
    lines = result.stdout.strip().split("\n")
    start = next((i for i, line in enumerate(lines) if "GPU0" in line), None)
    if start is None:
        custom_print("Failed to parse nvidia-smi output"); return
    headers = lines[start].split()
    gpu_count = len([h for h in headers if re.match(r"^GPU\d+$", h)])
    all_nvlink = all(
        all("NV" in conn for conn in row.split()[1:1 + gpu_count] if conn != "X")
        for row in lines[start + 1: start + 1 + gpu_count]
    )
    if all_nvlink:
        custom_print("All GPUs are fully connected via NVLink.")
    else:
        custom_print("Warning: not all GPUs are fully connected via NVLink — multi-GPU throughput may be suboptimal.")


def _check_amd_connectivity(custom_print):
    result = subprocess.run(["rocm-smi", "--showtopotype"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("Failed to run rocm-smi"); return
    lines = result.stdout.strip().split("\n")
    gpu_idx = next((i for i, line in enumerate(lines) if re.match(r"^\s*GPU0", line)), None)
    if gpu_idx is None or gpu_idx == 0:
        custom_print("Failed to parse rocm-smi output (no GPU headers found)"); return
    headers = lines[gpu_idx - 1].strip().split()
    gpu_count = len([h for h in headers if re.match(r"^GPU\d+$", h)])
    gpu_lines = [l.strip() for l in lines[gpu_idx: gpu_idx + gpu_count] if re.match(r"^\s*GPU\d+", l)]
    if len(gpu_lines) != gpu_count:
        custom_print("Mismatch in GPU count when parsing rocm-smi output"); return
    all_xgmi = all(
        all(conn in ("XGMI", "0") for conn in row.split()[1:1 + gpu_count])
        for row in gpu_lines
    )
    if all_xgmi:
        custom_print("All GPUs are fully connected via XGMI.")
    else:
        custom_print("Warning: not all GPUs are fully connected via XGMI — multi-GPU throughput may be suboptimal.")


# === Misc ===

def fix_and_load_json(s: str):
    """Tolerant JSON parser: strips trailing commas and inserts missing ones before retrying."""
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    s = re.sub(r'(?<=[}\]0-9truefalsenull"])\s*(\n\s*)"', r',\1"', s)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after fixing: {e}")


def create_finetuning_performance_report(training_time: float, device_type: str) -> str:
    lines = [
        "",
        "| ------------------------------------------------------",
        f"| Training Time : {training_time:.2f} s",
    ]
    if device_type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        lines.append(f"| Peak Memory   : {memory_used:.2f} GB")
    lines.append("| ------------------------------------------------------")
    return "\n".join(lines)







