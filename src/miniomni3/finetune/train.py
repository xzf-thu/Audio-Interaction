# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Full-parameter finetuning entry.

All hyperparameters and data sources live in a yaml config (see `config.yaml`),
passed via `--config`.
"""

import argparse
import dataclasses
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightning as L
import torch
import yaml
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from torchmetrics import RunningMean
from tqdm import tqdm

from src.miniomni3.args import EvalArgs, TrainArgs
from src.miniomni3.model import GPT, Block, Config
from src.miniomni3.utils import (
    CycleIterator,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    create_finetuning_performance_report,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    parse_devices,
)
from src.miniomni3.finetune.dataloader import get_dataloaders


def load_config(path):
    """Load a yaml config, expanding ${ENV_VAR} references in path strings."""
    with open(path) as f:
        return yaml.safe_load(os.path.expandvars(f.read()))


def setup(cfg: dict):
    """Construct fabric + TrainArgs/EvalArgs from cfg and launch training workers."""
    devices = parse_devices(cfg["training"]["devices"])
    num_nodes = cfg.get("num_nodes", 1)
    out_dir = init_out_dir(Path(cfg["paths"]["out_dir"]) / cfg["project"]["version"])
    checkpoint_dir = Path(cfg["paths"]["config_dir"])
    model_config = Config.from_file(checkpoint_dir / "model_config.yaml")

    precision = get_default_supported_precision(training=True)
    logger = choose_logger(
        cfg["logger_name"], out_dir,
        name=f"finetune-{model_config.name}",
        resume=bool(cfg.get("resume", False)),
        log_interval=cfg["training"]["log_interval"],
    )

    train_args = TrainArgs(
        save_interval=cfg["training"]["save_interval"],
        log_interval=cfg["training"]["log_interval"],
        global_batch_size=cfg["training"]["global_batch_size"],
        micro_batch_size=cfg["training"]["micro_batch_size"],
        lr_warmup_steps=cfg["training"]["lr_warmup_steps"],
        epochs=cfg["training"]["epochs"],
        max_steps=cfg["training"].get("max_steps"),
        max_seq_length=cfg["training"]["max_seq_length"],
    )
    eval_args = EvalArgs(
        interval=cfg["eval"]["eval_interval"],
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        max_iters=cfg["eval"]["max_iters"],
        initial_validation=True,
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices, num_nodes=num_nodes,
        strategy=strategy, precision=precision, loggers=logger,
    )
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main, cfg, devices, num_nodes, model_config,
        checkpoint_dir, out_dir, train_args, eval_args,
    )


def main(fabric, cfg, devices, num_nodes, model_config,
         checkpoint_dir, out_dir, train, eval):
    fabric.seed_everything(cfg["seed"])

    data_sources = {k: v for k, v in cfg["data_sources"].items() if v.get("enabled", True)}
    train_dataloader, val_dataloader, type_to_name = get_dataloaders(
        data_sources=data_sources,
        eval_data_percentage=cfg["eval"]["eval_data_percentage"],
        max_seq_len=cfg["training"]["max_seq_length"],
        train_batchsize=cfg["training"]["micro_batch_size"],
        eval_batchsize=cfg["eval"]["eval_batch_size"],
        seed=cfg["seed"],
    )
    fabric.print(f"Type to name mapping: {type_to_name}")
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices, num_nodes)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))
    print(f"steps_per_epoch: {steps_per_epoch}, lr_max_steps: {lr_max_steps}")

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(model_config)
    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    if fabric.global_rank == 0:
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
    model = fabric.setup(model)

    # float() handles YAML 1.1 parsing `2e-4` as a string (no dot, no upper E).
    optimizer = instantiate_torch_optimizer(
        cfg["training"]["optimizer"], model.parameters(), lr=float(cfg["training"]["max_lr"]),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps)

    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler,
             "iter_num": 0, "step_count": 0}

    resume = find_resume_path(cfg.get("resume", False), out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state, strict=False)
    else:
        load_checkpoint(fabric, state["model"], cfg["paths"]["init_checkpoint"])

    train_time = time.perf_counter()
    fit(
        fabric=fabric, cfg=cfg, state=state,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
        devices=devices, num_nodes=num_nodes, resume=resume,
        checkpoint_dir=checkpoint_dir, out_dir=out_dir,
        train=train, eval=eval, type_to_name=type_to_name,
    )
    training_time = time.perf_counter() - train_time
    fabric.print(create_finetuning_performance_report(training_time, fabric.device.type))

    # Final evaluation
    if eval.final_validation:
        val_losses = validate(
            fabric, model, val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            type_to_name,
        )
        metrics = {}
        for name, loss in val_losses.items():
            if loss is not None:
                metrics[f"val_loss_{name}"] = loss
                metrics[f"val_ppl_{name}"] = math.exp(loss)
        fabric.log_dict(metrics, step=state["iter_num"])
        loss_str = " | ".join(
            f"{name}: {loss.item():.3f}"
            for name, loss in val_losses.items() if loss is not None
        )
        fabric.print(f"Final evaluation | {loss_str}")

    # Final checkpoint
    save_path = out_dir / "final" / "lit_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fabric.save(save_path, {"model": state["model"]})
    if fabric.global_rank == 0:
        copy_config_files(checkpoint_dir, save_path.parent)


def fit(*, fabric, cfg, state, train_dataloader, val_dataloader,
        devices, num_nodes, resume, checkpoint_dir, out_dir, train, eval, type_to_name):
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    model.max_seq_length = train.max_seq_length

    fabric.print(f"max_seq_length={model.max_seq_length}, block_size={model.config.block_size}")

    # Initial validation
    if eval.initial_validation:
        val_losses = validate(
            fabric, model, val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            type_to_name,
        )
        val_loss_strs = {name: f"{loss:.3f}" if loss is not None else "n/a" for name, loss in val_losses.items()}
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, model, val_dataloader,
                 dataclasses.replace(eval, max_iters=2), type_to_name, verbose=False)
        val_loss_strs = {name: "n/a" for name in type_to_name.values()}

    initial_iter = state["iter_num"]
    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)

    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f}s")

    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices, num_nodes),
        sync_on_compute=False,
    ).to(fabric.device)
    fabric.barrier()

    best_list: List[Tuple[int, float]] = []  # top-K (step, loss) ascending
    save_best_after = cfg["eval"].get("save_best_after_step")

    while state["step_count"] < max_steps:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        if train_iterator.epoch >= train.epochs:
            break

        input_ids = batch["input_ids"].to(model.device)
        targets = batch["labels"].to(model.device)
        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            audio_info = {"feats_paths": batch["pt_list"], "audio_pos": batch["audio_pos"]}
            logits = model(input_ids, batch["tasks"], batch["batch_size"], audio_info)
            loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
            fabric.backward(loss / train.gradient_accumulation_iters(devices, num_nodes))

        running_loss.update(loss.detach())

        if not is_accumulating:
            next_step = state["step_count"] + 1
            # Looser clip during warmup, standard 1.0 afterwards.
            max_grad_norm = 10.0 if next_step <= train.lr_warmup_steps - 1 else 1.0
            fabric.clip_gradients(model, optimizer, max_norm=max_grad_norm, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] = next_step

        # Periodic train-loss log line
        if state["iter_num"] % train.log_interval == 0:
            train_loss = running_loss.compute().item()
            metrics = {
                "loss": train_loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": time.perf_counter() - iter_t0,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            val_str = " | ".join(
                f"val_{name}: {v if isinstance(v, str) else f'{v:.3f}'}"
                for name, v in val_loss_strs.items()
            )
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f} | {val_str} |"
                f" iter time: {metrics['iter_time']*1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''} lr: {metrics['learning_rate']:.3e}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        # Validation + best-K bookkeeping
        if not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_losses = validate(fabric, model, val_dataloader, eval, type_to_name)
            val_time = time.perf_counter() - t0

            val_loss_strs = {name: f"{loss:.3f}" if loss is not None else "n/a" for name, loss in val_losses.items()}
            loss_line = " | ".join(
                f"val_{name}: {loss.item():.4f}"
                for name, loss in val_losses.items() if loss is not None
            )
            fabric.print(f"iter {state['iter_num']}: {loss_line}, val time: {val_time*1000:.2f} ms")

            metrics = {}
            for name, loss in val_losses.items():
                if loss is not None:
                    metrics[f"val_loss_{name}"] = loss
                    metrics[f"val_ppl_{name}"] = math.exp(loss)
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

            if save_best_after is not None and state["step_count"] >= save_best_after:
                _maybe_save_best(
                    fabric, state, val_losses, best_list, best_k=3,
                    out_dir=out_dir, checkpoint_dir=checkpoint_dir,
                )

        # Periodic checkpoint
        if (train.save_interval is not None and not is_accumulating
                and state["step_count"] % train.save_interval == 0):
            ckpt = out_dir / f"step-{state['step_count']:06d}" / "lit_model.pth"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"Saving checkpoint to {str(ckpt.parent)!r}")
            fabric.save(ckpt, state)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, ckpt.parent)


def _maybe_save_best(fabric, state, val_losses, best_list, *, best_k, out_dir, checkpoint_dir):
    extra_loss = val_losses.get("train")
    if extra_loss is None:
        return
    loss_val = fabric.all_reduce(extra_loss.detach().clone(), reduce_op="mean").item()
    step = state["step_count"]

    if len(best_list) < best_k:
        best_list.append((step, loss_val))
    elif loss_val < best_list[-1][1]:
        evict_step = best_list[-1][0]
        best_list[-1] = (step, loss_val)
        if fabric.global_rank == 0:
            evict_dir = out_dir / f"best_step{evict_step:06d}"
            if evict_dir.exists():
                shutil.rmtree(evict_dir)
                fabric.print(f"Evicted worst checkpoint {evict_dir!r}")
    else:
        return

    best_list.sort(key=lambda x: x[1])
    best_dir = out_dir / f"best_step{step:06d}"
    best_dir.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving best checkpoint (loss {loss_val:.4f}) to {best_dir!r}")
    fabric.save(best_dir / "lit_model.pth", state)
    if fabric.global_rank == 0:
        copy_config_files(checkpoint_dir, best_dir)


@torch.no_grad()
def validate(fabric, model, val_dataloader, eval, type_to_name, verbose=True):
    """Run validation, returning {display_name: mean_loss} bucketed by data source type."""
    if verbose:
        fabric.print("Validating ...")
    model.eval()

    max_iters = min(len(val_dataloader), eval.max_iters)
    trackers = {
        name: {
            "losses": torch.zeros(max_iters, device=model.device),
            "used": torch.zeros(max_iters, dtype=torch.bool, device=model.device),
            "count": 0,
        }
        for name in type_to_name.values()
    }

    for k, batch in enumerate(tqdm(val_dataloader, disable=not verbose)):
        if k >= eval.max_iters:
            break
        input_ids = batch["input_ids"].to(model.device).contiguous()
        targets = batch["labels"].to(model.device).contiguous()
        audio_info = {"feats_paths": batch["pt_list"], "audio_pos": batch["audio_pos"]}
        logits = model(input_ids, batch["tasks"], batch["batch_size"], audio_info)
        loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)

        name = type_to_name.get(batch["types"][0])
        if name and name in trackers:
            trackers[name]["losses"][k] = loss
            trackers[name]["used"][k] = True
            trackers[name]["count"] += 1
        else:
            fabric.print(f"Warning: unknown type {batch['types'][0]!r} at iter {k}")

    results = {
        name: (t["losses"][t["used"]].mean() if t["used"].any() else None)
        for name, t in trackers.items()
    }
    if verbose:
        fabric.print("Data counts - " + ", ".join(f"{n}: {t['count']}" for n, t in trackers.items()))
    model.train()
    return results


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the audio-enhanced GPT.")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config.yaml",
        help="path to the training config yaml (default: alongside this script)",
    )
    args = parser.parse_args()
    setup(load_config(args.config))
