# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
import yaml
from typing_extensions import Self
from src.miniomni3.utils import find_multiple


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    # General size parameters
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    # Transformer block (structure, normalizations)
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_qk: bool = False
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    parallel_residual: bool = True
    shared_attention_norm: bool = False
    # Transformer block (self-attention)
    n_head: int = 32
    head_size: Optional[int] = None
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    attn_bias: bool = False
    attention_scores_scalar: Optional[int] = None
    sliding_window_size: Optional[int] = None
    sliding_window_layer_placing: Optional[Literal["all", "interleaved"]] = None
    # if `attention_logit_softcapping` is used, cannot use optimized
    # `torch.nn.functional.scaled_dot_product_attention` (which implements
    # Flash attention), may result in higher memory and runtime footprint.
    attention_logit_softcapping: Optional[float] = None
    # Rotary position embedding (RoPE)
    rope_base: int = 10000
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_adjustments: Optional[dict] = None
    # Transformer block (MLP)
    intermediate_size: Optional[int] = None
    bias: bool = True
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    n_expert: int = 0
    n_expert_per_token: int = 0
    # GPT before/after blocks
    scale_embeddings: bool = False
    lm_head_bias: bool = False
    final_logit_softcapping: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(f"The config {self.name!r}, needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None:
            self.sliding_window_layer_stride = (
                1 if (self.sliding_window_layer_placing is None or self.sliding_window_layer_placing == "all") else 2
            )

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @property
    def mlp_class(self) -> Type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        import src.miniomni3.model
        return getattr(src.miniomni3.model, self.mlp_class_name)

    @property
    def norm_class(self) -> Type:
        # `self.norm_class_name` cannot be the type to keep the config serializable

        from functools import partial

        if self.norm_class_name == "RMSNorm":

            from src.miniomni3.model import RMSNorm

            return partial(RMSNorm, add_unit_offset="Gemma" in self.name)

        if self.norm_class_name == "LayerNorm" and "OLMo" in self.name:
            # this makes it equivalent to `torch.nn.functional.layer_norm`
            # that is used by OLMo
            # Table 5 caption in the OLMo paper shows this - https://aclanthology.org/2024.acl-long.841
            return partial(torch.nn.LayerNorm, elementwise_affine=False)

        return getattr(torch.nn, self.norm_class_name)

