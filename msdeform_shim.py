"""Pure-PyTorch drop-in for the `MultiScaleDeformableAttention` CUDA op.

X-Pose's `models/UniPose/ops/functions/ms_deform_attn_func.py` does
`import MultiScaleDeformableAttention as MSDA` at module import time. The
prebuilt CUDA op in that repo fails to compile against modern PyTorch
(`at::DeprecatedTypeProperties` -> `c10::ScalarType`). Rather than patching
the vendored source, we register a fake module in `sys.modules` *before*
the X-Pose import happens. It forwards to a `grid_sample`-based fallback,
which X-Pose itself already ships as `ms_deform_attn_core_pytorch`.

Forward-only. Inference nodes do not need a backward pass.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F


_FAST_MOD = None
_FAST_LOAD_ERR = None


def _load_fast_mod():
    global _FAST_MOD, _FAST_LOAD_ERR
    if _FAST_MOD is not None or _FAST_LOAD_ERR is not None:
        return _FAST_MOD
    path = "/workspace/ms_deform_attn_fast.py"
    try:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location("ms_deform_attn_fast", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FAST_MOD = mod
        print(f"[ComfyUI-XPose shim] loaded fast deform-attn from {path}")
    except Exception as e:
        _FAST_LOAD_ERR = e
        print(f"[ComfyUI-XPose shim] fast deform-attn unavailable ({e!r}); using grid_sample fallback")
    return _FAST_MOD


def _grid_sample_loop_forward(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    N_, S_, M_, D_ = value.shape
    _, Lq_, M2_, L_, P_, _ = sampling_locations.shape
    assert M_ == M2_

    shapes = value_spatial_shapes.tolist()
    split_sizes = [int(H_) * int(W_) for H_, W_ in shapes]
    value_list = value.split(split_sizes, dim=1)

    sampling_grids = 2.0 * sampling_locations - 1.0
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(shapes):
        H_, W_ = int(H_), int(W_)
        value_l_ = (
            value_list[lid_]
            .flatten(2)
            .transpose(1, 2)
            .reshape(N_ * M_, D_, H_, W_)
        )
        sampling_grid_l_ = (
            sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        )
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


_GATHER_FAILED = True  # gather path is slower than batched grid_sample on H200 — skip


def _ms_deform_attn_forward(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    global _GATHER_FAILED
    fast = _load_fast_mod()
    if fast is not None and not _GATHER_FAILED:
        try:
            return fast.multi_scale_deformable_attn_gather(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
            )
        except Exception as e:
            _GATHER_FAILED = True
            print(f"[ComfyUI-XPose shim] gather path failed ({e!r}); falling back to batched grid_sample")
    if fast is not None:
        try:
            return fast.multi_scale_deformable_attn_fast(
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights,
            )
        except Exception as e:
            print(f"[ComfyUI-XPose shim] batched grid_sample failed ({e!r}); falling back to per-level loop")
    return _grid_sample_loop_forward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )


def _ms_deform_attn_backward(*args, **kwargs):
    raise RuntimeError(
        "ComfyUI-XPose shim: backward pass not implemented (inference-only)"
    )


def install() -> None:
    if "MultiScaleDeformableAttention" in sys.modules:
        return
    mod = types.ModuleType("MultiScaleDeformableAttention")
    mod.ms_deform_attn_forward = _ms_deform_attn_forward
    mod.ms_deform_attn_backward = _ms_deform_attn_backward
    mod._xpose_shim = True  # marker so runtime can tell shim from real op
    sys.modules["MultiScaleDeformableAttention"] = mod
