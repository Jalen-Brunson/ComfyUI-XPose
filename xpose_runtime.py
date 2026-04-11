"""Runtime setup for the vendored X-Pose source tree.

Must be imported before anything inside `xpose_src/` is touched. Installs the
MSDeformAttn pytorch shim (F.grid_sample) and adds `xpose_src/` to sys.path
so the repo's top-level imports resolve.

Why the shim and not the vendored CUDA op: on Hopper (H200) the modern
`F.grid_sample` kernel is ~12% faster than X-Pose's 2020-era custom CUDA
implementation. A prebuilt `.so` from `xpose_src/models/UniPose/ops/` can be
loaded by setting the env var COMFYUI_XPOSE_USE_CUDA_OP=1, but it's not the
default because benchmarks showed no benefit.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys

from . import msdeform_shim

_XPOSE_REPO_URL = "https://github.com/IDEA-Research/X-Pose.git"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_XPOSE_SRC = os.path.join(_THIS_DIR, "xpose_src")
_OPS_DIR = os.path.join(_XPOSE_SRC, "models", "UniPose", "ops")

_READY = False
_USING_REAL_CUDA_OP = False


def _try_load_real_cuda_op() -> bool:
    """Attempt to import the compiled MSDeformAttn CUDA extension."""
    if "MultiScaleDeformableAttention" in sys.modules:
        mod = sys.modules["MultiScaleDeformableAttention"]
        return hasattr(mod, "ms_deform_attn_forward") and not getattr(
            mod, "_xpose_shim", False
        )
    import torch  # noqa: F401  # required so libc10.so is on the loader path
    if _OPS_DIR not in sys.path:
        sys.path.insert(0, _OPS_DIR)
    try:
        importlib.import_module("MultiScaleDeformableAttention")
    except Exception as e:
        print(f"[ComfyUI-XPose] real CUDA op unavailable ({e}); using pytorch shim", flush=True)
        return False
    mod = sys.modules["MultiScaleDeformableAttention"]
    if not hasattr(mod, "ms_deform_attn_forward"):
        print("[ComfyUI-XPose] CUDA op loaded but missing symbols; using shim", flush=True)
        return False
    return True


def _ensure_xpose_src() -> None:
    """Clone IDEA-Research/X-Pose into xpose_src/ if missing.

    We don't ship X-Pose's source in this repo because its license is
    non-commercial research only and we'd rather not re-distribute it.
    """
    marker = os.path.join(_XPOSE_SRC, "inference_on_a_image.py")
    if os.path.isfile(marker):
        return
    print(
        f"[ComfyUI-XPose] X-Pose source missing, cloning from {_XPOSE_REPO_URL} "
        f"into {_XPOSE_SRC} ...",
        flush=True,
    )
    os.makedirs(os.path.dirname(_XPOSE_SRC), exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", _XPOSE_REPO_URL, _XPOSE_SRC],
            check=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to clone IDEA-Research/X-Pose. Either clone it manually "
            f"into {_XPOSE_SRC} or make sure `git` is on PATH. Error: {e}"
        ) from e
    print("[ComfyUI-XPose] X-Pose source cloned", flush=True)


def ensure_ready() -> None:
    global _READY, _USING_REAL_CUDA_OP
    if _READY:
        return
    _ensure_xpose_src()
    if os.environ.get("COMFYUI_XPOSE_USE_CUDA_OP") == "1" and _try_load_real_cuda_op():
        _USING_REAL_CUDA_OP = True
        print("[ComfyUI-XPose] MSDeformAttn CUDA op loaded (opt-in)", flush=True)
    else:
        msdeform_shim.install()
        _USING_REAL_CUDA_OP = False
    if _XPOSE_SRC not in sys.path:
        sys.path.insert(0, _XPOSE_SRC)
    _READY = True


def using_real_cuda_op() -> bool:
    return _USING_REAL_CUDA_OP


def xpose_src_dir() -> str:
    return _XPOSE_SRC
