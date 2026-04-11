"""ComfyUI-XPose — open-vocabulary keypoint detection via IDEA-Research/X-Pose."""
from __future__ import annotations

import sys
import traceback

from . import xpose_runtime

_BANNER = """
================================================================================
ComfyUI-XPose failed to load: {err_type}: {err_msg}

{hint}

Run this in your ComfyUI environment to install everything this node needs:

    pip install -r {req_path}

Or on RunPod / managed ComfyUI, rerun your custom-node install hook. See
https://github.com/Jalen-Brunson/ComfyUI-XPose#installation for details.
================================================================================
"""


def _hint_for(err: BaseException) -> str:
    msg = str(err).lower()
    if "clip" in msg or "openai" in msg:
        return (
            "Cause: OpenAI CLIP (openai-clip) is not installed. It ships only "
            "as a git package, not on PyPI by that name."
        )
    if "ftfy" in msg or "regex" in msg:
        return "Cause: CLIP's deps (ftfy / regex) are missing."
    if "timm" in msg:
        return "Cause: timm is not installed (needed by X-Pose's Swin backbone)."
    if "addict" in msg or "yapf" in msg:
        return "Cause: X-Pose's config loader needs addict and yapf."
    if "huggingface_hub" in msg:
        return "Cause: huggingface_hub is missing (used for auto-downloading the checkpoint)."
    return "Cause: a required dependency is missing."


try:
    xpose_runtime.ensure_ready()
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except (ImportError, ModuleNotFoundError) as _e:
    import os as _os

    _req = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "requirements.txt")
    print(
        _BANNER.format(
            err_type=type(_e).__name__,
            err_msg=_e,
            hint=_hint_for(_e),
            req_path=_req,
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
    # Empty mappings: ComfyUI logs the failure but doesn't crash.
    NODE_CLASS_MAPPINGS: dict = {}  # type: ignore[no-redef]
    NODE_DISPLAY_NAME_MAPPINGS: dict = {}  # type: ignore[no-redef]

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
