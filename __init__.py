"""ComfyUI-XPose — open-vocabulary keypoint detection via IDEA-Research/X-Pose."""
from . import xpose_runtime

xpose_runtime.ensure_ready()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
