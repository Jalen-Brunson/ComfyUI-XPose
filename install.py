"""Install hook for ComfyUI-Manager.

ComfyUI-Manager runs this file automatically after cloning the node pack.
We just pip-install everything in requirements.txt. The `clip` dep is a git
URL, which Manager's generic requirements handling sometimes skips — running
our own install step guarantees it lands.

The first time the node loads, xpose_runtime will also clone IDEA-Research/
X-Pose into xpose_src/ and the XPoseModelLoader will auto-download the
UniPose SwinT checkpoint if it's missing. No manual steps required.
"""
from __future__ import annotations

import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REQ = os.path.join(THIS_DIR, "requirements.txt")


def main() -> int:
    if not os.path.isfile(REQ):
        print(f"[ComfyUI-XPose install] requirements.txt missing at {REQ}")
        return 0
    print(f"[ComfyUI-XPose install] pip install -r {REQ}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", REQ],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
