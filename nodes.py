"""ComfyUI nodes wrapping IDEA-Research/X-Pose (ECCV 2024).

Two nodes:
- `XPoseModelLoader`  : load the UniPose SwinT checkpoint + embedded CLIP.
- `XPoseEstimator`    : run keypoint detection on an IMAGE batch and emit
                        a rendered pose IMAGE batch + POSE_KEYPOINT JSON.

The estimator supports X-Pose's core selling point: open-vocabulary keypoint
detection on arbitrary categories (person, animal, fly, car, chair, ...),
with optional multi-pass body+face+hand for person-style workflows.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from . import xpose_runtime

xpose_runtime.ensure_ready()

# These imports must happen AFTER ensure_ready().
from util.config import Config  # type: ignore  # noqa: E402
from util.utils import clean_state_dict  # type: ignore  # noqa: E402
from models import build_model  # type: ignore  # noqa: E402
import predefined_keypoints as _xpose_kpts  # type: ignore  # noqa: E402
import transforms as _xpose_T  # type: ignore  # noqa: E402
from torchvision.ops import nms  # noqa: E402
from util import box_ops  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

def _collect_presets() -> dict[str, dict]:
    presets: dict[str, dict] = {}
    for name, val in vars(_xpose_kpts).items():
        if name.startswith("_"):
            continue
        if isinstance(val, dict) and "keypoints" in val and "skeleton" in val:
            presets[name] = val
    return presets


CATEGORY_PRESETS = _collect_presets()
CATEGORY_NAMES = sorted(CATEGORY_PRESETS.keys())


# ---------------------------------------------------------------------------
# Model directory registration with ComfyUI
# ---------------------------------------------------------------------------

try:
    import folder_paths  # type: ignore

    _MODELS_DIR = os.path.join(folder_paths.models_dir, "xpose")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    if "xpose" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["xpose"] = (
            [_MODELS_DIR],
            folder_paths.supported_pt_extensions,
        )
    else:
        existing = folder_paths.folder_names_and_paths["xpose"][0]
        if _MODELS_DIR not in existing:
            existing.append(_MODELS_DIR)
except Exception:
    folder_paths = None  # type: ignore
    _MODELS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints"
    )
    os.makedirs(_MODELS_DIR, exist_ok=True)


_CONFIG_PATH = os.path.join(
    xpose_runtime.xpose_src_dir(), "config_model", "UniPose_SwinT.py"
)

_DOWNLOAD_HINT = (
    "\n\nDownload the UniPose SwinT checkpoint (unipose_swint.pth) from "
    "https://github.com/IDEA-Research/X-Pose#model-zoo "
    "(Google Drive / OpenXLab) and place it at:\n  {path}\n"
)

# Unofficial HF mirror of IDEA's unipose_swint.pth. NOT maintained by IDEA.
# Auto-download is opt-out via env var so nothing happens behind a user's back
# if they prefer to fetch the official copy themselves.
_HF_MIRROR_REPO = "OwlMaster/XPose"
_HF_MIRROR_FILE = "unipose_swint.pth"


def _maybe_download_checkpoint(target_path: str) -> bool:
    """Try to fetch unipose_swint.pth from the HF mirror if missing.

    Returns True if the file is present after the call (either pre-existing
    or freshly downloaded). Set COMFYUI_XPOSE_NO_AUTODOWNLOAD=1 to disable.
    """
    if os.path.isfile(target_path):
        return True
    if os.environ.get("COMFYUI_XPOSE_NO_AUTODOWNLOAD") == "1":
        return False
    if os.path.basename(target_path) != _HF_MIRROR_FILE:
        return False  # don't auto-fetch arbitrary filenames
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:
        print(
            f"[ComfyUI-XPose] huggingface_hub not available ({e}); "
            "skipping auto-download",
            flush=True,
        )
        return False
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    print(
        f"[ComfyUI-XPose] checkpoint missing at {target_path}\n"
        f"[ComfyUI-XPose] auto-downloading from unofficial HF mirror "
        f"{_HF_MIRROR_REPO}/{_HF_MIRROR_FILE} (~1.3 GB, one-time)...\n"
        f"[ComfyUI-XPose]   (set COMFYUI_XPOSE_NO_AUTODOWNLOAD=1 to disable)",
        flush=True,
    )
    try:
        downloaded = hf_hub_download(
            repo_id=_HF_MIRROR_REPO,
            filename=_HF_MIRROR_FILE,
            local_dir=target_dir,
        )
    except Exception as e:
        print(f"[ComfyUI-XPose] auto-download failed: {e}", flush=True)
        return False
    # hf_hub_download may place the file under its own filename; ensure it
    # ends up at the exact target path.
    if os.path.abspath(downloaded) != os.path.abspath(target_path):
        try:
            os.replace(downloaded, target_path)
        except Exception:
            pass
    ok = os.path.isfile(target_path)
    if ok:
        print(f"[ComfyUI-XPose] checkpoint ready at {target_path}", flush=True)
    return ok


def _list_checkpoints() -> list[str]:
    if folder_paths is not None:
        try:
            files = folder_paths.get_filename_list("xpose")
            if files:
                return sorted(files)
        except Exception:
            pass
    if not os.path.isdir(_MODELS_DIR):
        return []
    return sorted(
        f
        for f in os.listdir(_MODELS_DIR)
        if f.endswith((".pth", ".pt", ".bin", ".safetensors"))
    )


def _resolve_checkpoint(name: str) -> str:
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("xpose", name)
            if path:
                return path
        except Exception:
            pass
    return os.path.join(_MODELS_DIR, name)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@dataclass
class XPoseBundle:
    model: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    config_path: str
    checkpoint_path: str
    text_cache: dict = field(default_factory=dict)


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


_NORMALIZE_CACHE: dict[int, Any] = {}


def _get_normalize(short_edge: int):
    se = int(short_edge)
    cached = _NORMALIZE_CACHE.get(se)
    if cached is None:
        cached = _xpose_T.Compose(
            [
                _xpose_T.RandomResize([se], max_size=1333),
                _xpose_T.ToTensor(),
                _xpose_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        _NORMALIZE_CACHE[se] = cached
    return cached


def _prepare_input(pil: Image.Image, short_edge: int = 800) -> torch.Tensor:
    tensor, _ = _get_normalize(short_edge)(pil, None)
    return tensor


@contextmanager
def _noop_ctx():
    yield


@contextmanager
def _backbone_cache(model: torch.nn.Module):
    """Cache Swin backbone output during a multi-pass sequence on the same frames.

    X-Pose runs the full backbone for each text-prompted pass even though the
    image features are identical across passes. Monkey-patching
    `model.backbone.forward` with a keyed cache lets the 2nd/3rd pass reuse the
    first pass's features and skip the Swin forward entirely. Restored on exit.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        yield
        return
    orig_forward = backbone.forward
    cache: dict[tuple, Any] = {}

    def cached_forward(samples):
        t = samples.tensors if hasattr(samples, "tensors") else samples
        key = (t.data_ptr(), tuple(t.shape))
        hit = cache.get(key)
        if hit is not None:
            return hit
        result = orig_forward(samples)
        cache[key] = result
        return result

    backbone.forward = cached_forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        backbone.forward = orig_forward  # type: ignore[method-assign]


def _build_target(
    model: torch.nn.Module,
    instance_names: tuple[str, ...],
    keypoint_names: tuple[str, ...],
    device: torch.device,
) -> dict[str, Any]:
    import clip as _clip  # type: ignore

    ins_embs = []
    for cat in instance_names:
        desc = f"a photo of {cat.lower().replace('_', ' ').replace('-', ' ')}"
        tok = _clip.tokenize(desc).to(device)
        with torch.no_grad():
            feat = model.clip_model.encode_text(tok)
        ins_embs.append(feat)
    ins_embs = torch.cat(ins_embs, dim=0).float()

    kpt_embs = []
    for name in keypoint_names:
        desc = f"a photo of {name.lower().replace('_', ' ')}"
        tok = _clip.tokenize(desc).to(device)
        with torch.no_grad():
            feat = model.clip_model.encode_text(tok)
        kpt_embs.append(feat)
    kpt_embs = torch.cat(kpt_embs, dim=0).float()

    pad_len = 100 - kpt_embs.shape[0]
    if pad_len < 0:
        raise ValueError(
            f"X-Pose supports at most 100 keypoints per category; got {kpt_embs.shape[0]}"
        )
    kpt_pad = torch.zeros(pad_len, 512, device=device)
    kpt_vis = torch.cat(
        [
            torch.ones(kpt_embs.shape[0], device=device),
            torch.zeros(pad_len, device=device),
        ]
    )
    return {
        "instance_text_prompt": list(instance_names),
        "keypoint_text_prompt": list(keypoint_names),
        "object_embeddings_text": ins_embs,
        "kpts_embeddings_text": torch.cat([kpt_embs, kpt_pad], dim=0),
        "kpt_vis_text": kpt_vis,
    }


def _get_or_build_target(
    bundle: XPoseBundle,
    instance_names: list[str],
    keypoint_names: list[str],
) -> dict[str, Any]:
    key = (tuple(instance_names), tuple(keypoint_names))
    cached = bundle.text_cache.get(key)
    if cached is None:
        cached = _build_target(bundle.model, key[0], key[1], bundle.device)
        bundle.text_cache[key] = cached
    return cached


def _iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two [N,4] / [M,4] cxcywh tensors. Returns [N,M]."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(a.shape[0], b.shape[0])
    ax1 = a[:, 0] - a[:, 2] / 2
    ay1 = a[:, 1] - a[:, 3] / 2
    ax2 = a[:, 0] + a[:, 2] / 2
    ay2 = a[:, 1] + a[:, 3] / 2
    bx1 = b[:, 0] - b[:, 2] / 2
    by1 = b[:, 1] - b[:, 3] / 2
    bx2 = b[:, 0] + b[:, 2] / 2
    by2 = b[:, 1] + b[:, 3] / 2
    ix1 = torch.maximum(ax1[:, None], bx1[None, :])
    iy1 = torch.maximum(ay1[:, None], by1[None, :])
    ix2 = torch.minimum(ax2[:, None], bx2[None, :])
    iy2 = torch.minimum(ay2[:, None], by2[None, :])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_a = ((ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0))[:, None]
    area_b = ((bx2 - bx1).clamp(min=0) * (by2 - by1).clamp(min=0))[None, :]
    union = area_a + area_b - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


class _Tracker:
    """Per-category greedy IoU tracker with EMA smoothing and dropout hold."""

    def __init__(self, iou_thresh: float, alpha: float, hold_frames: int):
        self.iou_thresh = float(iou_thresh)
        self.alpha = float(alpha)  # weight of previous value (0 = no smoothing)
        self.hold_frames = int(hold_frames)
        self._tracks: list[dict] = []  # {box:[4], kpts:[K,3], missing:int, id:int}
        self._next_id = 0

    def update(
        self, boxes: torch.Tensor, kpts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = boxes.shape[0]
        prev = self._tracks
        prev_boxes = (
            torch.stack([t["box"] for t in prev]) if prev else torch.zeros(0, 4)
        )

        match_new_to_prev: dict[int, int] = {}
        match_prev_to_new: dict[int, int] = {}
        if N > 0 and prev_boxes.shape[0] > 0:
            ious = _iou_cxcywh(boxes, prev_boxes)  # [N, P]
            flat = [
                (float(ious[i, j]), i, j)
                for i in range(ious.shape[0])
                for j in range(ious.shape[1])
                if float(ious[i, j]) >= self.iou_thresh
            ]
            flat.sort(key=lambda x: -x[0])
            used_new: set[int] = set()
            used_prev: set[int] = set()
            for _score, i, j in flat:
                if i in used_new or j in used_prev:
                    continue
                used_new.add(i)
                used_prev.add(j)
                match_new_to_prev[i] = j
                match_prev_to_new[j] = i

        a = self.alpha
        new_tracks: list[dict] = []
        # Matched + new (emitted in stable order: existing tracks first, then new)
        emit_boxes: list[torch.Tensor] = []
        emit_kpts: list[torch.Tensor] = []

        for j, t in enumerate(prev):
            if j in match_prev_to_new:
                i = match_prev_to_new[j]
                new_box = a * t["box"] + (1 - a) * boxes[i]
                # Only smooth visible keypoints; invisibles fall through
                new_kp = kpts[i].clone()
                vis = new_kp[:, 2] > 0
                prev_kp = t["kpts"]
                if vis.any():
                    new_kp[vis, :2] = a * prev_kp[vis, :2] + (1 - a) * kpts[i, vis, :2]
                t = {"box": new_box, "kpts": new_kp, "missing": 0, "id": t["id"]}
                new_tracks.append(t)
                emit_boxes.append(new_box)
                emit_kpts.append(new_kp)
            else:
                if t["missing"] + 1 <= self.hold_frames:
                    t = {
                        "box": t["box"],
                        "kpts": t["kpts"],
                        "missing": t["missing"] + 1,
                        "id": t["id"],
                    }
                    new_tracks.append(t)
                    emit_boxes.append(t["box"])
                    emit_kpts.append(t["kpts"])
                # else: drop the track silently

        for i in range(N):
            if i in match_new_to_prev:
                continue
            t = {
                "box": boxes[i].clone(),
                "kpts": kpts[i].clone(),
                "missing": 0,
                "id": self._next_id,
            }
            self._next_id += 1
            new_tracks.append(t)
            emit_boxes.append(t["box"])
            emit_kpts.append(t["kpts"])

        self._tracks = new_tracks

        if not emit_boxes:
            K = kpts.shape[1] if kpts.ndim == 3 else 0
            return torch.zeros(0, 4), torch.zeros(0, K, 3)
        return torch.stack(emit_boxes, dim=0), torch.stack(emit_kpts, dim=0)


def _postprocess_one(
    pred_logits_i: torch.Tensor,
    pred_boxes_i: torch.Tensor,
    pred_kpts_i: torch.Tensor,
    num_keypoints: int,
    box_threshold: float,
    iou_threshold: float,
    max_instances: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = pred_logits_i.float()
    boxes = pred_boxes_i.float()
    kpts_flat = pred_kpts_i[:, : 2 * num_keypoints].float()

    scores = logits.max(dim=1)[0]
    keep_mask = scores > float(box_threshold)
    boxes = boxes[keep_mask]
    kpts_flat = kpts_flat[keep_mask]
    scores = scores[keep_mask]

    if boxes.numel() == 0:
        return (
            torch.zeros(0, 4),
            torch.zeros(0, num_keypoints, 3),
            torch.zeros(0),
        )

    keep = nms(
        box_ops.box_cxcywh_to_xyxy(boxes), scores, iou_threshold=float(iou_threshold)
    )
    boxes = boxes[keep]
    kpts_flat = kpts_flat[keep]
    scores = scores[keep]

    if max_instances > 0 and scores.shape[0] > max_instances:
        order = torch.argsort(scores, descending=True)[:max_instances]
        boxes = boxes[order]
        kpts_flat = kpts_flat[order]
        scores = scores[order]

    kp = kpts_flat.view(-1, num_keypoints, 2)
    vis = torch.ones(kp.shape[0], num_keypoints, 1)
    kp_xyv = torch.cat([kp, vis], dim=-1)
    return boxes, kp_xyv, scores


def _forward_chunk(
    bundle: XPoseBundle,
    pil_frames: list[Image.Image],
    target: dict[str, Any],
    num_keypoints: int,
    box_threshold: float,
    iou_threshold: float,
    max_instances: int,
    short_edge: int = 800,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Batched forward pass. Returns per-frame (boxes, kp_xyv, scores)."""
    device = bundle.device
    normed = [_prepare_input(p, short_edge) for p in pil_frames]
    shapes = {tuple(t.shape) for t in normed}

    def _one_forward(stacked: torch.Tensor) -> dict:
        targets_list = [target] * stacked.shape[0]
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=bundle.dtype,
                enabled=bundle.dtype in (torch.float16, torch.bfloat16),
            ):
                return bundle.model(stacked, targets_list)

    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    if len(shapes) > 1:
        # Mixed-shape fallback: one frame at a time.
        for t in normed:
            outputs = _one_forward(t.unsqueeze(0).to(device))
            logits = outputs["pred_logits"].sigmoid()[0].cpu()
            boxes = outputs["pred_boxes"][0].cpu()
            kpts = outputs["pred_keypoints"][0].cpu()
            results.append(
                _postprocess_one(
                    logits, boxes, kpts, num_keypoints,
                    box_threshold, iou_threshold, max_instances,
                )
            )
        return results

    stacked = torch.stack(normed, dim=0).to(device)
    outputs = _one_forward(stacked)
    pred_logits = outputs["pred_logits"].sigmoid().cpu()
    pred_boxes = outputs["pred_boxes"].cpu()
    pred_kpts = outputs["pred_keypoints"].cpu()

    for i in range(stacked.shape[0]):
        results.append(
            _postprocess_one(
                pred_logits[i], pred_boxes[i], pred_kpts[i],
                num_keypoints, box_threshold, iou_threshold, max_instances,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PALETTE = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]


def _color(i: int) -> tuple[int, int, int]:
    return _PALETTE[i % len(_PALETTE)]


def _draw_detection(
    canvas: Image.Image,
    boxes: torch.Tensor,
    kpts: torch.Tensor,
    skeleton: list[list[int]],
    draw_skeleton: bool,
    draw_keypoints: bool,
    draw_bboxes: bool,
    line_thickness: int,
    point_radius: int,
    min_kpt_conf: float,
) -> None:
    W, H = canvas.size
    draw = ImageDraw.Draw(canvas)

    sks = np.array(skeleton) if skeleton else np.zeros((0, 2), dtype=int)
    if sks.size and sks.min() == 1:
        sks = sks - 1

    for i, box in enumerate(boxes):
        cx, cy, bw, bh = box.tolist()
        x0 = (cx - bw / 2) * W
        y0 = (cy - bh / 2) * H
        x1 = (cx + bw / 2) * W
        y1 = (cy + bh / 2) * H
        color = _color(i)

        if draw_bboxes:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=max(1, line_thickness))

        kp = kpts[i]
        pts_px = [(float(k[0]) * W, float(k[1]) * H, float(k[2])) for k in kp]

        if draw_skeleton and sks.size:
            for sk in sks:
                a, b = int(sk[0]), int(sk[1])
                if a >= len(pts_px) or b >= len(pts_px):
                    continue
                xa, ya, va = pts_px[a]
                xb, yb, vb = pts_px[b]
                if va < min_kpt_conf or vb < min_kpt_conf:
                    continue
                draw.line([xa, ya, xb, yb], fill=color, width=max(1, line_thickness))

        if draw_keypoints:
            for k_idx, (x, y, v) in enumerate(pts_px):
                if v < min_kpt_conf:
                    continue
                r = max(1, point_radius)
                kc = _color(k_idx + 3)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=kc, outline=(0, 0, 0))


# ---------------------------------------------------------------------------
# POSE_KEYPOINT (OpenPose-style JSON) export
# ---------------------------------------------------------------------------

def _to_openpose_json(
    width: int,
    height: int,
    per_category: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict:
    """Emit a DWPose-compatible POSE_KEYPOINT frame dict.

    When `person` body + optional hand/face passes are present, this maps into
    OpenPose's canonical slots (18-pt body, 21-pt hands, 68-pt face). Other
    categories are emitted under the `custom` slot where downstream nodes can
    inspect them if needed. This is intentionally lossy for non-person
    categories — most OpenPose-consuming nodes only look at the person slots.
    """
    people: list[dict] = []

    body_boxes, body_kpts = per_category.get("person", (torch.zeros(0, 4), torch.zeros(0, 0, 3)))
    face_boxes, face_kpts = per_category.get("face", (torch.zeros(0, 4), torch.zeros(0, 0, 3)))
    hand_boxes, hand_kpts = per_category.get("hand", (torch.zeros(0, 4), torch.zeros(0, 0, 3)))

    N = body_boxes.shape[0]
    if N == 0 and face_boxes.shape[0] == 0 and hand_boxes.shape[0] == 0:
        return {"version": "xpose_1.0", "people": [], "canvas_width": width, "canvas_height": height}

    def _flat(kp: torch.Tensor, scale_x: float, scale_y: float) -> list[float]:
        out: list[float] = []
        for k in kp:
            out.extend([float(k[0]) * scale_x, float(k[1]) * scale_y, float(k[2])])
        return out

    face_used = [False] * face_boxes.shape[0]
    hand_used = [False] * hand_boxes.shape[0]

    def _box_center(b: torch.Tensor) -> tuple[float, float]:
        return float(b[0]), float(b[1])

    def _nearest(target_xy: tuple[float, float], boxes: torch.Tensor, used: list[bool]) -> int:
        best, best_d = -1, math.inf
        for j, b in enumerate(boxes):
            if used[j]:
                continue
            cx, cy = _box_center(b)
            d = (cx - target_xy[0]) ** 2 + (cy - target_xy[1]) ** 2
            if d < best_d:
                best_d = d
                best = j
        return best

    for i in range(N):
        person: dict[str, Any] = {}
        person["pose_keypoints_2d"] = _flat(body_kpts[i], width, height)

        # Associate nearest face to this person by box center.
        face_idx = -1
        if face_boxes.shape[0] > 0:
            head_xy = _box_center(body_boxes[i])  # approximate head with body center
            # Prefer top of box for head
            head_xy = (float(body_boxes[i][0]), float(body_boxes[i][1] - body_boxes[i][3] / 3))
            face_idx = _nearest(head_xy, face_boxes, face_used)
        if face_idx >= 0:
            face_used[face_idx] = True
            person["face_keypoints_2d"] = _flat(face_kpts[face_idx], width, height)

        # Associate up to 2 nearest hands (left/right) by wrist position.
        if hand_boxes.shape[0] > 0 and body_kpts.shape[1] >= 11:
            # COCO-17: idx 9 = left wrist, 10 = right wrist
            for target_idx, slot in ((9, "hand_left_keypoints_2d"), (10, "hand_right_keypoints_2d")):
                if body_kpts[i, target_idx, 2] <= 0:
                    continue
                wx = float(body_kpts[i, target_idx, 0])
                wy = float(body_kpts[i, target_idx, 1])
                h_idx = _nearest((wx, wy), hand_boxes, hand_used)
                if h_idx >= 0:
                    hand_used[h_idx] = True
                    person[slot] = _flat(hand_kpts[h_idx], width, height)
        people.append(person)

    return {
        "version": "xpose_1.0",
        "people": people,
        "canvas_width": width,
        "canvas_height": height,
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class XPoseModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        ckpts = _list_checkpoints()
        if not ckpts:
            ckpts = ["<place checkpoint in models/xpose/>"]
        return {
            "required": {
                "checkpoint": (ckpts,),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("XPOSE_MODEL",)
    RETURN_NAMES = ("xpose_model",)
    FUNCTION = "load"
    CATEGORY = "XPose"

    def load(
        self,
        checkpoint: str,
        precision: str,
        device: str,
    ):
        ckpt_path = _resolve_checkpoint(checkpoint)
        if not os.path.isfile(ckpt_path):
            # If user picked the default name, try the HF mirror auto-download.
            target = (
                ckpt_path
                if os.path.basename(ckpt_path) == _HF_MIRROR_FILE
                else os.path.join(_MODELS_DIR, _HF_MIRROR_FILE)
            )
            if _maybe_download_checkpoint(target):
                ckpt_path = target
            else:
                raise FileNotFoundError(
                    f"X-Pose checkpoint not found: {ckpt_path}"
                    + _DOWNLOAD_HINT.format(
                        path=os.path.join(_MODELS_DIR, _HF_MIRROR_FILE)
                    )
                )

        cfg = Config.fromfile(_CONFIG_PATH)
        cfg.device = device
        # Inference-only: training-time gradient checkpointing only wastes compute.
        cfg.use_checkpoint = False
        cfg.use_transformer_ckpt = False
        model = build_model(cfg)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        res = model.load_state_dict(clean_state_dict(state), strict=False)
        missing = getattr(res, "missing_keys", [])
        unexpected = getattr(res, "unexpected_keys", [])
        if missing:
            print(f"[ComfyUI-XPose] missing keys: {len(missing)} (e.g. {missing[:3]})", flush=True)
        if unexpected:
            print(f"[ComfyUI-XPose] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})", flush=True)
        model.eval()

        torch_device = torch.device(device)
        model.to(torch_device)

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map[precision]

        bundle = XPoseBundle(
            model=model,
            device=torch_device,
            dtype=dtype,
            config_path=_CONFIG_PATH,
            checkpoint_path=ckpt_path,
        )
        return (bundle,)


class XPoseEstimator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xpose_model": ("XPOSE_MODEL",),
                "image": ("IMAGE",),
                "category_preset": (CATEGORY_NAMES, {"default": "person"}),
                "detect_body": ("BOOLEAN", {"default": True}),
                "detect_face": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Extra pass with category=face (68 pts)"},
                ),
                "detect_hands": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Extra pass with category=hand (21 pts)"},
                ),
                "detect_feet": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "X-Pose person skeleton ends at ankles. "
                            "Enable only if you add toe keypoints via custom_keypoint_prompt."
                        ),
                    },
                ),
                "box_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "iou_threshold": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_instances": ("INT", {"default": 0, "min": 0, "max": 100}),
                "min_keypoint_confidence": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "render_mode": (["openpose_style", "on_source", "keypoints_only"],),
                "draw_skeleton": ("BOOLEAN", {"default": True}),
                "draw_keypoints": ("BOOLEAN", {"default": True}),
                "draw_bboxes": ("BOOLEAN", {"default": False}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 16}),
                "point_radius": ("INT", {"default": 3, "min": 1, "max": 16}),
                "temporal_smoothing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Track detections across frames by bbox IoU and EMA-smooth "
                            "keypoints. Single images: leave off."
                        ),
                    },
                ),
                "smoothing_strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": "0 = no smoothing, higher = more temporal inertia.",
                    },
                ),
                "tracking_iou": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Min bbox IoU to match a detection to an existing track.",
                    },
                ),
                "hold_on_dropout_frames": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 30,
                        "tooltip": "Keep a track alive at last position for N missing frames.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "tooltip": (
                            "Frames per model forward pass. 4-8 is a good sweet spot on "
                            "H200; drop to 1 if OOM."
                        ),
                    },
                ),
                "resize_short_edge": (
                    "INT",
                    {
                        "default": 800,
                        "min": 320,
                        "max": 1333,
                        "step": 16,
                        "tooltip": (
                            "Short-edge resize target. X-Pose default is 800. "
                            "Drop to 560 or 480 for ~2-3x speed at the cost of "
                            "small-face/hand precision."
                        ),
                    },
                ),
                "cache_backbone_across_passes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Run the Swin backbone once per frame and reuse its "
                            "features across body/face/hand passes. ~1.5x faster "
                            "for multi-pass, identical results."
                        ),
                    },
                ),
            },
            "optional": {
                "custom_instance_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Overrides category_preset's instance prompt if non-empty.",
                    },
                ),
                "custom_keypoint_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Comma-separated keypoint names to override the preset's list. "
                            "Leave empty to use the preset's keypoints."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("pose_image", "pose_keypoint")
    FUNCTION = "estimate"
    CATEGORY = "XPose"

    def _resolve_category(
        self,
        preset_name: str,
        custom_instance: str,
        custom_keypoints: str,
    ) -> tuple[str, list[str], list[list[int]]]:
        preset = CATEGORY_PRESETS.get(preset_name, CATEGORY_PRESETS["person"])
        instance = custom_instance.strip() or preset_name.replace("_", " ")
        if custom_keypoints.strip():
            kpts = [s.strip() for s in custom_keypoints.split(",") if s.strip()]
            skeleton: list[list[int]] = []
        else:
            kpts = list(preset["keypoints"])
            skeleton = [list(s) for s in preset["skeleton"]]
        return instance, kpts, skeleton

    def estimate(
        self,
        xpose_model: XPoseBundle,
        image: torch.Tensor,
        category_preset: str,
        detect_body: bool,
        detect_face: bool,
        detect_hands: bool,
        detect_feet: bool,
        box_threshold: float,
        iou_threshold: float,
        max_instances: int,
        min_keypoint_confidence: float,
        render_mode: str,
        draw_skeleton: bool,
        draw_keypoints: bool,
        draw_bboxes: bool,
        line_thickness: int,
        point_radius: int,
        temporal_smoothing: bool = False,
        smoothing_strength: float = 0.6,
        tracking_iou: float = 0.3,
        hold_on_dropout_frames: int = 2,
        batch_size: int = 1,
        resize_short_edge: int = 800,
        cache_backbone_across_passes: bool = True,
        custom_instance_prompt: str = "",
        custom_keypoint_prompt: str = "",
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor of shape [B,H,W,C], got {tuple(image.shape)}")

        is_person_preset = category_preset == "person"

        main_instance, main_kpts, main_skel = self._resolve_category(
            category_preset, custom_instance_prompt, custom_keypoint_prompt
        )

        passes: list[tuple[str, str, list[str], list[list[int]]]] = []
        if detect_body or not is_person_preset:
            passes.append(("person" if is_person_preset else category_preset,
                           main_instance, main_kpts, main_skel))
        if detect_face and "face" in CATEGORY_PRESETS:
            p = CATEGORY_PRESETS["face"]
            passes.append(("face", "face", list(p["keypoints"]), [list(s) for s in p["skeleton"]]))
        if detect_hands and "hand" in CATEGORY_PRESETS:
            p = CATEGORY_PRESETS["hand"]
            passes.append(("hand", "hand", list(p["keypoints"]), [list(s) for s in p["skeleton"]]))
        if detect_feet:
            print(
                "[ComfyUI-XPose] detect_feet: X-Pose has no predefined foot skeleton; "
                "using experimental custom prompt.",
                flush=True,
            )
            foot_kpts = ["left big toe", "left small toe", "left heel",
                         "right big toe", "right small toe", "right heel"]
            passes.append(("person", main_instance, foot_kpts, []))

        if not passes:
            raise ValueError("At least one of detect_body/face/hands must be enabled.")

        out_images: list[torch.Tensor] = []
        out_poses: list[dict] = []

        try:
            import comfy.utils  # type: ignore

            pbar = comfy.utils.ProgressBar(int(image.shape[0]))
        except Exception:
            pbar = None

        trackers: dict[int, _Tracker] = {}
        if temporal_smoothing:
            for pass_idx in range(len(passes)):
                trackers[pass_idx] = _Tracker(
                    iou_thresh=tracking_iou,
                    alpha=smoothing_strength,
                    hold_frames=hold_on_dropout_frames,
                )

        # Pre-build (and cache) text-prompt targets once per pass for the whole video.
        pass_targets: list[dict[str, Any]] = []
        for cat_key, instance, kpts, _skel in passes:
            instance_names = [s.strip() for s in instance.split(",") if s.strip()]
            if not instance_names:
                raise ValueError("instance_prompt must contain at least one category name")
            pass_targets.append(_get_or_build_target(xpose_model, instance_names, kpts))

        B = image.shape[0]
        bs = max(1, int(batch_size))
        log_every = max(1, B // 20) if B >= 20 else 1
        t_start = time.time()
        print(
            f"[ComfyUI-XPose] estimating {B} frame(s), passes={len(passes)}, "
            f"batch_size={bs}, short_edge={resize_short_edge}, "
            f"backbone_cache={'on' if cache_backbone_across_passes and len(passes) > 1 else 'off'}",
            flush=True,
        )

        frames_done = 0
        for chunk_start in range(0, B, bs):
            chunk_end = min(chunk_start + bs, B)
            chunk_pils = [_tensor_to_pil(image[b]) for b in range(chunk_start, chunk_end)]

            # Run every pass on the chunk. Backbone cache collapses the Swin
            # forward across the multi-pass sequence for identical image tensors.
            use_bb_cache = cache_backbone_across_passes and len(passes) > 1
            bb_ctx = (
                _backbone_cache(xpose_model.model)
                if use_bb_cache
                else _noop_ctx()
            )
            pass_chunk_results: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
            with bb_ctx:
                for p_idx, (_cat_key, _instance, kpts, _skel) in enumerate(passes):
                    chunk_dets = _forward_chunk(
                        xpose_model,
                        chunk_pils,
                        pass_targets[p_idx],
                        num_keypoints=len(kpts),
                        box_threshold=box_threshold,
                        iou_threshold=iou_threshold,
                        max_instances=max_instances,
                        short_edge=resize_short_edge,
                    )
                    pass_chunk_results.append(chunk_dets)

            # Post-process each frame in order: tracker, render, emit.
            for f_idx, pil in enumerate(chunk_pils):
                W, H = pil.size
                per_cat_results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                all_for_render: list[
                    tuple[torch.Tensor, torch.Tensor, list[list[int]]]
                ] = []

                for p_idx, (cat_key, _instance, _kpts, skel) in enumerate(passes):
                    boxes, kp_xyv, _scores = pass_chunk_results[p_idx][f_idx]
                    if temporal_smoothing:
                        boxes, kp_xyv = trackers[p_idx].update(boxes, kp_xyv)
                    if cat_key not in per_cat_results:
                        per_cat_results[cat_key] = (boxes, kp_xyv)
                    else:
                        eb, ek = per_cat_results[cat_key]
                        per_cat_results[cat_key] = (
                            torch.cat([eb, boxes], dim=0),
                            torch.cat([ek, kp_xyv], dim=0)
                            if ek.shape[1] == kp_xyv.shape[1]
                            else ek,
                        )
                    all_for_render.append((boxes, kp_xyv, skel))

                if render_mode == "openpose_style":
                    canvas = Image.new("RGB", (W, H), (0, 0, 0))
                elif render_mode == "on_source":
                    canvas = pil.copy()
                else:  # keypoints_only
                    canvas = Image.new("RGB", (W, H), (0, 0, 0))

                effective_skeleton = draw_skeleton and render_mode != "keypoints_only"

                for boxes, kp_xyv, skel in all_for_render:
                    if boxes.shape[0] == 0:
                        continue
                    _draw_detection(
                        canvas,
                        boxes,
                        kp_xyv,
                        skel,
                        draw_skeleton=effective_skeleton,
                        draw_keypoints=draw_keypoints,
                        draw_bboxes=draw_bboxes,
                        line_thickness=line_thickness,
                        point_radius=point_radius,
                        min_kpt_conf=min_keypoint_confidence,
                    )

                out_images.append(_pil_to_tensor(canvas))
                out_poses.append(_to_openpose_json(W, H, per_cat_results))

                frames_done += 1
                if pbar is not None:
                    pbar.update(1)
                if frames_done % log_every == 0 or frames_done == B:
                    elapsed = time.time() - t_start
                    fps = frames_done / max(elapsed, 1e-6)
                    eta = (B - frames_done) / max(fps, 1e-6)
                    print(
                        f"[ComfyUI-XPose] {frames_done}/{B}  {fps:.2f} fps  "
                        f"elapsed {elapsed:.1f}s  eta {eta:.1f}s",
                        flush=True,
                    )

        if out_images:
            heights = {t.shape[0] for t in out_images}
            widths = {t.shape[1] for t in out_images}
            if len(heights) == 1 and len(widths) == 1:
                batch = torch.stack(out_images, dim=0)
            else:
                target_h = max(t.shape[0] for t in out_images)
                target_w = max(t.shape[1] for t in out_images)
                padded = []
                for t in out_images:
                    h, w = t.shape[:2]
                    pad = torch.zeros(target_h, target_w, 3)
                    pad[:h, :w] = t
                    padded.append(pad)
                batch = torch.stack(padded, dim=0)
        else:
            batch = torch.zeros(1, image.shape[1], image.shape[2], 3)

        return (batch, out_poses)


NODE_CLASS_MAPPINGS = {
    "XPoseModelLoader": XPoseModelLoader,
    "XPoseEstimator": XPoseEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XPoseModelLoader": "X-Pose Model Loader",
    "XPoseEstimator": "X-Pose Estimator",
}
