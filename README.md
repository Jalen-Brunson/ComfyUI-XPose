# ComfyUI-XPose

ComfyUI custom nodes that wrap [IDEA-Research/X-Pose](https://github.com/IDEA-Research/X-Pose) (ECCV 2024) for open-vocabulary 2D keypoint detection on images and video batches.

X-Pose is a text-prompted, DETR-style keypoint detector. Unlike DWPose or YOLOv8-pose which are hard-wired to human skeletons, X-Pose takes an instance prompt (e.g. `"person"`, `"dog"`, `"fly"`, `"car"`) plus a list of keypoint names and localizes them. It ships with ~24 predefined categories and is especially strong on:

- Difficult human poses, occlusions, and unusual body orientations
- 68-point face landmarks and 21-point hand landmarks per side
- Animals: dogs, quadrupeds in general, insects, even fish
- Arbitrary objects via open-vocabulary prompting (clothing, furniture, vehicles)

## ⚠️ License — read this first

**This node wraps X-Pose, which is released under a "License for Non-commercial Scientific Research Purposes."** That license:

- Permits only non-commercial, non-military, non-surveillance, non-pornographic use
- Is revocable at IDEA's discretion
- Is not sublicensable

The wrapper code in this repo is MIT-licensed, but using the node as a whole means you're also bound by X-Pose's license. **Do not use this node in commercial work, advertising, or production pipelines that you bill for.** See [X-Pose's LICENSE.txt](https://github.com/IDEA-Research/X-Pose/blob/main/LICENSE.txt) for the full text.

If you need commercial-use keypoint detection, look at DWPose, YOLOv8-pose, or RTMPose instead.

## Installation

### Via ComfyUI-Manager

Search for "ComfyUI-XPose" and install. First run will auto-clone X-Pose and prompt for the checkpoint.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Jalen-Brunson/ComfyUI-XPose.git
cd ComfyUI-XPose
pip install -r requirements.txt
```

First time the node loads, it will auto-clone `IDEA-Research/X-Pose` into `xpose_src/` (via `git`). You need a working `git` on PATH.

### Checkpoint

Download `unipose_swint.pth` (~1.3 GB) and place it in `ComfyUI/models/xpose/unipose_swint.pth`:

- **Official (slow, requires account)**: [OpenXLab IDEA-Research/UniPose](https://openxlab.org.cn/models/detail/IDEA-Research/UniPose) or the Google Drive link in the [X-Pose README](https://github.com/IDEA-Research/X-Pose#model-zoo)
- **Unofficial HuggingFace mirror**: `https://huggingface.co/OwlMaster/XPose/resolve/main/unipose_swint.pth` — convenient, but not maintained by IDEA; at your own risk

OpenAI CLIP's ViT-B/32 will also download automatically on first model build (~338 MB).

## Nodes

### X-Pose Model Loader

Loads the UniPose SwinT checkpoint and returns an `XPOSE_MODEL` bundle.

| Input | Type | Default | Notes |
|---|---|---|---|
| `checkpoint` | dropdown | — | populated from `ComfyUI/models/xpose/` |
| `precision` | choice | `fp16` | `fp32` / `fp16` / `bf16` |
| `device` | choice | `cuda` | |

### X-Pose Estimator

Runs detection on an `IMAGE` batch. Returns a rendered pose `IMAGE` batch plus `POSE_KEYPOINT` JSON (OpenPose-compatible, consumable by `controlnet_aux` and similar nodes).

#### Main inputs

| Input | Default | Notes |
|---|---|---|
| `xpose_model` | — | from loader |
| `image` | — | batch `[B, H, W, C]` |
| `category_preset` | `person` | 24 presets: `person`, `face`, `hand`, `animal`, `animal_in_AP10K`, `fly`, `locust`, `car`, `chair`, clothing items, etc. |
| `detect_body` | True | runs the primary preset pass |
| `detect_face` | False | extra pass with `face` preset (68 kpts) |
| `detect_hands` | False | extra pass with `hand` preset (21 kpts × 2) |
| `detect_feet` | False | experimental, no predefined toe skeleton |
| `box_threshold` | 0.10 | detection confidence cutoff |
| `iou_threshold` | 0.90 | NMS |
| `max_instances` | 0 | cap detections per frame (0 = no cap) |
| `min_keypoint_confidence` | 0.0 | drop low-confidence keypoints before drawing |
| `render_mode` | `openpose_style` | `openpose_style` / `on_source` / `keypoints_only` |
| `draw_skeleton` / `draw_keypoints` / `draw_bboxes` | — | render toggles |
| `line_thickness` / `point_radius` | — | render sizes |

#### Speed / quality knobs

| Input | Default | Notes |
|---|---|---|
| `batch_size` | 1 | frames per model forward; 4–8 is the sweet spot on a 24 GB+ GPU |
| `resize_short_edge` | 800 | X-Pose default. Drop to 560 or 480 for ~15–30% speedup; small-face/hand precision suffers |
| `cache_backbone_across_passes` | True | run Swin once per frame and reuse features across body/face/hand passes |

#### Temporal smoothing (for video)

| Input | Default | Notes |
|---|---|---|
| `temporal_smoothing` | False | IoU-track detections across frames + EMA smooth keypoints |
| `smoothing_strength` | 0.6 | 0 = no smoothing, higher = more inertia |
| `tracking_iou` | 0.3 | min bbox IoU to match a new detection to an existing track |
| `hold_on_dropout_frames` | 2 | carry a track forward at its last position for N missing frames |

#### Open-vocabulary inputs

| Input | Notes |
|---|---|
| `custom_instance_prompt` | Overrides the preset's instance name (e.g. `"dalmatian"`, `"jumping spider"`) |
| `custom_keypoint_prompt` | Comma-separated keypoint names to override the preset's list |

## Performance

Measured on H200 / 480×848 / 16 frames, `person` preset, fp16:

| config | fps |
|---|---|
| body-only, `short_edge=480`, `batch_size=8` | ~4.6 |
| body-only, `short_edge=800`, `batch_size=8` | ~4.0 |
| body + face + hands, `short_edge=480`, `batch_size=8`, backbone cache on | ~1.6 |
| body + face + hands, `short_edge=800`, `batch_size=8`, backbone cache off | ~1.2 |

Rough planning: at `short_edge=480` with backbone cache on, a 2500-frame 480p clip takes ~10 min body-only or ~25 min body+face+hands. Scale linearly with frame count.

## How this compares to DWPose / YOLO-pose

| | X-Pose | DWPose | YOLOv8-pose |
|---|---|---|---|
| Speed (H200, 480×848) | ~4 fps body-only | ~20+ fps | ~60+ fps |
| Human body keypoints | very good, robust to occlusion | good | good |
| Face landmarks | 68 pts, strong | 68 pts, noisier on hard poses | none |
| Hand landmarks | 21 pts per side, strong | 21 pts per side | none |
| Animal pose | yes, many species | no | no |
| Commercial use | **no** | restricted (CC-BY-NC-SA) | yes (AGPL-3.0) |

Use X-Pose when you need the best quality on difficult poses, face, and hands, or when you're working with animals or custom categories. Use YOLOv8-pose when speed matters more than face/hand detail or when you need commercial use.

## Tips

- **Animal videos**: set `category_preset=animal` (or `animal_in_AP10K` for quadrupeds), `detect_body=True`, `detect_face=False`, `detect_hands=False`, and use `custom_instance_prompt` to be specific (e.g. `"dog"`, `"cat"`). Don't turn on face/hand detection on animals — those presets are trained on humans.
- **Long videos**: turn on `temporal_smoothing` to kill frame-to-frame jitter. `smoothing_strength=0.6` is a good starting point; bump to `0.8` for very shaky outputs.
- **Multiple people**: bump `max_instances` to match. Tracker will keep identities stable across frames.
- **Dropouts on fast motion**: raise `hold_on_dropout_frames` to 3–5 so tracks survive brief misses.

## Known limitations

- Per-frame detector with no native temporal component. The tracker + smoother we wrap around it help, but they're post-processing — they can't recover information the model didn't see.
- X-Pose's 17-point `person` skeleton is COCO-17 style (stops at ankles). There are no toe or foot keypoints in the predefined presets.
- `detect_feet` uses an experimental custom keypoint prompt and is not validated.
- The MSDeformAttn CUDA op from X-Pose's vendored source won't build on modern PyTorch without a small patch. We ship a pure-PyTorch `grid_sample` fallback (`msdeform_shim.py`) that is slightly *faster* than the real op on H200. If you're on an older GPU where the custom kernel would win, set `COMFYUI_XPOSE_USE_CUDA_OP=1` and build `xpose_src/models/UniPose/ops/` manually.

## Acknowledgments

- [IDEA-Research/X-Pose](https://github.com/IDEA-Research/X-Pose) (Yang, Zeng, Zhang, Zhang — ECCV 2024)
- [OpenAI CLIP](https://github.com/openai/CLIP) (text encoder for open-vocab prompts)
- The [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) MSDeformAttn op
