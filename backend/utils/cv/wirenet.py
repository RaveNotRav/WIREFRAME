"""Optional WireNet v2.12 ensemble architecture metadata loader and binary wire mask refiner.

Reads `v212_ensemble.json`, `v212_convnext_t.arch.json`, `v212_mitb1.arch.json`, and training logs
from `backend/models/`. When the optional `.pt` weights (`v212_convnext_t.pt`, `v212_mitb1.pt`,
`wirenet-r34.pt`) and `segmentation_models_pytorch` are installed on disk, can refine coarse
YOLOv11-seg masks to full pixel resolution; when weights are absent, degrades gracefully without error.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.utils.cv.types import SegmentationResult, normalize_wire_class

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# WireNet class channel order (1-indexed; 0 is background)
WIRENET_CLASS_MAP: dict[str, int] = {
    "power_distribution_line": 1,
    "communication_cable": 2,
    "dead_wire": 3,
    "vegetation": 4,
}


def load_wirenet_ensemble_metadata(models_dir: str | Path | None = None) -> dict[str, Any]:
    """Load v2.12 WireNet ensemble specification, member architecture configs, and availability."""
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    ensemble_file = mdir / "v212_ensemble.json"
    convnext_arch_file = mdir / "v212_convnext_t.arch.json"
    mitb1_arch_file = mdir / "v212_mitb1.arch.json"
    convnext_log_file = mdir / "v212_convnext_t.log.json"
    mitb1_log_file = mdir / "v212_mitb1.log.json"

    ensemble_data: dict[str, Any] = {}
    if ensemble_file.is_file():
        try:
            ensemble_data = json.loads(ensemble_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", ensemble_file, e)

    architectures: dict[str, dict[str, Any]] = {}
    for name, arch_path in (
        ("v212_convnext_t", convnext_arch_file),
        ("v212_mitb1", mitb1_arch_file),
    ):
        if arch_path.is_file():
            try:
                architectures[name] = json.loads(arch_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to parse %s: %s", arch_path, e)

    training_logs_present = {
        "v212_convnext_t": convnext_log_file.is_file(),
        "v212_mitb1": mitb1_log_file.is_file(),
    }

    members_info: list[dict[str, Any]] = []
    raw_members = ensemble_data.get("members", []) if isinstance(ensemble_data, dict) else []
    tot_weight = sum(float(m.get("weight", 0.0)) for m in raw_members if isinstance(m, dict)) or 1.0

    for m in raw_members:
        if not isinstance(m, dict):
            continue
        raw_w_path = str(m.get("weights", ""))
        w_name = Path(raw_w_path).name
        resolved_pt = mdir / w_name
        stem = Path(w_name).stem
        arch_cfg = architectures.get(stem, {"encoder": "resnet34", "arch": "Unet", "classes": 5})
        weight_val = float(m.get("weight", 0.0))
        members_info.append({
            "name": stem,
            "weights_file": w_name,
            "weights_path": str(resolved_pt),
            "weights_available": resolved_pt.is_file(),
            "weight": weight_val,
            "normalized_weight": round(weight_val / tot_weight, 4),
            "arch": arch_cfg,
        })

    all_weights_present = bool(members_info) and all(m["weights_available"] for m in members_info)
    return {
        "models_dir": str(mdir),
        "ensemble_config_loaded": bool(ensemble_data),
        "comment": ensemble_data.get("comment", ""),
        "members": members_info,
        "architectures": architectures,
        "training_logs_present": training_logs_present,
        "all_weights_available": all_weights_present,
    }


def is_wirenet_available(models_dir: str | Path | None = None) -> bool:
    """Return True only when all v2.12 ensemble `.pt` weights and `segmentation_models_pytorch` exist."""
    meta = load_wirenet_ensemble_metadata(models_dir)
    if not meta.get("all_weights_available", False):
        return False
    try:
        import segmentation_models_pytorch  # noqa: F401
        return True
    except ImportError:
        return False


def _predict_wirenet_probs(rgb_image: np.ndarray, models_dir: Path, device: str = "cpu") -> np.ndarray | None:
    """Run forward pass on loaded WireNet ensemble members if available."""
    import segmentation_models_pytorch as smp
    import torch

    meta = load_wirenet_ensemble_metadata(models_dir)
    if not meta.get("all_weights_available", False):
        return None

    h, w = rgb_image.shape[:2]
    pad_h = math.ceil(h / 32) * 32
    pad_w = math.ceil(w / 32) * 32
    padded = np.full((pad_h, pad_w, 3), 114, dtype=np.uint8)
    padded[:h, :w] = rgb_image
    x_norm = ((padded.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    tensor_in = torch.from_numpy(np.ascontiguousarray(x_norm)).to(device)

    fused_probs: np.ndarray | None = None
    with torch.no_grad():
        for member in meta["members"]:
            arch_info = member["arch"]
            enc = arch_info.get("encoder", "resnet34")
            arch_name = arch_info.get("arch", "Unet")
            n_cls = int(arch_info.get("classes", 5))
            net = getattr(smp, arch_name)(enc, encoder_weights=None, classes=n_cls)
            state = torch.load(member["weights_path"], map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            net = net.to(device).eval()
            probs = net(tensor_in).softmax(1)[0, :, :h, :w].float().cpu().numpy()
            w_norm = float(member["normalized_weight"])
            fused_probs = probs * w_norm if fused_probs is None else (fused_probs + probs * w_norm)

    return fused_probs


def refine_detections_with_wirenet(
    rgb_image: np.ndarray,
    seg_res: SegmentationResult,
    *,
    models_dir: str | Path | None = None,
    dilate_px: int = 6,
    min_pixels: int = 20,
    device: str = "cpu",
) -> SegmentationResult:
    """Refine coarse YOLOv11-seg wire masks using the WireNet v2.12 probability-fusion ensemble.

    If the optional `.pt` weights or `segmentation_models_pytorch` are not installed on disk,
    logs a debug message and returns `seg_res` unchanged.
    """
    if not seg_res.mask_available or not seg_res.masks:
        return seg_res

    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    if not is_wirenet_available(mdir):
        logger.debug("WireNet optional weights not present in %s; keeping YOLOv11-seg masks.", mdir)
        return seg_res

    try:
        probs = _predict_wirenet_probs(rgb_image, mdir, device=device)
        if probs is None:
            return seg_res
    except Exception as exc:
        logger.warning("WireNet refinement failed (%s); falling back to YOLOv11-seg masks.", exc)
        return seg_res

    cls_map = probs.argmax(axis=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))

    refined_masks: list[np.ndarray] = []
    for i, mask in enumerate(seg_res.masks):
        cls_idx = seg_res.classes[i] if i < len(seg_res.classes) else -1
        raw_name = seg_res.names.get(cls_idx, "")
        canon = normalize_wire_class(raw_name)
        wn_channel = WIRENET_CLASS_MAP.get(canon)

        # Vegetation or unknown/pole classes keep their YOLO mask directly
        if wn_channel is None or canon == "vegetation":
            refined_masks.append(mask)
            continue

        dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0
        refined_bin = dilated & (cls_map == wn_channel)
        if int(refined_bin.sum()) >= min_pixels:
            refined_masks.append(refined_bin.astype(np.uint8) * 255)
        else:
            refined_masks.append(mask)

    return SegmentationResult(
        mask_available=seg_res.mask_available,
        boxes=seg_res.boxes,
        classes=seg_res.classes,
        confidences=seg_res.confidences,
        masks=refined_masks,
        names=seg_res.names,
    )
