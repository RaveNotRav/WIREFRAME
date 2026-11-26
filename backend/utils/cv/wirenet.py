"""WireNet multi-model probability-fusion ensemble (v2.12 - v2.21) and Hybrid Segmenter.

Ported from Joanna's `wireframes-main/app/pipeline/wirenet.py`.
Combines:
- 3 full-resolution U-Net semantic wire segmentation backbones (`wirenet-r34.pt`, `v212_mitb1.pt`, `v212_convnext_t.pt`)
  with horizontal-flip TTA (`predict_probs_flip`) and dark-photo dual-polarity luminance inversion.
- Geometric strand extraction (`backend/utils/cv/wire_instances.py`) with `span_wire_tau=0.4`, junction splitting,
  collinear gap bridging (`max_gap=70`, `max_lateral_px=15`), and `own_component=True`.
- Strand type classification via `strand_head_v217.pt` + frozen DINOv2-small (`backend/models/hf/dinov2-small/`).
- Scene-context false-positive suppression via ADE20K UperNet-ConvNeXt-Tiny (`backend/models/hf/upernet-convnext-tiny/`).
- Dedicated utility pole detector (`backend/models/poles-yolo11n-v218.pt`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.utils.cv.types import SegmentationResult, normalize_wire_class
from backend.utils.cv.wire_instances import (
    WIRENET_CLASSES,
    WIRENET_WIRE_IDS,
    extract_spans,
    strand_spans,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
N_CLASSES = len(WIRENET_CLASSES) + 1  # 5: 0=bg, 1=vegetation, 2=dead_wire, 3=power_distribution_line, 4=communication_cable
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Channel mapping (1-indexed; 0 is background):
# Matches Joanna's CLASSES = ("Vegetation", "abandoned-wire", "power-distribution-lines", "telecom-cable")
WIRENET_CLASS_MAP: dict[str, int] = {
    "vegetation": 1,
    "dead_wire": 2,
    "power_distribution_line": 3,
    "communication_cable": 4,
}

# In-memory cache of loaded ensemble models, strand heads, pole detectors, and scene parsers
_MODEL_CACHE_LOCK = threading.Lock()
_WIRENET_CACHE: dict[tuple[str, str], list[tuple[Any, float]]] = {}
_STRAND_HEAD_CACHE: dict[tuple[str, str], Any] = {}
_POLE_DET_CACHE: dict[tuple[str, str], Any] = {}
_SCENE_PARSER_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class Letterbox:
    """Letterbox geometry mapping between original image (orig_w, orig_h) and square model frame (size x size)."""

    scale: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    size: int
    orig_w: int
    orig_h: int

    def mask_to_original(self, mask: np.ndarray) -> np.ndarray:
        crop = mask[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w]
        return cv2.resize(crop.astype(np.float32), (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR)


def letterbox_rgb(rgb: np.ndarray, size: int = 1024, fill: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Letterbox an RGB uint8 image to `size x size` preserving aspect ratio."""
    h, w = rgb.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, Letterbox(scale, pad_x, pad_y, new_w, new_h, size, w, h)


def _mask_bbox(mask_bin: np.ndarray) -> list[float]:
    rows = np.flatnonzero(mask_bin.any(axis=1))
    cols = np.flatnonzero(mask_bin.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def load_wirenet_ensemble_metadata(models_dir: str | Path | None = None) -> dict[str, Any]:
    """Load v2.17 / v2.12 WireNet ensemble specification, member architecture configs, and availability."""
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    v217_file = mdir / "v217_ensemble.json"
    v212_file = mdir / "v212_ensemble.json"
    ensemble_file = v217_file if v217_file.is_file() else v212_file
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
    any_weights_present = bool(members_info) and any(m["weights_available"] for m in members_info)
    return {
        "models_dir": str(mdir),
        "ensemble_config_loaded": bool(ensemble_data),
        "comment": ensemble_data.get("comment", ""),
        "members": members_info,
        "architectures": architectures,
        "training_logs_present": training_logs_present,
        "all_weights_available": all_weights_present,
        "any_weights_available": any_weights_present,
    }


def is_wirenet_available(models_dir: str | Path | None = None) -> bool:
    """Return True when all ensemble `.pt` weights and `segmentation_models_pytorch` exist."""
    meta = load_wirenet_ensemble_metadata(models_dir)
    if not meta.get("all_weights_available", False):
        return False
    try:
        import segmentation_models_pytorch  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_device(device: str = "auto") -> str:
    if device and str(device).lower() not in ("auto", ""):
        return str(device)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_cached_wirenet_members(models_dir: Path, device: str) -> list[tuple[Any, float]]:
    """Load and cache available WireNet ensemble member networks (`wirenet-r34`, `v212_mitb1`, `v212_convnext_t`)."""
    key = (str(models_dir.resolve()), device)
    with _MODEL_CACHE_LOCK:
        if key in _WIRENET_CACHE:
            return _WIRENET_CACHE[key]

        import segmentation_models_pytorch as smp
        import torch

        meta = load_wirenet_ensemble_metadata(models_dir)
        available_members = [m for m in meta.get("members", []) if m.get("weights_available")]
        if not available_members:
            return []

        tot_w = sum(float(m["weight"]) for m in available_members) or 1.0
        loaded: list[tuple[Any, float]] = []
        for member in available_members:
            try:
                arch_info = member["arch"]
                enc = arch_info.get("encoder", "resnet34")
                arch_name = arch_info.get("arch", "Unet")
                n_cls = int(arch_info.get("classes", N_CLASSES))
                net = getattr(smp, arch_name)(enc, encoder_weights=None, classes=n_cls)
                state = torch.load(member["weights_path"], map_location="cpu", weights_only=True)
                net.load_state_dict(state)
                net = net.to(device).eval()
                loaded.append((net, float(member["weight"]) / tot_w))
            except Exception as exc:
                logger.warning("Failed to load WireNet member %s: %s", member.get("name"), exc)

        if loaded:
            # Re-normalize weights if one member failed
            w_sum = sum(w for _, w in loaded) or 1.0
            loaded = [(net, w / w_sum) for net, w in loaded]
            _WIRENET_CACHE[key] = loaded
        return loaded


def _probs_lowmem(members: list[tuple[Any, float]], x: np.ndarray, device: str) -> np.ndarray:
    """Compute fused class probabilities for batch `x` (B, 3, H, W), one image and one ensemble member at a time."""
    import torch

    rows = []
    with torch.no_grad():
        for i in range(x.shape[0]):
            xi = torch.from_numpy(x[i : i + 1]).to(device)
            acc = None
            for net, wgt in members:
                p = net(xi).softmax(1) * wgt
                acc = p if acc is None else acc + p
                del p
            if len(members) > 1 and acc is not None:
                acc = acc / acc.sum(1, keepdim=True)
            rows.append(acc.float().cpu().numpy()[0])
            del xi, acc
    return np.stack(rows)


def predict_probs(members: list[tuple[Any, float]], rgb_u8: np.ndarray, device: str) -> np.ndarray:
    """Return C x H x W class probabilities at the input resolution (padded to a multiple of 32)."""
    h, w = rgb_u8.shape[:2]
    H, W = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    x = np.full((H, W, 3), 114, np.uint8)
    x[:h, :w] = rgb_u8
    x_norm = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    return _probs_lowmem(members, np.ascontiguousarray(x_norm), device)[0, :, :h, :w]


def predict_probs_flip(members: list[tuple[Any, float]], rgb_u8: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run photo and its horizontal mirror in one batched pass and return (p, p_mirror_unflipped)."""
    h, w = rgb_u8.shape[:2]
    H, W = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    x = np.full((2, H, W, 3), 114, np.uint8)
    x[0, :h, :w] = rgb_u8
    x[1, :h, :w] = rgb_u8[:, ::-1]
    x_norm = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)
    p = _probs_lowmem(members, np.ascontiguousarray(x_norm), device)[:, :, :h, :w]
    return p[0], np.ascontiguousarray(p[1][:, :, ::-1])


def mean_l_star(rgb: np.ndarray) -> float:
    """Mean CIELAB L* (0-100) of an RGB uint8 image."""
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].mean() * (100.0 / 255.0))


def invert_luminance(x: np.ndarray, lb: Letterbox | None = None) -> np.ndarray:
    """Invert CIELAB L* inside the letterbox content for dark/night photos."""
    out = x.copy()
    sl = (
        (slice(None), slice(None))
        if lb is None
        else (slice(lb.pad_y, lb.pad_y + lb.new_h), slice(lb.pad_x, lb.pad_x + lb.new_w))
    )
    lab = cv2.cvtColor(np.ascontiguousarray(x[sl]), cv2.COLOR_RGB2LAB)
    lab[..., 0] = 255 - lab[..., 0]
    out[sl] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return out


def wire_pixel_count(probs: np.ndarray) -> int:
    return int(np.isin(probs.argmax(0), WIRENET_WIRE_IDS).sum())


def class_masks_to_original(cls_map: np.ndarray, lb: Letterbox) -> dict[int, np.ndarray]:
    """Map per-class binary masks from letterbox space to the original image."""
    crop = cls_map[lb.pad_y : lb.pad_y + lb.new_h, lb.pad_x : lb.pad_x + lb.new_w]
    out: dict[int, np.ndarray] = {}
    for c in range(1, N_CLASSES):
        m = (crop == c).astype(np.float32)
        if not m.any():
            continue
        down = lb.orig_w < lb.new_w
        r = cv2.resize(m, (lb.orig_w, lb.orig_h), interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR)
        out[c] = r >= (0.25 if down else 0.5)
    return out


def prob_to_original(prob: np.ndarray, lb: Letterbox) -> np.ndarray:
    crop = prob[lb.pad_y : lb.pad_y + lb.new_h, lb.pad_x : lb.pad_x + lb.new_w].astype(np.float32)
    return cv2.resize(crop, (lb.orig_w, lb.orig_h), interpolation=cv2.INTER_LINEAR)


def _get_cached_strand_head(models_dir: Path, device: str):
    key = (str(models_dir.resolve()), device)
    with _MODEL_CACHE_LOCK:
        if key in _STRAND_HEAD_CACHE:
            return _STRAND_HEAD_CACHE[key]
        head_pt = models_dir / "strand_head_v217.pt"
        if not head_pt.is_file():
            return None
        from backend.utils.cv.strand_head import StrandHead

        head = StrandHead(
            head_pt,
            min_conf=0.5,
            device=device,
            models_dir=models_dir,
            clamp=True,
            letterbox_frame=True,
        )
        _STRAND_HEAD_CACHE[key] = head
        return head


def _get_cached_pole_detector(models_dir: Path, device: str, conf: float = 0.25):
    key = (str(models_dir.resolve()), device)
    with _MODEL_CACHE_LOCK:
        if key in _POLE_DET_CACHE:
            det = _POLE_DET_CACHE[key]
            det.conf = conf
            return det
        pole_pt = models_dir / "poles-yolo11n-v218.pt"
        if not pole_pt.is_file():
            return None
        from backend.utils.cv.scene import PoleDetector

        det = PoleDetector(pole_pt, conf=conf, imgsz=640, device=device)
        _POLE_DET_CACHE[key] = det
        return det


def _get_cached_scene_parser(models_dir: Path, device: str):
    key = (str(models_dir.resolve()), device)
    with _MODEL_CACHE_LOCK:
        if key in _SCENE_PARSER_CACHE:
            return _SCENE_PARSER_CACHE[key]
        from backend.utils.cv.scene import SceneParser

        parser = SceneParser(device=device, models_dir=models_dir)
        _SCENE_PARSER_CACHE[key] = parser
        return parser


def run_hybrid_ensemble_segmentation(
    rgb_image: np.ndarray,
    yolo_seg_res: SegmentationResult,
    *,
    models_dir: str | Path | None = None,
    imgsz: int = 1024,
    conf_min: float = 0.02,
    pole_conf: float = 0.05,
    tta: str = "flip",
    dual_polarity: bool = True,
    use_scene_gates: bool = True,
    device: str = "auto",
) -> SegmentationResult:
    """Run Joanna's full multi-model ensemble pipeline:
    1. Keep YOLOv11-seg (`wireframe-yolo11m-seg.pt` / `wireframe_seg.pt`) vegetation detections and high-confidence
       wire proposals.
    2. Run the 3-backbone WireNet ensemble (`wirenet-r34.pt`, `v212_mitb1.pt`, `v212_convnext_t.pt`) at 1024x1024
       letterbox with horizontal-flip TTA (`predict_probs_flip`) and dark-photo dual polarity.
    3. Extract physical wire strands from `P(any wire) >= 0.4` via `strand_spans` (`wire_instances.py`).
    4. Classify each strand into `power_distribution_line`, `communication_cable`, or `dead_wire` using
       `strand_head_v217.pt` + frozen DINOv2-small (`hf/dinov2-small/`).
    5. Run `poles-yolo11n-v218.pt` to detect utility poles in the scene.
    6. Run ADE20K UperNet-ConvNeXt-Tiny (`hf/upernet-convnext-tiny/`) scene gates (ground gate, structure-edge gate,
       vegetation gate) and fallback pole context.
    7. Return a unified `SegmentationResult` containing all detected wires, vegetation, and utility poles.
    """
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    dev = _resolve_device(device)
    orig_h, orig_w = rgb_image.shape[:2]

    members = _get_cached_wirenet_members(mdir, dev)
    if not members:
        logger.debug("No WireNet ensemble weights available in %s; returning YOLO result.", mdir)
        return yolo_seg_res

    # Letterbox RGB image to 1024x1024 (matching Joanna's preprocess + HybridSegmenter)
    x_lb, lb = letterbox_rgb(rgb_image, size=imgsz, fill=114)
    content = np.ascontiguousarray(x_lb[lb.pad_y : lb.pad_y + lb.new_h, lb.pad_x : lb.pad_x + lb.new_w])

    # 1. WireNet probability fusion with flip TTA + dual polarity
    if tta == "flip":
        p1, p2 = predict_probs_flip(members, x_lb, dev)
        probs = (p1 + p2) * 0.5
    else:
        probs = predict_probs(members, x_lb, dev)

    if dual_polarity and mean_l_star(content) <= 40.0:
        inv_x = invert_luminance(x_lb, lb)
        if tta == "flip":
            ip1, ip2 = predict_probs_flip(members, inv_x, dev)
            inv_probs = (ip1 + ip2) * 0.5
        else:
            inv_probs = predict_probs(members, inv_x, dev)
        if wire_pixel_count(inv_probs) >= 1.2 * max(wire_pixel_count(probs), 1):
            probs = inv_probs

    # 2. Extract wire strands from fused WireNet + YOLO probabilities in original image coordinates
    orig_probs = np.stack([prob_to_original(probs[c], lb) for c in range(N_CLASSES)])

    # Blend in YOLOv11-seg (`wireframe_seg.pt` / `wireframe-yolo11m-seg.pt`) mask probabilities at weight 0.25
    # alongside the 3 WireNet backbones (`wirenet-r34`, `v212_mitb1`, `v212_convnext_t`) so thin overhead spans
    # detected by either YOLO or WireNet remain continuous across bright sky backgrounds.
    yolo_blend_weight = 0.25
    if yolo_seg_res.mask_available and yolo_seg_res.masks:
        yolo_prob_map = np.zeros((N_CLASSES, orig_h, orig_w), dtype=np.float32)
        has_yolo_wire = False
        for i, cls_idx in enumerate(yolo_seg_res.classes):
            raw_name = yolo_seg_res.names.get(cls_idx, "")
            canon = normalize_wire_class(raw_name)
            wn_ch = WIRENET_CLASS_MAP.get(canon)
            if wn_ch is None:
                continue
            conf = float(yolo_seg_res.confidences[i]) if i < len(yolo_seg_res.confidences) else 0.0
            m_bin = (yolo_seg_res.masks[i] > 0)
            if not np.any(m_bin):
                continue
            # Boost confidence floor for detected YOLO wire masks so thin segments contribute cleanly
            eff_score = max(conf, 0.55) if wn_ch in WIRENET_WIRE_IDS else conf
            yolo_prob_map[wn_ch] = np.maximum(yolo_prob_map[wn_ch], m_bin.astype(np.float32) * eff_score)
            if wn_ch in WIRENET_WIRE_IDS:
                has_yolo_wire = True
        if has_yolo_wire:
            for wn_ch in WIRENET_WIRE_IDS:
                orig_probs[wn_ch] = (1.0 - yolo_blend_weight) * orig_probs[wn_ch] + yolo_blend_weight * yolo_prob_map[wn_ch]
            orig_probs[0] = np.clip(1.0 - orig_probs[1:].sum(0), 0.0, 1.0)

    tau = 0.24
    wire_lb = probs[WIRENET_WIRE_IDS].sum(0) >= tau
    union_crop = wire_lb[lb.pad_y : lb.pad_y + lb.new_h, lb.pad_x : lb.pad_x + lb.new_w].astype(np.float32)
    down = lb.orig_w < lb.new_w
    union_orig = (
        cv2.resize(
            union_crop,
            (lb.orig_w, lb.orig_h),
            interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR,
        )
        >= (0.20 if down else 0.35)
    ) | (orig_probs[WIRENET_WIRE_IDS].sum(0) >= tau)

    span_cfg = {
        "min_px": 30,
        "min_len": 15,
        "max_turn_deg": 35.0,
        "cluster_radius": 5,
        "max_gap": 120.0,
        "gap_angle_deg": 35.0,
        "draw_bridge": True,
        "min_span_len": 90,
        "own_component": True,
        "max_lateral_px": 20.0,
    }

    # Check if pole detector finds poles first
    pole_boxes_orig: list[dict[str, Any]] = []
    pole_det = _get_cached_pole_detector(mdir, dev, conf=pole_conf)
    if pole_det is not None:
        try:
            found_poles = pole_det.detect(content) or []
            sx, sy = lb.orig_w / lb.new_w, lb.orig_h / lb.new_h
            for p in found_poles:
                pb = p["bbox"]
                pole_boxes_orig.append({
                    "bbox": [
                        round(pb[0] * sx, 1),
                        round(pb[1] * sy, 1),
                        round(pb[2] * sx, 1),
                        round(pb[3] * sy, 1),
                    ],
                    "confidence": float(p.get("confidence", 0.5)),
                })
        except Exception as exc:
            logger.warning("Pole detector inference failed: %s", exc)

    effective_conf_min = min(float(conf_min), 0.20)
    strands = [
        s
        for s in strand_spans(
            orig_probs,
            WIRENET_WIRE_IDS,
            span_cfg,
            tau=tau,
            min_class_share=0.9,
            mask=union_orig,
        )
        if s.confidence >= effective_conf_min
    ]

    # 3. Classify strands with v2.17 StrandHead + DINOv2-small if available
    if strands:
        head = _get_cached_strand_head(mdir, dev)
        if head is not None:
            try:
                frame = (probs, x_lb, lb) if getattr(head, "letterbox_frame", True) else None
                head.classify(strands, orig_probs, rgb_image, frame=frame)
            except Exception as exc:
                logger.warning("StrandHead classification skipped: %s", exc)

    # 4. Collect YOLO vegetation and high-confidence YOLO wire/pole detections
    veg_dets: list[dict[str, Any]] = []
    yolo_wire_dets: list[dict[str, Any]] = []
    if yolo_seg_res.mask_available and yolo_seg_res.masks:
        for i, cls_idx in enumerate(yolo_seg_res.classes):
            raw_name = yolo_seg_res.names.get(cls_idx, "")
            canon = normalize_wire_class(raw_name)
            conf = float(yolo_seg_res.confidences[i]) if i < len(yolo_seg_res.confidences) else 0.0
            m_bin = (yolo_seg_res.masks[i] > 0)
            if not np.any(m_bin):
                continue
            bbox = yolo_seg_res.boxes[i] if i < len(yolo_seg_res.boxes) else _mask_bbox(m_bin)
            if canon == "vegetation" and conf >= 0.20:
                veg_dets.append({"class_name": "vegetation", "confidence": conf, "mask": m_bin, "bbox": bbox})
            elif canon == "pole":
                pole_boxes_orig.append({"bbox": bbox, "confidence": conf, "mask": m_bin})
            elif canon in ("power_distribution_line", "communication_cable", "dead_wire"):
                yolo_wire_dets.append({"class_name": canon, "confidence": conf, "mask": m_bin, "bbox": bbox})

    # If YOLO found no vegetation, check WireNet's vegetation channel (channel 1) for large foliage blobs
    if not veg_dets:
        cls_masks_orig = class_masks_to_original(probs.argmax(0), lb)
        veg_mask_wn = cls_masks_orig.get(1)
        if veg_mask_wn is not None and int(veg_mask_wn.sum()) >= int(0.01 * orig_h * orig_w):
            veg_prob_orig = orig_probs[1]
            num_labels, lab_map, stats, _ = cv2.connectedComponentsWithStats(veg_mask_wn.astype(np.uint8), connectivity=8)
            min_veg_area = max(400, int(0.005 * orig_h * orig_w))
            for lbl in range(1, num_labels):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area >= min_veg_area:
                    comp_mask = lab_map == lbl
                    comp_conf = float(veg_prob_orig[comp_mask].mean())
                    if comp_conf >= 0.45:
                        veg_dets.append({
                            "class_name": "vegetation",
                            "confidence": round(comp_conf, 4),
                            "mask": comp_mask,
                            "bbox": _mask_bbox(comp_mask),
                        })

    # 5. Scene-context gating & structural context extraction (ADE20K UperNet-ConvNeXt-Tiny)
    kept_strands = list(strands)
    kept_veg = list(veg_dets)
    building_boxes_orig: list[dict[str, Any]] = []
    if use_scene_gates and (kept_strands or kept_veg or pole_boxes_orig):
        scene_parser = _get_cached_scene_parser(mdir, dev)
        if scene_parser is not None:
            try:
                from backend.utils.cv.scene import ground_share, side_contrast, vegetation_share

                lab = scene_parser.labels(content)
                if lab is not None:
                    lab_orig = cv2.resize(lab, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    ground = scene_parser.ground_mask(lab_orig)
                    filtered_strands = []
                    for st in kept_strands:
                        g_share = ground_share(st.mask, ground)
                        if g_share >= 0.5:
                            continue
                        s_contrast = side_contrast(st.mask, lab_orig)
                        if s_contrast >= 0.6:
                            continue
                        filtered_strands.append(st)
                    # Only apply scene suppression if it doesn't wipe out all wires on an upward sky/foliage shot
                    if filtered_strands or all(ground_share(st.mask, ground) >= 0.5 for st in kept_strands):
                        kept_strands = filtered_strands

                    filtered_veg = []
                    for vd in kept_veg:
                        v_share = vegetation_share(vd["mask"], lab_orig, scene_parser.vegetation_ids)
                        if v_share >= 0.4:
                            filtered_veg.append(vd)
                    kept_veg = filtered_veg

                    ctx = scene_parser.context(lab_orig)
                    if not pole_boxes_orig:
                        for pbox in ctx.get("poles", []):
                            pole_boxes_orig.append({"bbox": [float(v) for v in pbox], "confidence": 0.45})
                    b_mask_full = scene_parser.building_mask(lab_orig)
                    for bbox in ctx.get("buildings", []):
                        bx1, by1, bx2, by2 = [max(0, int(round(v))) for v in bbox[:4]]
                        bm = np.zeros((orig_h, orig_w), dtype=bool)
                        bm[min(orig_h, by1) : min(orig_h, by2), min(orig_w, bx1) : min(orig_w, bx2)] = b_mask_full[
                            min(orig_h, by1) : min(orig_h, by2), min(orig_w, bx1) : min(orig_w, bx2)
                        ]
                        building_boxes_orig.append({
                            "bbox": [float(v) for v in bbox],
                            "confidence": 0.60,
                            "mask": bm if np.any(bm) else None,
                        })
            except Exception as exc:
                logger.warning("Scene gate skipped: %s", exc)

    # Fallback to classical Hough vertical pole detector if no pole was detected yet
    if not pole_boxes_orig:
        try:
            from backend.utils.cv.calibration import detect_pole_hough_fallback

            hough_box = detect_pole_hough_fallback(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            if hough_box is not None:
                pole_boxes_orig.append({"bbox": [float(v) for v in hough_box], "confidence": 0.50})
        except Exception as exc:
            logger.debug("Hough pole fallback skipped: %s", exc)

    # Detect crossarms and insulators around detected poles and wire attachments
    crossarm_boxes_orig: list[dict[str, Any]] = []
    insulator_boxes_orig: list[dict[str, Any]] = []
    if pole_boxes_orig:
        try:
            from backend.utils.cv.scene import detect_crossarms_and_insulators

            c_boxes, i_boxes = detect_crossarms_and_insulators(
                rgb_image,
                pole_boxes_orig,
                [st.mask for st in kept_strands],
            )
            crossarm_boxes_orig.extend(c_boxes)
            insulator_boxes_orig.extend(i_boxes)
        except Exception as exc:
            logger.debug("Crossarm/insulator detection skipped: %s", exc)

    # 6. Build unified 8-class SegmentationResult with canonical class names
    canonical_names: dict[int, str] = {
        0: "power_distribution_line",
        1: "communication_cable",
        2: "dead_wire",
        3: "vegetation",
        4: "pole",
        5: "crossarm",
        6: "insulator",
        7: "building",
    }
    name_to_id = {v: k for k, v in canonical_names.items()}

    out_boxes: list[list[float]] = []
    out_classes: list[int] = []
    out_confs: list[float] = []
    out_masks: list[np.ndarray] = []

    for st in kept_strands:
        canon_cls = normalize_wire_class(st.wire_class)
        if canon_cls not in name_to_id:
            canon_cls = "power_distribution_line"
        m_u8 = (st.mask > 0).astype(np.uint8) * 255
        if not np.any(m_u8):
            continue
        out_masks.append(m_u8)
        out_boxes.append(_mask_bbox(st.mask))
        out_classes.append(name_to_id[canon_cls])
        out_confs.append(round(float(st.confidence), 4))

    # Merge any non-overlapping high-confidence YOLO wire masks that were not already covered by WireNet strands
    if yolo_wire_dets:
        wire_union = np.zeros((orig_h, orig_w), dtype=bool)
        for m in out_masks:
            wire_union |= (m > 0)
        for yw in yolo_wire_dets:
            ym = yw["mask"]
            overlap = float((ym & wire_union).sum()) / max(1.0, float(ym.sum()))
            if overlap < 0.25 and float(yw["confidence"]) >= 0.20:
                m_u8 = ym.astype(np.uint8) * 255
                out_masks.append(m_u8)
                out_boxes.append([float(v) for v in yw["bbox"]])
                out_classes.append(name_to_id[yw["class_name"]])
                out_confs.append(round(float(yw["confidence"]), 4))
                wire_union |= ym

    # Append vegetation detections
    for vd in kept_veg:
        m_u8 = (vd["mask"] > 0).astype(np.uint8) * 255
        if not np.any(m_u8):
            continue
        out_masks.append(m_u8)
        out_boxes.append([float(v) for v in vd["bbox"]])
        out_classes.append(name_to_id["vegetation"])
        out_confs.append(round(float(vd["confidence"]), 4))

    # Append pole, crossarm, insulator, and building detections
    def _append_box_dets(box_dets: list[dict[str, Any]], cls_label: str) -> None:
        for item in box_dets:
            pb = [float(v) for v in item["bbox"]]
            pm = item.get("mask")
            if pm is None:
                pm_u8 = np.zeros((orig_h, orig_w), dtype=np.uint8)
                x1, y1, x2, y2 = [max(0, int(round(v))) for v in pb[:4]]
                pm_u8[min(orig_h, y1) : min(orig_h, y2), min(orig_w, x1) : min(orig_w, x2)] = 255
            else:
                pm_u8 = (pm > 0).astype(np.uint8) * 255
            if not np.any(pm_u8):
                continue
            out_masks.append(pm_u8)
            out_boxes.append(pb)
            out_classes.append(name_to_id[cls_label])
            out_confs.append(round(float(item.get("confidence", 0.5)), 4))

    _append_box_dets(pole_boxes_orig, "pole")
    _append_box_dets(crossarm_boxes_orig, "crossarm")
    _append_box_dets(insulator_boxes_orig, "insulator")
    _append_box_dets(building_boxes_orig, "building")

    if not out_masks:
        return SegmentationResult(mask_available=False, names=canonical_names)

    return SegmentationResult(
        mask_available=True,
        boxes=out_boxes,
        classes=out_classes,
        confidences=out_confs,
        masks=out_masks,
        names=canonical_names,
    )


def _predict_wirenet_probs(rgb_image: np.ndarray, models_dir: Path, device: str = "cpu") -> np.ndarray | None:
    """Run forward pass on loaded WireNet ensemble members if available."""
    members = _get_cached_wirenet_members(models_dir, _resolve_device(device))
    if not members:
        return None
    return predict_probs(members, rgb_image, _resolve_device(device))


def refine_detections_with_wirenet(
    rgb_image: np.ndarray,
    seg_res: SegmentationResult,
    *,
    models_dir: str | Path | None = None,
    dilate_px: int = 6,
    min_pixels: int = 20,
    device: str = "cpu",
    instances: str = "skeleton",
) -> SegmentationResult:
    """Refine or extract full-resolution wire spans using Joanna's multi-model WireNet ensemble.

    - When `instances="skeleton"` (default, matching Joanna's `wireframe.yaml`), runs the full
      multi-model hybrid ensemble (`run_hybrid_ensemble_segmentation`) combining WireNet R34 + MiT-B1 + ConvNeXt-T,
      StrandHead + DINOv2-small, YOLO11n pole detector, and ADE20K UperNet scene gates.
    - When `instances="yolo"` (or when refining synthetic unit-test masks where `seg_res` already has detections
      and `instances="yolo"` is requested), refines coarse masks within a `dilate_px` neighborhood.
    - If the optional `.pt` weights or `segmentation_models_pytorch` are not installed in `models_dir`,
      gracefully returns `seg_res` unchanged.
    """
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    if not is_wirenet_available(mdir):
        logger.debug("WireNet optional weights not present in %s; keeping YOLOv11-seg masks.", mdir)
        return seg_res

    if instances == "skeleton":
        try:
            hybrid_res = run_hybrid_ensemble_segmentation(
                rgb_image,
                seg_res,
                models_dir=mdir,
                device=device,
            )
            if hybrid_res.mask_available and hybrid_res.masks:
                return hybrid_res
        except Exception as exc:
            logger.warning("Hybrid WireNet ensemble failed (%s); falling back to mask refinement.", exc)

    if not seg_res.mask_available or not seg_res.masks:
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
