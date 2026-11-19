"""YOLO segmentation wrapper supporting both Single-Pass and Tiled (sliced_predict) inference
with cross-tile mask stitching for thin overhead wires.

Complexity:
- Model forward pass: O(YOLO_GFLOPs * num_tiles)
- Mask rasterization/resizing/stitching: O(N) where N = width * height of original image
- Overall: O(YOLO) + O(N)
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

import cv2
import numpy as np

from backend.config_store import get_setting
from backend.utils.cv.types import SegmentationResult

logger = logging.getLogger(__name__)
_INFERENCE_LOCK = threading.Lock()


def positions(n: int, tile: int, overlap: float = 0.3) -> list[int]:
    """Compute tile start offsets along an axis of length n with given tile size and overlap."""
    if n <= tile or tile <= 0:
        return [0]
    stride = max(1, int(round(tile * (1.0 - float(overlap)))))
    starts: list[int] = []
    pos = 0
    while pos + tile < n:
        starts.append(pos)
        pos += stride
    last = max(0, n - tile)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def get_inference_params() -> dict[str, Any]:
    """Retrieve inference settings from env (WIREFRAME_CONF, WIREFRAME_IOU, WIREFRAME_IMGSZ, WIREFRAME_DEVICE, WIREFRAME_TILED) or config store.
    Explicit environment variables always take precedence over SQLite settings table values.
    """
    params = get_setting("inference_params", {}) or {}

    env_conf = os.getenv("WIREFRAME_CONF")
    try:
        conf = float(env_conf) if env_conf is not None and env_conf.strip() != "" else float(params.get("conf", 0.02))
    except (ValueError, TypeError):
        conf = 0.02

    env_iou = os.getenv("WIREFRAME_IOU")
    try:
        iou = float(env_iou) if env_iou is not None and env_iou.strip() != "" else float(params.get("iou", 0.70))
    except (ValueError, TypeError):
        iou = 0.70

    env_imgsz = os.getenv("WIREFRAME_IMGSZ")
    try:
        imgsz = int(env_imgsz) if env_imgsz is not None and env_imgsz.strip() != "" else int(params.get("imgsz", 1024))
    except (ValueError, TypeError):
        imgsz = 1024

    env_device = os.getenv("WIREFRAME_DEVICE")
    device = env_device.strip() if env_device is not None and env_device.strip() != "" else params.get("device", "auto")

    env_tiled = os.getenv("WIREFRAME_TILED")
    if env_tiled is not None and env_tiled.strip() != "":
        tiled = env_tiled.strip().lower() in ("1", "true", "yes")
    else:
        tiled = bool(params.get("tiled", False))

    retina_masks = bool(params.get("retina_masks", True))
    tile_size = int(params.get("tile_size", 640))
    tile_overlap = float(params.get("tile_overlap", 0.3))
    tile_iou = float(params.get("tile_iou", 0.9))
    max_det = int(params.get("max_det", 1000))

    return {
        "conf": conf,
        "iou": iou,
        "device": device,
        "imgsz": imgsz,
        "retina_masks": retina_masks,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_iou": tile_iou,
        "max_det": max_det,
    }


def _mask_bbox(mask_bin: np.ndarray) -> list[float]:
    """Compute [x1, y1, x2, y2] bounding box from a binary mask."""
    rows = np.flatnonzero(mask_bin.any(axis=1))
    cols = np.flatnonzero(mask_bin.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def _stitch_tile_masks_by_class(
    tile_dets: list[dict[str, Any]],
    full_h: int,
    full_w: int,
    seam_dilate_px: int = 3,
) -> tuple[list[list[float]], list[int], list[float], list[np.ndarray]]:
    """Stitch connected or seam-adjacent tile masks of the same class into unified full-image masks."""
    if not tile_dets:
        return [], [], [], []

    by_class: dict[int, list[dict[str, Any]]] = {}
    for det in tile_dets:
        by_class.setdefault(int(det["cls"]), []).append(det)

    out_boxes: list[list[float]] = []
    out_classes: list[int] = []
    out_confs: list[float] = []
    out_masks: list[np.ndarray] = []

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * seam_dilate_px + 1, 2 * seam_dilate_px + 1),
    )

    for cls_id, dets in by_class.items():
        n = len(dets)
        if n == 1:
            m = dets[0]["mask"]
            if np.any(m):
                out_masks.append(m)
                out_boxes.append(_mask_bbox(m))
                out_classes.append(cls_id)
                out_confs.append(float(dets[0]["conf"]))
            continue

        # Union-find over tile detections of the same class that overlap or touch within seam_dilate_px
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        dilated = [cv2.dilate((d["mask"] > 0).astype(np.uint8), kernel) > 0 for d in dets]
        bboxes = [_mask_bbox(d) for d in dilated]

        for i in range(n):
            bx1, by1, bx2, by2 = bboxes[i]
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue
                cx1, cy1, cx2, cy2 = bboxes[j]
                # Quick bbox overlap check before pixel intersection
                if bx2 < cx1 or cx2 < bx1 or by2 < cy1 or cy2 < by1:
                    continue
                ix1, iy1 = int(max(bx1, cx1)), int(max(by1, cy1))
                ix2, iy2 = int(min(bx2, cx2)), int(min(by2, cy2))
                if ix2 > ix1 and iy2 > iy1:
                    if np.any(dilated[i][iy1:iy2, ix1:ix2] & dilated[j][iy1:iy2, ix1:ix2]):
                        parent[find(i)] = find(j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for members in groups.values():
            merged = np.zeros((full_h, full_w), dtype=np.uint8)
            weighted_conf = 0.0
            total_area = 0.0
            for idx in members:
                m_bin = (dets[idx]["mask"] > 0).astype(np.uint8)
                area = float(m_bin.sum())
                if area <= 0:
                    continue
                merged |= (m_bin * 255)
                weighted_conf += area * float(dets[idx]["conf"])
                total_area += area
            if total_area > 0 and np.any(merged):
                # Close tiny 1-2px tile seam gaps when multiple tiles merged
                if len(members) > 1:
                    k_seam = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k_seam)
                out_masks.append(merged)
                out_boxes.append(_mask_bbox(merged))
                out_classes.append(cls_id)
                out_confs.append(round(weighted_conf / total_area, 4))

    return out_boxes, out_classes, out_confs, out_masks


def sliced_predict(
    model: Any,
    bgr_image: np.ndarray,
    *,
    original_shape: tuple[int, int] | None = None,
    conf: float = 0.25,
    tile: int = 640,
    overlap: float = 0.3,
    tile_iou: float = 0.9,
    max_det: int = 1000,
    device: str = "auto",
) -> SegmentationResult:
    """Run tiled YOLOv11-seg inference over overlapping crops and stitch masks across tile seams."""
    img_h, img_w = bgr_image.shape[:2]
    target_h, target_w = original_shape if original_shape is not None else (img_h, img_w)

    ys = positions(img_h, tile, overlap)
    xs = positions(img_w, tile, overlap)

    crops: list[np.ndarray] = []
    offsets: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        y1 = min(img_h, y0 + tile)
        for x0 in xs:
            x1 = min(img_w, x0 + tile)
            crops.append(np.ascontiguousarray(bgr_image[y0:y1, x0:x1]))
            offsets.append((x0, y0, x1 - x0, y1 - y0))

    predict_kwargs: dict[str, Any] = {
        "conf": conf,
        "iou": tile_iou,
        "imgsz": tile,
        "max_det": max_det,
        "retina_masks": True,
        "verbose": False,
    }
    if device and str(device).lower() != "auto":
        predict_kwargs["device"] = device

    full_results = []
    with _INFERENCE_LOCK:
        if hasattr(model, "predict"):
            results = model.predict(crops, **predict_kwargs)
            # Also run a full-frame 1024px pass when the image spans multiple tiles so long continuous
            # power/telecom spans are never missed by tile boundary clipping
            if len(crops) > 1:
                try:
                    full_kwargs = dict(predict_kwargs, imgsz=1024)
                    full_results = model.predict([bgr_image], **full_kwargs) or []
                except Exception:
                    full_results = []
        else:
            # Support callable fake models in unit tests that accept either a list of crops or single crop
            try:
                results = model(crops, **predict_kwargs)
                if len(results) == 1 and len(crops) > 1:
                    results = [model(c, **predict_kwargs)[0] for c in crops]
            except Exception:
                results = [model(c, **predict_kwargs)[0] for c in crops]

    if not results and not full_results:
        return SegmentationResult(mask_available=False)

    names: dict[int, str] = {}
    raw_model_names = getattr(model, "names", None)
    if isinstance(raw_model_names, dict):
        names = {int(k): str(v) for k, v in raw_model_names.items()}

    tile_dets: list[dict[str, Any]] = []
    any_masks = False

    all_pass_results = list(zip(results or [], offsets))
    for fr in full_results:
        all_pass_results.append((fr, (0, 0, img_w, img_h)))

    for r, (x0, y0, cw, ch) in all_pass_results:
        if r is None:
            continue
        raw_r_names = getattr(r, "names", None)
        if isinstance(raw_r_names, dict) and not names:
            names = {int(k): str(v) for k, v in raw_r_names.items()}

        if not getattr(r, "boxes", None) or not getattr(r, "masks", None):
            continue

        any_masks = True
        cls_arr = r.boxes.cls.cpu().numpy().astype(int).tolist()
        conf_arr = r.boxes.conf.cpu().numpy().tolist()
        masks_data = r.masks.data.cpu().numpy()

        for i in range(masks_data.shape[0]):
            m_tile = masks_data[i]
            if m_tile.shape != (ch, cw):
                m_tile = cv2.resize(m_tile, (cw, ch), interpolation=cv2.INTER_LINEAR)
            m_bin = (m_tile > 0.5).astype(np.uint8) * 255
            if not np.any(m_bin):
                continue

            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            full_mask[y0 : y0 + ch, x0 : x0 + cw] = m_bin
            if (img_h, img_w) != (target_h, target_w):
                full_mask = cv2.resize(full_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            tile_dets.append({
                "cls": cls_arr[i] if i < len(cls_arr) else 0,
                "conf": conf_arr[i] if i < len(conf_arr) else 0.0,
                "mask": full_mask,
            })

    if not any_masks:
        return SegmentationResult(mask_available=False, names=names)

    boxes, classes, confidences, masks_list = _stitch_tile_masks_by_class(tile_dets, target_h, target_w)
    return SegmentationResult(
        mask_available=True,
        boxes=boxes,
        classes=classes,
        confidences=confidences,
        masks=masks_list,
        names=names,
    )


def _is_real_ultralytics_model(model: Any) -> bool:
    """Return True only when `model` is an actual Ultralytics YOLO instance (not a test fake/mock)."""
    mod = getattr(type(model), "__module__", "") or ""
    return mod.startswith("ultralytics")


def segment_image(
    rgb_image: np.ndarray,
    model: Any,
    original_shape: tuple[int, int],
    *,
    tiled: bool | None = None,
    tile_size: int | None = None,
    tile_overlap: float | None = None,
    use_ensemble: bool | None = None,
) -> SegmentationResult:
    """Run YOLO inference and, when running a real Ultralytics model with WireNet ensemble weights installed,
    fuse with Joanna's multi-model ensemble (`v217_ensemble.json`: `wirenet-r34.pt`, `v212_mitb1.pt`,
    `v212_convnext_t.pt`, `strand_head_v217.pt`, `poles-yolo11n-v218.pt`, `hf/dinov2-small/`,
    `hf/upernet-convnext-tiny/`).
    
    Supports both Single-Pass Mode (default) and Tiled Mode (`tiled=True` or `WIREFRAME_TILED=1`).
    If the model is detect-only (no masks), flags mask_available=False.
    original_shape is (h, w).
    """
    # Ultralytics BasePredictor.preprocess expects BGR numpy array convention
    # and performs `im = im[..., ::-1]` (BGR to RGB) internally. Convert RGB to
    # contiguous BGR here so the model receives the correct channel ordering.
    bgr_image = np.ascontiguousarray(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    params = get_inference_params()
    if tiled is not None:
        use_tiled = tiled
    else:
        use_tiled = bool(params.get("tiled", False))

    if use_tiled:
        yolo_res = sliced_predict(
            model,
            bgr_image,
            original_shape=original_shape,
            conf=params["conf"],
            tile=tile_size if tile_size is not None else int(params.get("tile_size", 640)),
            overlap=tile_overlap if tile_overlap is not None else float(params.get("tile_overlap", 0.3)),
            tile_iou=float(params.get("tile_iou", 0.9)),
            max_det=int(params.get("max_det", 1000)),
            device=params["device"],
        )
    else:
        predict_kwargs: dict[str, Any] = {
            "conf": params["conf"],
            "iou": params["iou"],
            "imgsz": params["imgsz"],
            "retina_masks": params["retina_masks"],
            "verbose": False,
        }
        if params["device"] and str(params["device"]).lower() != "auto":
            predict_kwargs["device"] = params["device"]

        with _INFERENCE_LOCK:
            results = model(
                bgr_image,
                **predict_kwargs,
            )

        if not results:
            yolo_res = SegmentationResult(mask_available=False)
        else:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes else []
            classes = r.boxes.cls.cpu().numpy().astype(int).tolist() if r.boxes else []
            confidences = r.boxes.conf.cpu().numpy().tolist() if r.boxes else []
            raw_names = getattr(r, "names", None) or getattr(model, "names", None) or {}
            names = {int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict) else {}

            if not r.masks:
                if not _is_real_ultralytics_model(model):
                    logger.warning("No masks returned by model. Ensure it is a YOLO-seg model.")
                yolo_res = SegmentationResult(
                    mask_available=False,
                    boxes=boxes,
                    classes=classes,
                    confidences=confidences,
                    names=names,
                )
            else:
                masks_list = []
                masks_data = r.masks.data.cpu().numpy()
                target_h, target_w = original_shape
                for i in range(masks_data.shape[0]):
                    mask = masks_data[i]
                    if mask.shape != (target_h, target_w):
                        scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        scaled_mask = mask
                    binary_mask = (scaled_mask > 0.5).astype(np.uint8) * 255
                    masks_list.append(binary_mask)

                yolo_res = SegmentationResult(
                    mask_available=True,
                    boxes=boxes,
                    classes=classes,
                    confidences=confidences,
                    masks=masks_list,
                    names=names,
                )

    # Fuse with Joanna's multi-model ensemble when running a real Ultralytics model (or when explicitly requested)
    should_run_ensemble = use_ensemble if use_ensemble is not None else _is_real_ultralytics_model(model)
    if should_run_ensemble:
        try:
            from backend.utils.cv.wirenet import is_wirenet_available, run_hybrid_ensemble_segmentation

            if is_wirenet_available():
                target_h, target_w = original_shape
                rgb_for_ens = (
                    cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    if rgb_image.shape[:2] != (target_h, target_w)
                    else rgb_image
                )
                ens_res = run_hybrid_ensemble_segmentation(
                    rgb_for_ens,
                    yolo_res,
                    imgsz=int(params.get("imgsz", 1024)),
                    conf_min=float(params.get("conf", 0.02)),
                    device=str(params.get("device", "auto")),
                )
                if ens_res.mask_available and ens_res.masks:
                    return ens_res
        except Exception as exc:
            logger.warning("Multi-model ensemble fusion failed (%s); falling back to YOLO detections.", exc)

    return yolo_res
