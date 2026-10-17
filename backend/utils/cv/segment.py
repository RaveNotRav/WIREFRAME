"""YOLO segmentation wrapper.

Complexity:
- Model forward pass: O(YOLO_GFLOPs)
- Mask rasterization/resizing: O(N) where N = width * height of original image
- Overall: O(YOLO) + O(N)
"""

import logging

import cv2
import numpy as np

import os
from typing import Any

from backend.config_store import get_setting
from backend.utils.cv.types import SegmentationResult

logger = logging.getLogger(__name__)


def get_inference_params() -> dict[str, Any]:
    """Retrieve inference settings from env (WIREFRAME_CONF, WIREFRAME_IOU, WIREFRAME_DEVICE) or config store."""
    params = get_setting("inference_params", {}) or {}

    env_conf = os.getenv("WIREFRAME_CONF")
    try:
        conf = float(env_conf) if env_conf is not None else float(params.get("conf", 0.25))
    except (ValueError, TypeError):
        conf = 0.25

    env_iou = os.getenv("WIREFRAME_IOU")
    try:
        iou = float(env_iou) if env_iou is not None else float(params.get("iou", 0.7))
    except (ValueError, TypeError):
        iou = 0.7

    env_device = os.getenv("WIREFRAME_DEVICE")
    device = env_device if env_device is not None else params.get("device", "auto")

    imgsz = int(params.get("imgsz", 1024))
    retina_masks = bool(params.get("retina_masks", True))

    return {
        "conf": conf,
        "iou": iou,
        "device": device,
        "imgsz": imgsz,
        "retina_masks": retina_masks,
    }


def segment_image(rgb_image: np.ndarray, model, original_shape: tuple[int, int]) -> SegmentationResult:
    """Run YOLO inference and scale masks back to original image size.
    
    If the model is detect-only (no masks), flags mask_available=False.
    original_shape is (h, w).
    """
    # Ultralytics BasePredictor.preprocess expects BGR numpy array convention
    # and performs `im = im[..., ::-1]` (BGR to RGB) internally. Convert RGB to
    # contiguous BGR here so the model receives the correct channel ordering.
    bgr_image = np.ascontiguousarray(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    params = get_inference_params()
    predict_kwargs = {
        "conf": params["conf"],
        "iou": params["iou"],
        "imgsz": params["imgsz"],
        "retina_masks": params["retina_masks"],
        "verbose": False,
    }
    if params["device"] and str(params["device"]).lower() != "auto":
        predict_kwargs["device"] = params["device"]

    results = model(
        bgr_image,
        **predict_kwargs,
    )
    
    if not results:
        return SegmentationResult(mask_available=False)
        
    r = results[0]
    
    boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes else []
    classes = r.boxes.cls.cpu().numpy().astype(int).tolist() if r.boxes else []
    confidences = r.boxes.conf.cpu().numpy().tolist() if r.boxes else []
    names = r.names if hasattr(r, 'names') else {}
    
    # Check if masks exist
    if not r.masks:
        logger.warning("No masks returned by model. Ensure it is a YOLO-seg model.")
        return SegmentationResult(
            mask_available=False,
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            names=names
        )
        
    # Scale masks to original image shape
    masks_list = []
    # r.masks.data is (N, H, W)
    masks_data = r.masks.data.cpu().numpy()
    
    target_h, target_w = original_shape
    
    for i in range(masks_data.shape[0]):
        # masks_data is float array (0.0 to 1.0)
        mask = masks_data[i]
        
        # Only resize back to original image size if shapes differ
        if mask.shape != (target_h, target_w):
            scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_mask = mask
        
        # Threshold to create binary mask
        binary_mask = (scaled_mask > 0.5).astype(np.uint8) * 255
        masks_list.append(binary_mask)
        
    return SegmentationResult(
        mask_available=True,
        boxes=boxes,
        classes=classes,
        confidences=confidences,
        masks=masks_list,
        names=names
    )
