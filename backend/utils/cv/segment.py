"""YOLO segmentation wrapper.

Complexity:
- Model forward pass: O(YOLO_GFLOPs)
- Mask rasterization/resizing: O(N) where N = width * height of original image
- Overall: O(YOLO) + O(N)
"""

import logging

import cv2
import numpy as np

from backend.utils.cv.types import SegmentationResult

logger = logging.getLogger(__name__)


def segment_image(rgb_image: np.ndarray, model, original_shape: tuple[int, int]) -> SegmentationResult:
    """Run YOLO inference and scale masks back to original image size.
    
    If the model is detect-only (no masks), flags mask_available=False.
    original_shape is (h, w).
    """
    results = model(rgb_image, verbose=False)
    
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
    # r.masks.data is (N, H, W) typically scaled down (e.g. 160x160)
    masks_data = r.masks.data.cpu().numpy()
    
    target_h, target_w = original_shape
    
    for i in range(masks_data.shape[0]):
        # masks_data is float array (0.0 to 1.0)
        mask = masks_data[i]
        
        # Resize back to original image size
        scaled_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
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
