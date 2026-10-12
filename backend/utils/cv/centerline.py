"""Zhang-Suen thinning and skeleton tracing.

Complexity:
- N = width * height of bounding box crop
- Zhang-Suen Thinning: O(N * max_iterations) ~ O(N * thickness)
- Skeleton Tracing: O(L) where L is length of skeleton
- Overall: O(N * thickness)
"""

import logging

import cv2
import numpy as np

from backend.utils.cv.types import CenterlineResult

logger = logging.getLogger(__name__)


def find_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    """Find endpoints of a 1px skeleton."""
    endpoints = []
    # Kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    filtered = cv2.filter2D(skel // 255, -1, kernel)
    # Endpoints have exactly 1 neighbor (10 + 1 = 11)
    # Isolated points have 0 neighbors (10)
    y_coords, x_coords = np.where(filtered == 11)
    
    for y, x in zip(y_coords, x_coords):
        endpoints.append((x, y))
        
    return endpoints


def trace_polyline(skel: np.ndarray, start_x: int, start_y: int) -> list[tuple[int, int]]:
    """Trace a 1px skeleton from a starting point."""
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    
    polyline = []
    curr_x, curr_y = start_x, start_y
    
    while True:
        polyline.append((curr_x, curr_y))
        visited[curr_y, curr_x] = True
        
        # Find next unvisited neighbor
        next_x, next_y = None, None
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = curr_y + dy, curr_x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skel[ny, nx] > 0 and not visited[ny, nx]:
                        next_x, next_y = nx, ny
                        break
            if next_x is not None:
                break
                
        if next_x is None:
            break
            
        curr_x, curr_y = next_x, next_y
        
    return polyline


MIN_MASK_COMPONENT_PIXELS = 15
MIN_COMPONENT_PIXELS = MIN_MASK_COMPONENT_PIXELS


def extract_centerline(mask: np.ndarray, bbox: list[float]) -> CenterlineResult:
    """Extract ordered polyline from a binary mask.
    
    Crops to bbox to save computation.
    Removes connected components of the cropped mask smaller than MIN_MASK_COMPONENT_PIXELS.
    Takes median y per column if W >= H, else median x per row if H > W.
    Returns points sorted along span, or invalid if < 2 points.
    Note that end caps add ~half wire thickness.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Add small padding
    pad = 5
    h, w = mask.shape
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    
    crop = mask[y1:y2, x1:x2].copy()
    crop_bin = (crop > 0).astype(np.uint8)
    
    if np.sum(crop_bin) == 0:
        return CenterlineResult(is_valid=False)
        
    # Remove connected components of the cropped mask smaller than MIN_MASK_COMPONENT_PIXELS
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crop_bin, connectivity=8)
    filtered = np.zeros_like(crop_bin)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= MIN_MASK_COMPONENT_PIXELS:
            filtered[labels == lbl] = 1

    if cv2.countNonZero(filtered) == 0:
        return CenterlineResult(is_valid=False)

    crop_h, crop_w = filtered.shape
    polyline_global: list[tuple[float, float]] = []

    # If W >= H: median y per column; else if H > W: median x per row
    if crop_w >= crop_h:
        for x_rel in range(crop_w):
            ys = np.where(filtered[:, x_rel] > 0)[0]
            if len(ys) > 0:
                med_y = float(np.median(ys))
                polyline_global.append((float(x_rel + x1), float(med_y + y1)))
    else:
        for y_rel in range(crop_h):
            xs = np.where(filtered[y_rel, :] > 0)[0]
            if len(xs) > 0:
                med_x = float(np.median(xs))
                polyline_global.append((float(med_x + x1), float(y_rel + y1)))

    if len(polyline_global) < 2:
        return CenterlineResult(is_valid=False)

    return CenterlineResult(polyline=polyline_global, is_valid=True)

