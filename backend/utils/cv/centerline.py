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


def extract_centerline(mask: np.ndarray, bbox: list[float]) -> CenterlineResult:
    """Extract ordered polyline from a binary mask.
    
    Crops to bbox to save computation.
    Applies Zhang-Suen thinning (cv2.ximgproc.thinning).
    Traces from one endpoint to another.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Add small padding
    pad = 5
    h, w = mask.shape
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    
    crop = mask[y1:y2, x1:x2].copy()
    
    if np.sum(crop) == 0:
        return CenterlineResult(is_valid=False)
        
    # Zhang-Suen Thinning
    try:
        # cv2.ximgproc is available in opencv-contrib-python
        # Since we use opencv-python-headless, we might not have it.
        # Fallback to standard morphology if not available.
        if hasattr(cv2, 'ximgproc'):
            skel = cv2.ximgproc.thinning(crop, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        else:
            # Custom naive skeletonization
            skel = np.zeros_like(crop)
            img = crop.copy()
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()
                if cv2.countNonZero(img) == 0:
                    break
    except Exception as e:
        logger.warning(f"Thinning failed: {e}")
        return CenterlineResult(is_valid=False)
        
    endpoints = find_endpoints(skel)
    
    if not endpoints:
        # Loop or isolated, just pick first non-zero
        ys, xs = np.where(skel > 0)
        if len(ys) == 0:
            return CenterlineResult(is_valid=False)
        start_pt = (xs[0], ys[0])
    else:
        start_pt = endpoints[0]
        
    polyline_local = trace_polyline(skel, start_pt[0], start_pt[1])
    
    # Map back to global coordinates
    polyline_global = [(x + x1, y + y1) for x, y in polyline_local]
    
    return CenterlineResult(polyline=polyline_global, is_valid=True)
