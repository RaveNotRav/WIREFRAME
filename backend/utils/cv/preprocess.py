"""Image preprocessing and quality validation.

Complexity:
- N = width * height (number of pixels)
- Grayscale conversion: O(N)
- Resize: O(N)
- Denoise: O(N)
- Overall Preprocessing: O(N)
"""

import logging

import cv2
import numpy as np

from backend.utils.cv.types import PreprocessResult

logger = logging.getLogger(__name__)


from PIL import Image, ImageOps


def validate_and_preprocess(image_path: str) -> PreprocessResult:
    """Validate image quality and prepare arrays for downstream stages.
    
    1. Corrupt file check
    2. Minimum resolution check (e.g. 100x100)
    3. Blurry check (Laplacian variance < 50)
    4. Dark check (Mean brightness < 15)
    
    Returns PreprocessResult with quality flags and prepped images.
    """
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return PreprocessResult(quality_ok=False, reason="corrupt_file")

    h, w = img.shape[:2]
    original_shape = (h, w)

    if h < 100 or w < 100:
        return PreprocessResult(quality_ok=False, reason="resolution_too_low", original_shape=original_shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mean_bright = np.mean(gray)
    if mean_bright < 15.0:
        return PreprocessResult(quality_ok=False, reason="too_dark", original_shape=original_shape)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50.0:
        return PreprocessResult(quality_ok=False, reason="too_blurry", original_shape=original_shape)

    # Prepare RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Optional enhancement (Gaussian denoise, contrast) could be done here
    # For now, just pass the clean arrays
    rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    logger.debug(f"Preprocessed {image_path}: {w}x{h}, blur={laplacian_var:.1f}, bright={mean_bright:.1f}")
    return PreprocessResult(
        quality_ok=True,
        reason="",
        gray=gray,
        rgb=rgb,
        original_shape=original_shape
    )
