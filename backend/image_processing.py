"""Image processing: VNI (FFT-based), enhancement, and authenticity checking."""

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_EXIF_MAKE = 0x010F       # 271
_EXIF_MODEL = 0x0110      # 272
_EXIF_SOFTWARE = 0x0131   # 305


def compute_visual_noise_index(image: np.ndarray) -> float:
    """FFT-based Visual Noise Index (delegates to paper-conformant compute_vni)."""
    from backend.utils.cv.vni import compute_vni
    return compute_vni(image).vni


def check_authenticity(image_path: str) -> dict[str, Any]:
    """Advisory authenticity scoring (never blocks processing).

    Start 100; deductions: -25 no EXIF, -15 no make/model,
    -40 AI software tag, -30 FFT peakiness>40.
    >=70 likely_authentic, 40-69 uncertain, <40 flagged_for_review.
    """
    score = 100
    reasons: list[str] = []

    try:
        pil_img = Image.open(image_path)
        exif = pil_img.getexif()
    except Exception as exc:
        logger.warning("Cannot read EXIF: %s", exc)
        exif = None

    if not exif:
        score -= 25
        reasons.append("no_exif_data")
    else:
        if exif.get(_EXIF_MAKE) is None and exif.get(_EXIF_MODEL) is None:
            score -= 15
            reasons.append("no_camera_make_model")
        software = str(exif.get(_EXIF_SOFTWARE, "")).lower()
        ai_tags = ["stable diffusion", "midjourney", "dall-e", "dalle",
                   "comfyui", "automatic1111", "novelai"]
        if any(t in software for t in ai_tags):
            score -= 40
            reasons.append("ai_generator_detected")

    try:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            mag = np.abs(np.fft.fftshift(
                np.fft.fft2(gray.astype(np.float64) / 255.0)))
            mean_m = np.mean(mag)
            if mean_m > 0 and float(np.max(mag) / mean_m) > 40:
                score -= 30
                reasons.append("fft_peakiness_high")
    except Exception as exc:
        logger.warning("FFT peakiness check failed: %s", exc)

    score = max(0, min(100, score))
    if score >= 70:
        classification = "likely_authentic"
    elif score >= 40:
        classification = "uncertain"
    else:
        classification = "flagged_for_review"

    logger.debug("Authenticity: score=%d class=%s", score, classification)
    return {"score": score, "classification": classification, "reasons": reasons}


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Classical enhancement pipeline on a COPY (original untouched).

    1. CLAHE (clip 2.0, tile 8x8) on LAB-L
    2. Gamma 1.4 if mean brightness < 90
    3. fastNlMeansDenoisingColored(5, 5, 7, 21)
    4. Unsharp mask (sigma=3, weights 1.5/-0.5)
    """
    out = image.copy()

    # 1 - CLAHE
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # 2 - Gamma
    mean_bright = float(np.mean(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)))
    if mean_bright < 90:
        inv_g = 1.0 / 1.4
        lut = np.clip(((np.arange(256) / 255.0) ** inv_g) * 255,
                      0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
        logger.debug("Gamma applied, brightness was %.1f", mean_bright)

    # 3 - Denoise
    out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)

    # 4 - Unsharp
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
    out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)

    logger.debug("Enhancement complete")
    return out
