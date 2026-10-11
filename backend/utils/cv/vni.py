"""FFT-based Visual Noise Index (VNI).

Complexity:
- N = width * height (number of pixels in resized image, up to 512x512)
- FFT2: O(N log N)
- FFTShift: O(N)
- Masking: O(N)
- Overall: O(N log N)
"""

import logging

import cv2
import numpy as np

from backend.utils.cv.types import VNIResult

logger = logging.getLogger(__name__)


def compute_vni(gray_image: np.ndarray) -> VNIResult:
    """Compute VNI and return the spectrum image for visualization.
    
    Grayscale -> resize longest 512 -> /255 -> fft2 -> fftshift -> magnitude.
    Mask out radius < 8%*min(H,W) and axis bands |x-cx|<2, |y-cy|<2.
    VNI = sum(mag*mask) / sum(mag), clipped [0,1].
    """
    h, w = gray_image.shape
    longest = max(h, w)
    
    if longest > 0:
        scale = 512.0 / longest
        gray = cv2.resize(gray_image, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    else:
        gray = gray_image

    h, w = gray.shape
    if h < 4 or w < 4:
        return VNIResult(vni=0.0)

    gray_norm = gray.astype(np.float64) / 255.0
    f = np.fft.fft2(gray_norm)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    total = np.sum(magnitude)
    
    if total == 0:
        return VNIResult(vni=0.0)

    cy, cx = h // 2, w // 2
    radius_thr = int(0.08 * min(h, w))
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    mask = np.ones((h, w), dtype=bool)
    mask[dist < radius_thr] = False
    mask[np.abs(yy - cy)[:, 0] < 2, :] = False
    mask[:, np.abs(xx - cx)[0, :] < 2] = False

    vni = float(np.clip(np.sum(magnitude[mask]) / total, 0.0, 1.0))
    
    # Generate log-magnitude spectrum image for visualization
    # Avoid log(0)
    mag_log = np.log(1 + magnitude)
    mag_norm = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Colorize or just leave grayscale. Let's make it an attractive heatmap
    spectrum_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
    
    # Encode to PNG
    _, buffer = cv2.imencode('.png', spectrum_color)
    
    logger.debug(f"VNI computed: {vni:.4f}")
    return VNIResult(
        vni=vni,
        spectrum_png=buffer.tobytes()
    )
