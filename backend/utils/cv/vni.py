"""FFT-based Visual Noise Index (VNI) (Section 3.6.3, Equations 1-3, Algorithm 2).

    F(u, v) = sum_x sum_y I(x, y) w(x, y) exp(-j 2pi (ux/M + vy/N))       (1)
    P(u, v) = |F(u, v)|^2                                                  (2)
    VNI_raw = sum_{r > r_L} P(u, v) / sum_{r > 0} P(u, v)                  (3)
    VNI     = clip((VNI_raw - q01) / (q99 - q01), 0, 1)

Complexity:
- N = width * height (number of pixels in resized image, 640x640)
- FFT2: O(N log N)
- FFTShift: O(N)
- Masking: O(N)
- Overall: O(N log N)
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
import logging

import cv2
import numpy as np

from backend.config_store import get_setting
from backend.utils.cv.types import VNIResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def hann_2d(h: int, w: int) -> np.ndarray:
    """Separable 2-D Hann window (suppresses spectral leakage across tile borders)."""
    return np.outer(np.hanning(h), np.hanning(w))


@lru_cache(maxsize=8)
def radial_distance(h: int, w: int) -> np.ndarray:
    """Distance from the shifted-spectrum centre (M/2, N/2) for every (u, v)."""
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.hypot(uu, vv)


def prepare_gray(gray_or_rgb: np.ndarray, size: int = 640, kernel: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Algorithm 2 lines 1-3: grayscale, resize to size x size, Gaussian pre-filter."""
    g = cv2.cvtColor(gray_or_rgb, cv2.COLOR_RGB2GRAY) if gray_or_rgb.ndim == 3 else gray_or_rgb
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    if kernel and kernel > 1:
        g = cv2.GaussianBlur(g, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return g


def power_spectrum(gray: np.ndarray, remove_mean: bool = True) -> np.ndarray:
    """Shifted power spectrum P(u, v) = |F(u, v)|^2 of the Hann-windowed image."""
    h, w = gray.shape
    x = gray.astype(np.float64)
    if remove_mean:
        x = x - x.mean()
    windowed = x * hann_2d(h, w)
    f = np.fft.fftshift(np.fft.fft2(windowed))
    return (f.real * f.real + f.imag * f.imag)


def vni_raw(
    gray_or_rgb: np.ndarray,
    *,
    size: int = 640,
    kernel: int = 5,
    sigma: float = 1.0,
    radius_factor: float = 0.15,
    remove_mean: bool = True,
) -> tuple[float, np.ndarray]:
    """Return (VNI_raw, shifted power spectrum) for one image."""
    g = prepare_gray(gray_or_rgb, size=size, kernel=kernel, sigma=sigma)
    p = power_spectrum(g, remove_mean=remove_mean)
    r = radial_distance(*g.shape)
    r_low = radius_factor * (min(g.shape) / 2.0)
    denom = float(p[r > 0].sum())
    if denom <= 0.0:
        return 0.0, p
    num = float(p[r > r_low].sum())
    return num / denom, p


def normalize(raw: float, q01: float | None, q99: float | None) -> tuple[float, list[str]]:
    """Quantile rescaling of Algorithm 2 line 10."""
    flags: list[str] = []
    if q01 is None or q99 is None or q99 <= q01:
        flags.append("uncalibrated_vni_quantiles")
        return float(np.clip(raw, 0.0, 1.0)), flags
    val = (raw - q01) / (q99 - q01)
    return float(np.clip(val, 0.0, 1.0)), flags


def get_vni_band(vni: float) -> str:
    """Table 1.1 / Section 3.6.3 VNI bands:
    0.00 - 0.25 -> "Low"
    >0.25 - 0.50 -> "Moderate"
    >0.50 - 0.75 -> "High"
    >0.75 - 1.00 -> "Severe"
    """
    vni_cfg = get_setting("vni_params", {}) or {}
    bands = vni_cfg.get("bands", {}) if isinstance(vni_cfg, dict) else {}
    low_max = float(bands.get("Low", 0.25))
    mod_max = float(bands.get("Moderate", 0.50))
    high_max = float(bands.get("High", 0.75))

    if vni <= low_max:
        return "Low"
    if vni <= mod_max:
        return "Moderate"
    if vni <= high_max:
        return "High"
    return "Severe"


def calibrate_quantiles(images: Iterable[np.ndarray]) -> tuple[float, float]:
    """Compute (q_0.01, q_0.99) of VNI_raw over a reference corpus (Algorithm 2)."""
    vni_cfg = get_setting("vni_params", {}) or {}
    size = int(vni_cfg.get("size", 640))
    kernel = int(vni_cfg.get("gaussian_kernel", 5))
    sigma = float(vni_cfg.get("gaussian_sigma", 1.0))
    radius_factor = float(vni_cfg.get("radius_factor", 0.15))
    remove_mean = bool(vni_cfg.get("remove_mean", True))

    raws = [
        vni_raw(
            im,
            size=size,
            kernel=kernel,
            sigma=sigma,
            radius_factor=radius_factor,
            remove_mean=remove_mean,
        )[0]
        for im in images
    ]
    if len(raws) < 2:
        raise ValueError("need at least 2 images to calibrate VNI quantiles")
    q01, q99 = np.quantile(raws, [0.01, 0.99])
    return float(q01), float(q99)


def compute_vni(gray_image: np.ndarray) -> VNIResult:
    """Compute paper-conformant 2D-FFT VNI and return the spectrum image for visualization.

    Algorithm 2:
    1. Convert to grayscale (if needed) and check minimum dimensions.
    2. Resize to 640x640 (cv2.INTER_AREA) and apply 5x5 Gaussian blur (sigma=1.0).
    3. Subtract mean and apply 2D Hann window w(x, y) before 2D FFT and fftshift.
    4. Compute power spectrum P(u, v) = |F(u, v)|^2 and r_L = 0.15 * min(M, N) / 2.
    5. VNI_raw = sum_{r > r_L} P(u, v) / sum_{r > 0} P(u, v).
    6. Rescale using q01 and q99 clipped to [0.0, 1.0] and classify band.
    """
    if gray_image is None or gray_image.size == 0:
        return VNIResult(vni=0.0, vni_raw=0.0, band="Low")

    if gray_image.ndim == 3:
        gray_in = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_in = gray_image

    h, w = gray_in.shape[:2]
    if h < 4 or w < 4:
        return VNIResult(vni=0.0, vni_raw=0.0, band="Low")

    vni_cfg = get_setting("vni_params", {}) or {}
    size = int(vni_cfg.get("size", 640))
    kernel = int(vni_cfg.get("gaussian_kernel", 5))
    sigma = float(vni_cfg.get("gaussian_sigma", 1.0))
    radius_factor = float(vni_cfg.get("radius_factor", 0.15))
    remove_mean = bool(vni_cfg.get("remove_mean", True))
    q01 = vni_cfg.get("q01", 0.034001)
    q99 = vni_cfg.get("q99", 0.243397)

    raw_val, p_spec = vni_raw(
        gray_in,
        size=size,
        kernel=kernel,
        sigma=sigma,
        radius_factor=radius_factor,
        remove_mean=remove_mean,
    )
    val, flags = normalize(raw_val, q01, q99)
    band = get_vni_band(val)

    # Generate log-power spectrum image for visualization
    mag_log = np.log1p(np.sqrt(p_spec))
    mag_norm = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    spectrum_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_MAGMA)
    _, buffer = cv2.imencode(".png", spectrum_color)

    logger.debug("VNI computed: vni=%.4f (raw=%.4f, band=%s)", val, raw_val, band)
    return VNIResult(
        vni=round(val, 6),
        spectrum_png=buffer.tobytes(),
        vni_raw=round(raw_val, 6),
        band=band,
        flags=flags,
    )
