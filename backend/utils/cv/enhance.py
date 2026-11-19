"""Adaptive enhancement and scene segmentation helpers (Section 3.6.2, Table 3.4, Algorithm 1).

Provides:
- CLAHE on CIELAB L channel
- Immerkaer (1996) noise estimation + NLM / bilateral denoising
- Blur-adaptive unsharp masking (Laplacian variance)
- Auto-gamma low-light recovery and glare/line-visibility checks
- HSV sky mask and Excess Green (ExG = 2G - R - B) vegetation mask helpers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# Immerkaer (1996) 3x3 mask for fast noise variance estimation
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


@dataclass(frozen=True)
class Quality:
    luminance_mean: float
    contrast_std: float
    noise_sigma: float
    laplacian_var: float
    glare_fraction: float = 0.0
    line_visibility_ok: bool = True


@dataclass(frozen=True)
class Enhanced:
    rgb: np.ndarray
    gray: np.ndarray
    before: Quality
    after: Quality
    steps: tuple[str, ...]
    flags: tuple[str, ...] = field(default_factory=tuple)


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise standard deviation estimator on a uint8 grayscale image."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, _IMMERKAER, borderType=cv2.BORDER_DEFAULT)
    inner = conv[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2.0) * np.abs(inner).sum() / (6.0 * (w - 2) * (h - 2)))


def check_glare_and_line_visibility(gray: np.ndarray) -> tuple[float, bool]:
    """Estimate specular glare fraction and whether dark linear features remain distinguishable."""
    if gray.size == 0:
        return 0.0, False
    glare_frac = float(np.mean(gray >= 250))
    # Check local gradient energy so severe overexposure / washout is flagged
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_p90 = float(np.percentile(np.hypot(gx, gy), 90))
    visible = glare_frac < 0.65 and grad_p90 >= 8.0
    return glare_frac, visible


def measure(rgb: np.ndarray) -> Quality:
    """Measure luminance, contrast, Immerkaer noise sigma, Laplacian variance, and glare."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    glare_frac, vis_ok = check_glare_and_line_visibility(gray)
    return Quality(
        luminance_mean=float(gray.mean()),
        contrast_std=float(gray.std()),
        noise_sigma=noise_sigma(gray),
        laplacian_var=float(lap.var()),
        glare_fraction=glare_frac,
        line_visibility_ok=vis_ok,
    )


def auto_gamma_rgb(rgb: np.ndarray, target_mean: float = 115.0) -> tuple[np.ndarray, float]:
    """Apply automatic gamma correction on luminance when scene is underexposed or overexposed."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.clip(gray.mean(), 1.0, 254.0))
    gamma = math.log(target_mean / 255.0) / math.log(mean_val / 255.0)
    gamma = float(np.clip(gamma, 0.45, 2.2))
    if abs(gamma - 1.0) < 0.05:
        return rgb, 1.0
    lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(rgb, lut), gamma


def clahe_lab(rgb: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L channel in CIELAB space; A and B channels are untouched."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l_eq = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2RGB)


def denoise(rgb: np.ndarray, h: float = 6.0, template: int = 7, search: int = 21) -> np.ndarray:
    """Non-local means denoising on colour image."""
    return cv2.fastNlMeansDenoisingColored(rgb, None, h, h, template, search)


def edge_denoise(rgb: np.ndarray, d: int = 7, sigma_color: float = 35.0, sigma_space: float = 35.0) -> np.ndarray:
    """Bilateral filter: smooths sky and foliage texture while keeping 1-3 px wire edges intact."""
    return cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def unsharp(rgb: np.ndarray, sigma: float = 1.0, amount: float = 0.6) -> np.ndarray:
    """Unsharp mask: I + amount * (I - G_sigma * I), clipped to [0, 255]."""
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(rgb, 1.0 + amount, blur, -amount, 0)


def enhance(
    rgb: np.ndarray,
    *,
    low_light_mean: float = 85.0,
    high_glare_mean: float = 200.0,
    low_contrast_std: float = 42.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple[int, int] = (8, 8),
    noise_sigma_threshold: float = 8.0,
    nlm_h: float = 6.0,
    nlm_template_window: int = 7,
    nlm_search_window: int = 21,
    blur_laplacian_threshold: float = 120.0,
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 0.6,
) -> Enhanced:
    """Algorithm 1 lines 5-17. Each stage runs only when its trigger fires."""
    before = measure(rgb)
    out = rgb
    steps: list[str] = []
    flags: list[str] = []

    if before.luminance_mean < low_light_mean:
        out, g = auto_gamma_rgb(out)
        if abs(g - 1.0) >= 0.05:
            steps.append(f"auto_gamma(gamma={g:.2f})")
        flags.append("low_light_enhanced")

    if before.glare_fraction > 0.15:
        flags.append("high_glare_detected")
    if not before.line_visibility_ok:
        flags.append("low_line_visibility")

    if (
        before.luminance_mean < low_light_mean
        or before.luminance_mean > high_glare_mean
        or before.contrast_std < low_contrast_std
    ):
        out = clahe_lab(out, clahe_clip_limit, clahe_tile_grid[0])
        steps.append(f"clahe_lab(clip={clahe_clip_limit},tile={clahe_tile_grid[0]})")

    if before.noise_sigma > noise_sigma_threshold:
        out = denoise(out, nlm_h, nlm_template_window, nlm_search_window)
        steps.append(f"nlm_denoise(h={nlm_h})")

    current_lap = float(cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    if current_lap < blur_laplacian_threshold:
        out = unsharp(out, unsharp_sigma, unsharp_amount)
        steps.append(f"unsharp(sigma={unsharp_sigma},amount={unsharp_amount})")

    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    after = measure(out)
    return Enhanced(rgb=out, gray=gray, before=before, after=after, steps=tuple(steps), flags=tuple(flags))


def enhance_strong(
    rgb: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile: int = 8,
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 0.6,
) -> Enhanced:
    """Unconditional visibility pass for fallback segmentation retry:
    CLAHE on LAB L -> bilateral edge-preserving denoise -> gentle unsharp mask.
    """
    before = measure(rgb)
    out = clahe_lab(rgb, clahe_clip_limit * 1.25, clahe_tile)
    out = edge_denoise(out)
    out = unsharp(out, unsharp_sigma, max(unsharp_amount, 0.8))
    steps = (
        f"clahe_lab(clip={clahe_clip_limit * 1.25:.2f},tile={clahe_tile})",
        "bilateral(d=7,sigma=35)",
        f"unsharp(sigma={unsharp_sigma},amount={max(unsharp_amount, 0.8)})",
    )
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    return Enhanced(rgb=out, gray=gray, before=before, after=measure(out), steps=steps)


def detect_sky_mask_hsv(rgb: np.ndarray) -> np.ndarray:
    """Return boolean mask of sky pixels using HSV + top-connected component heuristics."""
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue_sky = (hue >= 85) & (hue <= 135) & (sat >= 20) & (val >= 110)
    overcast = (sat <= 55) & (val >= 165)
    cand = ((blue_sky | overcast) & (np.arange(h)[:, None] < int(0.85 * h))).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(cand, connectivity=8)
    if num_labels <= 1:
        return np.zeros((h, w), dtype=bool)
    top_labels = set(np.unique(labels[: max(1, int(0.15 * h)), :])) - {0}
    if not top_labels:
        return np.zeros((h, w), dtype=bool)
    return np.isin(labels, list(top_labels))


def detect_vegetation_exg(rgb: np.ndarray, exg_min: float = 25.0, min_area: int = 500) -> list[np.ndarray]:
    """Detect foliage masks using Excess Green index ExG = 2G - R - B + HSV green gate."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (exg >= exg_min) & (hue >= 22) & (hue <= 98) & (sat >= 30) & (val >= 25) & (g > r) & (g > b)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    masks: list[np.ndarray] = []
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            masks.append(labels == lbl)
    return masks
