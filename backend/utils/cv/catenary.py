"""Catenary fitting and sag metric calculations.

Complexity:
- n = number of points in polyline
- p = number of parameters (a, b, x0)
- Curve Fit: O(n * iter_count) via Levenberg-Marquardt
- Chord distance: O(n)
"""

import logging

import numpy as np
from scipy.optimize import curve_fit

from backend.utils.cv.types import CatenaryMetrics

logger = logging.getLogger(__name__)


def cosh_func(x, a, b, x0):
    # a * cosh((x - x0) / a) + b
    # Cap exponent to avoid overflow
    arg = np.clip((x - x0) / a, -700, 700)
    return a * np.cosh(arg) + b


def calculate_sag_metrics(polyline: list[tuple[int, int]]) -> CatenaryMetrics:
    """Fit a catenary and compute legacy + true sag metrics."""
    if len(polyline) < 3:
        return CatenaryMetrics()
        
    pts = np.array(polyline, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Sort by X
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    # Legacy Metrics
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]
    dx = xmax - xmin
    if dx == 0:
        sag_ratio = 0.0
    else:
        # Based on phase 0 prompt: sag_ratio = (ymax-ymin)/(xmax-xmin)
        # Assuming y is positive downwards (image coords)
        # Actually in phase 0 context: sag_ratio = abs(ymax-ymin)/dx
        # Let's use the bounding box ratio
        sag_ratio = (np.max(y) - np.min(y)) / dx

    chord_mid = (y[0] + y[-1]) / 2.0
    sag_down = max(0.0, float(np.max(y) - chord_mid))
    sag_up = max(0.0, float(chord_mid - np.min(y)))
    x0_guess = (xmin + xmax) / 2.0

    a_arch = max(1.0, (dx**2) / (8.0 * max(1.0, sag_up if sag_up > 0 else 1.0)))
    b_arch = np.min(y) - a_arch

    a_hang = -max(1.0, (dx**2) / (8.0 * max(1.0, sag_down if sag_down > 0 else 1.0)))
    b_hang = np.max(y) - a_hang

    candidates = [
        [a_arch, b_arch, x0_guess],
        [a_hang, b_hang, x0_guess],
    ]

    best_residual = float('inf')
    best_popt = None
    best_y_fit = None

    for idx, p0 in enumerate(candidates):
        try:
            popt, _ = curve_fit(cosh_func, x, y, p0=p0, maxfev=1000)
            y_fit = cosh_func(x, *popt)
            res = float(np.sqrt(np.mean((y - y_fit)**2)))
            if res < best_residual:
                best_residual = res
                best_popt = popt
                best_y_fit = y_fit
        except Exception as e:
            logger.debug(f"Candidate {idx} fit failed: {e}")

    if best_popt is not None:
        residual = best_residual
        fitted_points = list(zip(x.tolist(), best_y_fit.tolist()))
        fitted_a = float(best_popt[0])
    else:
        logger.warning("Catenary fit failed")
        residual = float('inf')
        fitted_points = []
        fitted_a = 0.0
        
    # True Metrics
    span_px = float(np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2))
    
    sag_px = 0.0
    if span_px > 0:
        # Line from p1 to p2: Ax + By + C = 0
        # Max orthogonal distance
        # Wait, the prompt says "max vertical deviation of the fitted curve from the chord"
        # Vertical deviation means delta Y at the same X.
        # Chord Y at X:
        if dx > 0:
            slope = (ymax - ymin) / dx
            chord_y = ymin + slope * (x - xmin)
            # In image coords, y increases downwards. Sag is when y is > chord_y.
            deviations = y - chord_y
            sag_px = float(np.max(deviations))
        
    # Cap at 0 (if curve bows upwards, not a sag)
    sag_px = max(0.0, sag_px)
    sag_to_span = sag_px / span_px if span_px > 0 else 0.0
    
    return CatenaryMetrics(
        sag_ratio=float(sag_ratio),
        residual=residual,
        span_px=span_px,
        sag_px=sag_px,
        sag_to_span=sag_to_span,
        fitted_a=fitted_a,
        fitted_points=fitted_points
    )
