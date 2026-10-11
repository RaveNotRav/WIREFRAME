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

    # Initial guess for curve fit
    # a: roughly dx^2 / (8 * sag) if parabolic
    sag_approx = max(1.0, np.max(y) - (y[0]+y[-1])/2)
    a_guess = max(1.0, (dx**2) / (8 * sag_approx))
    x0_guess = (xmin + xmax) / 2
    b_guess = np.min(y) - a_guess
    
    try:
        popt, _ = curve_fit(cosh_func, x, y, p0=[a_guess, b_guess, x0_guess], maxfev=1000)
        y_fit = cosh_func(x, *popt)
        residual = float(np.sqrt(np.mean((y - y_fit)**2)))
        fitted_points = list(zip(x.tolist(), y_fit.tolist()))
        fitted_a = float(popt[0])
    except Exception as e:
        logger.warning(f"Catenary fit failed: {e}")
        popt = None
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
