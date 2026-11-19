"""Catenary fitting and sag metric calculations (Section 3.6.5, Equations 6-8, Algorithm 5).

Complexity:
- n = number of points in polyline
- p = number of parameters (a, b, x0)
- Curve Fit: O(n * iter_count) via Levenberg-Marquardt
- Chord distance: O(n)
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.optimize import curve_fit

from backend.utils.cv.types import CatenaryMetrics

logger = logging.getLogger(__name__)


def cosh_func(x, a, b, x0):
    # a * cosh((x - x0) / a) + b
    # Cap exponent to avoid overflow
    arg = np.clip((x - x0) / a, -700, 700)
    return a * np.cosh(arg) + b


def _perpendicular_chord_sag(
    x: np.ndarray,
    y_eval: np.ndarray,
    xa: float,
    ya: float,
    xb: float,
    yb: float,
) -> float:
    """Largest perpendicular distance delta from a downward-sagging curve to the endpoint chord."""
    dx = xb - xa
    dy = yb - ya
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return 0.0
    # Signed vertical offset below chord (positive when y_eval > chord_y in image coordinates)
    if abs(dx) > 1e-9:
        slope = dy / dx
        chord_y = ya + slope * (x - xa)
        below = y_eval - chord_y
        # Perpendicular distance from (x, y_eval) to the line through (xa, ya)-(xb, yb)
        perp = ((y_eval - ya) * dx - (x - xa) * dy) / length
        valid = perp[below > 0]
        return float(np.max(valid)) if len(valid) > 0 else 0.0
    return 0.0


def calculate_sag_metrics(polyline: list[tuple[int, int]] | list[tuple[float, float]]) -> CatenaryMetrics:
    """Fit a catenary and compute legacy + paper-conformant perpendicular chord sag metrics."""
    if len(polyline) < 3:
        return CatenaryMetrics(fit_failed=True, flags=["too_short_to_fit"])

    pts = np.array(polyline, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]

    # Sort by X
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Legacy Metrics
    xmin, xmax = float(x[0]), float(x[-1])
    ymin, ymax = float(y[0]), float(y[-1])
    dx = xmax - xmin
    if dx == 0:
        sag_ratio = 0.0
    else:
        sag_ratio = float((np.max(y) - np.min(y)) / dx)

    chord_mid = (y[0] + y[-1]) / 2.0
    sag_down = max(0.0, float(np.max(y) - chord_mid))
    sag_up = max(0.0, float(chord_mid - np.min(y)))
    x0_guess = (xmin + xmax) / 2.0

    a_arch = max(1.0, (dx**2) / (8.0 * max(1.0, sag_up if sag_up > 0 else 1.0)))
    b_arch = float(np.min(y)) - a_arch

    a_hang = -max(1.0, (dx**2) / (8.0 * max(1.0, sag_down if sag_down > 0 else 1.0)))
    b_hang = float(np.max(y)) - a_hang

    candidates = [
        [a_arch, b_arch, x0_guess],
        [a_hang, b_hang, x0_guess],
    ]

    best_residual = float("inf")
    best_popt = None
    best_y_fit = None

    for idx, p0 in enumerate(candidates):
        try:
            popt, _ = curve_fit(cosh_func, x, y, p0=p0, maxfev=1000)
            y_fit = cosh_func(x, *popt)
            res = float(np.sqrt(np.mean((y - y_fit) ** 2)))
            if res < best_residual:
                best_residual = res
                best_popt = popt
                best_y_fit = y_fit
        except Exception as e:
            logger.debug("Candidate %d fit failed: %s", idx, e)

    flags: list[str] = []
    if best_popt is not None and np.all(np.isfinite(best_popt)):
        residual = best_residual
        fitted_points = list(zip(x.tolist(), best_y_fit.tolist()))
        fitted_a = float(best_popt[0])
        # Note: in image coordinates (y downwards), a hanging catenary has fitted_a < 0
        # (equivalently a_up = -fitted_a > 0 in flipped y-up frame).
        fit_failed = False
    else:
        logger.warning("Catenary fit failed")
        residual = float("inf")
        fitted_points = []
        fitted_a = 0.0
        fit_failed = True
        flags.append("catenary_fit_failed")

    # Chord length L = span_px between the endpoints
    span_px = float(np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2))

    sag_px = 0.0
    perp_sag_px = 0.0
    if span_px > 0 and dx > 0:
        slope = (ymax - ymin) / dx
        chord_y = ymin + slope * (x - xmin)
        # In image coords, y increases downwards. Sag is when y > chord_y.
        deviations = y - chord_y
        sag_px = float(np.max(deviations))
        y_curve = best_y_fit if best_y_fit is not None else y
        perp_sag_px = _perpendicular_chord_sag(
            x,
            y_curve,
            xmin,
            float(y_curve[0]),
            xmax,
            float(y_curve[-1]),
        )

    # Cap at 0 (if curve bows upwards, not a sag)
    sag_px = max(0.0, sag_px)
    perp_sag_px = max(0.0, perp_sag_px)
    sag_to_span = sag_px / span_px if span_px > 0 else 0.0
    perp_sag_ratio = perp_sag_px / span_px if span_px > 0 else 0.0

    # If curve bows upward (arch: fitted_a > 0 in y-down frame and sag_px == 0), record non-hanging
    if not fit_failed and fitted_a > 0 and sag_px == 0.0:
        flags.append("arch_orientation")

    return CatenaryMetrics(
        sag_ratio=float(sag_ratio),
        residual=residual,
        span_px=span_px,
        sag_px=sag_px,
        sag_to_span=sag_to_span,
        fitted_a=fitted_a,
        fitted_points=fitted_points,
        fit_failed=fit_failed,
        perp_sag_px=perp_sag_px,
        perp_sag_ratio=perp_sag_ratio,
        flags=flags,
    )
