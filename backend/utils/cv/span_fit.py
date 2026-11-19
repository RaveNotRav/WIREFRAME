"""S15 (register D-32): measure a traced wire only where it behaves like one hanging span.

A wire hanging between two supports is a smooth catenary: one curve explains its whole trace. A trace that switches to
a neighbouring cable at a crossing, or runs through a pole attachment, is two near-straight pieces meeting at an angle.
When the catenary leaves a noticeable residual, the trace is also fitted as two straight lines broken at one point.
If two lines explain it much better than the curve (`kink_ratio`), the trace is kinked, not sagging: it is split at the
break and each piece is fitted on its own.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from backend.utils.cv.catenary import calculate_sag_metrics
from backend.utils.cv.types import CatenaryMetrics

ALGORITHM = "catenary+kink-split/1"


def _line_sse(p: np.ndarray) -> float:
    """Sum of squared perpendicular distances to the total-least-squares line through ``p``."""
    if len(p) < 3:
        return 0.0
    q = p - p.mean(0)
    s = np.linalg.svd(q, compute_uv=False)
    return float(s[-1] ** 2)


def two_line(points: np.ndarray, min_piece: float = 0.15, steps: int = 48) -> tuple[float, int]:
    """(RMS perpendicular residual of the best two-line fit, index of its break point)."""
    p = np.asarray(points, float)
    n = len(p)
    lo, hi = max(3, int(min_piece * n)), min(n - 3, int((1 - min_piece) * n))
    if hi <= lo:
        return float("inf"), -1
    best, k_best = float("inf"), -1
    for k in np.unique(np.linspace(lo, hi, steps).astype(int)):
        sse = _line_sse(p[: k + 1]) + _line_sse(p[k:])
        if sse < best:
            best, k_best = sse, int(k)
    return float(np.sqrt(best / n)), k_best


def fit_span(
    points: list[tuple[float, float]] | np.ndarray,
    *,
    kink_ratio: float = 0.5,
    kink_check_from: float = 0.004,
    n_min: int = 20,
) -> CatenaryMetrics:
    """Fit a catenary to `points`, splitting kinked traces into their best single-curve sub-span when appropriate."""
    pts = np.asarray(points, dtype=float)
    base_list = [(float(x), float(y)) for x, y in pts]
    f = calculate_sag_metrics(base_list)
    if f.fit_failed or f.span_px <= 0 or f.residual <= 0:
        return f

    rel_residual = f.residual / f.span_px
    if rel_residual < kink_check_from:
        return f

    rms2, k = two_line(pts)
    if k < 0 or rms2 >= kink_ratio * f.residual:
        return f

    pieces = [pts[: k + 1], pts[k:]]
    candidates: list[tuple[CatenaryMetrics, int]] = []
    for pc in pieces:
        if len(pc) >= n_min:
            sub_fit = calculate_sag_metrics([(float(x), float(y)) for x, y in pc])
            if not sub_fit.fit_failed and sub_fit.span_px > 0:
                candidates.append((sub_fit, len(pc)))

    if not candidates:
        return replace(f, fit_failed=True, flags=list(dict.fromkeys([*f.flags, "kinked_trace"])))

    best_fit, _ = max(candidates, key=lambda item: item[0].span_px)
    return replace(best_fit, flags=list(dict.fromkeys([*best_fit.flags, "kink_split"])))
