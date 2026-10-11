"""Dataclasses for communication between CV pipeline stages."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PreprocessResult:
    """Output of the preprocessing stage."""
    quality_ok: bool
    reason: str
    gray: np.ndarray | None = None
    rgb: np.ndarray | None = None
    original_shape: tuple[int, int] | None = None  # (h, w)


@dataclass
class VNIResult:
    """Output of the VNI (Visual Noise Index) stage."""
    vni: float
    spectrum_png: bytes | None = None


@dataclass
class SegmentationResult:
    """Output of the YOLO segmentation stage."""
    mask_available: bool
    boxes: list[list[float]] = field(default_factory=list)  # [x1, y1, x2, y2]
    classes: list[int] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)  # Binary masks (H, W) aligned to original image
    names: dict[int, str] = field(default_factory=dict)


@dataclass
class CenterlineResult:
    """Output of the skeletonization and tracing stage."""
    polyline: list[tuple[int, int]] = field(default_factory=list)  # Ordered (x,y) points
    is_valid: bool = False


@dataclass
class CatenaryMetrics:
    """Output of the catenary fitting stage."""
    sag_ratio: float = 0.0
    residual: float = 0.0
    span_px: float = 0.0
    sag_px: float = 0.0
    sag_to_span: float = 0.0
    fitted_a: float = 0.0
    fitted_points: list[tuple[float, float]] = field(default_factory=list)
