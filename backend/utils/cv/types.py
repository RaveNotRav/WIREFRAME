"""Dataclasses for communication between CV pipeline stages."""

from dataclasses import dataclass, field

import numpy as np


def normalize_class_name(name: str) -> str:
    """Normalize class name: strip, lowercase, spaces/hyphens/slashes to underscore."""
    import re
    cleaned = re.sub(r"[\s\-/]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


CANONICAL_WIRE_CLASSES = ("power_line", "telecom_cable", "fiber_optic", "dead_abandoned")

DEFAULT_CLASS_ALIASES: dict[str, str] = {
    # Power Line variants
    "power_line": "power_line",
    "power-line": "power_line",
    "power line": "power_line",
    "power": "power_line",
    "powerline": "power_line",
    "power_cable": "power_line",
    "power_wire": "power_line",
    "electric_line": "power_line",
    "electric_wire": "power_line",
    "electric_cable": "power_line",
    "electrical_line": "power_line",
    "electrical_wire": "power_line",
    "electrical_cable": "power_line",
    "high_voltage": "power_line",
    "low_voltage": "power_line",
    "primary_line": "power_line",
    "secondary_line": "power_line",
    "service_drop": "power_line",
    "distribution_line": "power_line",
    "overhead_power": "power_line",
    "lv_line": "power_line",
    "hv_line": "power_line",
    # Telecom Cable variants
    "telecom_cable": "telecom_cable",
    "telecom-cable": "telecom_cable",
    "telecom cable": "telecom_cable",
    "telecom": "telecom_cable",
    "telco": "telecom_cable",
    "telco_cable": "telecom_cable",
    "telecom_wire": "telecom_cable",
    "telecom_line": "telecom_cable",
    "communication_cable": "telecom_cable",
    "communications_cable": "telecom_cable",
    "comm_cable": "telecom_cable",
    "coax": "telecom_cable",
    "coaxial": "telecom_cable",
    "coaxial_cable": "telecom_cable",
    "catv": "telecom_cable",
    "telephone": "telecom_cable",
    "telephone_line": "telecom_cable",
    "telephone_cable": "telecom_cable",
    "copper_cable": "telecom_cable",
    # Fiber Optic variants
    "fiber_optic": "fiber_optic",
    "fiber-optic": "fiber_optic",
    "fiber optic": "fiber_optic",
    "fiber": "fiber_optic",
    "fibre": "fiber_optic",
    "fiber_cable": "fiber_optic",
    "fibre_cable": "fiber_optic",
    "fiber_optic_cable": "fiber_optic",
    "fibre_optic_cable": "fiber_optic",
    "optical_fiber": "fiber_optic",
    "optical_fibre": "fiber_optic",
    "fo_cable": "fiber_optic",
    "ftth": "fiber_optic",
    # Dead / Abandoned variants
    "dead_abandoned": "dead_abandoned",
    "dead-abandoned": "dead_abandoned",
    "dead abandoned": "dead_abandoned",
    "dead": "dead_abandoned",
    "dead_wire": "dead_abandoned",
    "dead_cable": "dead_abandoned",
    "dead_line": "dead_abandoned",
    "abandoned": "dead_abandoned",
    "abandoned_wire": "dead_abandoned",
    "abandoned_cable": "dead_abandoned",
    "abandoned_line": "dead_abandoned",
    "unused_wire": "dead_abandoned",
    "dangling_wire": "dead_abandoned",
    "cut_wire": "dead_abandoned",
    "broken_wire": "dead_abandoned",
    "dropped_wire": "dead_abandoned",
    # Utility Pole variants
    "pole": "pole",
    "utility_pole": "pole",
    "pole_utility": "pole",
    "electric_pole": "pole",
    "power_pole": "pole",
    "telecom_pole": "pole",
    "wooden_pole": "pole",
    "concrete_pole": "pole",
    "utility_post": "pole",
    "post": "pole",
}

IGNORED_NON_WIRE_CLASSES = {
    "clock",
    "traffic_light",
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "tree",
    "building",
    "sky",
    "background",
}


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
