"""Dataclasses for communication between CV pipeline stages."""

from dataclasses import dataclass, field

import numpy as np


def normalize_class_name(name: str) -> str:
    """Normalize class name: strip, lowercase, spaces/hyphens/slashes to underscore."""
    import re
    cleaned = re.sub(r"[\s\-/]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


CANONICAL_WIRE_CLASSES = ("power_distribution_line", "communication_cable", "dead_wire")
CANONICAL_CONTEXT_CLASSES = ("vegetation",)

DEFAULT_CLASS_ALIASES: dict[str, str] = {
    # Power Line variants -> power_distribution_line
    "power_distribution_line": "power_distribution_line",
    "power_distribution_lines": "power_distribution_line",
    "power-distribution-line": "power_distribution_line",
    "power-distribution-lines": "power_distribution_line",
    "power distribution line": "power_distribution_line",
    "power distribution lines": "power_distribution_line",
    "power_line": "power_distribution_line",
    "power_lines": "power_distribution_line",
    "power-line": "power_distribution_line",
    "power-lines": "power_distribution_line",
    "power line": "power_distribution_line",
    "power lines": "power_distribution_line",
    "power": "power_distribution_line",
    "powerline": "power_distribution_line",
    "powerlines": "power_distribution_line",
    "power_cable": "power_distribution_line",
    "power_wire": "power_distribution_line",
    "electric_line": "power_distribution_line",
    "electric_wire": "power_distribution_line",
    "electric_cable": "power_distribution_line",
    "electrical_line": "power_distribution_line",
    "electrical_wire": "power_distribution_line",
    "electrical_cable": "power_distribution_line",
    "high_voltage": "power_distribution_line",
    "low_voltage": "power_distribution_line",
    "primary_line": "power_distribution_line",
    "secondary_line": "power_distribution_line",
    "service_drop": "power_distribution_line",
    "distribution_line": "power_distribution_line",
    "overhead_power": "power_distribution_line",
    "lv_line": "power_distribution_line",
    "hv_line": "power_distribution_line",
    # Communication / Telecom / Fiber Optic variants -> communication_cable
    "communication_cable": "communication_cable",
    "communication-cable": "communication_cable",
    "communication cable": "communication_cable",
    "communication_cables": "communication_cable",
    "communication_wire": "communication_cable",
    "communication-wire": "communication_cable",
    "communication wire": "communication_cable",
    "communication_line": "communication_cable",
    "communications_cable": "communication_cable",
    "comm_cable": "communication_cable",
    "telecom_cable": "communication_cable",
    "telecom-cable": "communication_cable",
    "telecom cable": "communication_cable",
    "telecom_wires": "communication_cable",
    "telecom-wires": "communication_cable",
    "telecom wires": "communication_cable",
    "telecom": "communication_cable",
    "telco": "communication_cable",
    "telco_cable": "communication_cable",
    "telecom_wire": "communication_cable",
    "telecom_line": "communication_cable",
    "coax": "communication_cable",
    "coaxial": "communication_cable",
    "coaxial_cable": "communication_cable",
    "catv": "communication_cable",
    "telephone": "communication_cable",
    "telephone_line": "communication_cable",
    "telephone_cable": "communication_cable",
    "copper_cable": "communication_cable",
    "fiber_optic": "communication_cable",
    "fiber-optic": "communication_cable",
    "fiber optic": "communication_cable",
    "fiber_optic_drop": "communication_cable",
    "fiber-optic-drop": "communication_cable",
    "fiber optic drop": "communication_cable",
    "fiber": "communication_cable",
    "fibre": "communication_cable",
    "fiber_cable": "communication_cable",
    "fibre_cable": "communication_cable",
    "fiber_optic_cable": "communication_cable",
    "fibre_optic_cable": "communication_cable",
    "optical_fiber": "communication_cable",
    "optical_fibre": "communication_cable",
    "fo_cable": "communication_cable",
    "ftth": "communication_cable",
    # Dead / Abandoned variants -> dead_wire
    "dead_wire": "dead_wire",
    "dead-wire": "dead_wire",
    "dead wire": "dead_wire",
    "dead_wires": "dead_wire",
    "dead_abandoned": "dead_wire",
    "dead-abandoned": "dead_wire",
    "dead abandoned": "dead_wire",
    "dead": "dead_wire",
    "dead_cable": "dead_wire",
    "dead_line": "dead_wire",
    "abandoned": "dead_wire",
    "abandoned_wire": "dead_wire",
    "abandoned-wire": "dead_wire",
    "abandoned wire": "dead_wire",
    "abandoned_cable": "dead_wire",
    "abandoned_line": "dead_wire",
    "unused_wire": "dead_wire",
    "dangling_wire": "dead_wire",
    "cut_wire": "dead_wire",
    "broken_wire": "dead_wire",
    "dropped_wire": "dead_wire",
    # Vegetation variants -> vegetation
    "vegetation": "vegetation",
    "tree": "vegetation",
    "trees": "vegetation",
    "foliage": "vegetation",
    "branch": "vegetation",
    "branches": "vegetation",
    "plants": "vegetation",
    "plant": "vegetation",
    "shrub": "vegetation",
    "vines": "vegetation",
    "vine": "vegetation",
    "canopy": "vegetation",
    # Utility Pole variants -> pole
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
    "building",
    "sky",
    "background",
}


def normalize_wire_class(name: str) -> str:
    """Normalize wire or context class name to canonical 4-class registry (plus pole/other)."""
    norm = normalize_class_name(name)
    if norm in DEFAULT_CLASS_ALIASES:
        return DEFAULT_CLASS_ALIASES[norm]
    return norm


@dataclass
class PreprocessResult:
    """Output of the preprocessing stage."""
    quality_ok: bool
    reason: str
    gray: np.ndarray | None = None
    rgb: np.ndarray | None = None
    original_shape: tuple[int, int] | None = None  # (h, w)
    metadata: dict[str, object] = field(default_factory=dict)
    envelope: list[dict[str, object]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    enhanced_rgb: np.ndarray | None = None


@dataclass
class VNIResult:
    """Output of the VNI (Visual Noise Index) stage."""
    vni: float
    spectrum_png: bytes | None = None
    vni_raw: float | None = None
    band: str | None = None
    flags: list[str] = field(default_factory=list)


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
    flags: list[str] = field(default_factory=list)
    skeleton_pixels: int = 0
    pruned_pixels: int = 0
    reason: str | None = None


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
    fit_failed: bool = False
    perp_sag_px: float = 0.0
    perp_sag_ratio: float = 0.0
    flags: list[str] = field(default_factory=list)
