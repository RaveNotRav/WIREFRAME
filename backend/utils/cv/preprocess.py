"""Image preprocessing, quality validation, and EXIF acquisition envelope checks.

Complexity:
- N = width * height (number of pixels)
- Grayscale conversion: O(N)
- Resize: O(N)
- Denoise: O(N)
- Overall Preprocessing: O(N)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from backend.utils.cv.enhance import enhance as enhance_adaptive
from backend.utils.cv.types import PreprocessResult

logger = logging.getLogger(__name__)

_TAG_BY_NAME = {v: k for k, v in ExifTags.TAGS.items()}
_GPS_BY_NAME = {v: k for k, v in ExifTags.GPSTAGS.items()}


@dataclass(frozen=True)
class EnvelopeCheck:
    parameter: str
    value: float | str | None
    status: str  # "in_envelope" | "outside_envelope" | "unverified"
    bound: str


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError, ZeroDivisionError):
        if isinstance(val, tuple) and len(val) == 2 and val[1]:
            return float(val[0]) / float(val[1])
        return None


def _parse_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def read_exif(img: Image.Image) -> dict[str, Any]:
    """Extract capture metadata needed by the acquisition envelope checks."""
    out: dict[str, Any] = {
        "orientation": 1,
        "focal_length_35mm": None,
        "focal_length_mm": None,
        "subject_distance_m": None,
        "datetime": None,
        "make": None,
        "model": None,
        "elevation_angle_deg": None,
    }
    try:
        exif = img.getexif()
    except Exception:
        return out
    if not exif:
        return out

    out["orientation"] = int(exif.get(_TAG_BY_NAME["Orientation"], 1))
    out["make"] = exif.get(_TAG_BY_NAME["Make"])
    out["model"] = exif.get(_TAG_BY_NAME["Model"])
    dt = exif.get(_TAG_BY_NAME["DateTimeOriginal"]) or exif.get(_TAG_BY_NAME["DateTime"])

    try:
        ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        ifd = {}
    if ifd:
        dt = dt or ifd.get(_TAG_BY_NAME["DateTimeOriginal"])
        out["focal_length_35mm"] = _to_float(ifd.get(_TAG_BY_NAME["FocalLengthIn35mmFilm"]))
        out["focal_length_mm"] = _to_float(ifd.get(_TAG_BY_NAME["FocalLength"]))
        dist = _to_float(ifd.get(_TAG_BY_NAME["SubjectDistance"]))
        if dist is not None and 0 < dist < 1e6:
            out["subject_distance_m"] = dist
    out["datetime"] = _parse_datetime(dt)
    return out


def check_acquisition_envelope(
    meta: dict[str, Any],
    *,
    focal_length_35mm_range: tuple[float, float] = (24.0, 50.0),
    distance_m_range: tuple[float, float] = (5.0, 15.0),
    elevation_deg_range: tuple[float, float] = (30.0, 60.0),
    daylight_hours: tuple[int, int] = (6, 18),
    user_overrides: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify capture metadata against Table 3.3 acquisition envelope without rejecting images."""
    ov = user_overrides or {}
    checks: list[EnvelopeCheck] = []
    flags: list[str] = []

    def bound_check(name: str, val: float | None, lo: float, hi: float, unit: str) -> None:
        bound = f"[{lo}, {hi}] {unit}"
        if val is None:
            checks.append(EnvelopeCheck(name, None, "unverified", bound))
            return
        ok = lo <= val <= hi
        status = "in_envelope" if ok else "outside_envelope"
        checks.append(EnvelopeCheck(name, val, status, bound))
        if not ok:
            flags.append(f"outside_envelope:{name}")

    fl = ov.get("focal_length_35mm") or meta.get("focal_length_35mm")
    bound_check("focal_length_35mm", fl, *focal_length_35mm_range, "mm")

    dist = ov.get("distance_m") or meta.get("subject_distance_m")
    bound_check("distance_m", dist, *distance_m_range, "m")

    elev = ov.get("elevation_angle_deg") or meta.get("elevation_angle_deg")
    bound_check("elevation_angle_deg", elev, *elevation_deg_range, "deg")

    dt: datetime | None = meta.get("datetime")
    h_lo, h_hi = daylight_hours
    bound = f"[{h_lo:02d}:00, {h_hi:02d}:00]"
    if dt is None:
        checks.append(EnvelopeCheck("capture_hour", None, "unverified", bound))
    else:
        hour = dt.hour + dt.minute / 60.0
        ok = h_lo <= hour <= h_hi
        checks.append(EnvelopeCheck("capture_hour", round(hour, 2), "in_envelope" if ok else "outside_envelope", bound))
        if not ok:
            flags.append("outside_envelope:capture_hour")

    if any(c.status == "unverified" for c in checks):
        flags.append("acquisition_unverified")

    serialized_checks = [
        {"parameter": c.parameter, "value": c.value, "status": c.status, "bound": c.bound}
        for c in checks
    ]
    return serialized_checks, flags


def validate_and_preprocess(
    image_path: str,
    user_overrides: dict[str, float] | None = None,
) -> PreprocessResult:
    """Validate image quality, check EXIF acquisition envelope, and prepare arrays for downstream stages.
    
    1. Corrupt file check
    2. Minimum resolution check (e.g. 100x100)
    3. Dark check (Mean brightness < 15)
    4. Blurry check (Laplacian variance < 50)
    5. EXIF acquisition envelope check (flags only, never rejects)
    
    Returns PreprocessResult with quality flags, envelope metadata, and prepped images.
    """
    try:
        pil_img = Image.open(image_path)
        exif_meta = read_exif(pil_img)
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return PreprocessResult(quality_ok=False, reason="corrupt_file")

    h, w = img.shape[:2]
    original_shape = (h, w)

    if h < 100 or w < 100:
        return PreprocessResult(quality_ok=False, reason="resolution_too_low", original_shape=original_shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mean_bright = float(np.mean(gray))
    if mean_bright < 15.0:
        return PreprocessResult(quality_ok=False, reason="too_dark", original_shape=original_shape)

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 50.0:
        return PreprocessResult(quality_ok=False, reason="too_blurry", original_shape=original_shape)

    envelope_checks, flags = check_acquisition_envelope(exif_meta, user_overrides=user_overrides)
    exif_meta["width"] = w
    exif_meta["height"] = h

    # Prepare RGB
    rgb_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enhanced = enhance_adaptive(rgb_raw)
    for f in enhanced.flags:
        if f not in flags:
            flags.append(f)

    rgb = cv2.GaussianBlur(rgb_raw, (3, 3), 0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    logger.debug(
        "Preprocessed %s: %dx%d, blur=%.1f, bright=%.1f, flags=%s",
        image_path, w, h, laplacian_var, mean_bright, flags,
    )
    return PreprocessResult(
        quality_ok=True,
        reason="",
        gray=gray,
        rgb=rgb,
        original_shape=original_shape,
        metadata=exif_meta,
        envelope=envelope_checks,
        flags=flags,
        enhanced_rgb=enhanced.rgb,
    )
