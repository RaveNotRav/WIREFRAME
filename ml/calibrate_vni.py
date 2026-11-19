"""Calibrate the VNI quantiles q01 and q99 (Equation 4) and compare radius factors.

    python -m ml.calibrate_vni path/to/calibration_images [--write]

Runs Algorithm 2's grayscale path and Equation 3 on every image, reports q01/q99 for the
radius factors 0.10, 0.15 and 0.20 examined in Section 3.6.3, and with --write updates the
values in the WIREFRAME config store. Calibration images must be kept separate from test images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from backend.config_store import get_setting, set_setting
from backend.utils.cv.vni import vni_raw

EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def calibrate_folder(folder: Path, min_images: int = 2) -> dict:
    """Compute (q01, q99) across radius factors 0.10, 0.15, and 0.20 for images in folder."""
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXT)
    if len(files) < min_images:
        raise ValueError(f"{len(files)} images found; need at least {min_images}")

    vni_cfg = get_setting("vni_params", {}) or {}
    size = int(vni_cfg.get("size", 640))
    kernel = int(vni_cfg.get("gaussian_kernel", 5))
    sigma = float(vni_cfg.get("gaussian_sigma", 1.0))
    remove_mean = bool(vni_cfg.get("remove_mean", True))

    raws: dict[float, list[float]] = {0.10: [], 0.15: [], 0.20: []}
    for f in files:
        with Image.open(f) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        for r in raws:
            val, _ = vni_raw(
                rgb,
                size=size,
                kernel=kernel,
                sigma=sigma,
                radius_factor=r,
                remove_mean=remove_mean,
            )
            raws[r].append(val)

    report = {}
    for r, vals in raws.items():
        q01, q99 = np.quantile(vals, [0.01, 0.99])
        report[str(r)] = {"q01": round(float(q01), 6), "q99": round(float(q99), 6)}

    return {
        "images": len(files),
        "remove_mean": remove_mean,
        "quantiles": report,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-images", type=int, default=20)
    a = ap.parse_args()

    result = calibrate_folder(a.folder, min_images=a.min_images)
    print(json.dumps(result, indent=2))
    if a.write:
        vni_cfg = dict(get_setting("vni_params", {}) or {})
        rf = str(vni_cfg.get("radius_factor", 0.15))
        if rf in result["quantiles"]:
            vni_cfg["q01"] = result["quantiles"][rf]["q01"]
            vni_cfg["q99"] = result["quantiles"][rf]["q99"]
            set_setting("vni_params", vni_cfg, updated_by="ml.calibrate_vni")
            print("Updated vni_params in config_store.")


if __name__ == "__main__":
    main()
