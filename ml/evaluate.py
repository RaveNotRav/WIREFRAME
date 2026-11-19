"""Evaluation runs (Sections 3.9 - 3.11) for WIREFRAME geometric and segmentation layers.

Usage:
    python -m ml.evaluate geometry --scenes 20
    python -m ml.evaluate ultralytics --data datasets/wireframes_v1/data.yaml --split train
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.formulas import compute_allowable_sag, compute_mspi, compute_paper_crs
from backend.utils.cv.catenary import calculate_sag_metrics, cosh_func
from backend.utils.cv.centerline import extract_centerline
from backend.utils.cv.types import CANONICAL_CONTEXT_CLASSES, CANONICAL_WIRE_CLASSES

ALL_CANONICAL_CLASSES = (*CANONICAL_WIRE_CLASSES, *CANONICAL_CONTEXT_CLASSES)


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def summarize(xs: list[float]) -> dict[str, float | int | None]:
    clean = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "n": len(clean),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "p90": round(float(np.quantile(arr, 0.90)), 6),
        "max": round(float(arr.max()), 6),
    }


def geometry_benchmark(n_scenes: int = 20, seed: int = 0) -> dict[str, Any]:
    """Run Algorithms 4-5 (Zhang-Suen centerline + Catenary sag fitting) on synthetic wires with known ground truth."""
    rng = np.random.default_rng(seed)
    residuals: list[float] = []
    sag_errors: list[float] = []
    failures = 0

    for i in range(n_scenes):
        h, w = 300, 500
        mask = np.zeros((h, w), dtype=np.uint8)
        a_true = float(rng.uniform(180.0, 350.0))
        x0_true = 250.0
        b_true = 60.0 - a_true
        xs = np.arange(60, 440, dtype=np.float64)
        y_up = cosh_func(xs, a_true, b_true, x0_true)
        y_hang = float(np.max(y_up)) - (y_up - 60.0)
        true_sag = float(np.max(y_hang) - 0.5 * (y_hang[0] + y_hang[-1]))

        for xi, yi in zip(xs, y_hang):
            cv2.circle(mask, (int(xi), int(round(yi))), 3, 255, -1)

        bbox = [60.0, 55.0, 440.0, float(np.max(y_hang) + 6.0)]
        cline = extract_centerline(mask, bbox)
        if not cline.is_valid:
            failures += 1
            continue
        cat = calculate_sag_metrics(cline.polyline)
        if cat.fit_failed or cat.span_px <= 0:
            failures += 1
            continue
        residuals.append(cat.residual)
        if true_sag > 0:
            sag_errors.append(abs(cat.sag_px - true_sag) / true_sag)

    return {
        "synthetic": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "scenes": n_scenes,
        "fit_failures": failures,
        "residual_px": summarize(residuals),
        "relative_sag_error": summarize(sag_errors),
        "canonical_classes": list(ALL_CANONICAL_CLASSES),
    }


def ultralytics_eval(data_yaml: Path, split: str = "train", weights: str = "backend/models/wireframe_seg.pt", imgsz: int = 640) -> dict[str, Any]:
    """Run Ultralytics YOLO11-seg validator on a dataset split."""
    from ultralytics import YOLO

    m = YOLO(weights).val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        iou=0.45,
        conf=0.001,
        batch=1,
        retina_masks=True,
        plots=False,
        verbose=False,
    )
    return {
        "split": split,
        "weights": weights,
        "imgsz": imgsz,
        "box_map50": float(m.box.map50),
        "mask_map50": float(m.seg.map50),
        "mask_map50_95": float(m.seg.map),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--scenes", type=int, default=20)
    u = sub.add_parser("ultralytics")
    u.add_argument("--data", type=Path, required=True)
    u.add_argument("--split", default="train")
    u.add_argument("--weights", default="backend/models/wireframe_seg.pt")
    u.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    if a.cmd == "geometry":
        payload = geometry_benchmark(a.scenes)
    else:
        payload = ultralytics_eval(a.data, a.split, a.weights, a.imgsz)
    print(json.dumps(_clean(payload), indent=2))


if __name__ == "__main__":
    main()
