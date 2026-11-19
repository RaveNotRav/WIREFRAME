"""End-to-end and per-stage runtime benchmark for the WIREFRAME CV pipeline.

Usage:
    python -m ml.benchmark --images datasets/wireframes_v1/train/images/*.jpg --reps 1 --out outputs/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import time
from pathlib import Path

import numpy as np

from backend.utils.cv.catenary import calculate_sag_metrics
from backend.utils.cv.centerline import extract_centerline
from backend.utils.cv.preprocess import validate_and_preprocess
from backend.utils.cv.vni import compute_vni


def stats(xs: list[float]) -> dict[str, float | int]:
    xs = sorted(xs)
    if not xs:
        return {}
    rank = lambda q: xs[min(len(xs) - 1, max(0, int(np.ceil(q * len(xs))) - 1))]  # noqa: E731
    return {
        "n": len(xs),
        "min": round(xs[0], 4),
        "max": round(xs[-1], 4),
        "mean": round(st.mean(xs), 4),
        "median": round(st.median(xs), 4),
        "p90": round(rank(0.9), 4),
        "p95": round(rank(0.95), 4),
        "std": round(st.pstdev(xs), 4),
    }


def benchmark_image(image_path: str | Path) -> dict:
    """Benchmark preprocessing, VNI, and mock/real segmentation on a single image."""
    t0 = time.perf_counter()
    prep = validate_and_preprocess(str(image_path))
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    vni_res = compute_vni(prep.gray) if prep.quality_ok and prep.gray is not None else None
    t_vni = time.perf_counter() - t1

    return {
        "image": Path(image_path).name,
        "quality_ok": prep.quality_ok,
        "vni": vni_res.vni if vni_res else None,
        "vni_band": vni_res.band if vni_res else None,
        "preprocess_s": round(t_prep, 4),
        "vni_s": round(t_vni, 4),
        "total_s": round(t_prep + t_vni, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    runs = []
    for p in a.images:
        for _ in range(a.reps):
            runs.append(benchmark_image(p))

    report = {
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "summary": {
            "preprocess_s": stats([r["preprocess_s"] for r in runs]),
            "vni_s": stats([r["vni_s"] for r in runs]),
            "total_s": stats([r["total_s"] for r in runs]),
        },
        "runs": runs,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
