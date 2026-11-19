"""Pretrained ADE20K scene parsing (`backend/models/hf/upernet-convnext-tiny/`) and YOLO11n pole detection
(`backend/models/poles-yolo11n-v218.pt`) ported from Joanna's `wireframes-main/app/pipeline/{scene,poles}.py`.

Used by the multi-model ensemble segmenter for:
1. Ground-structure gate (`ground_share >= 0.5`): suppresses pavement/kerb/manhole lines lying on the road.
2. Structure-edge gate (`side_contrast >= 0.6`): suppresses building/flyover/canopy roof outlines against the sky.
3. Vegetation gate (`vegetation_share < 0.4`): suppresses false vegetation regions on shadowed concrete.
4. Utility pole detection (`poles-yolo11n-v218.pt` + ADE20K pole fallback): returns cleaned pole bounding boxes.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.utils.cv.strand_head import hf_source
from backend.utils.cv.wire_instances import _crop_skeleton

logger = logging.getLogger(__name__)

MODEL_ID = "openmmlab/upernet-convnext-tiny"
INPUT_PX = 384
VEGETATION_CLASSES = ("tree", "grass", "plant", "palm", "flower", "field")
POLE_CLASSES = ("pole", "streetlight", "tower")
GROUND_CLASSES = ("road", "sidewalk", "floor", "earth", "path", "grass", "field", "sand", "dirt track", "runway", "rug")


class SceneParser:
    """ADE20K UperNet-ConvNeXt-Tiny scene parser loaded from `backend/models/hf/upernet-convnext-tiny`."""

    def __init__(self, device: str = "auto", models_dir: Path | None = None) -> None:
        self.device = device
        self.models_dir = models_dir
        self._net = None
        self._proc = None
        self._dev = "cpu"
        self._lock = threading.Lock()
        self.error: str | None = None
        self.ground_ids: list[int] = []
        self.vegetation_ids: list[int] = []
        self.pole_ids: list[int] = []

    def load(self) -> bool:
        if self._net is not None:
            return True
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

            dev = (
                self.device
                if self.device not in ("auto", "", None)
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            src = hf_source("upernet-convnext-tiny", MODEL_ID, self.models_dir)
            self._proc = AutoImageProcessor.from_pretrained(src)
            self._net = AutoModelForSemanticSegmentation.from_pretrained(src).to(dev).eval()
            self._dev = dev
            names = {v: int(k) for k, v in self._net.config.id2label.items()}
            self.ground_ids = [names[n] for n in GROUND_CLASSES if n in names]
            self.vegetation_ids = [names[n] for n in VEGETATION_CLASSES if n in names]
            self.pole_ids = [names[n] for n in POLE_CLASSES if n in names]
            return True
        except Exception as exc:
            self.error = f"scene parser unavailable: {exc}"
            logger.warning(self.error)
            return False

    def labels(self, rgb: np.ndarray) -> np.ndarray | None:
        """Return ADE20K class index per pixel at `rgb`'s resolution (uint8), or None if unavailable."""
        if not self.load():
            return None
        import torch
        from PIL import Image

        with self._lock, torch.no_grad():
            x = self._proc(
                images=Image.fromarray(rgb),
                return_tensors="pt",
                size={"height": INPUT_PX, "width": INPUT_PX},
            )["pixel_values"].to(self._dev)
            lo = self._net(pixel_values=x).logits
            lab = torch.nn.functional.interpolate(lo, size=rgb.shape[:2], mode="bilinear").argmax(1)[0]
            return lab.to(torch.uint8).cpu().numpy()

    def ground_mask(self, labels: np.ndarray) -> np.ndarray:
        return np.isin(labels, self.ground_ids)

    def context(self, labels: np.ndarray, min_frac: float = 0.002) -> dict[str, Any]:
        """Extract pole-like regions (ADE20K pole / streetlight / tower) as bounding boxes."""
        poles = np.isin(labels, self.pole_ids).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(poles, connectivity=8)
        area_min = min_frac * labels.size
        boxes = [
            [int(x), int(y), int(x + w), int(y + h)]
            for x, y, w, h, a in stats[1:]
            if a >= area_min and h > w
        ]
        return {"poles": boxes}


def ground_share(span_mask: np.ndarray, ground: np.ndarray) -> float:
    """Share of a span's pixels that lie on ground classes."""
    n = int(span_mask.sum())
    return float((span_mask & ground).sum() / n) if n else 0.0


def vegetation_share(mask: np.ndarray, labels: np.ndarray, vegetation_ids: list[int]) -> float:
    """Share of a region's pixels that the scene parser labels as vegetation."""
    n = int(mask.sum())
    return float(np.isin(labels[mask], vegetation_ids).sum() / n) if n else 0.0


def side_contrast(mask: np.ndarray, labels: np.ndarray, offset_frac: float = 0.012, min_offset: float = 8.0) -> float:
    """Share of a span's skeleton where the scene class at +d and -d along the span's normal differ."""
    sk = _crop_skeleton(mask).astype(np.float64)
    if len(sk) < 5:
        return 0.0
    c = sk.mean(0)
    _, _, vt = np.linalg.svd(sk - c, full_matrices=False)
    nrm = np.array([-vt[0][1], vt[0][0]])
    d = max(min_offset, offset_frac * max(labels.shape))
    h, w = labels.shape
    a = np.rint(sk + d * nrm).astype(int)
    b = np.rint(sk - d * nrm).astype(int)
    ok = (
        (a[:, 0] >= 0)
        & (a[:, 0] < h)
        & (a[:, 1] >= 0)
        & (a[:, 1] < w)
        & (b[:, 0] >= 0)
        & (b[:, 0] < h)
        & (b[:, 1] >= 0)
        & (b[:, 1] < w)
    )
    if ok.sum() < 5:
        return 0.0
    return float((labels[a[ok, 0], a[ok, 1]] != labels[b[ok, 0], b[ok, 1]]).mean())


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def clean_pole_boxes(dets: list[dict[str, Any]], iou: float = 0.5, group_n: int = 2) -> list[dict[str, Any]]:
    """Greedy NMS on pole boxes, then drop any box containing the centers of `group_n` or more kept boxes."""
    keep: list[dict[str, Any]] = []
    for d in sorted(dets, key=lambda item: -float(item.get("confidence", 0.0))):
        if all(_iou(d["bbox"], k["bbox"]) < iou for k in keep):
            keep.append(d)
    out: list[dict[str, Any]] = []
    for d in keep:
        x0, y0, x1, y1 = d["bbox"]
        inside = sum(
            1
            for k in keep
            if k is not d
            and x0 <= (k["bbox"][0] + k["bbox"][2]) / 2 <= x1
            and y0 <= (k["bbox"][1] + k["bbox"][3]) / 2 <= y1
        )
        if inside < group_n:
            out.append(d)
    return out


class PoleDetector:
    """Fine-tuned YOLO11n utility pole detector (`poles-yolo11n-v218.pt`)."""

    def __init__(self, weights: Path, conf: float = 0.25, imgsz: int = 640, device: str = "auto") -> None:
        self.weights = Path(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self._dev = "cpu"
        self._lock = threading.Lock()
        self.error: str | None = None

    def load(self) -> bool:
        if self._model is not None:
            return True
        if not self.weights.is_file():
            self.error = f"no pole weights at {self.weights.name}"
            return False
        try:
            import torch
            from ultralytics import YOLO

            self._dev = (
                self.device
                if self.device not in ("auto", "", None)
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._model = YOLO(str(self.weights))
            return True
        except Exception as exc:
            self.error = f"pole detector unavailable: {exc}"
            logger.warning(self.error)
            return False

    def detect(self, rgb: np.ndarray) -> list[dict[str, Any]] | None:
        """Return [{"bbox": [x0, y0, x1, y1], "confidence": float}] in `rgb` pixel coordinates."""
        if not self.load():
            return None
        with self._lock:
            r = self._model.predict(
                np.ascontiguousarray(rgb[..., ::-1]),
                imgsz=self.imgsz,
                conf=self.conf,
                device=self._dev,
                verbose=False,
            )[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        raw = [
            {"bbox": [round(float(v), 1) for v in b], "confidence": round(float(c), 3)}
            for b, c in zip(boxes, confs, strict=True)
        ]
        return clean_pole_boxes(raw)
