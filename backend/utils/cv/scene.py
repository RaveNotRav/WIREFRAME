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
BUILDING_CLASSES = ("building", "house", "skyscraper", "wall", "awning", "canopy", "bridge")
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
        self.building_ids: list[int] = []

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
            self.building_ids = [names[n] for n in BUILDING_CLASSES if n in names]
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

    def building_mask(self, labels: np.ndarray) -> np.ndarray:
        return np.isin(labels, self.building_ids)

    def context(self, labels: np.ndarray, min_frac: float = 0.002) -> dict[str, Any]:
        """Extract pole-like and building regions from ADE20K labels as bounding boxes."""
        poles = np.isin(labels, self.pole_ids).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(poles, connectivity=8)
        area_min = min_frac * labels.size
        boxes = [
            [int(x), int(y), int(x + w), int(y + h)]
            for x, y, w, h, a in stats[1:]
            if a >= area_min and h > w
        ]

        buildings = np.isin(labels, self.building_ids).astype(np.uint8)
        _, _, b_stats, _ = cv2.connectedComponentsWithStats(buildings, connectivity=8)
        b_area_min = max(area_min * 4.0, 0.01 * labels.size)
        b_boxes = [
            [int(x), int(y), int(x + w), int(y + h)]
            for x, y, w, h, a in b_stats[1:]
            if a >= b_area_min
        ]
        return {"poles": boxes, "buildings": b_boxes}


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


def detect_crossarms_and_insulators(
    rgb: np.ndarray,
    pole_boxes: list[dict[str, Any]],
    wire_masks: list[np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect utility crossarms and insulators near upper pole sections and wire attachment clusters.

    Uses Sobel horizontal/vertical gradient structure and local contrast peaks inside the upper 45%
    of each detected pole region (or wire endpoint convergence zones), returning:
    - `crossarms`: `[{"bbox": [x1, y1, x2, y2], "confidence": float}]`
    - `insulators`: `[{"bbox": [x1, y1, x2, y2], "confidence": float}]`
    """
    if rgb is None or rgb.size == 0 or not pole_boxes:
        return [], []

    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    crossarms: list[dict[str, Any]] = []
    insulators: list[dict[str, Any]] = []

    # Collect wire skeleton endpoints / masks near poles to corroborate crossarm & insulator attachment points
    wire_union = np.zeros((h, w), dtype=bool)
    for wm in wire_masks or []:
        if wm is not None and wm.shape[:2] == (h, w):
            wire_union |= (wm > 0)

    for p in pole_boxes:
        pb = p.get("bbox")
        if not pb or len(pb) < 4:
            continue
        px1, py1, px2, py2 = [float(v) for v in pb[:4]]
        pw = max(6.0, px2 - px1)
        ph = max(20.0, py2 - py1)
        pcx = 0.5 * (px1 + px2)

        # Search window around upper 45% of the pole for horizontal crossarm bars (typically 2.5x - 5.5x pole width)
        roi_x1 = max(0, int(round(pcx - 3.2 * pw)))
        roi_x2 = min(w, int(round(pcx + 3.2 * pw)))
        roi_y1 = max(0, int(round(py1 - 0.05 * ph)))
        roi_y2 = min(h, int(round(py1 + 0.48 * ph)))
        if roi_x2 - roi_x1 < 16 or roi_y2 - roi_y1 < 12:
            continue

        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        edges = cv2.Canny(roi_gray, 45, 135)
        min_arm_len = max(14, int(round(1.6 * pw)))
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=20,
            minLineLength=min_arm_len,
            maxLineGap=max(6, int(round(0.4 * pw))),
        )

        best_arm: list[float] | None = None
        best_arm_len = 0.0
        if lines is not None:
            for ln in lines[:, 0, :]:
                lx1, ly1, lx2, ly2 = map(float, ln)
                dx, dy = abs(lx2 - lx1), abs(ly2 - ly1)
                seg_len = float(np.hypot(dx, dy))
                angle_deg = float(np.degrees(np.arctan2(dy, max(dx, 1e-6))))
                # Crossarm is near-horizontal (<= 22 deg) and crosses or adjoins the pole center
                gx1, gx2 = roi_x1 + min(lx1, lx2), roi_x1 + max(lx1, lx2)
                gy1, gy2 = roi_y1 + min(ly1, ly2), roi_y1 + max(ly1, ly2)
                if angle_deg <= 22.0 and gx1 <= pcx + pw and gx2 >= pcx - pw and seg_len > best_arm_len:
                    arm_half_h = max(4.0, 0.18 * pw)
                    ym = 0.5 * (gy1 + gy2)
                    best_arm = [
                        round(max(0.0, gx1), 1),
                        round(max(0.0, ym - arm_half_h), 1),
                        round(min(float(w - 1), gx2), 1),
                        round(min(float(h - 1), ym + arm_half_h), 1),
                    ]
                    best_arm_len = seg_len

        # If wires cross the upper pole zone without a wide wooden/steel crossarm (armless / spool rack construction),
        # check wire-to-pole attachment band for insulator pins
        wire_roi = wire_union[roi_y1:roi_y2, max(0, int(px1 - pw)) : min(w, int(px2 + pw))]
        if best_arm is not None:
            crossarms.append({"bbox": best_arm, "confidence": round(min(0.85, 0.45 + best_arm_len / max(w, 1)), 3)})
            ax1, ay1, ax2, ay2 = best_arm
            ins_r = max(4.0, 0.22 * pw)
            for ix in (ax1 + ins_r, 0.5 * (ax1 + ax2), ax2 - ins_r):
                insulators.append({
                    "bbox": [
                        round(max(0.0, ix - ins_r), 1),
                        round(max(0.0, ay1 - 1.6 * ins_r), 1),
                        round(min(float(w - 1), ix + ins_r), 1),
                        round(min(float(h - 1), ay2), 1),
                    ],
                    "confidence": 0.58,
                })
        elif np.any(wire_roi):
            wy, _ = np.nonzero(wire_roi)
            attach_y = float(roi_y1 + np.median(wy))
            ins_r = max(4.0, 0.24 * pw)
            insulators.append({
                "bbox": [
                    round(max(0.0, pcx - ins_r), 1),
                    round(max(0.0, attach_y - ins_r), 1),
                    round(min(float(w - 1), pcx + ins_r), 1),
                    round(min(float(h - 1), attach_y + ins_r), 1),
                ],
                "confidence": 0.52,
            })

    return crossarms, insulators

