"""Mock YOLOv11 detector for WIREFRAME_MOCK=1 mode."""

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MOCK_CLASS_NAMES: dict[int, str] = {
    0: "power_line",
    1: "telecom_cable",
    2: "fiber_optic",
    3: "dead_abandoned",
    4: "pole",
}


def is_mock_mode() -> bool:
    return os.environ.get("WIREFRAME_MOCK", "0") == "1"


@dataclass
class MockBox:
    """Mimics ultralytics Boxes."""
    _xyxy: np.ndarray
    _conf: np.ndarray
    _cls: np.ndarray

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    @property
    def conf(self) -> np.ndarray:
        return self._conf

    @property
    def cls(self) -> np.ndarray:
        return self._cls

    def __len__(self) -> int:
        return len(self._xyxy)


@dataclass
class MockResult:
    """Mimics ultralytics Results."""
    boxes: MockBox
    names: dict[int, str]
    orig_img: np.ndarray
    orig_shape: tuple[int, int]
    path: str = ""

    def plot(self) -> np.ndarray:
        return self.orig_img.copy()


def generate_mock_detections(image: np.ndarray, image_path: str = "") -> list[MockResult]:
    """Fixed synthetic detections proportional to image size."""
    h, w = image.shape[:2]
    bboxes = np.array([
        [w * 0.05, h * 0.25, w * 0.95, h * 0.30],
        [w * 0.10, h * 0.40, w * 0.90, h * 0.45],
        [w * 0.15, h * 0.55, w * 0.85, h * 0.58],
        [w * 0.80, h * 0.10, w * 0.85, h * 0.90],
    ], dtype=np.float32)
    confs = np.array([0.92, 0.85, 0.78, 0.95], dtype=np.float32)
    classes = np.array([0, 0, 1, 4], dtype=np.float32)

    result = MockResult(
        boxes=MockBox(_xyxy=bboxes, _conf=confs, _cls=classes),
        names=MOCK_CLASS_NAMES,
        orig_img=image,
        orig_shape=(h, w),
        path=image_path,
    )
    logger.info("Mock: %d detections for %s", len(result.boxes), image_path or "<image>")
    return [result]


def validate_mock_result(result: Any) -> bool:
    """Strict shape check against Ultralytics result contract."""
    try:
        boxes = result.boxes
        xyxy = boxes.xyxy
        assert isinstance(xyxy, np.ndarray) and xyxy.ndim == 2 and xyxy.shape[1] == 4
        n = len(xyxy)
        assert boxes.conf.shape == (n,)
        assert boxes.cls.shape == (n,)
        assert isinstance(result.names, dict)
        assert isinstance(result.orig_img, np.ndarray)
        assert hasattr(result, "orig_shape")
        return True
    except (AssertionError, AttributeError) as exc:
        logger.error("Mock validation failed: %s", exc)
        return False
