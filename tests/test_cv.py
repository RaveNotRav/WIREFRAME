"""Tests for the CV Pipeline."""

import cv2
import numpy as np

from backend.utils.cv.catenary import calculate_sag_metrics, cosh_func
from backend.utils.cv.centerline import extract_centerline
from backend.utils.cv.preprocess import validate_and_preprocess
from backend.utils.cv.vni import compute_vni


def test_quality_validation(tmp_path):
    # Too small
    img_small = np.zeros((50, 50, 3), dtype=np.uint8)
    small_path = tmp_path / "small.jpg"
    cv2.imwrite(str(small_path), img_small)
    res = validate_and_preprocess(str(small_path))
    assert not res.quality_ok
    assert res.reason == "resolution_too_low"

    # Too dark
    img_dark = np.zeros((200, 200, 3), dtype=np.uint8)
    dark_path = tmp_path / "dark.jpg"
    cv2.imwrite(str(dark_path), img_dark)
    res = validate_and_preprocess(str(dark_path))
    assert not res.quality_ok
    assert res.reason == "too_dark"

    # Too blurry (solid color = 0 variance)
    img_blur = np.full((200, 200, 3), 128, dtype=np.uint8)
    blur_path = tmp_path / "blur.jpg"
    cv2.imwrite(str(blur_path), img_blur)
    res = validate_and_preprocess(str(blur_path))
    assert not res.quality_ok
    assert res.reason == "too_blurry"


def test_vni_order():
    # Clean image (smooth gradient)
    img_clean = np.linspace(0, 255, 256).astype(np.uint8)
    img_clean = np.tile(img_clean, (256, 1))
    
    # Cluttered image (white noise)
    rng = np.random.RandomState(42)
    img_cluttered = rng.randint(0, 255, (256, 256), dtype=np.uint8)
    
    res_clean = compute_vni(img_clean)
    res_cluttered = compute_vni(img_cluttered)
    
    assert res_clean.vni < res_cluttered.vni
    assert res_clean.spectrum_png is not None


def test_skeleton_and_catenary():
    # Generate a synthetic binary mask with a thick catenary curve
    mask = np.zeros((200, 400), dtype=np.uint8)
    
    # True parameters
    a_true = 200.0
    x0_true = 200.0
    b_true = 50.0 - a_true  # So y = 50 at x = x0
    
    x = np.arange(50, 350)
    y = cosh_func(x, a_true, b_true, x0_true)
    
    # Draw thick curve
    for xi, yi in zip(x, y):
        cv2.circle(mask, (int(xi), int(yi)), 5, 255, -1)
        
    bbox = [50, 50, 350, int(np.max(y)) + 5]
    
    cline = extract_centerline(mask, bbox)
    assert cline.is_valid
    
    # Check width (by checking length against expected span approximately)
    # Morphological thinning tends to erode endpoints, so it won't be exactly 300.
    assert len(cline.polyline) > 100
    assert len(cline.polyline) < 350
    
    cat = calculate_sag_metrics(cline.polyline)
    assert cat.sag_px > 0
    assert cat.span_px > 150
    
    # The fitted parameter 'a' is highly unstable for shallow curves, so we don't assert its exact value.
    # Instead, we just ensure the fit residual is low.
    assert cat.residual < 10.0  # Very tight fit for synthetic data


def test_segment_image_options_and_channel_order(monkeypatch):
    import torch
    from backend.config_store import set_setting
    from backend.utils.cv.segment import segment_image

    recorded_call = {}

    class FakeBoxes:
        def __init__(self):
            self.xyxy = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
            self.cls = torch.tensor([0])
            self.conf = torch.tensor([0.95])

    class FakeMasks:
        def __init__(self, shape):
            self.data = torch.ones(shape, dtype=torch.float32)

    class FakeResult:
        def __init__(self, shape):
            self.boxes = FakeBoxes()
            self.masks = FakeMasks(shape)
            self.names = {0: "power_line"}

    def fake_model(bgr_input, **kwargs):
        recorded_call["input_shape"] = bgr_input.shape
        recorded_call["sample_pixel"] = bgr_input[0, 0].tolist()
        recorded_call["kwargs"] = kwargs
        # Return masks matching original image shape to test skipping resize
        return [FakeResult((1, 100, 100))]

    set_setting("inference_params", {
        "conf": 0.35,
        "iou": 0.65,
        "imgsz": 800,
        "retina_masks": True,
    })

    # Pure Red image in RGB format: (255, 0, 0)
    rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_img[:, :] = (255, 0, 0)

    # In segment_image, RGB is converted to BGR, so pixel becomes (0, 0, 255)
    seg_res = segment_image(rgb_img, fake_model, original_shape=(100, 100))

    assert seg_res.mask_available is True
    assert recorded_call["sample_pixel"] == [0, 0, 255]  # Red in BGR is (0, 0, 255)
    assert recorded_call["kwargs"]["conf"] == 0.35
    assert recorded_call["kwargs"]["iou"] == 0.65
    assert recorded_call["kwargs"]["imgsz"] == 800
    assert recorded_call["kwargs"]["retina_masks"] is True
    # Mask shape matches original_shape
    assert seg_res.masks[0].shape == (100, 100)

