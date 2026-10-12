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
    # Note: Corrected to sagging orientation (y = ymax - (y_up - ymin)) so the wire sags
    # downward towards the ground in image coordinates instead of bowing upward as an arch.
    mask = np.zeros((200, 400), dtype=np.uint8)
    
    # True parameters
    a_true = 200.0
    x0_true = 200.0
    b_true = 50.0 - a_true  # So y = 50 at x = x0
    
    x = np.arange(50, 350)
    y_up = cosh_func(x, a_true, b_true, x0_true)
    ymin = 50.0
    ymax = float(np.max(y_up))
    y = ymax - (y_up - ymin)
    
    # Draw thick curve
    for xi, yi in zip(x, y):
        cv2.circle(mask, (int(xi), int(round(yi))), 5, 255, -1)
        
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


def test_skeleton_and_catenary_span_200():
    # Additional test: x in [100, 300] span, hanging orientation
    mask = np.zeros((200, 400), dtype=np.uint8)
    a_true = 200.0
    x0_true = 200.0
    b_true = 50.0 - a_true
    x = np.arange(100, 300)
    y_up = cosh_func(x, a_true, b_true, x0_true)
    ymin = 50.0
    ymax = float(np.max(y_up))
    y = ymax - (y_up - ymin)
    for xi, yi in zip(x, y):
        cv2.circle(mask, (int(xi), int(round(yi))), 5, 255, -1)
    bbox = [100, 50, 300, int(np.max(y)) + 5]
    cline = extract_centerline(mask, bbox)
    assert cline.is_valid
    assert len(cline.polyline) > 100
    cat = calculate_sag_metrics(cline.polyline)
    assert cat.sag_px > 0
    assert cat.span_px > 150
    assert cat.residual < 10.0


def test_arch_curve_has_zero_sag():
    # An upward-bowing curve (arch) in image coordinates has y <= chord_y everywhere,
    # so sag_px is capped at 0.0 by design in calculate_sag_metrics.
    mask = np.zeros((200, 400), dtype=np.uint8)
    x = np.arange(50, 350)
    y = cosh_func(x, 200.0, 50.0 - 200.0, 200.0)
    for xi, yi in zip(x, y):
        cv2.circle(mask, (int(xi), int(yi)), 5, 255, -1)
    bbox = [50, 50, 350, int(np.max(y)) + 5]
    cline = extract_centerline(mask, bbox)
    assert cline.is_valid
    cat = calculate_sag_metrics(cline.polyline)
    assert cat.sag_px == 0.0
    assert cat.residual < 10.0



def test_segment_image_options_and_channel_order(monkeypatch):
    import torch
    from backend.config_store import get_setting, set_setting
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

    orig_params = get_setting("inference_params")
    try:
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
    finally:
        set_setting("inference_params", orig_params)


def test_centerline_two_fragments():
    # Mask with two disconnected wire fragments separated by an occlusion gap
    mask = np.zeros((100, 300), dtype=np.uint8)
    # Fragment 1: x in [20, 120), y = 50 (100 pixels length)
    cv2.line(mask, (20, 50), (119, 50), 255, thickness=2)
    # Fragment 2: x in [160, 260), y = 50 (100 pixels length)
    cv2.line(mask, (160, 50), (259, 50), 255, thickness=2)

    bbox = [10, 40, 270, 60]
    res = extract_centerline(mask, bbox)
    assert res.is_valid is True
    # ~200 points total across both fragments
    assert 180 <= len(res.polyline) <= 220
    # Ordered by x
    xs = [pt[0] for pt in res.polyline]
    assert all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    # Covers both fragments
    assert xs[0] <= 25
    assert xs[-1] >= 255
    assert any(x < 125 for x in xs)
    assert any(x > 155 for x in xs)


def test_centerline_noise_speckle_and_empty():
    # Empty mask
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    res_empty = extract_centerline(empty_mask, [10, 10, 50, 50])
    assert res_empty.is_valid is False

    # 3-pixel noise speckle only (< MIN_COMPONENT_PIXELS=5)
    speckle_mask = np.zeros((100, 100), dtype=np.uint8)
    speckle_mask[20, 20:23] = 255  # exactly 3 px
    res_speckle = extract_centerline(speckle_mask, [15, 15, 25, 25])
    assert res_speckle.is_valid is False

    # 3-pixel noise speckle near a real wire (wire has 50 px, speckle has 3 px)
    mask_with_noise = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(mask_with_noise, (10, 50), (59, 50), 255, thickness=2)  # real wire
    mask_with_noise[80, 80:83] = 255  # 3 px speckle
    res_with_noise = extract_centerline(mask_with_noise, [5, 5, 95, 95])
    assert res_with_noise.is_valid is True
    # Verify no points from speckle (y around 80) were included
    assert all(pt[1] < 70 for pt in res_with_noise.polyline)


def test_catenary_hanging_and_arch_residuals():
    x = np.arange(50, 350, dtype=float)
    y_up = cosh_func(x, 200.0, -150.0, 200.0)

    # Pure hanging catenary
    y_hang = float(np.max(y_up)) - (y_up - 50.0)
    res_hang = calculate_sag_metrics(list(zip(x, y_hang)))
    assert res_hang.residual < 1.0
    assert res_hang.sag_px > 0.0

    # Pure arch catenary
    res_arch = calculate_sag_metrics(list(zip(x, y_up)))
    assert res_arch.residual < 1.0
    assert res_arch.sag_px == 0.0


def test_catenary_shallow_noisy():
    # Shallow wire with 2% sag over 300 px span (true sag = 6 px) with +/- 1 px noise
    x = np.arange(0, 301, dtype=float)
    y_true = 100.0 + 6.0 * (1.0 - ((x - 150.0) / 150.0) ** 2)
    rng = np.random.RandomState(42)
    noise = rng.uniform(-1.0, 1.0, size=len(x))
    y_noisy = y_true + noise

    res = calculate_sag_metrics(list(zip(x, y_noisy)))
    assert res.residual < 3.0
    assert abs(res.sag_px - 6.0) < 2.0


def test_centerline_steep_slope_2_to_1():
    # 2:1 slope line (dy=100, dx=50) covering span
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.line(mask, (50, 50), (100, 150), 255, thickness=4)
    bbox = [50, 50, 100, 150]

    res = extract_centerline(mask, bbox)
    assert res.is_valid is True
    assert len(res.polyline) >= 100
    # Covers vertical span
    ys = [p[1] for p in res.polyline]
    assert max(ys) - min(ys) >= 95.0


def test_centerline_gap_40px():
    # Wire with a 40-px gap
    mask = np.zeros((100, 300), dtype=np.uint8)
    cv2.line(mask, (20, 50), (100, 50), 255, thickness=4)
    cv2.line(mask, (140, 50), (220, 50), 255, thickness=4)
    bbox = [20, 45, 220, 55]

    res = extract_centerline(mask, bbox)
    assert res.is_valid is True
    assert len(res.polyline) > 150
    # Gap from 105 to 135 has no centerline points
    gap_points = [p for p in res.polyline if 105 < p[0] < 135]
    assert len(gap_points) == 0


def test_centerline_thicknesses_hanging_curve():
    # Thicknesses 3, 5, 8, 11 px on a hanging curve all return ~1 point per column
    for th in [3, 5, 8, 11]:
        mask = np.zeros((200, 400), dtype=np.uint8)
        x = np.arange(100, 300)
        y_up = cosh_func(x, 200.0, -150.0, 200.0)
        y = float(np.max(y_up)) - (y_up - 50.0)
        for xi, yi in zip(x, y):
            cv2.circle(mask, (int(xi), int(round(yi))), th // 2, 255, -1)
        bbox = [100, 50, 300, int(np.max(y)) + 5]

        res = extract_centerline(mask, bbox)
        assert res.is_valid is True
        xs = [p[0] for p in res.polyline]
        # Exactly 1 point per column spanned
        assert len(res.polyline) == len(set(xs))
        assert 195 <= len(res.polyline) <= 220



