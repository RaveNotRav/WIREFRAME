"""Characterization tests for backend.image_processing."""

import numpy as np
from PIL import Image

from backend.image_processing import (
    check_authenticity,
    compute_visual_noise_index,
    enhance_image,
)
from backend.utils.cv.preprocess import validate_and_preprocess


class TestExifOrientation:
    def _make_exif_jpeg(self, path, orientation):
        # Create an image that passes the resolution, dark, and blurry checks
        rng = np.random.RandomState(42)
        arr = np.clip(128 + rng.randint(-50, 51, (150, 200, 3)), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        exif = img.getexif()
        exif[0x0112] = orientation # 274 is Orientation
        img.save(str(path), "JPEG", exif=exif.tobytes())

    def test_exif_orientation_6_and_8(self, tmp_path):
        p6 = tmp_path / "ori6.jpg"
        self._make_exif_jpeg(p6, 6) # 90 CW. 200x150 should become 150x200
        res6 = validate_and_preprocess(str(p6))
        assert res6.quality_ok
        assert res6.original_shape == (200, 150) # h=200, w=150
        
        p8 = tmp_path / "ori8.jpg"
        self._make_exif_jpeg(p8, 8) # 90 CCW. 
        res8 = validate_and_preprocess(str(p8))
        assert res8.quality_ok
        assert res8.original_shape == (200, 150)

class TestComputeVisualNoiseIndex:
    def test_clean_low_vni(self, clean_image):
        vni = compute_visual_noise_index(clean_image)
        assert 0.0 <= vni <= 1.0
        assert vni < 0.1, f"Clean image should have low VNI, got {vni}"

    def test_noisy_higher_vni(self, noisy_image):
        vni = compute_visual_noise_index(noisy_image)
        assert 0.0 <= vni <= 1.0
        assert vni > 0.3, f"Noisy image should have higher VNI, got {vni}"

    def test_noisy_greater_than_clean(self, clean_image, noisy_image):
        clean_vni = compute_visual_noise_index(clean_image)
        noisy_vni = compute_visual_noise_index(noisy_image)
        assert noisy_vni > clean_vni

    def test_grayscale_input(self):
        gray = np.full((480, 640), 128, dtype=np.uint8)
        vni = compute_visual_noise_index(gray)
        assert 0.0 <= vni <= 1.0

    def test_range_always_0_1(self, noisy_image):
        vni = compute_visual_noise_index(noisy_image)
        assert 0.0 <= vni <= 1.0

    def test_tiny_image(self):
        tiny = np.full((2, 2, 3), 128, dtype=np.uint8)
        vni = compute_visual_noise_index(tiny)
        assert vni == 0.0


class TestCheckAuthenticity:
    def _make_jpeg(self, path, *, make=None, model=None, software=None):
        rng = np.random.RandomState(99)
        arr = np.clip(128 + rng.randint(-20, 21, (100, 100, 3)),
                      0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        exif = img.getexif()
        if make:
            exif[0x010F] = make
        if model:
            exif[0x0110] = model
        if software:
            exif[0x0131] = software
        if exif:
            img.save(str(path), "JPEG", exif=exif.tobytes())
        else:
            img.save(str(path), "JPEG")

    def _make_jpeg_no_exif(self, path):
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        img.save(str(path), "JPEG")

    def test_full_exif_likely_authentic(self, tmp_path):
        p = tmp_path / "good.jpg"
        self._make_jpeg(p, make="Canon", model="EOS R5")
        result = check_authenticity(str(p))
        assert result["score"] >= 70
        assert result["classification"] == "likely_authentic"

    def test_no_exif_deduction(self, tmp_path):
        p = tmp_path / "noexif.jpg"
        self._make_jpeg_no_exif(p)
        result = check_authenticity(str(p))
        assert result["score"] <= 75
        assert "no_exif_data" in result["reasons"]

    def test_ai_software_flagged(self, tmp_path):
        p = tmp_path / "ai.jpg"
        self._make_jpeg(p, make="AI", model="Gen",
                        software="Stable Diffusion v2")
        result = check_authenticity(str(p))
        assert "ai_generator_detected" in result["reasons"]
        assert result["score"] <= 60

    def test_nonexistent_file(self, tmp_path):
        result = check_authenticity(str(tmp_path / "missing.jpg"))
        assert result["score"] <= 75

    def test_classification_tiers(self, tmp_path):
        p = tmp_path / "test.jpg"
        self._make_jpeg(p, make="Canon", model="EOS")
        result = check_authenticity(str(p))
        score = result["score"]
        if score >= 70:
            assert result["classification"] == "likely_authentic"
        elif score >= 40:
            assert result["classification"] == "uncertain"
        else:
            assert result["classification"] == "flagged_for_review"


class TestEnhanceImage:
    def test_output_shape_matches_input(self, clean_image):
        out = enhance_image(clean_image)
        assert out.shape == clean_image.shape

    def test_original_not_modified(self, clean_image):
        original = clean_image.copy()
        enhance_image(clean_image)
        np.testing.assert_array_equal(clean_image, original)

    def test_dark_image_brightened(self, dark_image):
        out = enhance_image(dark_image)
        assert float(np.mean(out)) > float(np.mean(dark_image))

    def test_dtype_preserved(self, clean_image):
        out = enhance_image(clean_image)
        assert out.dtype == np.uint8

    def test_output_is_copy(self, clean_image):
        out = enhance_image(clean_image)
        assert out is not clean_image
