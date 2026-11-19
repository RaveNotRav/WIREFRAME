# WIREFRAME — Team Handoff & Deployment Guide (`HANDOFF.md`)

This document explains how to run **WIREFRAME** in **Mock Testing Mode** vs. **Real Model Mode**, where to drop trained YOLOv11-seg and WireNet `.pt` checkpoints, how the 4-class taxonomy and thesis-paper mathematical pipeline work, and how to run the automated test suites.

---

## 1. Canonical 4-Class Registry

The entire backend (`backend/utils/cv/`, `backend/formulas.py`, `backend/config_store.py`), mock detector, frontend UI (`analyze.html`, `dashboard.html`, `history.html`), and PDF/CSV/GeoJSON exports are unified around the 4 canonical classes:

| Canonical Class Key | Roboflow / `data.yaml` Alias | UI Display Label | Category |
| :--- | :--- | :--- | :--- |
| `power_distribution_line` | `power-distribution-lines` | **Power Line** | Overhead Wire |
| `communication_cable` | `telecom-cable`, `fiber-optic-drop` | **Communication Wire** | Overhead Wire |
| `dead_wire` | `abandoned-wire` | **Abandoned Wire** | Overhead Wire |
| `vegetation` | `Vegetation` | **Vegetation** | Context / Encroachment |

> **Backward Compatibility:** Legacy database records (`power_line`, `telecom_cable`, `fiber_optic`, `dead_abandoned`) automatically map to their canonical counterparts everywhere in the API, UI, and reports.

---

## 2. Where to Drop Trained `.pt` Model Weights

All model weight files (`.pt`) are git-ignored (`*.pt` in `.gitignore`) and reside in `backend/models/` (manifest documented in `backend/models/MODEL_MANIFEST.json`):

1. **Primary YOLOv11-seg Detector (Required for Real Mode):**
   - **Target path:** `backend/models/wireframe_seg.pt`
   - *(Already populated on this workstation from `runs/segment/runs/segment/wireframe_seg-5/weights/best.pt`; to update with newly trained weights, replace `backend/models/wireframe_seg.pt`.)*
2. **Optional WireNet v2.12 Pixel-Resolution Ensemble Refiner:**
   - The ensemble specification and member architecture configs are already committed in `backend/models/`:
     - `backend/models/v212_ensemble.json`
     - `backend/models/v212_convnext_t.arch.json` & `v212_convnext_t.log.json`
     - `backend/models/v212_mitb1.arch.json` & `v212_mitb1.log.json`
   - To enable full-resolution WireNet refinement on top of YOLOv11-seg proposals, drop the trained checkpoints into:
     - `backend/models/v212_convnext_t.pt`
     - `backend/models/v212_mitb1.pt`
     - `backend/models/wirenet-r34.pt`
   - *Graceful Degradation:* If the WireNet `.pt` files are absent, the pipeline automatically logs a debug message and uses the YOLOv11-seg masks directly without error.

---

## 3. Running the Server: Mock Mode vs. Real Model Mode

### A. Mock Testing Mode (`WIREFRAME_MOCK=1`)
Use Mock Mode for UI demos, grading scenarios (`low`, `moderate`, `high`, `critical`, `cycle`), or testing without GPU inference:

```powershell
# PowerShell (Windows)
$env:WIREFRAME_MOCK = "1"
$env:WIREFRAME_MOCK_SCENARIO = "cycle"   # optional: random | low | moderate | high | critical | cycle
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Or use the demo launcher script:
```powershell
.\run_demo.ps1 -MockScenario cycle
```

### B. Real Model Mode (`WIREFRAME_MOCK=0`)
Use Real Model Mode to run live YOLOv11-seg inference (`backend/models/wireframe_seg.pt`) with optional tiled inference (`sliced_predict` with cross-tile seam stitching) and automatic Hough vertical-edge pole detection fallback:

```powershell
# PowerShell (Windows)
$env:WIREFRAME_MOCK = "0"
$env:WIREFRAME_TILED = "1"                 # Enables 640x640 tiled inference (30% overlap) + cross-tile mask stitching
$env:WIREFRAME_HOUGH_POLE_FALLBACK = "1"   # Enables classical Hough pole detection when 4-class model outputs no pole
$env:WIREFRAME_BRIDGE_POLES = "1"          # Optional: bridges wire masks occluded by a detected utility pole
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### C. Verifying Live Health & Model Readiness (`GET /api/health`)
Open `http://127.0.0.1:8000/api/health` to inspect live diagnostics:
```json
{
  "status": "ok",
  "ready": true,
  "model_status": "loaded",
  "model_loaded": true,
  "schema_version": 7,
  "mock_mode": false,
  "weights_present": true,
  "wirenet_weights_present": false,
  "tiled_mode": true,
  "canonical_classes": [
    "power_distribution_line",
    "communication_cable",
    "dead_wire",
    "vegetation"
  ]
}
```

---

## 4. Thesis-Paper CV & Mathematical Modules

All thesis-conformant algorithms from Joanna's `wireframes-main` pipeline are integrated into `backend/utils/cv/`, `backend/formulas.py`, and `ml/`:

- **Acquisition Envelope & Adaptive Enhancement (`backend/utils/cv/preprocess.py` & `enhance.py`):**
  - Checks EXIF 35mm-equivalent focal length ($24\text{--}50\text{ mm}$), subject distance ($5\text{--}15\text{ m}$), elevation angle ($30\text{--}60^\circ$), and daylight capture hours ($06\text{:}00\text{--}18\text{:}00$), attaching `"acquisition_unverified"` or `"outside_envelope:<param>"` flags without rejecting valid images.
  - Provides auto-gamma low-light recovery, CIELAB $L^*$ CLAHE, Immerkaer noise estimation + NLM/bilateral denoising, blur-adaptive unsharp masking, HSV sky masking, and Excess Green ($\text{ExG} = 2G - R - B$) vegetation masks.
- **2D-FFT Visual Noise Index (`backend/utils/cv/vni.py` & `ml/calibrate_vni.py`):**
  - Resizes grayscale to $640\times 640$ (`cv2.INTER_AREA`), applies $5\times 5$ Gaussian pre-filter ($\sigma = 1.0$), mean subtraction, and 2D Hann window (`hann_2d`) before 2D-FFT power spectrum $|F(u,v)|^2$ ($r_L = 0.15 \cdot \min(M,N)/2$), rescaled via `q01`/`q99` into `Low` ($\le 0.25$), `Moderate` ($\le 0.50$), `High` ($\le 0.75$), and `Severe` ($> 0.75$) bands.
- **Zhang-Suen Skeletonization, Good Continuation & Span Fitting (`centerline.py`, `continuation.py`, `span_fit.py`, `catenary.py`):**
  - Vectorized two-subiteration Zhang-Suen thinning (`zhang_suen`), m-adjacent pixel graph (`pixel_graph`), spur pruning (`prune_spurs`, $L_{\min} = 30\text{ px}$), junction continuation (`continuation_path`), kink-split span fitting (`fit_span`), and perpendicular chord sag ratio $\delta_{\text{rel}} = \delta / L$.
- **Paper CRS (0–10) & Multi-Site Priority Index (`backend/formulas.py`):**
  - `compute_allowable_sag()` (Equation 9 + Kiessling change-of-state tension and `"default_parameters"` flag).
  - `compute_paper_crs()` (Equation 10 $0\text{--}10$ CRS & Table 1.2 `Low` $<3.0$, `Medium` $3.0\text{--}6.0$, `High` $>6.0$).
  - `compute_mspi()` & `compute_mspi_sensitivity_grid()` (Equation 11 & 66-point simplex sensitivity grid with Spearman's $\rho$ and top-$k$ overlap).

---

## 5. Running the Automated Test Suites

Both test suites run in isolated temporary environments (`WIREFRAME_MOCK=1` with temporary SQLite databases) and never touch real local SQLite data:

```powershell
# 1. Backend Python Test Suite (280 tests)
python -m pytest -q

# 2. Frontend Node.js ES-Module Test Suite (163 tests)
node --test "frontend/assets/js/**/*.test.js"
```
