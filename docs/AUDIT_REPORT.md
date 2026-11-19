# WIREFRAME — Full Read-Only Audit Report

**Date:** 2026-10-20
**Auditor:** opencode (read-only audit; no application code modified)
**Repository:** `C:\Users\brain\Downloads\WIREFRAME` → `https://github.com/RaveNotRav/WIREFRAME.git`

**Revision audited (working tree, not clean):**
- HEAD: `01f3ad0` — *"Data integrity: exclude public tips from MPC ranking, fix trend cutoff format, shared GPS validation for single+batch, purge all mock files safely, safer path containment"*
- Modified but uncommitted: `backend/auth.py`, `backend/routes/analysis.py`, `backend/routes/stats.py`, `main.py`, `run_demo.ps1`, `scripts/seed_demo_admin.py`, `tests/test_seed_demo_admin.py`
- Untracked: `tests/test_audit_batch_c.py`

> The audit reflects the **dirty working tree**, i.e. the "Audit Batch C" changes already present on disk are treated as part of the system under review. If a clean-HEAD audit is needed, re-run against `01f3ad0` only.

**Method:** Static reading of every backend module and route with `file:line` citations, database migration review, endpoint inventory, security review, and one full read-only test execution. No source files were edited. The audit report itself is the only file created.

---

## 1. Executive summary

WIREFRAME is a FastAPI + SQLite web application that ingests street-level photos, runs a YOLOv11 segmentation model (or a deterministic mock), fits a catenary to each detected wire, estimates ground clearance and sag, and produces a per-wire and per-image risk/safety verdict plus PDF/CSV/GeoJSON reports.

**Overall state: functional and well-tested, but with several provisional/placeholder engineering decisions that matter for a thesis defense.**

Strengths:
- Clean layered backend: `main.py` → `backend/routes/*` → `backend/services/*` → `backend/utils/*`, with pure math isolated in `backend/formulas.py`.
- 245 automated tests pass in ~151 s (0 failures).
- Solid input hardening: filename sanitization, path containment, extension allowlists, image verification, decompression-bomb guard, CSV injection guard, XSS escaping, PBKDF2 password hashing, CSRF + RBAC.
- Configurable thresholds via a DB-backed settings store with an in-memory cache.

Top concerns (details in §6):
1. **PEC minimum clearance is a single provisional 5.5 m constant for every wire class** — explicitly flagged `TODO(VERIFY-WITH-ADVISER)` in the code.
2. **Anonymous/public uploads are effectively un-rate-limited**: `check_anon_rate_limit()` exists and is imported but never called.
3. **PDF per-wire table prints blank Band / Span / Clearance columns** because it reads wire keys that do not exist in the `wires` schema.
4. **`risk_bands` is stored as a list but validated as a dict**, making `PUT /api/config` risky/broken for that key.
5. **Duplicate legacy analysis engine** (`analysis_engine.analyze_image`) diverges from the production pipeline (`utils/pipeline.execute_pipeline`).

---

## 2. Phase 1 — Architecture & structure

### 2.1 Stack & entry points
- Stack (`README.md:6-10`): Python 3.12, FastAPI, Uvicorn, SQLite; Ultralytics YOLOv11, OpenCV, NumPy, SciPy, Pillow; vanilla HTML/CSS/JS.
- ASGI app + lifespan: `main.py:118`, `main.py:49-116`.
- Dev run (`README.md:23-27`, `main.py:275-279`): `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` / `run_demo.ps1`.
- Health: `GET /health` and `GET /api/health` (`main.py:190-213`).
- Routers registered (`main.py:143-151`): auth, users, config, sites, reporting, batches, analysis, admin, stats.
- Global handlers: `RequestValidationError → 422` (`main.py:124-129`) and catch-all `Exception → 500` with an `error_id` (`main.py:131-141`).

### 2.2 Directory map
```
WIREFRAME/
  main.py                      FastAPI app, lifespan, health, result endpoint, HTML routes
  make_demo_images.py          demo image generator (used by tests)
  backend/
    auth.py                    PBKDF2 auth, sessions, RBAC deps, CSRF, rate limiter
    config_store.py            DEFAULT_SETTINGS + SQLite-backed settings cache
    database.py                connection + 11 ordered migrations
    formulas.py                CRS, barangay risk, severity, bands, safety, clearance status
    logging_config.py          stderr logging setup
    main_state.py              global model/config-failure flags
    mock_detector.py           mock ultralytics-compatible detector
    routes/                    admin, analysis, auth, batches, config, reporting, sites, stats, users
    services/                  sites.py (normalization/creation), worker.py (batch thread)
    utils/
      analysis_engine.py       legacy facade/duplicate pipeline
      clearance_helpers.py     pure per-wire evaluate/rollup helpers
      pipeline.py              production state machine
      report_pdf.py            ReportLab PDF builders
      uploads.py               filename sanitizer + path containment
      cv/                      calibration, catenary, centerline, preprocess, segment, vni, types, mock_generator, cli*
    ml/                        offline training/eval scripts (not used at runtime)
    models/                    model weight location (default backend/models/wireframe_seg.pt)
  frontend/
    pages/                     index, login, analyze, dashboard, history, states-preview, design-system
    assets/js/                 main, analyze, dashboard, presentation, loaders + lib/* + *.test.js
    assets/css/                tokens, components, bento, neumorphism, global-header
  tests/                       34 test modules (245 tests)
  scripts/seed_demo_admin.py   demo admin seeder (--add/--reset/--ensure)
  _quarantine/                 archived scratch scripts/audit artifacts (git-ignored)
```

### 2.3 Runtime flow (production path)
1. Client uploads photo(s) → `POST /api/analysis/upload` (`analysis.py:68`) or `POST /api/batches` (`batches.py:30`).
2. Upload validates type/size/content/GPS/location, saves a sanitized file under `uploads/`, inserts a `batches` row and one `analyses` row per file (status `queued`), then `enqueue_batch(batch_id)` (`analysis.py:200-205`, `batches.py:161-164`).
3. Background worker thread (`services/worker.py:16-74`) marks the batch `running` and calls `process_analysis` → `execute_pipeline` per item (`worker.py:45`).
4. `execute_pipeline` (`pipeline.py:90-345`) is the real state machine: preprocess/quality gate → mock or real segmentation → centerline → catenary → CRS → calibration/clearance → risk rollup → persist `wires` + analysis, write annotated JPEG and VNI spectrum PNG (`pipeline.py:246-325`).
5. Client polls `GET /api/batches/{id}` (`batches.py:176`) and `GET /api/analysis/{id}/events` (`analysis.py:375`); results via `GET /api/analysis/{id}/result` (`main.py:216`).
6. Reports: PDF (`reporting.py:122`), summary PDF (`reporting.py:193`), GeoJSON (`reporting.py:279`), CSV (`reporting.py:369`), MPC ranking (`reporting.py:22`), dashboard stats (`stats.py:9`).
7. Manual scale calibration: `POST /api/analysis/{id}/calibrate` (`analysis.py:491`) recomputes clearance from a 2-point reference and re-rolls up the analysis.

### 2.4 Configuration & environment
- Defaults (`config_store.py:15-89`): `crs_params`, `site_risk_weights`, `risk_bands`, `class_weights`, `pole_heights`, `pec_min_clearance_by_class`, `margin_of_error_pct`, `model_path`, `class_aliases`, `inference_params`, `severity_weights`, `severity_thresholds`, `safety_mapping`, `action_mapping`.
- Seeding/cache: `init_config_store` (`config_store.py:92-99`), `_seed_defaults` (`102-115`), `get_setting` (`126-138`), `set_setting` (`141-156`).
- Env vars (`.env.example`, README): `WIREFRAME_MOCK`, `WIREFRAME_DB_PATH`, `WIREFRAME_LOG_LEVEL`, `WIREFRAME_HOST`, `WIREFRAME_PORT`, plus `WIREFRAME_MODEL_PATH` (`main.py:64`), `WIREFRAME_CONF/IOU/DEVICE` (`segment.py:27-40`), `WIREFRAME_ENV=test` lowers PBKDF2 iterations (`auth.py:15`), `TRUSTED_PROXY` affects rate-limit IP (`auth.py:173-174`).
- Logging: single stderr handler, level from env (`logging_config.py:8-26`).

---

## 3. Phase 2 — Database, API, PEC rules, security

### 3.1 Database schema & migrations
`backend/database.py` — SQLite, WAL, `busy_timeout=3000`, `foreign_keys=ON`, `Row` factory (`database.py:10-19`). Schema versioning table `schema_version` (`22-37`). 11 ordered migrations (`256-269`):

| Ver | Name | Effect | Ref |
|----|------|--------|-----|
| 1 | base_tables | `analyses`, `detections`, `reports` + 4 indexes | `44-97` |
| 2 | settings_table | `settings(key, value_json, updated_at, updated_by)` | `100-108` |
| 3 | wires_table | `wires` + index | `111-130` |
| 4 | orchestrator | `batches`, `pipeline_events`; add `analyses.batch_id`, `current_state` + indexes | `132-156` |
| 5 | rbac_and_sites | `users`, `sessions`, `sites`, `calibrations`; add analyses `created_by/site_id/gps_*/spectrum_path/width/height`; add `batches.created_by` | `158-207` |
| 6 | mock_flag | add `analyses.is_mock` | `209-212` |
| 7 | company_street | add `analyses.company/street`, `sites.street/barangay_norm/street_norm` + unique normalized index + backfill | `214-230` |
| 8 | dashboard_indexes | `analyses(created_at)`, `(is_mock)`, `(site_id)` | `232-237` |
| 9 | analysis_notes | add `analyses.notes` | `239-242` |
| 10 | public_submissions | add `guest_session_id`, `is_public` to `batches` and `analyses` | `245-249` |
| 11 | ground_reference_and_margin | add `analyses.pole_bbox_json`, `margin_error_m` | `252-254` |

**`wires` columns** (`database.py:113-128`): `id, analysis_id, class_name, confidence, crs, band, sag_px, span_px, sag_m, clearance_m, clearance_status, safety_level, centerline_json, params_json`. **No `span_m`, no `clearance_min_m/max_m`, no `crs_band`.**

**`analyses` effective columns** (base `46-68` + migrations): `id, filename, original_path, annotated_path, status, barangay, sector, summary, risk_score, vni, max_crs, dead_wire_count, barangay_risk, enhancement, authenticity, sag_meters, clearance_ok, clearance_available, pec_min_clearance_m, assumed_pole_height_m, created_at, batch_id, current_state, created_by, site_id, gps_lat, gps_lng, gps_source, spectrum_path, width, height, is_mock, company, street, notes, guest_session_id, is_public, pole_bbox_json, margin_error_m`.

> **Dead schema:** `detections` (`database.py:70-83`) and its index (`93`) are created but **never inserted into** (grep of `INSERT INTO detections` = no matches). `analyses.enhancement` / `authenticity` are also unwritten.

### 3.2 API inventory (29 route handlers; 9 modules + main)

Auth (`/api/auth`, `backend/routes/auth.py`): `POST /login` (csrf, `22`), `POST /logout` (csrf, `62`), `GET /me` (`69`).

Users (`/api/users`, `users.py`): `POST ""` (csrf + admin, `14`).

Config (`/api/config`, `config.py`): `GET ""` (admin, `19`), `GET /public` (whitelist `40`), `PUT ""` (csrf + admin, `46`).

Sites (`/api/sites`, `sites.py`): `GET ""` (authed, `16`), `POST ""` (csrf + admin/lgu, `35`).

Reporting (`/api`, `reporting.py`): `GET /mpc/ranking` (lgu/admin, `22`), `GET /reports/{analysis_id}.pdf` (`122`), `POST /reports/summary` (csrf, `193`), `POST /reports/geojson` (csrf, `279`), `POST /reports/csv` (csrf, `369`).

Batches (`/api/batches`, `batches.py`): `POST ""` (csrf, `30`), `GET /{batch_id}` (`176`), `POST /{batch_id}/cancel` (csrf, `212`).

Analysis (`/api/analysis`, `analysis.py`): `POST /upload` (csrf, `68`), `GET ""` (current user, `226`), `DELETE /{analysis_id}` (csrf, `330`), `GET /{analysis_id}/events` (csrf, `375`), `GET /{analysis_id}/wires` (`394`), `GET /{analysis_id}/original` (`433`), `GET /{analysis_id}/overlay` (`451`), `GET /{analysis_id}/annotated` (`455`), `GET /{analysis_id}/spectrum` (`459`), `POST /{analysis_id}/calibrate` (csrf, `491`).

Admin (`/api/admin`, `admin.py`): `DELETE /purge_mock` (csrf + admin, `12`).

Stats (`/api/stats`, `stats.py`): `GET /dashboard` (admin/lgu, `9`).

Main (`main.py`): `GET /health`, `GET /api/health` (`190-191`), `GET /api/analysis/{id}/result` (`216`), HTML pages `/index.html /login.html /analyze.html /states-preview.html /dashboard.html /history.html /` (`156-184, 270`), static mounts `/assets`, `/pages` (`187-188`).

**Access model:** `get_current_user` (`auth.py:64-93`), `get_optional_user` (`96-120`), `verify_analysis_access` (`122-138`) — private rows require auth; public/guest rows require the `guest_session` cookie hash. `require_role` treats `admin` as superset (`auth.py:140-145`). `verify_csrf` (`147-164`) passes GET and accepts `X-Requested-With: XMLHttpRequest` or `Origin.host == Host`.

### 3.3 PEC / standard / formula audit

| Quantity | Value / formula | Location |
|---|---|---|
| PEC min clearance (per class + default) | **5.5 m for all** | `config_store.py:42-48` |
| PEC margin of error | 20 % | `config_store.py:49` (`calibration.py:105`) |
| Assumed pole heights | `[9, 10, 12]`, first used as reference | `config_store.py:41` (`calibration.py:52`) |
| CRS sag offset / scale | 0.03 / 0.17 | `config_store.py:18-19` |
| CRS residual offset / gain / cap | 2.0 / 5.0 / 20 | `config_store.py:20-22` |
| CRS formula | `clip(sag_term + residual_pen, 0, 100)` | `formulas.py:16-33` |
| Barangay risk weights | vni 0.45 / crs 0.45 / dead 0.10 | `config_store.py:24-28` |
| Barangay risk formula | `w_vni·VNI·100 + w_crs·max_CRS + w_dead·density·100` | `formulas.py:36-50` |
| Risk bands | Low ≤.33, Moderate ≤.66, High ≤.85, Critical ≤1.00 (0–1 scale) | `config_store.py:29-34` (`formulas.py:88-104`) |
| Severity thresholds | high-conf ≥0.7, med ≥0.4 (inverted: high conf = low severity) | `config_store.py:73-76` (`formulas.py:53-63`) |
| Severity weights | high 1.0 / medium 0.5 / low 0.1 | `config_store.py:68-72` |
| Detection risk | mean severity weight × 100 | `formulas.py:71-80` |
| Final risk score | 0.5·detection + 0.5·barangay | `formulas.py:83-85` |
| Clearance status | `low ≥ min → ok`, `high < min → below`, else `marginal` | `formulas.py:107-126` |
| Safety mapping | CRS/clearance ranks → Safe/Caution/Unsafe/Critical | `config_store.py:77-81` (`formulas.py:129-170`) |
| Scale calibration | manual > pole bbox > unavailable; pole touching bottom edge ⇒ unavailable | `calibration.py:19-58` |
| Clearance estimate | `(ground_y − lowest_wire_y) · m/px`, ±margin | `calibration.py:61-115` |

**PEC / standard concerns (not verified against the actual PEC 2017 text — flagged by the code itself at `formulas.py:4-5, 23, 57, 180, 220`):**
- Using one 5.5 m minimum for `power_line`, `telecom_cable`, `fiber_optic`, and `dead_abandoned` is almost certainly not PEC-accurate (supply vs communication conductors have different requirements and different crossing rules). The code marks this `TODO(VERIFY-WITH-ADVISER)`.
- CRS constants (`sag_offset`, `sag_scale`, residual gain/cap) are explicitly `PROVISIONAL` (`config_store.py:16`) and require field recalibration.
- The severity mapping is intentionally **inverted** (high model confidence ⇒ low "severity"/risk). This means a confidently-detected hazardous wire contributes *less* to detection risk. This may be a deliberate design (confidence = certainty of clear detection), but it is counter-intuitive for a safety score and should be defended or reversed with the adviser.
- `compute_barangay_risk` (`formulas.py:36-50`) is computed **per image**, not aggregated per barangay, despite its name; the only barangay-level aggregation is the MPC ranking average (`reporting.py:29-36`).

### 3.4 Security posture

Implemented controls:
- Password hashing: PBKDF2-HMAC-SHA256, 600 000 iterations (1 000 under `WIREFRAME_ENV=test`), 16-byte salt, `hmac.compare_digest` (`auth.py:15, 27-46`).
- Sessions: 32-byte URL-safe tokens stored only as SHA-256 hashes, 8 h expiry, expiry cleanup (`auth.py:48-62, 79-86`).
- CSRF: requested-with header or Origin==Host (`auth.py:147-164`).
- Upload safety: extension allowlist (`.jpg/.jpeg/.png`), sanitized collision-free filenames with Windows-reserved-name handling and stem truncation (`uploads.py:17-81`), `is_relative_to` path containment (`uploads.py:84-104`), 15 MB per-file / 60 MB per-batch limits (`analysis.py:112`, `batches.py:67,108`), PIL content verification and `DecompressionBombError` handling (`analysis.py:116-127`, `batches.py:122-141`).
- SQL injection: parameterized queries throughout; LIKE wildcards escaped (`analysis.py:282-286`, `analysis.py:352-357` uses resolved containment before delete).
- Report safety: CSV formula-injection guard (`reporting.py:361-367`), XML-escaped PDF text (`report_pdf.py:42-55`).
- Cookies: `httponly`, `samesite=lax`, `secure` when HTTPS (`auth.py:49-57`, `analysis.py:213-221`, `batches.py:171-172`).

Gaps/limitations:
- **No anonymous rate limiting is applied** despite the function existing (see §6 F2).
- Rate limiter is in-process memory only (`auth.py:168-237`): resets on restart and is not shared across Uvicorn workers; `login` 5/15 min, authenticated `upload` 30/60 s.
- No security response headers (CSP, `X-Content-Type-Options`, `X-Frame-Options`, HSTS). No CORS middleware is configured (defaults to same-origin, which is safe but implicit).
- `PUT /api/config` can select any `.pt` inside `backend/models/` and load it immediately (`config.py:88-117`) — admin-only, but a high-impact endpoint.

---

## 4. Phase 3 — Feature implementation status

| # | Feature | Status | Evidence |
|---|---|---|---|
| 1 | Image upload (single) | Implemented | `analysis.py:68-223` |
| 2 | Batch upload (≤10 files, ≤60 MB) | Implemented | `batches.py:30-173` |
| 3 | Guest/anonymous submissions + isolation | Implemented | `analysis.py:83-90, 212-221`; `auth.py:122-138` |
| 4 | GPS from form or EXIF, PH-bounds warning | Implemented | `analysis.py:39-66, 129-156`; `services/sites.py:8-46` |
| 5 | Location validation & site de-dup | Implemented | `sites.py:50-124` |
| 6 | Background batch worker + cancel | Implemented | `services/worker.py:16-107`; `batches.py:212-235` |
| 7 | Startup zombie recovery | Implemented | `main.py:94-103` |
| 8 | Image quality gate (corrupt/res/low/blur/dark) | Implemented | `cv/preprocess.py:24-72` |
| 9 | YOLOv11 segmentation ψ mock detector | Implemented | `cv/segment.py:54-130`; `mock_detector.py`, `cv/mock_generator.py` |
| 10 | VNI (FFT visual-noise index) + spectrum PNG | Implemented | `cv/vni.py:21-78` |
| 11 | Centerline extraction (median per column/row) | Implemented | `cv/centerline.py:77-130` |
| 12 | Catenary fit + true sag-to-span | Implemented | `cv/catenary.py:27-125` |
| 13 | CRS scoring + class weights | Implemented | `formulas.py:16-33`; `pipeline.py:221-244` |
| 14 | Clearance estimation + margin band | Implemented | `cv/calibration.py:61-115` |
| 15 | 2-point manual calibration + recompute | Implemented | `analysis.py:491-680` |
| 16 | Safety verdict (worst-of CRS/clearance) | Implemented | `formulas.py:129-170`; `clearance_helpers.py:189-206` |
| 17 | Per-wire persistence | Implemented | `pipeline.py:246-258`; `database.py:113-128` |
| 18 | Annotated image + spectrum persistence | Implemented | `pipeline.py:260-293` |
| 19 | Risk score + barangay risk (per image) | Implemented | `formulas.py:36-85`; `pipeline.py:295-298` |
| 20 | Real-time pipeline events | Implemented | `pipeline.py:76-88`; `analysis.py:375-391` |
| 21 | Analysis result API mapping | Implemented | `main.py:216-264` |
| 22 | PDF report (single) | Implemented (with defect F3) | `reporting.py:122-186`; `report_pdf.py:98-333` |
| 23 | Summary PDF | Implemented | `reporting.py:188-278`; `report_pdf.py:335-416` |
| 24 | CSV export | Implemented | `reporting.py:369-438` |
| 25 | GeoJSON export | Implemented | `reporting.py:279-359` |
| 26 | MPC ranking | Implemented | `reporting.py:22-39` |
| 27 | Dashboard stats (KPI/dist/heatmap/trend) | Implemented | `stats.py:9-136` |
| 28 | Config store + admin API + public whitelist | Implemented (defect F4) | `config_store.py`; `config.py` |
| 29 | Auth login/logout/me | Implemented | `auth.py` (route) |
| 30 | Users create (admin) | Implemented | `users.py:14-43` |
| 31 | Sites list/create | Implemented | `sites.py:16-46` |
| 32 | Mock-data purge | Implemented | `admin.py:12-61` |
| 33 | Demo admin seeding (`--add/--reset/--ensure`) | Implemented | `scripts/seed_demo_admin.py` |
| 34 | ML training/eval tooling | Present, offline | `backend/ml/*.py` |
| 35 | Frontend pages + JS helpers + JS tests | Implemented | `frontend/pages/*`, `frontend/assets/js/*` |

**Not implemented / stubbed:**
- Per-wire `span_m` and clearance range are computed then not stored (`analysis_engine.py:122`; `pipeline.py:234-235` drops span).
- `detections` table is unused.
- Report generation is synchronous on request, not part of the pipeline (`pipeline.py:186, 331` comments "In a real app we'd build a report PDF here").
- No barangay-level aggregation model (only site averaging).
- No `package.json` scripts / JS test runner configured, despite `*.test.js` files.

---

## 5. Phase 4 — Test & lint verification

**Environment used:** `fresh_venv` (Python 3.12, pytest 8.3.4, httpx 0.28.1, torch 2.14.0, ultralytics 8.3.52, opencv-python-headless 4.10.0.84).

**Command**
```powershell
.\fresh_venv\Scripts\python.exe -m pytest tests -q --tb=short -p no:cacheprovider
```
(ran from the repository root; **must** be run there because `main.py:187` mounts `frontend/assets` relative to CWD and `tests/conftest.py:64` imports `main`).

**Result**
```
245 passed, 1 warning in 151.06s (0:02:31)
```
- 245 collected, 245 passed, **0 failed, 0 reported skipped**. The single warning is a Starlette/anyio deprecation (`BlockingPortal` alias), not a project defect.

**Test inventory (34 modules):** auth, rbac, csrf, anonymous isolation, upload regression/hardening, filename sanitization, batch limits/path traversal/decompression bomb, calibration recompute & rollback, clearance rollup & API status mapping, pipeline state + zombie recovery, worker failure recovery, CV (preprocess/VNI/centerline/catenary), formulas & scoring, config store/API, database migrations, export (CSV/GeoJSON), PDF, stats, MPC ranking, migrations, mock-mode E2E, ML pipeline, model integration, demo-admin seeding, and the new `test_audit_batch_c.py`.

**Operational warning:** `tests/test_pipeline.py:19-23, 33-37` deletes `wireframe.db` in the repo root during setup/teardown. Running the suite **deleted the working `wireframe.db`**; this audit backed it up beforehand and restored it. Add a guard so tests never touch a real database.

**Missing tooling (not installed):**
- `pytest-asyncio` and `pytest-cov` are not present (`pip show` reports "not found"). The suite still runs (the async test was collected/executed successfully under the available anyio support), but strict asyncio mode and coverage reporting are unavailable.
  ```powershell
  .\fresh_venv\Scripts\python.exe -m pip install pytest-asyncio pytest-cov
  ```
- `ruff` is declared in `requirements-dev.txt` (`ruff==0.3.3`) but **not installed** in either venv, so the configured lint gate (`ruff.toml`: `select = ["F", "E722"]`) did not run.
  ```powershell
  .\fresh_venv\Scripts\python.exe -m pip install -r requirements-dev.txt
  ```
- Fallback verification performed: `python -m compileall backend main.py scripts` → exit 0 (no syntax errors).

---

## 6. Findings register

Severity: **P0** = correctness/safety or security, **P1** = significant quality/consistency, **P2** = cleanup/maintainability.

### P0

**F1 — Provisional, single-value PEC clearance.**
`pec_min_clearance_by_class` is 5.5 m for every class and the `default` (`config_store.py:42-48`); clearance status compares against it (`formulas.py:107-126`); code flags `TODO(VERIFY-WITH-ADVISER)` (`formulas.py:5, 23, 180`). Risk: safety verdicts that do not reflect actual PEC 2017 class/voltage requirements.

**F2 — Anonymous upload rate limiting defined but never enforced.**
`check_anon_rate_limit()` (`auth.py:215-237`) is imported in `analysis.py:23` and `batches.py:20` but **never called** (grep confirms only definition + imports). Guest uploads (`analysis.py:85-90`, `batches.py:47-53`) therefore have no rate limit; the 429 message at `auth.py:234` is dead code.

**F3 — PDF per-wire table reads non-existent wire keys.**
`report_pdf.py:284-289` reads `crs_band`, `span_m`, `clearance_min_m`, `clearance_max_m`, but `wires` rows only contain `band`, `sag_m`, `clearance_m` (`database.py:113-128`). Result: Band/Span/Clearance columns render empty/`None` in the single-report PDF wire table, while data exists in `params_json`/`band`.

### P1

**F4 — `risk_bands` type mismatch between store and API.**
Stored/consumed as a **list** of `{min,max,label}` (`config_store.py:29-34`; `formulas.py:99-104`; `stats.py:92-97`), but `PUT /api/config` validates it as a **dict** keyed `Low/Moderate/High/Critical` (`config.py:56-61`) and `main.py:39-43` hedges. Sending a dict would break `compute_crs_band`; sending a valid list would fail contiguity validation. The admin config update for risk bands is effectively inconsistent.

**F5 — Duplicate legacy analysis engine diverges from production.**
`analysis_engine.analyze_image` (`analysis_engine.py:26-185`) duplicates `pipeline.execute_pipeline` but differs: no class alias resolution, `str(dict)` params (not JSON), computes `span_m` but the table lacks it, no `safety_reasons`, different dead-wire handling. It is exercised only by tests (`test_pipeline.py:283`, `test_audit_fixes.py:1687`) and `cv/cli_phase3.py:143,176` — a maintenance trap.

**F6 — Inverted severity semantics.**
`get_severity` maps high confidence → low severity (`formulas.py:53-63`) and feeds `compute_detection_risk` (`formulas.py:71-80`). A high-confidence hazardous detection lowers detection risk. Deliberate or not, it needs an explicit rationale in the thesis and ideally a clearer name (e.g. `confidence_to_uncertainty_weight`).

**F7 — "Barangay risk" is per-image.**
`compute_barangay_risk` (`formulas.py:36-50`) is called once per image (`pipeline.py:297`) and stored on the analysis. No barangay aggregation exists beyond the site-average MPC ranking (`reporting.py:29-36`). Naming vs. semantics mismatch.

**F8 — Tests delete the real `wireframe.db`.**
`tests/test_pipeline.py:19-23,33-37` removes `wireframe.db` in CWD (repo root). Any developer running `pytest` loses the demo database. Should use only `tmp_path`/env-configured DBs.

**F9 — `_validate_config` is effectively a no-op.**
`main.py:29-47` returns `True` and its contiguity branch is commented/pass. Startup "integrity check" does not actually validate bands/clearance config.

### P2

**F10 — Dead code/schema:** `detections` table + index (`database.py:70-83,93`), `compute_metric_clearance` (deprecated, self-described wrong, `formulas.py:173-215`), `rank_sites` (`formulas.py:217-231`), `analyses.enhancement/authenticity` columns.
**F11 — Unused imports / mid-file imports:** `stats.py:1-2` (`sqlite3`, `List/Dict/Any`), `analysis.py:464-466` imports mid-file, duplicate `PIL.Image` import (`analysis.py:19,116`).
**F12 — `web/reporting.py` "verdict" field is just `status`** (`reporting.py:327, 414`) — misleading export field name.
**F13 — `sites.create_site` lacks lat/lng range validation** (`sites.py:35-45`) unlike `validate_gps_coordinates`.
**F14 — `get_analyses` has no upper bound on `limit`** (`analysis.py:229`).
**F15 — `GET /api/analysis/{id}/events` declares `verify_csrf` on a GET** (`analysis.py:375`) — harmless but misleading.
**F16 — HTML routes without explicit encoding** (`main.py:157-159, 272`) vs utf-8 elsewhere.
**F17 — Mock filename "reject" becomes `no_wires`, not `rejected`** (`mock_generator.py:47-49` → `pipeline.py:181-188`).
**F18 — No security headers / implicit CORS.** No middleware configuring CSP, `X-Content-Type-Options`, `X-Frame-Options`, or CORS (`main.py`).
**F19 — Global in-memory rate-limit and report cache invalidation.** Rate limiter per-process (`auth.py:168`); a single config write deletes **all** cached reports (`config.py:131-143`).
**F20 — `frontend` JS tests exist but no runner/scripts** (`package.json` only lists `puppeteer`; no `scripts`).

---

## 7. Phase 5 — Recommended plan (NOT YET IMPLEMENTED)

> Per instruction, **no code has been modified**. The following is a proposal awaiting approval. Suggested execution order, each item independently testable and source-bound to a commit.

### Milestone 1 — Safety-critical correctness (P0)
1. **PEC clearance table (F1).** Replace the single 5.5 m map with a class/voltage-keyed table sourced from PEC 2017; keep 5.5 m only as a clearly-labeled fallback; add a migration/setting docs entry and a test asserting per-class thresholds. Resolve `TODO(VERIFY-WITH-ADVISER)`.
   - Files: `backend/config_store.py`, `backend/formulas.py`, migrations in `backend/database.py`.
   - Acceptance: `compute_clearance_status` uses the correct class minimum; tests cover power/telecom/dead with distinct minima.
2. **Enforce public rate limiting (F2).** Call `check_anon_rate_limit` on guest branches of `POST /api/analysis/upload` and `POST /api/batches`; add tests for the 4th guest upload in 15 min → 429.
   - Files: `backend/routes/analysis.py`, `backend/routes/batches.py`.
3. **Fix PDF wire table (F3).** Emit the real wire fields (use `band`, add derived span, use `clearance_m` and `params_json` low/high). Add a PDF test asserting non-empty Band/Span/Clearance.
   - Files: `backend/utils/report_pdf.py`, `backend/routes/reporting.py`.

### Milestone 2 — Consistency (P1)
4. **Unify `risk_bands` representation (F4).** Choose list (recommended, matches consumers), fix `PUT /api/config` validation and `_validate_config`; add round-trip test.
   - Files: `backend/routes/config.py`, `main.py`, `backend/config_store.py`.
5. **Decide the fate of the legacy engine (F5).** Either delete `analysis_engine.analyze_image` + `cli_phase3` usage and port tests to `pipeline`, or make it a thin wrapper that calls `execute_pipeline`. Add a single source-of-truth test.
6. **Clarify/reverse severity semantics (F6)** and document the "barangay risk" definition (F7) — code rename and thesis text.
7. **Test isolation (F8).** Ensure `tests/test_pipeline.py` never deletes the repo `wireframe.db`; use `tmp_path` + `WIREFRAME_DB_PATH`.

### Milestone 3 — Hardening & cleanup (P2)
8. Remove dead schema/code (F10–F11); fix export field names (F12); add `sites` validation (F13); cap `limit` (F14); add security headers (F18); add `scripts`/JS test runner (F20).
9. Add CI that runs `pytest` + `ruff` from a clean checkout, with `pytest-cov` threshold.

### Risks / dependencies
- PEC values require adviser confirmation before changing any safety threshold (blocking F1).
- Any PEC/CRS change invalidates cached reports; the existing global report bust (`config.py:131-143`) will cover it, but budgets/regeneration cost should be checked.
- Keep changes bound to an exact commit; run the full 245-test suite plus new tests after each milestone.

---

## Appendix A — Component/file inventory (backend)
- Entry: `main.py`
- Routes: `backend/routes/{admin,analysis,auth,batches,config,reporting,sites,stats,users}.py`
- Services: `backend/services/{sites,worker}.py`
- Utils: `backend/utils/{analysis_engine,clearance_helpers,pipeline,report_pdf,uploads}.py`
- CV: `backend/utils/cv/{calibration,catenary,centerline,preprocess,segment,vni,types,mock_generator,cli,cli_phase3,cli_phase4}.py`
- Core: `backend/{auth,config_store,database,formulas,logging_config,main_state,mock_detector}.py`
- ML (offline): `backend/ml/{train_seg,eval_seg,dataset_stats,split_dataset,merge_datasets,clean_dataset,export_best,audit_agreement}.py`, `classes_map.yaml`, `wireframe_seg.yaml`
- Frontend: `frontend/pages/*.html`, `frontend/assets/js/**`, `frontend/assets/css/**`
- Tests: `tests/*.py` (34 modules)
- Scripts: `scripts/seed_demo_admin.py`; `make_demo_images.py`

## Appendix B — Commands run during this audit
```powershell
# test suite (read-only w.r.t. source; DB backed up/restored)
.\fresh_venv\Scripts\python.exe -m pytest tests -q --tb=short -p no:cacheprovider
# -> 245 passed, 1 warning in 151.06s

# syntax check
.\fresh_venv\Scripts\python.exe -m compileall -q backend main.py scripts   # exit 0

# dependency checks
.\fresh_venv\Scripts\python.exe -m pip show pytest-asyncio pytest-cov      # not found
.\fresh_venv\Scripts\python.exe -m pip list
```

**End of report.** No application code was changed. Awaiting Phase 5 approval before any implementation.
