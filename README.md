# WIREFRAME

Computer-vision web system for analyzing street-level photos of overhead utility wires.
Computes wire sag, safety clearance, and risk scores.

## Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, SQLite
- **CV/ML**: Ultralytics YOLOv11, OpenCV, NumPy, SciPy, Pillow
- **Frontend**: Vanilla HTML/CSS/JS

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
.\run_demo.ps1
# or directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Test

```bash
pytest -v
```

See `.env.example` for configuration options.
