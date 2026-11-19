# WIREFRAME Dataset Workspace (`datasets/`)

This directory holds local YOLOv11-seg training, validation, and evaluation datasets and is **git-ignored** (`datasets/` in `.gitignore`) so large image archives are never committed to the repository.

## 1. Staged Dataset: `datasets/wireframes_v1/`
Extracted from `Wireframes-Prelabelled-Datasets.v1i.yolov11.zip` (965 annotated images, CC BY 4.0, exported from Roboflow workspace `ravens-workspace-g6fye/wireframes-prelabelled-datasets`).

### Directory Layout
```text
datasets/
├── README.md
└── wireframes_v1/
    ├── data.yaml
    ├── README.dataset.txt
    ├── README.roboflow.txt
    └── train/
        ├── images/   (965 pre-labelled JPG images, 432x432 auto-oriented)
        └── labels/   (965 YOLOv11-seg polygon annotation .txt files)
```
*(When running curated splits via `ml/`, images are partitioned into a 70% train / 20% valid / 10% test split.)*

## 2. Canonical 4-Class Mapping (`data.yaml` → WIREFRAME System)

The dataset `data.yaml` defines `nc: 4` with the following class indices and names, which resolve automatically through `DEFAULT_CLASS_ALIASES` (`backend/utils/cv/types.py`) and `ml/class_map.yaml`:

| `data.yaml` Index | `data.yaml` Name | WIREFRAME Canonical Class | UI Display Label | Category |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `Vegetation` | `vegetation` | **Vegetation** | Context / Encroachment |
| `1` | `abandoned-wire` | `dead_wire` | **Abandoned Wire** | Overhead Wire |
| `2` | `power-distribution-lines` | `power_distribution_line` | **Power Line** | Overhead Wire |
| `3` | `telecom-cable` | `communication_cable` | **Communication Wire** | Overhead Wire |

> **Note:** `telecom-cable` / `communication_cable` unifies standard telecommunication lines and fiber-optic drops under a single canonical class, while legacy database rows (`power_line`, `telecom_cable`, `fiber_optic`, `dead_abandoned`) remain 100% backward-compatible.
