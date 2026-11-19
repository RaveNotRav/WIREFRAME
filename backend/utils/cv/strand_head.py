"""v2.17 strand type head: one wire type per physical strand, from the whole strand and its scene.

Ported from Joanna's `wireframes-main/app/pipeline/strand_head.py`.
Given the strands of one photo, the photo and the fused WireNet probabilities, the head returns
P(communication_cable | power_or_communication) per strand from model evidence, geometry, scene position,
appearance and frozen DINOv2-small patch features (`backend/models/hf/dinov2-small/`) pooled along the strand.
Strands whose top pixel evidence is `dead_wire` (`abandoned-wire`) keep their `dead_wire` label;
the head never removes a strand.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from backend.utils.cv.centerline import zhang_suen

logger = logging.getLogger(__name__)

# Matches Joanna's CLASSES = ("Vegetation", "abandoned-wire", "power-distribution-lines", "telecom-cable")
# 1-indexed in WireNet probability tensor (0 = background):
# 1 = vegetation, 2 = dead_wire, 3 = power_distribution_line, 4 = communication_cable
WIRE_IDS = [2, 3, 4]
IDX_DEAD = 2
IDX_POWER = 3
IDX_TELECOM = 4

PT, TC, DW = "power_distribution_line", "communication_cable", "dead_wire"
DINO_ID = "facebook/dinov2-small"
DINO_PX = 1008  # 72 x 72 patches of 14 px
N_DINO = 32     # PCA dimensions kept
REF_SIDE = 1024
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def hf_source(local_name: str, hub_id: str, models_dir: Path | None = None) -> str:
    """Resolve local Hugging Face directory in `backend/models/hf/<local_name>` or fall back to hub_id."""
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    local_dir = mdir / "hf" / local_name
    if (local_dir / "config.json").is_file() and (local_dir / "model.safetensors").is_file():
        return str(local_dir)
    return hub_id


class Dino:
    """Frozen DINOv2-small patch feature extractor loaded from `backend/models/hf/dinov2-small`."""

    def __init__(self, device: str | None = None, models_dir: Path | None = None):
        import torch
        from transformers import AutoModel

        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        src = hf_source("dinov2-small", DINO_ID, models_dir)
        self.model = AutoModel.from_pretrained(src).to(self.dev).eval()

    def grid(self, rgb: np.ndarray) -> np.ndarray:
        """(72, 72, 384) patch features of the photo resized to 1008 px."""
        import torch

        x = cv2.resize(rgb, (DINO_PX, DINO_PX), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(self.dev)
        with torch.no_grad():
            h = self.model(pixel_values=t).last_hidden_state[0, 1:]  # drop CLS
        g = DINO_PX // 14
        out = h.reshape(g, g, -1).float().cpu().numpy()
        return out


def _skel_points(m: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "ximgproc"):
        s = cv2.ximgproc.thinning((m > 0).astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    else:
        s = zhang_suen(m > 0)
    ys, xs = np.nonzero(s)
    return np.stack([xs, ys], 1).astype(np.float32) if len(xs) else np.zeros((0, 2), np.float32)


def strand_features(
    strands: list,
    probs: np.ndarray,
    rgb: np.ndarray,
    grid: np.ndarray | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (hand features (n, F), pooled DINO features (n, 384)) for the strands of one image."""
    H, W = probs.shape[1:]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    p_wire = probs[WIRE_IDS].sum(0)
    ent = -(probs * np.log(np.clip(probs, 1e-6, 1))).sum(0)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    pts = [_skel_points(s.mask) for s in strands]
    ymeans = np.array([p[:, 1].mean() if len(p) else H for p in pts])
    hand, dino = [], []
    for s, p, ym in zip(strands, pts, ymeans, strict=True):
        n = max(len(p), 1)
        m = s.mask > 0
        area = int(m.sum())
        if len(p) >= 2:
            c = p.mean(0)
            _, _, vt = np.linalg.svd(p - c, full_matrices=False)
            d = vt[0]
            proj = (p - c) @ d
            chord = float(proj.max() - proj.min()) + 1
            perp = (p - c) @ np.array([-d[1], d[0]])
            sag = float(perp.max() - perp.min()) / chord
            ang = abs(float(np.arctan2(d[1], d[0])))
            ang = min(ang, np.pi - ang)
        else:
            chord, sag, ang = 1.0, 0.0, 0.0
        ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & ~m
        g_in = float(gray[m].mean()) if m.any() else 0.0
        g_out = float(gray[ring].mean()) if ring.any() else float(gray.mean())
        share = getattr(s, "class_share", None) or {}
        sh = [
            float(share.get(PT, share.get("power-distribution-lines", 0.0))),
            float(share.get(TC, share.get("telecom-cable", 0.0))),
            float(share.get(DW, share.get("abandoned-wire", 0.0))),
        ]
        above = float((ymeans < ym).mean()) if len(ymeans) > 1 else 0.0
        idx_pt = IDX_POWER if probs.shape[0] > IDX_POWER else 1
        idx_tc = IDX_TELECOM if probs.shape[0] > IDX_TELECOM else 2
        hand.append([
            *sh, sh[1] - sh[0],
            float(p_wire[m].mean()) if m.any() else 0.0,
            float(ent[m].mean()) if m.any() else 0.0,
            float(probs[idx_pt][m].mean()) if m.any() else 0.0,
            float(probs[idx_tc][m].mean()) if m.any() else 0.0,
            np.log1p(n * scale), n / chord, sag, ang, area / n * scale,
            ym / H, (float(p[:, 1].min()) / H) if len(p) else 1.0, above, len(strands) / 20,
            (g_out - g_in) / 255, g_in / 255, g_out / 255,
            float(hsv[..., 1][m].mean()) / 255 if m.any() else 0.0,
            float(hsv[..., 0][m].mean()) / 180 if m.any() else 0.0,
            float(lap[ring].mean()) / 255 if ring.any() else 0.0,
            float(rgb[..., 2][m].mean() - rgb[..., 0][m].mean()) / 255 if m.any() else 0.0,
        ])
        if grid is not None and len(p):
            g = grid.shape[0]
            ix = np.clip((p[:, 0] / W * g).astype(int), 0, g - 1)
            iy = np.clip((p[:, 1] / H * g).astype(int), 0, g - 1)
            cells = np.unique(iy * g + ix)
            dino.append(grid.reshape(g * g, -1)[cells].mean(0))
        else:
            dino.append(np.zeros(grid.shape[-1] if grid is not None else 384, np.float32))
    return np.asarray(hand, np.float32), np.asarray(dino, np.float32)


class Head:
    """Standardise -> [hand, PCA(dino)] -> 2-layer MLP (power vs telecom)."""

    def __init__(self, stats: dict, state: dict | None = None, bounds: dict | None = None):
        import torch

        self.stats = stats
        self.bounds = (np.asarray(bounds["lo"]), np.asarray(bounds["hi"])) if bounds else None
        d = len(stats["mu"]) + N_DINO
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1),
        )
        if state:
            self.net.load_state_dict(state)
        self.net.eval()

    def x(self, hand: np.ndarray, dino: np.ndarray) -> np.ndarray:
        s = self.stats
        if self.bounds is not None:
            hand = np.clip(hand, *self.bounds)
        h = (hand - np.asarray(s["mu"])) / np.asarray(s["sd"])
        dz = (dino - np.asarray(s["dmu"])) @ np.asarray(s["pca"]).T / np.asarray(s["psd"])
        return np.concatenate([h, dz], 1).astype(np.float32)

    def p_telecom(self, hand: np.ndarray, dino: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return torch.sigmoid(self.net(torch.from_numpy(self.x(hand, dino)))).numpy()[:, 0]


def load_head(path: Path, clamp: bool = False) -> Head:
    import torch

    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    bounds = None
    if clamp:
        bp = path.with_suffix(".bounds.json")
        if bp.is_file():
            bounds = json.loads(bp.read_text(encoding="utf-8"))
            if len(bounds["lo"]) != len(meta["stats"]["mu"]):
                raise ValueError(f"{bp.name} does not match {path.name}")
    return Head(meta["stats"], torch.load(path, map_location="cpu", weights_only=True), bounds)


def _ends(m: np.ndarray, k: int = 25) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """The two ends of a strand: (end point, outward unit direction from a fit to the last ``k`` skeleton px)."""
    p = _skel_points(m)
    if len(p) < 4:
        return None
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    t = (p - c) @ vt[0]
    order = p[np.argsort(t)]
    out = []
    for seg, sign in ((order[:k], -1), (order[-k:], 1)):
        sc = seg.mean(0)
        _, _, v = np.linalg.svd(seg - sc, full_matrices=False)
        d = v[0] * (1 if v[0] @ vt[0] * sign >= 0 else -1)
        out.append((order[0] if sign < 0 else order[-1], d / (np.linalg.norm(d) + 1e-9)))
    return out


def continuation_groups(strands: list, max_gap: float, max_deg: float = 20.0) -> list[int]:
    """Union-find over strands whose ends meet head-on."""
    parent = list(range(len(strands)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ends = [_ends(st.mask) for st in strands]
    cos = np.cos(np.radians(max_deg))
    for i in range(len(strands)):
        for j in range(i + 1, len(strands)):
            if ends[i] is None or ends[j] is None:
                continue
            for pi, di in ends[i]:
                for pj, dj in ends[j]:
                    v = pj - pi
                    dist = float(np.linalg.norm(v))
                    if dist > max_gap:
                        continue
                    if dist < 3:
                        ok = float(di @ -dj) >= cos
                    else:
                        u = v / dist
                        ok = float(di @ u) >= cos and float(dj @ -u) >= cos
                    if ok:
                        parent[find(i)] = find(j)
    return [find(i) for i in range(len(strands))]


def group_probabilities(p: np.ndarray, lengths: np.ndarray, groups: list[int]) -> np.ndarray:
    """Length-weighted mean log-odds per continuation group."""
    lo = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
    g = np.asarray(groups)
    out = p.astype(np.float64).copy()
    for k in set(groups):
        sel = g == k
        if sel.sum() > 1:
            out[sel] = 1 / (1 + np.exp(-np.average(lo[sel], weights=lengths[sel])))
    return out


class _Boxed:
    __slots__ = ("mask", "class_share")

    def __init__(self, mask: np.ndarray, class_share: dict | None):
        self.mask, self.class_share = mask, class_share


def to_letterbox(mask: np.ndarray, lb) -> np.ndarray:
    """Map an original-size strand mask into the size x size letterbox frame."""
    down = lb.new_w < lb.orig_w
    m = cv2.resize(
        mask.astype(np.float32),
        (lb.new_w, lb.new_h),
        interpolation=cv2.INTER_AREA if down else cv2.INTER_LINEAR,
    )
    out = np.zeros((lb.size, lb.size), bool)
    out[lb.pad_y : lb.pad_y + lb.new_h, lb.pad_x : lb.pad_x + lb.new_w] = m >= (0.25 if down else 0.5)
    if not out.any() and mask.any():
        out[lb.pad_y : lb.pad_y + lb.new_h, lb.pad_x : lb.pad_x + lb.new_w] = cv2.resize(
            mask.astype(np.uint8),
            (lb.new_w, lb.new_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return out


class StrandHead:
    """Lazy, thread-safe wrapper for v2.17 strand classification (`strand_head_v217.pt` + DINOv2-small)."""

    def __init__(
        self,
        path: Path,
        min_conf: float = 0.5,
        device: str | None = None,
        group_gap: float = 0.0,
        models_dir: Path | None = None,
        clamp: bool = True,
        letterbox_frame: bool = True,
    ) -> None:
        self.path = Path(path)
        self.min_conf = min_conf
        self.device = device
        self.group_gap = group_gap
        self.models_dir = Path(models_dir) if models_dir is not None else self.path.parent
        self.clamp = clamp
        self.letterbox_frame = letterbox_frame
        self._head: Head | None = None
        self._dino: Dino | None = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def load(self) -> bool:
        if self._head is not None:
            return True
        if not self.path.is_file():
            self.error = f"strand head weights not found at {self.path.name}"
            return False
        try:
            self._head = load_head(self.path, clamp=self.clamp)
            try:
                self._dino = Dino(self.device, self.models_dir)
            except Exception as dino_exc:
                logger.warning("DINOv2-small unavailable (%s); using zero DINO features for strand head.", dino_exc)
                self._dino = None
            return True
        except Exception as exc:
            self.error = f"strand head unavailable: {exc}"
            return False

    def classify(self, strands: list, probs: np.ndarray, rgb: np.ndarray, frame: tuple | None = None) -> bool:
        """Mutate each strand's `wire_class` / `class_status` (and record `head_p_telecom` in `class_share`)."""
        if not strands or not self.load():
            return False
        with self._lock:
            if frame is not None:
                lp, lx, lb = frame
                boxed = [_Boxed(to_letterbox(st.mask, lb), st.class_share) for st in strands]
                grid = self._dino.grid(lx) if self._dino is not None else None
                hand, dn = strand_features(boxed, lp, lx, grid, scale=REF_SIDE / lb.size)
            else:
                grid = self._dino.grid(rgb) if self._dino is not None else None
                hand, dn = strand_features(strands, probs, rgb, grid, scale=REF_SIDE / max(rgb.shape[:2]))
            p = self._head.p_telecom(hand, dn)
        if self.group_gap > 0 and len(strands) > 1:
            keep = [
                i
                for i, st in enumerate(strands)
                if not (
                    st.class_share
                    and max(st.class_share, key=st.class_share.get) in (DW, "abandoned-wire")
                )
            ]
            if len(keep) > 1:
                sub = [strands[i] for i in keep]
                groups = continuation_groups(sub, self.group_gap / (REF_SIDE / max(rgb.shape[:2])))
                lengths = np.array([max(int(len(_skel_points(st.mask))), 1) for st in sub], float)
                p = p.astype(np.float64)
                p[keep] = group_probabilities(p[keep], lengths, groups)
        for s, pt in zip(strands, p, strict=True):
            share = s.class_share or {}
            if share and max(share, key=share.get) in (DW, "abandoned-wire"):
                continue
            s.wire_class = TC if pt >= 0.5 else PT
            s.class_status = "confident" if max(pt, 1 - pt) >= self.min_conf else "uncertain"
            s.class_share = {**share, "head_p_telecom": round(float(pt), 4)}
        return True
