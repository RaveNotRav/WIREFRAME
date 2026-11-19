"""v2.17 strand type head: one wire type per physical strand, from the whole strand and its scene.

Given the strands of one photo, the photo and the fused WireNet probabilities, the head returns
P(communication_cable | power_or_communication) per strand from model evidence, geometry, scene position,
appearance and optional DINOv2-small patch features pooled along the strand. Strands whose top pixel
evidence is dead_wire keep their dead_wire label; the head never removes a strand.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np

from backend.utils.cv.centerline import zhang_suen

CANONICAL_CLASSES = ("power_distribution_line", "communication_cable", "dead_wire", "vegetation")
WIRE_IDS = [1, 2, 3]
PT, TC, DW = "power_distribution_line", "communication_cable", "dead_wire"
DINO_ID = "facebook/dinov2-small"
DINO_PX = 1008  # 72 x 72 patches of 14 px
N_DINO = 32     # PCA dimensions kept
REF_SIDE = 1024


def _skel_points(m: np.ndarray) -> np.ndarray:
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
        area = int(s.mask.sum())
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
        m = s.mask > 0
        ring = cv2.dilate(m.astype(np.uint8), ring_k).astype(bool) & ~m
        g_in = float(gray[m].mean()) if m.any() else 0.0
        g_out = float(gray[ring].mean()) if ring.any() else float(gray.mean())
        share = getattr(s, "class_share", None) or {}
        sh = [
            share.get(PT, share.get("power-distribution-lines", 0.0)),
            share.get(TC, share.get("telecom-cable", 0.0)),
            share.get(DW, share.get("abandoned-wire", 0.0)),
        ]
        above = float((ymeans < ym).mean()) if len(ymeans) > 1 else 0.0
        hand.append([
            *sh, sh[1] - sh[0],
            float(p_wire[m].mean()) if m.any() else 0.0,
            float(ent[m].mean()) if m.any() else 0.0,
            float(probs[1][m].mean()) if m.any() else 0.0,
            float(probs[2][m].mean()) if m.any() else 0.0,
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


class StrandHead:
    """Lazy, thread-safe wrapper for optional v2.17 strand classification."""

    def __init__(self, path: Path, min_conf: float = 0.5, device: str | None = None, group_gap: float = 0.0) -> None:
        self.path, self.min_conf, self.device, self.group_gap = Path(path), min_conf, device, group_gap
        self._head = None
        self._lock, self.error = threading.Lock(), None

    def load(self) -> bool:
        if self._head is not None:
            return True
        if not self.path.is_file():
            self.error = f"strand head weights not found at {self.path.name}"
            return False
        try:
            import torch
            meta = json.loads(self.path.with_suffix(".json").read_text())
            self._head = {"meta": meta, "state": torch.load(self.path, map_location="cpu", weights_only=True)}
            return True
        except Exception as exc:
            self.error = f"strand head unavailable: {exc}"
            return False
