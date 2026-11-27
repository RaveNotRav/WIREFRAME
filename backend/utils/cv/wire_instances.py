"""Span instances from a full-resolution wire class map (v2.7 - v2.17).

Ported from Joanna's `wireframes-main/app/pipeline/wire_instances.py` and adapted to
WIREFRAME's canonical 4-class taxonomy:
- `power_distribution_line`
- `communication_cable`
- `dead_wire`
- `vegetation`

Extracts wire spans geometrically from the fused WireNet probability map:
1. Per wire class (or on the union wire mask `P(any wire) >= tau` for `strand_spans`),
   remove connected components smaller than `min_px`.
2. Zhang-Suen skeletonization.
3. Remove junction pixels (>= 3 skeleton neighbours) so the skeleton splits into pieces.
4. Re-join collinear pieces meeting at a junction within `max_turn_deg` via union-find.
5. Assign every mask pixel to its nearest skeleton piece (Euclidean distance transform)
   within its own connected component (`own_component=True`).
6. Bridge collinear gaps up to `max_gap` (with optional `max_lateral_px` and pole occluder support).
7. Classify each physical strand across its observed pixels (and optional v2.17 `StrandHead`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np
from scipy import ndimage as ndi

from backend.utils.cv.centerline import zhang_suen
from backend.utils.cv.types import normalize_wire_class

_NEIGH = np.ones((3, 3), np.uint8)

# WireNet model channel order (1-indexed; 0 = background):
# Matches Joanna's CLASSES = ("Vegetation", "abandoned-wire", "power-distribution-lines", "telecom-cable")
# mapped to canonical WIREFRAME names:
WIRENET_CLASSES: tuple[str, ...] = (
    "vegetation",
    "dead_wire",
    "power_distribution_line",
    "communication_cable",
)
WIRENET_WIRE_IDS: list[int] = [2, 3, 4]  # dead_wire=2, power_distribution_line=3, communication_cable=4


def _skeleton(mask: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(
            mask.astype(np.uint8) * 255,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        ) > 0
    return zhang_suen(mask > 0)


def _label_coords(lab: np.ndarray, n: int) -> list[np.ndarray]:
    """[(row, col) pixels of label i + 1] for i < n, each in row-major order, from one pass."""
    flat = np.flatnonzero(lab)
    vals = lab.ravel()[flat]
    order = np.argsort(vals, kind="stable")
    flat, vals = flat[order], vals[order]
    bounds = np.searchsorted(vals, np.arange(1, n + 2))
    rc = np.stack(np.unravel_index(flat, lab.shape), axis=1)
    return [rc[bounds[i] : bounds[i + 1]] for i in range(n)]


def _end_direction(coords: np.ndarray, end: np.ndarray, k: int = 8) -> np.ndarray:
    """Unit vector pointing *out of* a piece at `end`, from its k nearest pixels."""
    d = np.hypot(*(coords - end).T)
    near = coords[np.argsort(d)[: min(k, len(coords))]]
    v = end - near.mean(axis=0)
    n = np.hypot(*v)
    return v / n if n > 0 else v


def _far_direction(coords: np.ndarray, end: np.ndarray, near: float = 4.0, far: float = 20.0) -> np.ndarray:
    """Outward unit tangent at `end` from piece pixels `near`-`far` px away."""
    d = np.hypot(*(coords - end).T)
    sel = coords[(d >= near) & (d <= far)]
    if len(sel) < 4:
        return _end_direction(coords, end)
    c = sel.mean(axis=0)
    _, _, vt = np.linalg.svd(sel - c, full_matrices=False)
    v = vt[0]
    if np.dot(end - c, v) < 0:
        v = -v
    return v / (np.hypot(*v) or 1.0)


def _perfect_matchings(items: list[int]):
    if len(items) < 2:
        yield []
        return
    a, rest = items[0], items[1:]
    for k, b in enumerate(rest):
        for m in _perfect_matchings(rest[:k] + rest[k + 1 :]):
            yield [(a, b)] + m


def _xjunction_pairs(members, tol_deg: float, min_cross_deg: float):
    """Best perfect matching of piece ends at one crossing (4 or 6 distinct pieces), or None."""
    ids = {}
    for i, d in members:
        ids.setdefault(i, d)
    if len(ids) not in (4, 6):
        return None
    keys = list(ids)
    cos_tol = np.cos(np.deg2rad(180 - tol_deg))
    best, best_cost = None, None
    for m in _perfect_matchings(list(range(len(keys)))):
        dots = [float(np.dot(ids[keys[a]], ids[keys[b]])) for a, b in m]
        if max(dots) > cos_tol:
            continue
        cost = sum(1.0 + x for x in dots)
        if best_cost is None or cost < best_cost:
            best, best_cost = m, cost
    if best is None:
        return None
    lines = [ids[keys[a]] - ids[keys[b]] for a, b in best]
    for x in range(len(lines)):
        for y in range(x + 1, len(lines)):
            u, v = lines[x], lines[y]
            c = abs(float(np.dot(u, v))) / ((np.hypot(*u) * np.hypot(*v)) or 1.0)
            if np.degrees(np.arccos(np.clip(c, 0, 1))) <= min_cross_deg:
                return None
    return [(keys[a], keys[b]) for a, b in best]


def spans_from_class_mask(
    mask: np.ndarray,
    prob: np.ndarray,
    min_px: int = 30,
    min_len: int = 15,
    max_turn_deg: float = 35.0,
    junction_radius: int = 3,
    cluster_radius: int = 5,
    xjunction_tol_deg: float = 0.0,
    xjunction_min_cross_deg: float = 30.0,
    xjunction_far_tangent: bool = True,
    own_component: bool = True,
) -> list[tuple[np.ndarray, float]]:
    """Return [(span mask, confidence)] for one wire mask."""
    lab, n = ndi.label(mask, structure=_NEIGH)
    if n == 0:
        return []
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    mask = np.isin(lab, np.flatnonzero(sizes >= min_px) + 1)
    if not mask.any():
        return []
    skel = _skeleton(mask)
    nb = ndi.convolve(skel.astype(np.uint8), _NEIGH, mode="constant") - 1
    junction = skel & (nb >= 3)
    jzone = ndi.binary_dilation(junction, iterations=junction_radius)
    pieces = skel & ~jzone
    plab, npieces = ndi.label(pieces, structure=_NEIGH)
    if npieces == 0:
        return []
    coords = _label_coords(plab, npieces)
    pnb = ndi.convolve((plab > 0).astype(np.uint8), _NEIGH, mode="constant") - 1
    parent = list(range(npieces))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cluster = ndi.binary_dilation(junction, iterations=cluster_radius)
    jlab, _ = ndi.label(cluster, structure=_NEIGH)
    touch: dict[int, list[tuple[int, np.ndarray]]] = {}
    grown = ndi.binary_dilation(cluster, iterations=1)
    bridges = set()
    for i, c in enumerate(coords):
        if cluster[tuple(c.T)].all():
            bridges.add(i)
            continue
        ends = [p for p in c if pnb[tuple(p)] <= 1] or [c[0]]
        for e in ends:
            if grown[tuple(e)]:
                y, x = e
                win = jlab[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
                ids = np.unique(win[win > 0])
                for j in ids:
                    touch.setdefault(int(j), []).append(
                        (
                            i,
                            _far_direction(c, e)
                            if xjunction_tol_deg > 0 and xjunction_far_tangent
                            else _end_direction(c, e),
                        )
                    )
    cos_max = np.cos(np.deg2rad(180 - max_turn_deg))
    for members in touch.values():
        if xjunction_tol_deg > 0:
            xp = _xjunction_pairs(members, xjunction_tol_deg, xjunction_min_cross_deg)
            if xp is not None:
                for ia, ib in xp:
                    if find(ia) != find(ib):
                        parent[find(ia)] = find(ib)
                continue
        pairs = sorted(
            (
                (float(np.dot(da, db)), ia, ib)
                for x, (ia, da) in enumerate(members)
                for (ib, db) in members[x + 1 :]
                if ia != ib
            ),
            key=lambda t: t[0],
        )
        used: set[int] = set()
        for c, ia, ib in pairs:
            if c <= cos_max and ia not in used and ib not in used:
                parent[find(ia)] = find(ib)
                used.update((ia, ib))
    groups: dict[int, list[int]] = {}
    for i in range(npieces):
        if i not in bridges:
            groups.setdefault(find(i), []).append(i)
    if not groups:
        return []

    seeds = plab.copy()
    if bridges:
        seeds[np.isin(seeds, np.fromiter((i + 1 for i in bridges), np.int32))] = 0
    _, (iy, ix) = ndi.distance_transform_edt(seeds == 0, return_indices=True)
    owner = seeds[iy, ix]
    owner[~mask] = 0
    if own_component:
        clab, _ = ndi.label(mask, structure=_NEIGH)
        owner[clab != clab[iy, ix]] = 0
    remap = np.zeros(npieces + 1, np.int32)
    for g, members in enumerate(groups.values(), start=1):
        remap[np.array(members) + 1] = g
    span_lab = remap[owner]
    boxes = ndi.find_objects(span_lab)
    out = []
    for g, members in enumerate(groups.values(), start=1):
        length = sum(len(coords[i]) for i in members)
        if length < min_len or g > len(boxes) or boxes[g - 1] is None:
            continue
        sl = boxes[g - 1]
        sub = span_lab[sl] == g
        if sub.sum() < min_px:
            continue
        m = np.zeros(mask.shape, bool)
        m[sl] = sub
        out.append((m, float(prob[sl][sub].mean())))
    return out


def _crop_skeleton(mask: np.ndarray) -> np.ndarray:
    """Skeleton pixel coordinates (row, col) in image space, computed on the bounding-box crop."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return np.zeros((0, 2))
    y0, x0 = max(rows[0] - 1, 0), max(cols[0] - 1, 0)
    crop = mask[y0 : rows[-1] + 2, x0 : cols[-1] + 2]
    return np.argwhere(_skeleton(crop)) + (y0, x0)


def skeleton_length(mask: np.ndarray) -> int:
    return len(_crop_skeleton(mask))


def span_ends(mask: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Endpoints and outward unit tangents of a span."""
    sk = _crop_skeleton(mask).astype(np.float64)
    if len(sk) < 2:
        sk = np.argwhere(mask).astype(np.float64)
    c = sk.mean(axis=0)
    _, _, vt = np.linalg.svd(sk - c, full_matrices=False)
    t = (sk - c) @ vt[0]
    order = np.argsort(t)
    sk = sk[order]
    a, b = sk[0], sk[-1]
    ta = a - sk[: min(k, len(sk))].mean(axis=0)
    tb = b - sk[-min(k, len(sk)) :].mean(axis=0)
    ta = ta / (np.hypot(*ta) or 1.0)
    tb = tb / (np.hypot(*tb) or 1.0)
    return a, ta, b, tb, float(len(sk))


def _stitch_candidates(ends, max_gap: float, cos_lim: float) -> list:
    n = len(ends)
    if n < 2:
        return []
    pts = np.array([[e[0], e[2]] for e in ends], np.float64).reshape(2 * n, 2)
    tan = np.array([[e[1], e[3]] for e in ends], np.float64).reshape(2 * n, 2)
    owner = np.repeat(np.arange(n), 2)
    a, b = np.nonzero(owner[:, None] < owner[None, :])
    gap = pts[b] - pts[a]
    d = np.hypot(gap[:, 0], gap[:, 1])
    ok = d <= max_gap
    tt = tan[a, 0] * tan[b, 0] + tan[a, 1] * tan[b, 1]
    ok &= ~(-tt < cos_lim)
    long_ = ok & (d > 1.0)
    u = np.zeros_like(gap)
    u[long_] = gap[long_] / d[long_, None]
    da = tan[a, 0] * u[:, 0] + tan[a, 1] * u[:, 1]
    db = tan[b, 0] * -u[:, 0] + tan[b, 1] * -u[:, 1]
    ok &= ~(long_ & ((da < cos_lim) | (db < cos_lim)))
    sel = np.flatnonzero(ok)
    sel = sel[np.lexsort((b[sel] % 2, a[sel] % 2, owner[b[sel]], owner[a[sel]]))]
    return [(float(d[k]), int(owner[a[k]]), int(owner[b[k]]), pts[a[k]], pts[b[k]]) for k in sel]


def bridge_support(prob: np.ndarray, ea: np.ndarray, eb: np.ndarray) -> float:
    n = int(np.ceil(np.hypot(*(eb - ea))))
    if n < 2:
        return 1.0
    t = np.linspace(0.0, 1.0, n + 1)[1:-1]
    pts = np.rint(ea[None] + t[:, None] * (eb - ea)[None]).astype(int)
    pts[:, 0] = pts[:, 0].clip(0, prob.shape[0] - 1)
    pts[:, 1] = pts[:, 1].clip(0, prob.shape[1] - 1)
    return float(prob[pts[:, 0], pts[:, 1]].mean())


def lateral_offset(end_i, end_j, ea: np.ndarray, eb: np.ndarray) -> float:
    def tangent(e, p):
        return e[1] if np.allclose(e[0], p) else e[3]

    d = eb - ea
    ta, tb = tangent(end_i, ea), tangent(end_j, eb)
    return float(max(abs(ta[0] * d[1] - ta[1] * d[0]), abs(tb[0] * d[1] - tb[1] * d[0])))


def _gap_crosses(ea: np.ndarray, eb: np.ndarray, boxes) -> bool:
    t = np.linspace(0.0, 1.0, 33)
    pts = ea[None] + t[:, None] * (eb - ea)[None]
    ys, xs = pts[:, 0], pts[:, 1]
    return any(bool(((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)).any()) for x0, y0, x1, y1 in boxes)


def stitch_spans(
    spans: list[tuple[np.ndarray, float]],
    max_gap: float = 120.0,
    max_angle_deg: float = 35.0,
    draw_bridge: bool = True,
    bridge_width: int = 2,
    prob: np.ndarray | None = None,
    bridge_min_prob: float = 0.0,
    bridges_out: list | None = None,
    max_lateral_px: float = 20.0,
    occluders: list | None = None,
    occluder_max_gap: float = 0.0,
) -> list[tuple[np.ndarray, float]]:
    """Join spans whose facing endpoints are collinear across small gaps or occluders."""
    if len(spans) < 2:
        return spans
    ends = [span_ends(m) for m, _ in spans]
    cos_lim = np.cos(np.deg2rad(max_angle_deg))
    long_gap = occluder_max_gap if occluders and occluder_max_gap > max_gap else max_gap
    cands = _stitch_candidates(ends, long_gap, cos_lim)
    if long_gap > max_gap:
        cands = [c for c in cands if c[0] <= max_gap or _gap_crosses(c[3], c[4], occluders)]
    parent = list(range(len(spans)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bridges: dict[int, list] = {}
    used_ends: set = set()
    for _d, i, j, ea, eb in sorted(cands, key=lambda c: c[0]):
        ka, kb = (i, tuple(ea)), (j, tuple(eb))
        if ka in used_ends or kb in used_ends or find(i) == find(j):
            continue
        if bridge_min_prob > 0 and prob is not None and bridge_support(prob, ea, eb) < bridge_min_prob:
            continue
        if max_lateral_px > 0 and lateral_offset(ends[i], ends[j], ea, eb) > max_lateral_px:
            continue
        used_ends.update((ka, kb))
        parent[find(i)] = find(j)
        bridges.setdefault(j, []).append((ea, eb))
        if bridges_out is not None:
            bridges_out.append(((float(ea[1]), float(ea[0])), (float(eb[1]), float(eb[0]))))
    groups: dict[int, list[int]] = {}
    for i in range(len(spans)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        m = np.zeros_like(spans[members[0]][0])
        area = conf = 0.0
        for i in members:
            m |= spans[i][0]
            a = float(spans[i][0].sum())
            area += a
            conf += a * spans[i][1]
        if draw_bridge:
            canvas = m.astype(np.uint8)
            for i in members:
                for ea, eb in bridges.get(i, []):
                    cv2.line(canvas, (int(ea[1]), int(ea[0])), (int(eb[1]), int(eb[0])), 1, bridge_width)
            m = canvas.astype(bool)
        out.append((m, conf / max(area, 1.0)))
    return out


def extract_spans(
    class_mask: np.ndarray,
    prob: np.ndarray,
    *,
    min_px: int = 20,
    min_len: int = 12,
    max_turn_deg: float = 35.0,
    cluster_radius: int = 5,
    max_gap: float = 120.0,
    gap_angle_deg: float = 35.0,
    draw_bridge: bool = True,
    min_span_len: int = 55,
    xjunction_tol_deg: float = 0.0,
    xjunction_min_cross_deg: float = 30.0,
    xjunction_far_tangent: bool = True,
    bridge_min_prob: float = 0.0,
    bridges_out: list | None = None,
    max_lateral_px: float = 20.0,
    own_component: bool = True,
    occluders: list | None = None,
    occluder_max_gap: float = 0.0,
) -> list[tuple[np.ndarray, float]]:
    """v2.8-v2.17 span extraction: junction-split spans, collinear gap bridging, and stitched length gate."""
    spans = spans_from_class_mask(
        class_mask,
        prob,
        min_px=min_px,
        min_len=min_len,
        max_turn_deg=max_turn_deg,
        cluster_radius=cluster_radius,
        xjunction_tol_deg=xjunction_tol_deg,
        xjunction_min_cross_deg=xjunction_min_cross_deg,
        xjunction_far_tangent=xjunction_far_tangent,
        own_component=own_component,
    )
    if max_gap > 0 or (occluders and occluder_max_gap > 0):
        spans = stitch_spans(
            spans,
            max_gap,
            gap_angle_deg,
            draw_bridge,
            prob=prob,
            bridge_min_prob=bridge_min_prob,
            bridges_out=bridges_out,
            max_lateral_px=max_lateral_px,
            occluders=occluders,
            occluder_max_gap=occluder_max_gap,
        )
    if min_span_len > 0:
        spans = [(m, c) for m, c in spans if skeleton_length(m) >= min_span_len]
    return spans


@dataclass
class Strand:
    """One physical wire strand: geometry built once on the class-agnostic wire mask, one class for all of it."""

    wire_class: str
    mask: np.ndarray
    confidence: float
    class_share: dict[str, float] = field(default_factory=dict)
    class_status: str = "confident"


def strand_spans(
    probs: np.ndarray,
    wire_ids: list[int],
    params: dict,
    *,
    tau: float | None = 0.22,
    min_class_share: float = 0.9,
    bridges_out: list | None = None,
    class_names: tuple[str, ...] = WIRENET_CLASSES,
    mask: np.ndarray | None = None,
) -> list[Strand]:
    """v2.16-v2.17: extract spans from the union of wire classes (`P(wire) >= tau`), then assign one class per strand."""
    p_wire = probs[wire_ids].sum(0)
    if mask is not None:
        mask = mask.astype(bool)
    elif tau is None:
        mask = np.isin(probs.argmax(0), wire_ids)
    else:
        mask = p_wire >= tau
    if not mask.any():
        return []
    spans = extract_spans(mask, p_wire, bridges_out=bridges_out, **params)
    out: list[Strand] = []
    for m, conf in spans:
        obs = m & mask
        sel = obs if obs.any() else m
        mass = probs[wire_ids][:, sel].sum(1).astype(np.float64)
        tot = float(mass.sum()) or 1.0
        share = {
            normalize_wire_class(class_names[c - 1]): round(float(v) / tot, 4)
            for c, v in zip(wire_ids, mass, strict=True)
        }
        k = int(np.argmax(mass))
        top_cls = normalize_wire_class(class_names[wire_ids[k] - 1])
        status = "uncertain" if mass[k] / tot < min_class_share else "confident"
        out.append(Strand(top_cls, m, conf, share, status))
    return out
