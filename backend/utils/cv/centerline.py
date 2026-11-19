"""Zhang-Suen thinning, branch pruning, and skeleton tracing (Section 3.6.4, Equations 4-5, Algorithm 4).

Complexity:
- N = width * height of bounding box crop
- Zhang-Suen Thinning: O(N * max_iterations) ~ O(N * thickness)
- Skeleton Tracing: O(L) where L is length of skeleton
- Overall: O(N * thickness)
"""

from __future__ import annotations

from collections import deque
import logging

import cv2
import numpy as np

from backend.utils.cv.types import CenterlineResult

logger = logging.getLogger(__name__)

MIN_MASK_COMPONENT_PIXELS = 15
MIN_COMPONENT_PIXELS = MIN_MASK_COMPONENT_PIXELS
BORDER_TRUNCATION_MARGIN_PX = 10


def zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Two-subiteration parallel thinning of Zhang & Suen (1984), Equations 4-5.

    Input: 2-D boolean or uint8 array (foreground != 0).
    Output: 2-D boolean array, 8-connected 1-px skeleton.
    """
    im = np.pad((mask != 0).astype(np.uint8), 1, mode="constant")
    while True:
        changed = False
        for sub in (0, 1):
            p2 = im[:-2, 1:-1]
            p3 = im[:-2, 2:]
            p4 = im[1:-1, 2:]
            p5 = im[2:, 2:]
            p6 = im[2:, 1:-1]
            p7 = im[2:, :-2]
            p8 = im[1:-1, :-2]
            p9 = im[:-2, :-2]
            center = im[1:-1, 1:-1]

            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if sub == 0:
                cond = (center == 1) & (b >= 2) & (b <= 6) & (a == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (center == 1) & (b >= 2) & (b <= 6) & (a == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if np.any(cond):
                center[cond] = 0
                changed = True
        if not changed:
            break
    return im[1:-1, 1:-1].astype(bool)


_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def pixel_graph(skel: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Build the 8-adjacency graph of skeleton pixels (coords as (x, y) = (col, row)),
    dropping diagonal edges that share a 4-neighbour (m-adjacency).
    """
    ys, xs = np.nonzero(skel)
    nodes = {(int(x), int(y)) for x, y in zip(xs, ys, strict=True)}
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {p: set() for p in nodes}
    for x, y in nodes:
        for dx, dy in _OFFSETS:
            nb = (x + dx, y + dy)
            if nb not in nodes:
                continue
            if dx != 0 and dy != 0 and ((x + dx, y) in nodes or (x, y + dy) in nodes):
                continue
            adj[(x, y)].add(nb)
    return adj


def prune_spurs(adj: dict[tuple[int, int], set[tuple[int, int]]], l_min: int = 30) -> None:
    """Algorithm 4 lines 3-9: remove degree-1 branches shorter than l_min pixels in-place."""
    if l_min <= 0 or not adj:
        return
    endpoints = [v for v, nbs in adj.items() if len(nbs) == 1]
    to_remove: list[list[tuple[int, int]]] = []
    for ep in endpoints:
        if ep not in adj or len(adj[ep]) != 1:
            continue
        chain = [ep]
        prev = ep
        cur = next(iter(adj[ep]))
        while len(adj.get(cur, ())) == 2 and len(chain) < l_min:
            chain.append(cur)
            nbs = adj[cur]
            nxt = next(n for n in nbs if n != prev)
            prev, cur = cur, nxt
        # Only prune if it terminates at a junction (deg >= 3); never delete an isolated simple curve
        if len(adj.get(cur, ())) >= 3 and len(chain) < l_min:
            to_remove.append(chain)
    for chain in to_remove:
        for v in chain:
            for nb in list(adj.get(v, ())):
                adj[nb].discard(v)
            adj.pop(v, None)


def _bfs_farthest(
    start: tuple[int, int],
    adj: dict[tuple[int, int], set[tuple[int, int]]],
    allowed: set[tuple[int, int]],
) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    last = start
    while q:
        u = q.popleft()
        last = u
        for v in adj[u]:
            if v in allowed and v not in parent:
                parent[v] = u
                q.append(v)
    return last, parent


def longest_path(adj: dict[tuple[int, int], set[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Return the longest simple path (by pixel count) across all connected components."""
    seen: set[tuple[int, int]] = set()
    best: list[tuple[int, int]] = []
    for node in adj:
        if node in seen or not adj[node]:
            continue
        _, p1 = _bfs_farthest(node, adj, set(adj))
        comp = set(p1)
        seen |= comp
        u, _ = _bfs_farthest(node, adj, comp)
        v, parent = _bfs_farthest(u, adj, comp)
        path = []
        cur: tuple[int, int] | None = v
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        if len(path) > len(best):
            best = path
    return best


def find_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    """Find endpoints of a 1px skeleton."""
    endpoints = []
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D((skel > 0).astype(np.uint8), -1, kernel)
    y_coords, x_coords = np.where(filtered == 11)
    for y, x in zip(y_coords, x_coords):
        endpoints.append((int(x), int(y)))
    return endpoints


def trace_polyline(skel: np.ndarray, start_x: int, start_y: int) -> list[tuple[int, int]]:
    """Trace a 1px skeleton from a starting point."""
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    polyline = []
    curr_x, curr_y = start_x, start_y
    while True:
        polyline.append((curr_x, curr_y))
        visited[curr_y, curr_x] = True
        next_x, next_y = None, None
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = curr_y + dy, curr_x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if skel[ny, nx] > 0 and not visited[ny, nx]:
                        next_x, next_y = nx, ny
                        break
            if next_x is not None:
                break
        if next_x is None:
            break
        curr_x, curr_y = next_x, next_y
    return polyline


def _is_endpoint_near_border(pt: tuple[float, float], width: int, height: int, margin: int = BORDER_TRUNCATION_MARGIN_PX) -> bool:
    x, y = pt
    return x <= margin or y <= margin or x >= (width - 1 - margin) or y >= (height - 1 - margin)


def extract_centerline(
    mask: np.ndarray,
    bbox: list[float] | None = None,
    *,
    l_min: int = 30,
    n_min: int = 20,
    border_margin_px: int = BORDER_TRUNCATION_MARGIN_PX,
) -> CenterlineResult:
    """Extract ordered polyline from a binary mask using Zhang-Suen skeletonization,
    branch pruning, and span-sorted median extraction.

    - Crops to bbox (or mask bounds if bbox is None) to save computation.
    - Removes connected components smaller than MIN_MASK_COMPONENT_PIXELS.
    - Runs Zhang-Suen thinning + pixel_graph + prune_spurs(l_min=30) + longest_path.
    - Extracts 1 point per column (if W >= H) or per row (if H > W) across valid components.
    - Flags centerlines with < n_min points as "too_short_to_fit".
    - Flags centerlines whose endpoints lie within border_margin_px (default 10 px) of the
      image border as "truncated_wire".
    """
    if mask is None or mask.size == 0:
        return CenterlineResult(is_valid=False, reason="empty_mask")

    h, w = mask.shape[:2]
    if bbox is not None and len(bbox) >= 4:
        x1, y1, x2, y2 = map(int, bbox[:4])
    else:
        ys_nz, xs_nz = np.nonzero(mask)
        if len(xs_nz) == 0:
            return CenterlineResult(is_valid=False, reason="empty_mask")
        x1, y1, x2, y2 = int(xs_nz.min()), int(ys_nz.min()), int(xs_nz.max()) + 1, int(ys_nz.max()) + 1

    pad = 5
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    crop = mask[y1:y2, x1:x2].copy()
    crop_bin = (crop > 0).astype(np.uint8)

    if np.sum(crop_bin) == 0:
        return CenterlineResult(is_valid=False, reason="empty_mask")

    # Remove connected components smaller than MIN_MASK_COMPONENT_PIXELS
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crop_bin, connectivity=8)
    filtered = np.zeros_like(crop_bin)
    valid_components = 0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= MIN_MASK_COMPONENT_PIXELS:
            filtered[labels == lbl] = 1
            valid_components += 1

    if cv2.countNonZero(filtered) == 0:
        return CenterlineResult(is_valid=False, reason="empty_mask")

    # Run Zhang-Suen skeletonization + branch pruning (Algorithm 4)
    skel = zhang_suen(filtered)
    skel_px = int(skel.sum())
    adj = pixel_graph(skel)
    prune_spurs(adj, l_min=l_min)
    pruned_px = len(adj)

    crop_h, crop_w = filtered.shape
    polyline_global: list[tuple[float, float]] = []

    # Extract clean 1-point-per-column (when W >= H) or 1-point-per-row (when H > W)
    if crop_w >= crop_h:
        for x_rel in range(crop_w):
            ys = np.where(filtered[:, x_rel] > 0)[0]
            if len(ys) > 0:
                med_y = float(np.median(ys))
                polyline_global.append((float(x_rel + x1), float(med_y + y1)))
    else:
        for y_rel in range(crop_h):
            xs = np.where(filtered[y_rel, :] > 0)[0]
            if len(xs) > 0:
                med_x = float(np.median(xs))
                polyline_global.append((float(med_x + x1), float(y_rel + y1)))

    # Fallback to longest_path on pruned Zhang-Suen graph if needed
    if len(polyline_global) < 2 and adj:
        path = longest_path(adj)
        if len(path) >= 2:
            if path[0][0] > path[-1][0]:
                path.reverse()
            polyline_global = [(float(px + x1), float(py + y1)) for px, py in path]

    if len(polyline_global) < 2:
        return CenterlineResult(
            is_valid=False,
            flags=["too_short_to_fit"],
            skeleton_pixels=skel_px,
            pruned_pixels=pruned_px,
            reason="too_short_to_fit",
        )

    flags: list[str] = []
    if len(polyline_global) < n_min:
        flags.append("too_short_to_fit")

    if (
        _is_endpoint_near_border(polyline_global[0], w, h, border_margin_px)
        or _is_endpoint_near_border(polyline_global[-1], w, h, border_margin_px)
    ):
        flags.append("truncated_wire")

    return CenterlineResult(
        polyline=polyline_global,
        is_valid=True,
        flags=flags,
        skeleton_pixels=skel_px,
        pruned_pixels=pruned_px,
        reason=None,
    )
