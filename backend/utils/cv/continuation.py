"""S16 (register D-34): trace one wire through crossings by good continuation, not by the longest skeleton path.

``centerline.longest_path`` returns the graph diameter of the pruned skeleton. Where two wires cross inside one mask,
the diameter can run along one wire to the crossing and leave along the other; the catenary then reads the turn as sag.

A physical wire keeps its direction through a crossing. Here the skeleton is reduced to junctions and the pixel chains
between them; at every junction the chains are paired greedily by collinearity (the straightest pair first, each chain
used once), and a path may pass a junction only between paired chains. With at most one continuation per chain end,
every path is a chain of chains; the longest one is the centreline.
"""

from __future__ import annotations

import numpy as np

Pixel = tuple[int, int]  # (x, y) or (row, col), matching backend/utils/cv/centerline.py pixel_graph

DIR_PX = 10  # pixels along a chain used for its direction at a junction


def _junction_clusters(adj: dict[Pixel, set[Pixel]]) -> dict[Pixel, int]:
    """Junction pixels (degree >= 3), 8-connected ones merged into one cluster id."""
    junc = {p for p, nb in adj.items() if len(nb) >= 3}
    cid: dict[Pixel, int] = {}
    k = 0
    for p in junc:
        if p in cid:
            continue
        stack = [p]
        cid[p] = k
        while stack:
            q = stack.pop()
            for r in adj[q]:
                if r in junc and r not in cid:
                    cid[r] = k
                    stack.append(r)
        k += 1
    return cid


def _chains(adj: dict[Pixel, set[Pixel]], cid: dict[Pixel, int]) -> list[dict]:
    """Pixel chains between junction clusters / endpoints. Each: pixels, and for each end the cluster id or None."""
    seen_edge: set[tuple[Pixel, Pixel]] = set()
    chains = []
    starts = [p for p in adj if p in cid or len(adj[p]) == 1]
    for s in starts:
        for nxt in adj[s]:
            if nxt in cid and s in cid:
                continue  # inside a cluster
            if (s, nxt) in seen_edge:
                continue
            path, prev, cur = [s, nxt], s, nxt
            while cur not in cid and len(adj[cur]) == 2:
                a, b = tuple(adj[cur])
                prev, cur = cur, (b if a == prev else a)
                path.append(cur)
            seen_edge.add((path[0], path[1]))
            seen_edge.add((path[-1], path[-2]))
            chains.append({"px": path, "ends": (cid.get(path[0]), cid.get(path[-1]))})
    return chains


def _dir_into(px: list[Pixel], at_start: bool) -> np.ndarray:
    """Unit vector along the chain pointing into the junction at the given end."""
    seg = px[: DIR_PX + 1] if at_start else px[-(DIR_PX + 1):][::-1]
    a, b = np.asarray(seg[-1], float), np.asarray(seg[0], float)
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n else v


def continuation_path(adj: dict[Pixel, set[Pixel]]) -> list[Pixel] | None:
    """The longest path that passes every junction straight through. None when the skeleton has no junction."""
    cid = _junction_clusters(adj)
    if not cid:
        return None
    chains = _chains(adj, cid)
    if not chains:
        return None
    at: dict[int, list[tuple[int, int]]] = {}
    for i, c in enumerate(chains):
        for e, k in enumerate(c["ends"]):
            if k is not None:
                at.setdefault(k, []).append((i, e))
    link: dict[tuple[int, int], tuple[int, int]] = {}
    for _, br in at.items():
        dirs = {b: _dir_into(chains[b[0]]["px"], b[1] == 0) for b in br}
        pairs = sorted(
            ((float(1 + dirs[a] @ dirs[b]), a, b) for i, a in enumerate(br) for b in br[i + 1:] if a[0] != b[0]),
            key=lambda t: t[0],
        )
        used: set[tuple[int, int]] = set()
        for cost, a, b in pairs:
            if a in used or b in used or cost >= 1.0:
                continue
            link[a], link[b] = b, a
            used |= {a, b}
    length = [len(c["px"]) for c in chains]
    best: list[tuple[int, bool]] = []
    best_len = -1
    for i in range(len(chains)):
        for start_end in (0, 1):
            if (i, start_end) in link:
                continue
            seq, total, cur, entered, visited = [], 0, i, start_end, set()
            while cur not in visited:
                visited.add(cur)
                seq.append((cur, entered == 0))
                total += length[cur]
                nxt = link.get((cur, 1 - entered))
                if nxt is None:
                    break
                cur, entered = nxt
            if total > best_len:
                best, best_len = seq, total
    if not best:
        return None
    out: list[Pixel] = []
    for i, fwd in best:
        px = chains[i]["px"] if fwd else chains[i]["px"][::-1]
        out.extend(px if not out or out[-1] != px[0] else px[1:])
    return out
