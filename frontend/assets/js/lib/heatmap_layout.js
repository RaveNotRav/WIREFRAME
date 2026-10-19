/**
 * heatmap_layout.js - Pure helper to compute adaptive grid dimensions and cell sizes
 * for the risk heatmap tile.
 */

/**
 * Computes an adaptive grid layout for the risk heatmap.
 * Maximizes cell size s in [minCell, maxCell] such that cols * rows >= n fits in W x H with gap.
 * If n exceeds container capacity at minCell, caps visible cells to capacity (most recent first).
 *
 * @param {number} n - Total number of cells to layout
 * @param {number} W - Available container width in pixels
 * @param {number} H - Available container height in pixels
 * @param {number} [gap=8] - Gap between cells in pixels
 * @param {number} [minCell=16] - Minimum cell size in pixels
 * @param {number} [maxCell=64] - Maximum cell size in pixels
 * @returns {{ cols: number, rows: number, cellSize: number, maxVisible: number, capped: boolean }}
 */
export function computeHeatmapLayout(n, W, H, gap = 8, minCell = 16, maxCell = 64) {
    if (!n || n <= 0 || W <= 0 || H <= 0) {
        return {
            cols: 6,
            rows: 1,
            cellSize: minCell,
            maxVisible: 0,
            capped: false
        };
    }

    const maxCols = Math.max(1, Math.floor((W + gap) / (minCell + gap)));
    const maxRows = Math.max(1, Math.floor((H + gap) / (minCell + gap)));
    const capacity = maxCols * maxRows;

    let effectiveN = n;
    let capped = false;
    let maxVisible = n;

    if (n > capacity) {
        effectiveN = capacity;
        capped = true;
        maxVisible = capacity;
    }

    let bestCols = 1;
    let bestRows = effectiveN;
    let bestSize = 0;

    for (let c = 1; c <= Math.min(effectiveN, maxCols); c++) {
        const r = Math.ceil(effectiveN / c);
        if (r > maxRows) continue;

        const availW = W - (c - 1) * gap;
        const availH = H - (r - 1) * gap;
        if (availW <= 0 || availH <= 0) continue;

        const s = Math.min(availW / c, availH / r);
        if (s > bestSize) {
            bestSize = s;
            bestCols = c;
            bestRows = r;
        }
    }

    if (bestSize === 0) {
        bestCols = maxCols;
        bestRows = maxRows;
        bestSize = minCell;
        capped = true;
        maxVisible = capacity;
    }

    const clampedSize = Math.max(minCell, Math.min(maxCell, Math.floor(bestSize)));

    return {
        cols: bestCols,
        rows: bestRows,
        cellSize: clampedSize,
        maxVisible: capped ? maxVisible : n,
        capped
    };
}
