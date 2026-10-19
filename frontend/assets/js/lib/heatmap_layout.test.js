import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeHeatmapLayout } from './heatmap_layout.js';

describe('Heatmap Adaptive Layout Helper Tests', () => {
    const W = 360;
    const H = 200;
    const gap = 8;

    it('n = 1 clamps to maxCell size s = 64', () => {
        const layout = computeHeatmapLayout(1, W, H, gap);
        assert.equal(layout.cellSize, 64, 'Single item should clamp to maxCell 64px');
        assert.equal(layout.cols, 1, 'Single item should use 1 column');
        assert.equal(layout.rows, 1, 'Single item should use 1 row');
        assert.equal(layout.capped, false, 'n=1 is not capped');
        assert.equal(layout.maxVisible, 1);
    });

    it('n = 6 in 360x200 yields cols = 3 and s = 64', () => {
        const layout = computeHeatmapLayout(6, W, H, gap);
        assert.equal(layout.cols, 3, '6 items in 360x200 should pick cols = 3 (3x2 grid)');
        assert.equal(layout.rows, 2, '6 items with 3 cols should have 2 rows');
        assert.equal(layout.cellSize, 64, 'Cell size should clamp to 64px');
        assert.equal(layout.capped, false, 'n=6 is not capped');
        assert.equal(layout.maxVisible, 6);
    });

    it('n = 60 yields 16 <= s <= 64 fitting within W x H', () => {
        const layout = computeHeatmapLayout(60, W, H, gap);
        assert.ok(
            layout.cellSize >= 16 && layout.cellSize <= 64,
            `cellSize ${layout.cellSize} must be in [16, 64]`
        );
        assert.ok(
            layout.cols * layout.rows >= 60,
            `Grid capacity ${layout.cols * layout.rows} must accommodate 60 items`
        );
        const totalW = layout.cols * layout.cellSize + (layout.cols - 1) * gap;
        const totalH = layout.rows * layout.cellSize + (layout.rows - 1) * gap;
        assert.ok(totalW <= W, `Grid width ${totalW}px must fit inside container width ${W}px`);
        assert.ok(totalH <= H, `Grid height ${totalH}px must fit inside container height ${H}px`);
        assert.equal(layout.capped, false, 'n=60 is within container capacity');
        assert.equal(layout.maxVisible, 60);
    });

    it('n = 500 yields s = 16 with capping to container capacity', () => {
        const layout = computeHeatmapLayout(500, W, H, gap);
        assert.equal(layout.cellSize, 16, 'Overflowing items must shrink to minCell 16px');
        assert.equal(layout.capped, true, 'n=500 must be marked as capped');
        // Capacity in 360x200 at 16px with 8px gap:
        // maxCols = floor((360 + 8) / 24) = 15
        // maxRows = floor((200 + 8) / 24) = 8
        // Capacity = 15 * 8 = 120
        assert.equal(layout.maxVisible, 120, 'maxVisible must equal container capacity (120)');
        const totalW = layout.cols * layout.cellSize + (layout.cols - 1) * gap;
        const totalH = layout.rows * layout.cellSize + (layout.rows - 1) * gap;
        assert.ok(totalW <= W, `Grid width ${totalW}px must fit inside container width ${W}px`);
        assert.ok(totalH <= H, `Grid height ${totalH}px must fit inside container height ${H}px`);
    });

    it('handles edge cases (n = 0, negative dimensions)', () => {
        const zeroLayout = computeHeatmapLayout(0, W, H, gap);
        assert.equal(zeroLayout.maxVisible, 0);
        assert.equal(zeroLayout.capped, false);

        const invalidWLayout = computeHeatmapLayout(10, 0, H, gap);
        assert.equal(invalidWLayout.maxVisible, 0);

        const invalidHLayout = computeHeatmapLayout(10, W, -10, gap);
        assert.equal(invalidHLayout.maxVisible, 0);
    });
});
