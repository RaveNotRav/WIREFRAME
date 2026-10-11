import test from 'node:test';
import assert from 'node:assert';
import { getNaturalCoordinates } from './coordinates.js';

test('getNaturalCoordinates computes properly (contain)', () => {
    const mockElement = {
        getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 })
    };
    const mockEvent = { clientX: 110, clientY: 70 };
    // Image is 1000x500. Container is 200x100. Scale = 0.2. No offset.
    // Client clicked exactly center (110-10=100, 70-20=50).
    const coords = getNaturalCoordinates(mockEvent, mockElement, 1000, 500, 'contain');
    assert.strictEqual(coords.x, 500);
    assert.strictEqual(coords.y, 250);
});
