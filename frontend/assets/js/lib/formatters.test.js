// tests.js - Pure logic node test script
import test from 'node:test';
import assert from 'node:assert';
import { getNaturalCoordinates } from './coordinates.js';
import { escapeHtml, formatConfidence, getWireClassColor } from './formatters.js';

test('escapeHtml handles strings correctly', () => {
    // Mock document
    global.document = {
        createElement: () => ({
            appendChild: () => {},
            set textContent(val) { this._val = val; },
            get innerHTML() { 
                return this._val.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); 
            }
        })
    };
    
    assert.strictEqual(escapeHtml('<script>alert(1)</script>'), '&lt;script&gt;alert(1)&lt;/script&gt;');
    assert.strictEqual(escapeHtml(null), '');
    assert.strictEqual(escapeHtml(undefined), '');
});

test('formatConfidence formats correctly', () => {
    assert.strictEqual(formatConfidence(0.852), '85%');
    assert.strictEqual(formatConfidence('invalid'), '0%');
});

test('getWireClassColor maps correctly', () => {
    assert.strictEqual(getWireClassColor('power_line'), 'var(--wire-power)');
    assert.strictEqual(getWireClassColor('unknown'), 'var(--text-muted)');
});

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
