// tests.js - Pure logic node test script
import test from 'node:test';
import assert from 'node:assert';
import { getNaturalCoordinates } from './coordinates.js';
import { escapeHtml, formatConfidence, getWireClassColor, parseDbTimestamp, formatDbTimestamp } from './formatters.js';

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

test('parseDbTimestamp parses UTC SQLite strings', () => {
    const d = parseDbTimestamp('2026-10-11 06:49:00');
    assert.ok(d instanceof Date);
    assert.strictEqual(d.toISOString(), '2026-10-11T06:49:00.000Z');
    assert.strictEqual(parseDbTimestamp(null), null);
    assert.strictEqual(parseDbTimestamp(''), null);
});

test('formatDbTimestamp formats in Asia/Manila (UTC+8)', () => {
    // 06:49 UTC is 14:49 (02:49 PM) in Asia/Manila
    const formatted = formatDbTimestamp('2026-10-11 06:49:00');
    assert.strictEqual(formatted, '10/11/2026, 02:49 PM');
    
    const dateOnly = formatDbTimestamp('2026-10-11 06:49:00', { includeTime: false });
    assert.strictEqual(dateOnly, '10/11/2026');
    
    assert.strictEqual(formatDbTimestamp(null), '--');
});
