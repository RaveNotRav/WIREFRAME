// tests.js - Pure logic node test script
import test from 'node:test';
import assert from 'node:assert';
import { getNaturalCoordinates } from './coordinates.js';
import { escapeHtml, formatConfidence, getWireClassColor, formatWireClass, parseDbTimestamp, formatDbTimestamp } from './formatters.js';

test('escapeHtml handles strings correctly', () => {
    // XSS payload
    assert.strictEqual(escapeHtml('<script>alert("xss")</script>'), '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;');
    
    // Quotes (both double and single)
    assert.strictEqual(escapeHtml(`"hello" and 'world'`), '&quot;hello&quot; and &#39;world&#39;');
    
    // Null and undefined
    assert.strictEqual(escapeHtml(null), '');
    assert.strictEqual(escapeHtml(undefined), '');
    
    // Numbers
    assert.strictEqual(escapeHtml(123), '123');
    assert.strictEqual(escapeHtml(0), '0');
    assert.strictEqual(escapeHtml(-45.67), '-45.67');
    
    // Booleans
    assert.strictEqual(escapeHtml(true), 'true');
    assert.strictEqual(escapeHtml(false), 'false');
    
    // Normal strings and ampersand
    assert.strictEqual(escapeHtml('Hello World'), 'Hello World');
    assert.strictEqual(escapeHtml('Salt & Pepper'), 'Salt &amp; Pepper');
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

test('formatWireClass formats known classes and falls back to title case', () => {
    assert.strictEqual(formatWireClass('power_line'), 'Power Line');
    assert.strictEqual(formatWireClass('telecom_cable'), 'Telecom Cable');
    assert.strictEqual(formatWireClass('utility_pole'), 'Utility Pole');
    assert.strictEqual(formatWireClass('high_voltage-cable'), 'High Voltage Cable');
    assert.strictEqual(formatWireClass(''), 'Unknown');
    assert.strictEqual(formatWireClass(null), 'Unknown');
});
