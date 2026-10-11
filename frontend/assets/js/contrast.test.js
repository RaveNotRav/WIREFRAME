const { describe, it } = require('node:test');
const assert = require('node:assert');

function lum(c) {
  let [r,g,b] = [parseInt(c.slice(1,3),16), parseInt(c.slice(3,5),16), parseInt(c.slice(5,7),16)].map(v => {
    v /= 255;
    return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function getContrast(c1, c2) {
  let l1 = lum(c1), l2 = lum(c2);
  return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
}

describe('Contrast Requirements', () => {
    it('Dark theme border vs surface >= 3.0:1', () => {
        // Dark theme background
        const bg = '#1E1E24';
        // Updated subtle border
        const border = '#6A6A74';
        const ratio = getContrast(border, bg);
        assert.ok(ratio >= 3.0, `Expected ratio >= 3.0, got ${ratio.toFixed(2)}`);
    });
});
