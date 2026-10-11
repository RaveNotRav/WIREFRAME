const { describe, it } = require('node:test');
const assert = require('node:assert');
const { statusPresentation, safetyPresentation } = require('./presentation');

describe('Presentation Logic', () => {
    describe('statusPresentation', () => {
        it('should handle standard statuses', () => {
            assert.deepStrictEqual(statusPresentation('complete'), { text: 'Complete', cssClass: 'badge-complete' });
            assert.deepStrictEqual(statusPresentation('rejected'), { text: 'Rejected', cssClass: 'badge-rejected' });
        });
        it('should handle unavailable/Unknown edge cases', () => {
            assert.deepStrictEqual(statusPresentation('unavailable'), { text: 'Unknown', cssClass: 'badge-unknown' });
            assert.deepStrictEqual(statusPresentation('Unknown'), { text: 'Unknown', cssClass: 'badge-unknown' });
            assert.deepStrictEqual(statusPresentation(null), { text: 'Unknown', cssClass: 'badge-unknown' });
            assert.deepStrictEqual(statusPresentation(''), { text: 'Unknown', cssClass: 'badge-unknown' });
        });
    });

    describe('safetyPresentation', () => {
        it('should handle Safe correctly', () => {
            assert.deepStrictEqual(safetyPresentation('safe'), { text: 'Safe', cssClass: 'badge-safe' });
        });
        it('should never render unavailable or Unknown as Safe', () => {
            assert.notDeepStrictEqual(safetyPresentation('unavailable'), { text: 'Safe', cssClass: 'badge-safe' });
            assert.notDeepStrictEqual(safetyPresentation('Unknown'), { text: 'Safe', cssClass: 'badge-safe' });
            assert.deepStrictEqual(safetyPresentation('unavailable'), { text: 'Unknown', cssClass: 'badge-unknown' });
        });
    });
});
