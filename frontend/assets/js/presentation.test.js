import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { statusPresentation, safetyPresentation } from './presentation.js';

describe('Presentation Logic (re-exported via presentation.js)', () => {
    describe('statusPresentation', () => {
        it('should handle standard statuses', () => {
            assert.equal(statusPresentation('complete').text, 'Complete');
            assert.equal(statusPresentation('rejected').text, 'Rejected');
        });
        it('should handle unavailable/Unknown edge cases', () => {
            assert.equal(statusPresentation('unavailable').text, 'Unknown');
            assert.equal(statusPresentation('Unknown').text, 'Unknown');
            assert.equal(statusPresentation(null).text, 'Unknown');
            assert.equal(statusPresentation('').text, 'Unknown');
        });
    });

    describe('safetyPresentation', () => {
        it('should handle Safe correctly', () => {
            assert.equal(safetyPresentation('safe').text, 'Safe');
        });
        it('should never render unavailable or Unknown as Safe', () => {
            assert.notEqual(safetyPresentation('unavailable').text, 'Safe');
            assert.notEqual(safetyPresentation('Unknown').text, 'Safe');
            assert.equal(safetyPresentation('unavailable').text, 'Unknown');
        });
    });
});
