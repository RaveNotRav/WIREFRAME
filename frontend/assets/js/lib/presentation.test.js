import { test, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { statusPresentation, safetyPresentation, formatRejectionReason, shouldShowPublicConfirmation, isTerminalStatus, TERMINAL_STATUSES } from './presentation.js';

describe('Unified presentation.js tests', () => {
    describe('statusPresentation', () => {
        it('handles standard statuses and edge cases', () => {
            assert.equal(statusPresentation('complete').text, 'Complete');
            assert.equal(statusPresentation('rejected').text, 'Rejected');
            const noWiresPres = statusPresentation('no_wires');
            assert.equal(noWiresPres.text, 'No Wires');
            assert.equal(noWiresPres.style, 'warning');
            assert.notEqual(noWiresPres.icon, 'check');
            assert.notEqual(noWiresPres.style, 'success');
            assert.equal(statusPresentation('error').text, 'Error');
            assert.equal(statusPresentation('queued').text, 'Queued');
            assert.equal(statusPresentation('processing').text, 'Processing');
            assert.equal(statusPresentation('interrupted').text, 'Interrupted');
            assert.equal(statusPresentation('unavailable').text, 'Unknown');
            assert.equal(statusPresentation(null).text, 'Unknown');
        });

        it('handles analysis object input and formats rejection reasons', () => {
            assert.equal(statusPresentation({ status: 'complete' }).text, 'Complete');
            assert.equal(statusPresentation({ status: 'rejected', summary: 'resolution_too_low' }).text, 'Rejected');
            assert.equal(
                statusPresentation({ status: 'rejected', summary: 'resolution_too_low' }).reason,
                'The photo is too small. Use the original photo from your camera, not one sent through Telegram or Messenger.'
            );
        });
    });

    describe('formatRejectionReason', () => {
        it('formats known rejection codes into friendly English', () => {
            assert.equal(
                formatRejectionReason('resolution_too_low'),
                'The photo is too small. Use the original photo from your camera, not one sent through Telegram or Messenger.'
            );
            assert.equal(
                formatRejectionReason('too_dark'),
                'The photo is too dark. Retake it in better lighting or daytime.'
            );
            assert.equal(
                formatRejectionReason('too_blurry'),
                'The photo is too blurry. Hold the camera steady and tap to focus on the wires.'
            );
            assert.equal(
                formatRejectionReason('corrupt_file'),
                'The file is corrupt or unreadable. Please choose a valid JPG or PNG image.'
            );
        });

        it('falls back to generic message for unknown rejection codes', () => {
            assert.equal(
                formatRejectionReason('some_unknown_error'),
                'The photo could not be analyzed. Please retake it.'
            );
        });

        it('returns empty string for falsy input', () => {
            assert.equal(formatRejectionReason(null), '');
            assert.equal(formatRejectionReason(undefined), '');
            assert.equal(formatRejectionReason(''), '');
        });
    });

    describe('safetyPresentation matrix: 6 clearance statuses x 4 CRS bands', () => {
        const matrix = [
            // [clearance, crs, expectedText]
            ['above', 'Low', 'Safe'],
            ['above', 'Moderate', 'Caution'],
            ['above', 'High', 'Unsafe'],
            ['above', 'Critical', 'Critical'],

            ['near', 'Low', 'Caution'],
            ['near', 'Moderate', 'Caution'],
            ['near', 'High', 'Unsafe'],
            ['near', 'Critical', 'Critical'],

            ['below', 'Low', 'Critical'],
            ['below', 'Moderate', 'Critical'],
            ['below', 'High', 'Critical'],
            ['below', 'Critical', 'Critical'],

            ['no_pole', 'Low', 'Incomplete'],
            ['no_pole', 'Moderate', 'Incomplete'],
            ['no_pole', 'High', 'Unsafe'],
            ['no_pole', 'Critical', 'Critical'],

            ['no_wires', 'Low', 'Incomplete'],
            ['no_wires', 'Moderate', 'Incomplete'],
            ['no_wires', 'High', 'Unsafe'],
            ['no_wires', 'Critical', 'Critical'],

            ['unavailable', 'Low', 'Unknown'],
            ['unavailable', 'Moderate', 'Caution'],
            ['unavailable', 'High', 'Unsafe'],
            ['unavailable', 'Critical', 'Critical'],
        ];

        for (const [clr, crs, expectedText] of matrix) {
            it(`clearance=${clr} + crs=${crs} -> ${expectedText}`, () => {
                const res = safetyPresentation({ ground_clearance_status: clr, crs_band: crs });
                assert.equal(res.text, expectedText);

                // Assert that "Safe" NEVER appears for no_pole, no_wires or unavailable
                if (['no_pole', 'no_wires', 'unavailable'].includes(clr)) {
                    assert.notEqual(res.text, 'Safe');
                    assert.notEqual(res.label, 'Safe');
                    assert.notEqual(res.risk, 'low');
                }
            });
        }
    });

    describe('safetyPresentation public user English messages', () => {
        it('no_pole provides the approved message in reasons', () => {
            const res = safetyPresentation({ ground_clearance_status: 'no_pole', crs_band: 'Low' });
            assert.equal(res.text, 'Incomplete');
            assert.ok(res.reasons.includes('The whole pole and the ground were not visible, so ground clearance could not be checked. Retake the photo showing the full pole down to the ground.'));
        });

        it('no_wires provides the approved message in reasons', () => {
            const res = safetyPresentation({ ground_clearance_status: 'no_wires', crs_band: 'Low' });
            assert.equal(res.text, 'Incomplete');
            assert.ok(res.reasons.includes('No wires were detected in this photo. Retake it with the overhead wires clearly in view.'));
        });

        it('no_pole with High CRS preserves Unsafe text AND includes the no_pole message', () => {
            const res = safetyPresentation({ ground_clearance_status: 'no_pole', crs_band: 'High' });
            assert.equal(res.text, 'Unsafe');
            assert.ok(res.reasons.includes('The whole pole and the ground were not visible, so ground clearance could not be checked. Retake the photo showing the full pole down to the ground.'));
            assert.ok(res.reasons.includes('CRS band is High'));
        });

        it('no_wires with Critical CRS preserves Critical text AND includes the no_wires message', () => {
            const res = safetyPresentation({ ground_clearance_status: 'no_wires', crs_band: 'Critical' });
            assert.equal(res.text, 'Critical');
            assert.ok(res.reasons.includes('No wires were detected in this photo. Retake it with the overhead wires clearly in view.'));
            assert.ok(res.reasons.includes('CRS band is Critical'));
        });
    });

    describe('shouldShowPublicConfirmation', () => {
        it('opens ONLY for status complete with at least one wire for a guest', () => {
            assert.equal(shouldShowPublicConfirmation({ status: 'complete', wires: [{ id: 1 }] }), true);
            assert.equal(shouldShowPublicConfirmation({ status: 'complete', wire_count: 2 }), true);
        });

        it('does NOT open for complete with 0 wires', () => {
            assert.equal(shouldShowPublicConfirmation({ status: 'complete', wires: [] }), false);
            assert.equal(shouldShowPublicConfirmation({ status: 'complete', wire_count: 0 }), false);
            assert.equal(shouldShowPublicConfirmation({ status: 'complete' }), false);
        });

        it('does NOT open for other statuses (no_wires, rejected, error, interrupted, queued)', () => {
            assert.equal(shouldShowPublicConfirmation({ status: 'no_wires', wires: [] }), false);
            assert.equal(shouldShowPublicConfirmation({ status: 'rejected', wires: [] }), false);
            assert.equal(shouldShowPublicConfirmation({ status: 'error', wires: [] }), false);
            assert.equal(shouldShowPublicConfirmation({ status: 'interrupted', wires: [] }), false);
            assert.equal(shouldShowPublicConfirmation({ status: 'queued', wires: [] }), false);
        });

        it('does NOT open if user is logged in', () => {
            const user = { id: 1, username: 'admin', role: 'admin' };
            assert.equal(shouldShowPublicConfirmation({ status: 'complete', wires: [{ id: 1 }] }, user), false);
        });

        it('handles null or invalid analysis safely', () => {
            assert.equal(shouldShowPublicConfirmation(null), false);
            assert.equal(shouldShowPublicConfirmation(undefined), false);
        });
    });

    describe('isTerminalStatus', () => {
        it('identifies terminal statuses correctly', () => {
            for (const s of TERMINAL_STATUSES) {
                assert.equal(isTerminalStatus(s), true);
                assert.equal(isTerminalStatus(s.toUpperCase()), true);
            }
        });

        it('returns false for non-terminal statuses', () => {
            assert.equal(isTerminalStatus('queued'), false);
            assert.equal(isTerminalStatus('processing'), false);
            assert.equal(isTerminalStatus('pending'), false);
            assert.equal(isTerminalStatus('unavailable'), false);
        });

        it('handles null, undefined, empty safely', () => {
            assert.equal(isTerminalStatus(null), false);
            assert.equal(isTerminalStatus(undefined), false);
            assert.equal(isTerminalStatus(''), false);
            assert.equal(isTerminalStatus(123), false);
        });
    });
});
