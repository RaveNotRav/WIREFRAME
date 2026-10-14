import { test, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { statusPresentation, safetyPresentation } from './presentation.js';

describe('Unified presentation.js tests', () => {
    describe('statusPresentation', () => {
        it('handles standard statuses and edge cases', () => {
            assert.equal(statusPresentation('complete').text, 'Complete');
            assert.equal(statusPresentation('rejected').text, 'Rejected');
            assert.equal(statusPresentation('no_wires').text, 'No Wires');
            assert.equal(statusPresentation('error').text, 'Error');
            assert.equal(statusPresentation('queued').text, 'Queued');
            assert.equal(statusPresentation('processing').text, 'Processing');
            assert.equal(statusPresentation('interrupted').text, 'Interrupted');
            assert.equal(statusPresentation('unavailable').text, 'Unknown');
            assert.equal(statusPresentation(null).text, 'Unknown');
        });

        it('handles analysis object input', () => {
            assert.equal(statusPresentation({ status: 'complete' }).text, 'Complete');
            assert.equal(statusPresentation({ status: 'rejected', summary: 'Blurry' }).text, 'Rejected');
            assert.equal(statusPresentation({ status: 'rejected', summary: 'Blurry' }).reason, 'Blurry');
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
});
