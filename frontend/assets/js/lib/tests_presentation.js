// tests_presentation.js - Pure logic node test script for presentation
import test from 'node:test';
import assert from 'node:assert';
import { statusPresentation, safetyPresentation } from './presentation.js';

test('Status: queued', () => {
    assert.deepStrictEqual(statusPresentation({ status: 'queued' }), { text: 'QUEUED', risk: 'Unknown', reason: '' });
});
test('Status: processing', () => {
    assert.deepStrictEqual(statusPresentation({ status: 'processing' }), { text: 'PROCESSING', risk: 'Unknown', reason: '' });
});
test('Status: complete', () => {
    assert.deepStrictEqual(statusPresentation({ status: 'complete', reason: 'Done' }), { text: 'COMPLETE', risk: 'Safe', reason: 'Done' });
});
test('Status: rejected', () => {
    assert.deepStrictEqual(statusPresentation({ status: 'rejected', reason: 'Blurry' }), { text: 'REJECTED', risk: 'Critical', reason: 'Blurry' });
});
test('Status: no_wires', () => {
    assert.deepStrictEqual(statusPresentation({ status: 'no_wires' }), { text: 'NO WIRES', risk: 'Unknown', reason: 'No wires were detected in the image.' });
});
test('Status: error', () => {
    assert.deepStrictEqual(statusPresentation({ status: 'error', reason: 'DB Failed' }), { text: 'ERROR', risk: 'Critical', reason: 'An error occurred during analysis.' }); // Sanitized
});
test('Status: interrupted', () => {
    assert.deepStrictEqual(statusPresentation({ status: 'interrupted' }), { text: 'INTERRUPTED', risk: 'Critical', reason: 'Analysis was cancelled.' });
});

test('Safety: Safe', () => {
    const res = safetyPresentation({ safety_level: 'Safe', crs: 0.1, crs_band: 'Low', clearance_status: 'ok', safety_reasons: ['All good'] });
    assert.strictEqual(res.text, 'Safe');
    assert.strictEqual(res.risk, 'Safe');
});
test('Safety: Caution', () => {
    const res = safetyPresentation({ safety_level: 'Caution', crs: 0.4, crs_band: 'Moderate', clearance_status: 'ok', safety_reasons: ['Caution'] });
    assert.strictEqual(res.text, 'Caution');
    assert.strictEqual(res.risk, 'Caution');
});
test('Safety: Unsafe', () => {
    const res = safetyPresentation({ safety_level: 'Unsafe', crs: 0.7, crs_band: 'High', clearance_status: 'ok', safety_reasons: ['Unsafe'] });
    assert.strictEqual(res.text, 'Unsafe');
    assert.strictEqual(res.risk, 'Unsafe');
});
test('Safety: Critical', () => {
    const res = safetyPresentation({ safety_level: 'Critical', crs: 0.9, crs_band: 'Critical', clearance_status: 'ok', safety_reasons: ['Critical'] });
    assert.strictEqual(res.text, 'Critical');
    assert.strictEqual(res.risk, 'Critical');
});
test('Safety: Unknown', () => {
    const res = safetyPresentation({ safety_level: 'Unknown', crs_band: 'unavailable', safety_reasons: [] });
    assert.strictEqual(res.text, 'Unknown');
    assert.strictEqual(res.risk, 'Unknown');
});
test('Safety: clearance unavailable', () => {
    const res = safetyPresentation({ safety_level: 'Caution', crs: 0.5, crs_band: 'Moderate', clearance_status: 'unavailable', basis: 'crs_only', safety_reasons: [] });
    assert.ok(res.reasons.includes('Clearance could not be estimated; verdict is based on CRS only.'));
});
test('Safety: CRS unavailable', () => {
    const res = safetyPresentation({ safety_level: 'Unknown', crs_band: 'unavailable', safety_reasons: [] });
    assert.strictEqual(res.crsText, '--');
    assert.strictEqual(res.crsColor, 'var(--text-muted)');
    assert.ok(res.reasons.includes('CRS could not be computed due to missing mask or failed fit.'));
});
test('Safety: basis=crs_only', () => {
    const res = safetyPresentation({ safety_level: 'Caution', crs: 0.5, crs_band: 'Moderate', clearance_status: 'unavailable', basis: 'crs_only', safety_reasons: [] });
    assert.strictEqual(res.text, 'Caution');
});
test('Safety: explicit assertion unavailable/Unknown never render as Safe', () => {
    const res = safetyPresentation({ safety_level: 'Safe', crs_band: 'unavailable', clearance_status: 'unavailable' });
    assert.notStrictEqual(res.risk, 'Safe');
    assert.strictEqual(res.risk, 'Unknown');
});
