import { test } from 'node:test';
import assert from 'node:assert/strict';
import { statusPresentation, safetyPresentation } from './presentation.js';

test('statusPresentation returns correct output', (t) => {
    assert.deepEqual(statusPresentation('queued'), { label: 'Queued', style: 'neutral', icon: 'hourglass' });
    assert.deepEqual(statusPresentation('processing'), { label: 'Processing', style: 'neutral', icon: 'spinner' });
    assert.deepEqual(statusPresentation('complete'), { label: 'Complete', style: 'success', icon: 'check' });
    assert.deepEqual(statusPresentation('rejected', { summary: 'Blurry' }), { label: 'Rejected: Blurry', style: 'danger', icon: 'x' });
    assert.deepEqual(statusPresentation('no_wires'), { label: 'No Wires', style: 'success', icon: 'check' });
    assert.deepEqual(statusPresentation('error', { summary: 'Should not leak details' }), { label: 'Error', style: 'danger', icon: 'alert' });
    assert.deepEqual(statusPresentation('interrupted'), { label: 'Interrupted', style: 'neutral', icon: 'pause' });
});

test('safetyPresentation handles unavailable states', (t) => {
    // Assert that unavailable/Unknown NEVER renders as Safe and always uses a distinct neutral style, icon
    const unknownRes = { label: 'Unknown', style: 'neutral', icon: 'help' };
    
    assert.deepEqual(safetyPresentation('Safe', 'unavailable', 'Low'), unknownRes);
    assert.deepEqual(safetyPresentation('Safe', 'below', 'unavailable'), unknownRes);
    assert.deepEqual(safetyPresentation('Unknown', 'below', 'Low'), unknownRes);
    assert.deepEqual(safetyPresentation(null, null, null), unknownRes);
});

test('safetyPresentation handles known states', (t) => {
    assert.deepEqual(safetyPresentation('Safe', 'above', 'Low'), { label: 'Safe', style: 'success', icon: 'check' });
    assert.deepEqual(safetyPresentation('Caution', 'near', 'Moderate'), { label: 'Caution', style: 'warning', icon: 'alert' });
    assert.deepEqual(safetyPresentation('Unsafe', 'below', 'High'), { label: 'Unsafe', style: 'danger', icon: 'x' });
    assert.deepEqual(safetyPresentation('Critical', 'below', 'Critical'), { label: 'Critical', style: 'danger', icon: 'alert' });
});
