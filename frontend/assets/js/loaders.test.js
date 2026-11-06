import test from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fetchApi, unwrapList } from './main.js';
import {
    loadHistoryData,
    NO_VERIFIED_RECORDS_MSG,
    resolveExportSelection,
    formatSkippedNoGpsMessage,
    requestHistoryExport
} from './loaders.js';

test('History/Sites Loaders', async (t) => {
    const originalFetch = global.fetch;

    await t.test('normal array', async () => {
        global.fetch = async (url) => {
            return {
                ok: true,
                status: 200,
                json: async () => ({ data: [{ id: 1 }, { id: 2 }], total: 2, limit: 12 })
            };
        };
        const res = await loadHistoryData();
        assert.deepStrictEqual(res.items, [{ id: 1 }, { id: 2 }]);
        assert.strictEqual(res.total, 2);
    });

    await t.test('paginated object (unwrap simulation in unwrapList)', async () => {
        const input = { items: [{ id: 1 }], total: 1 };
        assert.deepStrictEqual(unwrapList(input), [{ id: 1 }]);
    });
    
    await t.test('null response unwrap', async () => {
        assert.deepStrictEqual(unwrapList(null), []);
    });

    await t.test('500 internal server error', async () => {
        global.fetch = async (url) => {
            return {
                ok: false,
                status: 500,
                json: async () => ({ detail: "Internal Server Error" })
            };
        };
        try {
            await loadHistoryData();
            assert.fail('Should have thrown');
        } catch (err) {
            assert.strictEqual(err.message, 'Internal Server Error');
        }
    });

    await t.test('network failure', async () => {
        global.fetch = async (url) => {
            throw new Error('Network failed');
        };
        try {
            await loadHistoryData();
            assert.fail('Should have thrown');
        } catch (err) {
            assert.strictEqual(err.message, 'Network failed');
        }
    });

    await t.test('a failed secondary fetch does not block the list', async () => {
        global.fetch = async (url) => {
            if (url.includes('/api/sites')) {
                throw new Error('Secondary fetch failed');
            }
            return {
                ok: true,
                status: 200,
                json: async () => ({ data: [{ id: 100 }], total: 1, limit: 12 })
            };
        };
        const res = await loadHistoryData();
        assert.deepStrictEqual(res.items, [{ id: 100 }]);
    });

    await t.test('include_invalid appends query parameter', async () => {
        let requestedUrl = '';
        global.fetch = async (url) => {
            if (!url.includes('/api/sites')) {
                requestedUrl = url;
            }
            return {
                ok: true,
                status: 200,
                json: async () => ({ data: [], total: 0, limit: 12 })
            };
        };
        await loadHistoryData({ include_invalid: true });
        assert.ok(requestedUrl.includes('include_invalid=true'), `Expected include_invalid=true in ${requestedUrl}`);

        await loadHistoryData({ include_invalid: false });
        assert.ok(!requestedUrl.includes('include_invalid'), `Did not expect include_invalid in ${requestedUrl}`);
    });

    await t.test('history export requests include include_mock from checkbox and fallback to filtered items when none checked', async () => {
        const loaded = [
            { id: 10, status: 'complete', is_mock: 0, is_public: 0 },
            { id: 11, status: 'complete', is_mock: 1, is_public: 1 },
            { id: 12, status: 'rejected', is_mock: 0, is_public: 0 }
        ];

        // When checkbox unchecked (includeMock=false) and no cards manually checked -> only verified complete items
        const idsFalse = resolveExportSelection([], loaded, { includeMock: false, canIncludeInvalid: false });
        assert.deepStrictEqual(idsFalse, [10]);

        // When checkbox checked (includeMock=true) and no cards manually checked -> includes mock & public complete items
        const idsTrue = resolveExportSelection([], loaded, { includeMock: true, canIncludeInvalid: false });
        assert.deepStrictEqual(idsTrue, [10, 11]);

        // When specific cards are checked, respects selection while filtering out mock/public if includeMock=false
        assert.deepStrictEqual(resolveExportSelection([11], loaded, { includeMock: false }), []);
        assert.deepStrictEqual(resolveExportSelection([11], loaded, { includeMock: true }), [11]);

        // Verify requestHistoryExport sends include_mock boolean to all three export endpoints
        const sentRequests = [];
        global.fetch = async (url, options) => {
            sentRequests.push({ url, body: JSON.parse(options.body), headers: options.headers });
            return {
                ok: true,
                status: 200,
                headers: new Map([['X-Skipped-Count', '0'], ['X-Skipped-No-Gps', '0']]),
                blob: async () => new Blob(['test'])
            };
        };

        for (const ep of ['/api/reports/summary', '/api/reports/geojson', '/api/reports/csv']) {
            await requestHistoryExport(ep, { analysis_ids: [10, 11], include_mock: true, include_invalid: false });
            await requestHistoryExport(ep, { analysis_ids: [10], include_mock: false, include_invalid: false });
        }

        assert.strictEqual(sentRequests.length, 6);
        assert.strictEqual(sentRequests[0].body.include_mock, true);
        assert.strictEqual(sentRequests[1].body.include_mock, false);
        assert.strictEqual(sentRequests[2].body.include_mock, true);
        assert.strictEqual(sentRequests[3].body.include_mock, false);
        assert.strictEqual(sentRequests[4].body.include_mock, true);
        assert.strictEqual(sentRequests[5].body.include_mock, false);
    });

    await t.test('zero-records message, skipped-no-GPS message, and server error detail shown', async () => {
        assert.strictEqual(
            NO_VERIFIED_RECORDS_MSG,
            "No verified records to export. Public tips and mock data are excluded from official exports. Check 'Include Mock' to export test data (marked as unverified)."
        );
        assert.strictEqual(formatSkippedNoGpsMessage(1), '1 record(s) skipped (no GPS).');
        assert.strictEqual(formatSkippedNoGpsMessage(3), '3 record(s) skipped (no GPS).');
        assert.strictEqual(formatSkippedNoGpsMessage(0), '');

        // Verify 400/403/422/429/500 export responses preserve server detail instead of generic fallback
        for (const status of [400, 403, 422, 429, 500]) {
            global.fetch = async () => ({
                ok: false,
                status,
                json: async () => ({ detail: `Server error detail for ${status}` })
            });
            await assert.rejects(
                async () => requestHistoryExport('/api/reports/geojson', { analysis_ids: [1], include_mock: false }),
                { message: `Server error detail for ${status}` }
            );
        }

        // Verify history.html wires NO_VERIFIED_RECORDS_MSG, formatSkippedNoGpsMessage, and requestHistoryExport
        const historyHtml = fs.readFileSync(path.resolve(process.cwd(), 'frontend/pages/history.html'), 'utf8');
        assert.ok(historyHtml.includes('NO_VERIFIED_RECORDS_MSG'));
        assert.ok(historyHtml.includes('formatSkippedNoGpsMessage'));
        assert.ok(historyHtml.includes('requestHistoryExport'));
    });

    global.fetch = originalFetch;
});
