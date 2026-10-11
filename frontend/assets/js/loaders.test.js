import test from 'node:test';
import assert from 'node:assert';
import { fetchApi, unwrapList } from './main.js';
import { loadHistoryData } from './loaders.js';

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

    global.fetch = originalFetch;
});
