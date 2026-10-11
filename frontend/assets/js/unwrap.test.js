import test from 'node:test';
import assert from 'node:assert';
import { unwrapList } from './main.js';

test('unwrapList', async (t) => {
    await t.test('array', () => {
        const input = [1, 2, 3];
        assert.deepStrictEqual(unwrapList(input), [1, 2, 3]);
    });
    
    await t.test('{items: [...]}', () => {
        const input = { items: [1, 2, 3], total: 3 };
        assert.deepStrictEqual(unwrapList(input), [1, 2, 3]);
    });
    
    await t.test('null', () => {
        assert.deepStrictEqual(unwrapList(null), []);
        assert.deepStrictEqual(unwrapList(undefined), []);
    });
    
    await t.test('malformed', () => {
        const input = { error: "bad request", message: "fail" };
        assert.deepStrictEqual(unwrapList(input), []);
        assert.deepStrictEqual(unwrapList("string"), []);
        assert.deepStrictEqual(unwrapList(123), []);
    });
});
