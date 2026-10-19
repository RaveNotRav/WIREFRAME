import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

describe('Analyze Upload Queue Reset & Retry Tests', () => {
    const analyzeJsPath = path.resolve(process.cwd(), 'frontend/assets/js/analyze.js');
    const analyzeJs = fs.readFileSync(analyzeJsPath, 'utf8');

    describe('Static Code & Architecture Checks', () => {
        it('defines clearUploadQueue resetting currentFiles, fileThumbURLs, thumbnails container, and file-input value', () => {
            assert.match(
                analyzeJs,
                /function\s+clearUploadQueue\s*\(\)\s*\{[\s\S]*?currentFiles\s*=\s*\[\]/i,
                'clearUploadQueue must reset currentFiles to an empty array'
            );
            assert.match(
                analyzeJs,
                /URL\.revokeObjectURL/i,
                'clearUploadQueue must call URL.revokeObjectURL to release memory'
            );
            assert.match(
                analyzeJs,
                /fileThumbURLs\s*=\s*\[\]/i,
                'clearUploadQueue must reset fileThumbURLs to an empty array'
            );
            assert.match(
                analyzeJs,
                /document\.getElementById\(['"]thumbnails['"]\)[\s\S]*?innerHTML\s*=\s*['"]['"]/i,
                'clearUploadQueue must empty the thumbnails UI'
            );
            assert.match(
                analyzeJs,
                /document\.getElementById\(['"]file-input['"]\)[\s\S]*?value\s*=\s*['"]['"]/i,
                'clearUploadQueue must reset file-input.value to empty string'
            );
        });

        it('startAnalysis calls clearUploadQueue inside try block after successful submit', () => {
            // Find startAnalysis function body
            const startAnalysisMatch = analyzeJs.match(/async\s+function\s+startAnalysis\s*\(\)\s*\{([\s\S]*?)\n\}/);
            assert.ok(startAnalysisMatch, 'startAnalysis function must exist in analyze.js');
            const body = startAnalysisMatch[1];

            // Verify clearUploadQueue is inside try block
            const tryBlockMatch = body.match(/try\s*\{([\s\S]*?)\}\s*catch/);
            assert.ok(tryBlockMatch, 'startAnalysis must have a try/catch structure');
            const tryContent = tryBlockMatch[1];
            assert.match(
                tryContent,
                /clearUploadQueue\s*\(\)/,
                'clearUploadQueue() must be called inside the try block after submission succeeds'
            );

            // Verify catch block does NOT call clearUploadQueue (so files are kept on failure for retry)
            const catchBlockMatch = body.match(/catch\s*\([^)]*\)\s*\{([\s\S]*?)\}/);
            assert.ok(catchBlockMatch, 'catch block must exist');
            const catchContent = catchBlockMatch[1];
            assert.doesNotMatch(
                catchContent,
                /clearUploadQueue\s*\(\)/,
                'clearUploadQueue() must NOT be called in the catch block on failure'
            );
        });

        it('single and batch paths validate response before clearing upload queue', () => {
            // Both single and batch should verify res.ok or ID presence before completing
            assert.match(
                analyzeJs,
                /!res\.ok\s*\|\|\s*!data\.analysis_id/,
                'Single upload should verify ok response and analysis_id'
            );
            assert.match(
                analyzeJs,
                /!res\.ok\s*\|\|\s*!data\.batch_id/,
                'Batch upload should verify ok response and batch_id'
            );
        });
    });

    describe('Unit & Integration Behavior', async () => {
        let analyzeModule;
        let revokedUrls = [];
        let mockElements = {};

        const makeMockElement = (props = {}) => ({
            style: {},
            innerHTML: '',
            textContent: '',
            value: '',
            addEventListener: () => {},
            removeEventListener: () => {},
            setAttribute: () => {},
            getAttribute: () => null,
            classList: { add: () => {}, remove: () => {}, contains: () => false },
            querySelector: () => makeMockElement(),
            querySelectorAll: () => [],
            appendChild: () => {},
            remove: () => {},
            click: () => {},
            ...props
        });

        beforeEach(async () => {
            revokedUrls = [];
            mockElements = {
                'thumbnails': makeMockElement({ innerHTML: '<div>thumb1</div><div>thumb2</div>' }),
                'file-input': makeMockElement({ value: 'C:\\fakepath\\wire1.jpg' }),
                'input-company': makeMockElement({ value: 'Meralco' }),
                'input-barangay': makeMockElement({ value: 'San Antonio' }),
                'input-street': makeMockElement({ value: 'Ortigas Ave' }),
                'input-notes': makeMockElement({ value: 'Pole test' }),
                'input-lat': makeMockElement({ value: '14.58' }),
                'input-lng': makeMockElement({ value: '121.06' }),
                'tile-batch': makeMockElement(),
                'tile-progress': makeMockElement(),
                'status-reason': makeMockElement({ textContent: '' }),
                'status-badge': makeMockElement(),
                'batch-tabs': makeMockElement(),
                'stepper': makeMockElement(),
                'step-upload': makeMockElement(),
                'step-processing': makeMockElement(),
                'step-results': makeMockElement(),
                'btn-cancel': makeMockElement()
            };

            globalThis.document = {
                getElementById: (id) => mockElements[id] || makeMockElement(),
                querySelectorAll: () => [],
                createDocumentFragment: () => ({
                    children: [],
                    appendChild(child) { this.children.push(child); }
                }),
                createElement: (tag) => makeMockElement({ tagName: tag.toUpperCase() }),
                addEventListener: () => {}
            };

            globalThis.URL = {
                createObjectURL: (file) => `blob:http://localhost/${file.name}`,
                revokeObjectURL: (url) => { revokedUrls.push(url); }
            };

            globalThis.FormData = class MockFormData {
                constructor() { this.entries = []; }
                append(key, val) { this.entries.push([key, val]); }
            };

            globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
            globalThis.cancelAnimationFrame = (id) => clearTimeout(id);

            analyzeModule = await import(`./analyze.js?t=${Date.now()}`);
        });

        afterEach(() => {
            analyzeModule?.stopAllPolling?.();
        });

        it('clearUploadQueue() resets files, revokes URLs, and clears DOM', () => {
            analyzeModule.setCurrentFiles([{ name: 'img1.jpg' }, { name: 'img2.jpg' }]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/img1.jpg', 'blob:http://localhost/img2.jpg']);

            assert.strictEqual(analyzeModule.getCurrentFiles().length, 2);
            assert.strictEqual(analyzeModule.getFileThumbURLs().length, 2);

            analyzeModule.clearUploadQueue();

            assert.deepStrictEqual(analyzeModule.getCurrentFiles(), []);
            assert.deepStrictEqual(analyzeModule.getFileThumbURLs(), []);
            assert.deepStrictEqual(revokedUrls, ['blob:http://localhost/img1.jpg', 'blob:http://localhost/img2.jpg']);
            assert.strictEqual(mockElements['thumbnails'].innerHTML, '');
            assert.strictEqual(mockElements['file-input'].value, '');
        });

        it('startAnalysis() on successful single upload resets queue and UI', async () => {
            const fakeFile = { name: 'pole1.jpg', size: 1024, type: 'image/jpeg' };
            analyzeModule.setCurrentFiles([fakeFile]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/pole1.jpg']);

            // Mock fetch to succeed
            globalThis.fetch = async (url) => {
                if (url.includes('/api/analysis/upload')) {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({ analysis_id: 101 })
                    };
                }
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({})
                };
            };

            await analyzeModule.startAnalysis();

            // After successful submit: queue is emptied and thumbnails revoked/cleared
            assert.deepStrictEqual(analyzeModule.getCurrentFiles(), []);
            assert.deepStrictEqual(analyzeModule.getFileThumbURLs(), []);
            assert.deepStrictEqual(revokedUrls, ['blob:http://localhost/pole1.jpg']);
            assert.strictEqual(mockElements['thumbnails'].innerHTML, '');
            assert.strictEqual(mockElements['file-input'].value, '');
        });

        it('startAnalysis() on failed upload preserves files for retry', async () => {
            const fakeFile = { name: 'retry_me.jpg', size: 2048, type: 'image/jpeg' };
            analyzeModule.setCurrentFiles([fakeFile]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/retry_me.jpg']);
            mockElements['thumbnails'].innerHTML = '<div>thumbnail_preview</div>';
            mockElements['file-input'].value = 'C:\\fakepath\\retry_me.jpg';

            // Mock fetch to reject with 500 error
            globalThis.fetch = async (url) => {
                return {
                    ok: false,
                    status: 500,
                    json: async () => ({ detail: 'Internal Server Error' })
                };
            };

            await analyzeModule.startAnalysis();

            // After failure: files and previews are PRESERVED so user can retry
            assert.strictEqual(analyzeModule.getCurrentFiles().length, 1);
            assert.strictEqual(analyzeModule.getCurrentFiles()[0].name, 'retry_me.jpg');
            assert.strictEqual(analyzeModule.getFileThumbURLs().length, 1);
            assert.strictEqual(revokedUrls.length, 0, 'No object URLs should be revoked on failure');
            assert.strictEqual(mockElements['thumbnails'].innerHTML, '<div>thumbnail_preview</div>');
            assert.strictEqual(mockElements['file-input'].value, 'C:\\fakepath\\retry_me.jpg');
        });

        it('a new upload after previous analysis starts a fresh single analysis', async () => {
            // Run analysis 1 successfully
            analyzeModule.setCurrentFiles([{ name: 'first.jpg' }]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/first.jpg']);

            let uploadedUrls = [];
            globalThis.fetch = async (url) => {
                uploadedUrls.push(url);
                if (url.includes('/api/analysis/upload')) {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({ analysis_id: 201 })
                    };
                }
                return { ok: true, status: 200, json: async () => ({}) };
            };

            await analyzeModule.startAnalysis();
            assert.strictEqual(analyzeModule.getCurrentFiles().length, 0, 'Queue cleared after first analysis');

            // User selects a single new file
            const newFile = { name: 'second.jpg', size: 1024, type: 'image/jpeg' };
            analyzeModule.setCurrentFiles([newFile]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/second.jpg']);

            // Start analysis again
            uploadedUrls = [];
            await analyzeModule.startAnalysis();

            // Verifies it called single upload (/api/analysis/upload) and NOT batch (/api/batches)
            assert.ok(uploadedUrls.some(u => u.includes('/api/analysis/upload')), 'Must call single analysis endpoint');
            assert.ok(!uploadedUrls.some(u => u.includes('/api/batches')), 'Must not call batch endpoint');
            assert.strictEqual(analyzeModule.getCurrentFiles().length, 0, 'Queue cleared after second analysis');
        });
    });
});
