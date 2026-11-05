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

        let testIndex = 0;

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

        it('#3 & #2: single upload sets thumb to server original URL and records batch_id', async () => {
            analyzeModule.setCurrentFiles([{ name: 'test_thumb.jpg' }]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/test_thumb.jpg']);

            globalThis.fetch = async (url) => {
                if (url.includes('/api/analysis/upload')) {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({ analysis_id: 305, batch_id: 99 })
                    };
                }
                return { ok: true, status: 200, json: async () => ({}) };
            };

            await analyzeModule.startAnalysis();
            const items = analyzeModule.getBatchItems();
            assert.strictEqual(items.length, 1);
            assert.strictEqual(items[0].thumb, '/api/analysis/305/original', 'Thumb must use server endpoint, not revoked blob URL');
            assert.strictEqual(analyzeModule.getCurrentBatchId(), 99, 'currentBatchId must be set from data.batch_id');
        });

        it('#11: rejected result (no wires, no site) renders REJECTED badge and friendly message without throwing', () => {
            const rejectedResult = {
                id: 42,
                status: 'rejected',
                summary: 'too_blurry'
            };
            const rejectedEvents = [
                { state: 'Initializing', detail: 'Starting analysis' },
                { state: 'ImageAcquired', detail: 'Loaded blurry.jpg' },
                { state: 'ImageRejected', detail: 'too_blurry' },
                { state: 'Idle', detail: 'Pipeline finished (rejected)' }
            ];

            assert.doesNotThrow(() => {
                analyzeModule.renderTerminalAnalysisState(rejectedResult, rejectedEvents);
            });

            assert.match(mockElements['status-badge'].innerHTML, /Rejected/i, 'Badge must stay Rejected');
            assert.doesNotMatch(mockElements['status-badge'].innerHTML, /Error/i, 'Badge must not flip to Error');
            assert.strictEqual(
                mockElements['status-reason'].textContent,
                'The photo is too blurry. Hold the camera steady and tap to focus on the wires.',
                'Friendly rejection reason must be shown'
            );
        });

        it('#11: startAnalysis does not throw Cannot read properties of undefined (reading "name") if currentFiles cleared mid-flight', async () => {
            analyzeModule.setCurrentFiles([{ name: 'blurry_phone.jpg' }]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/blurry_phone.jpg']);

            globalThis.fetch = async (url) => {
                if (url.includes('/api/analysis/upload')) {
                    // Simulate currentFiles being cleared or mutated while upload is in-flight
                    analyzeModule.clearUploadQueue();
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({ analysis_id: 404, batch_id: 12 })
                    };
                }
                if (url.includes('/events')) {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ([
                            { state: 'ImageRejected', detail: 'too_blurry' },
                            { state: 'Idle', detail: 'Pipeline finished (rejected)' }
                        ])
                    };
                }
                if (url.includes('/result')) {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({ id: 404, status: 'rejected', summary: 'too_blurry' })
                    };
                }
                return { ok: true, status: 200, json: async () => ({ status: 'complete' }) };
            };

            await analyzeModule.startAnalysis();

            assert.doesNotMatch(
                mockElements['status-reason'].textContent,
                /Cannot read properties of undefined/i,
                'status-reason must never show TypeError message'
            );
            assert.strictEqual(analyzeModule.getBatchItems()[0].name, 'blurry_phone.jpg');
        });

        it('GPS fix persists across clearUploadQueue() until successful submit, and is sent in both single and batch FormData', async () => {
            // 1. Simulate browser geolocation fix
            mockElements['input-lat'].value = '';
            mockElements['input-lng'].value = '';
            mockElements['gps-source-text'] = makeMockElement({ textContent: '' });
            mockElements['btn-gps'] = makeMockElement({ disabled: false });

            let capturedOptions = null;
            Object.defineProperty(globalThis, 'navigator', {
                value: {
                    geolocation: {
                        getCurrentPosition: (onSuccess, _onError, opts) => {
                            capturedOptions = opts;
                            onSuccess({ coords: { latitude: 14.6051796, longitude: 120.962724 } });
                        }
                    }
                },
                configurable: true,
                writable: true
            });

            analyzeModule.requestGpsLocation();
            assert.deepStrictEqual(capturedOptions, { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 });
            assert.strictEqual(mockElements['input-lat'].value, '14.6051796');
            assert.strictEqual(mockElements['input-lng'].value, '120.962724');
            assert.strictEqual(analyzeModule.getLastGpsFix().lat, '14.6051796');
            assert.strictEqual(analyzeModule.getLastGpsFix().lng, '120.962724');

            // 2. Queue reset does NOT wipe GPS fix
            analyzeModule.clearUploadQueue();
            assert.strictEqual(analyzeModule.getLastGpsFix().lat, '14.6051796');
            assert.strictEqual(analyzeModule.getLastGpsFix().lng, '120.962724');

            // Even if DOM input value was cleared by mobile autofill/picker, lastGpsFix fallback supplies it
            mockElements['input-lat'].value = '';
            mockElements['input-lng'].value = '';

            // 3. Single upload FormData includes gps_lat and gps_lng
            let singleFormEntries = [];
            analyzeModule.setCurrentFiles([{ name: 'single_gps.jpg' }]);
            globalThis.fetch = async (url, opts) => {
                if (url.includes('/api/analysis/upload')) {
                    singleFormEntries = opts.body.entries;
                    return { ok: true, status: 200, json: async () => ({ analysis_id: 501, batch_id: 51 }) };
                }
                return { ok: true, status: 200, json: async () => ({ status: 'complete' }) };
            };
            await analyzeModule.startAnalysis();
            analyzeModule.stopAllPolling();
            assert.ok(singleFormEntries.some(([k, v]) => k === 'gps_lat' && v === '14.6051796'), 'Single upload FormData must include gps_lat');
            assert.ok(singleFormEntries.some(([k, v]) => k === 'gps_lng' && v === '120.962724'), 'Single upload FormData must include gps_lng');

            // After successful submit, GPS state is cleared
            assert.strictEqual(analyzeModule.getLastGpsFix().lat, '');

            // 4. Batch upload FormData ALSO includes gps_lat and gps_lng
            analyzeModule.setLastGpsFix('14.42', '120.94', 'Browser (Device)');
            mockElements['input-lat'].value = '14.42';
            mockElements['input-lng'].value = '120.94';
            analyzeModule.setCurrentFiles([{ name: 'batch1.jpg' }, { name: 'batch2.jpg' }]);

            let batchFormEntries = [];
            globalThis.fetch = async (url, opts) => {
                if (url.endsWith('/api/batches')) {
                    batchFormEntries = opts.body.entries;
                    return { ok: true, status: 200, json: async () => ({ batch_id: 77 }) };
                }
                if (url.includes('/api/batches/77')) {
                    return {
                        ok: true,
                        status: 200,
                        json: async () => ({ analyses: [{ id: 701, status: 'queued' }, { id: 702, status: 'queued' }] })
                    };
                }
                return { ok: true, status: 200, json: async () => ({ status: 'complete' }) };
            };
            await analyzeModule.startAnalysis();
            analyzeModule.stopAllPolling();
            assert.ok(batchFormEntries.some(([k, v]) => k === 'gps_lat' && v === '14.42'), 'Batch upload FormData must include gps_lat');
            assert.ok(batchFormEntries.some(([k, v]) => k === 'gps_lng' && v === '120.94'), 'Batch upload FormData must include gps_lng');
        });

        it('GPS error-code mapping (1, 2, 3), keeps previous good fix on error, and checks isSecureContext', () => {
            assert.match(analyzeModule.formatGpsError({ code: 1 }), /Chrome or Safari/i);
            assert.match(analyzeModule.formatGpsError({ code: 2 }), /Location/i);
            assert.match(analyzeModule.formatGpsError({ code: 3 }), /timed out|try again/i);

            // Keep previous good fix on error
            analyzeModule.setLastGpsFix('14.6051796', '120.962724', 'Browser (Device)');
            mockElements['input-lat'].value = '14.6051796';
            mockElements['input-lng'].value = '120.962724';
            mockElements['gps-source-text'] = makeMockElement({ textContent: 'Source: Browser (Device)' });

            Object.defineProperty(globalThis, 'navigator', {
                value: {
                    geolocation: {
                        getCurrentPosition: (_onSuccess, onError) => {
                            onError({ code: 3 });
                        }
                    }
                },
                configurable: true,
                writable: true
            });
            analyzeModule.requestGpsLocation();
            assert.strictEqual(mockElements['input-lat'].value, '14.6051796', 'Previous lat fix must be preserved on error');
            assert.strictEqual(mockElements['input-lng'].value, '120.962724', 'Previous lng fix must be preserved on error');
            assert.strictEqual(analyzeModule.getLastGpsFix().lat, '14.6051796');

            // Insecure context check
            analyzeModule.clearGpsState();
            globalThis.window = { isSecureContext: false };
            analyzeModule.requestGpsLocation();
            assert.match(mockElements['gps-source-text'].textContent, /secure|HTTPS/i);
            delete globalThis.window;
        });

        it('In-app browser UA detection (FBAN, FBAV, FB_IAB, Messenger, Instagram, Line) and input attributes', () => {
            assert.strictEqual(analyzeModule.isInAppBrowser('Mozilla/5.0 [FBAN/FBIOS;FBAV/400.0]'), true);
            assert.strictEqual(analyzeModule.isInAppBrowser('Mozilla/5.0 [FB_IAB/FB4A]'), true);
            assert.strictEqual(analyzeModule.isInAppBrowser('Mozilla/5.0 MessengerForiOS'), false);
            assert.strictEqual(analyzeModule.isInAppBrowser('Mozilla/5.0 Messenger/100.0'), true);
            assert.strictEqual(analyzeModule.isInAppBrowser('Mozilla/5.0 Instagram 300.0'), true);
            assert.strictEqual(analyzeModule.isInAppBrowser('Mozilla/5.0 Line/13.0.0'), true);
            assert.strictEqual(analyzeModule.isInAppBrowser('Mozilla/5.0 Chrome/120.0.0.0 Mobile Safari/537.36'), false);

            const analyzeHtml = fs.readFileSync(path.resolve(process.cwd(), 'frontend/pages/analyze.html'), 'utf8');
            const latTag = analyzeHtml.match(/<input[^>]*id="input-lat"[^>]*>/i)?.[0] || '';
            const lngTag = analyzeHtml.match(/<input[^>]*id="input-lng"[^>]*>/i)?.[0] || '';
            for (const tag of [latTag, lngTag]) {
                assert.match(tag, /autocomplete="off"/i);
                assert.match(tag, /inputmode="decimal"/i);
                assert.match(tag, /spellcheck="false"/i);
                assert.match(tag, /name="wf_coord_(lat|lng)"/i);
            }
            assert.match(analyzeHtml, /For GPS and uploads, open this page in Chrome or Safari \(⋯ → Open in browser\)\./);
        });

        it('updateTilesFromJson shows "Risk score 44.6 (Moderate)" under Overall Safety and labels per-wire risk separately', () => {
            mockElements['safety-badge'] = makeMockElement();
            mockElements['safety-risk-summary'] = makeMockElement();
            mockElements['safety-reasons'] = makeMockElement();
            mockElements['action-text'] = makeMockElement();
            mockElements['crs-value'] = makeMockElement();
            mockElements['crs-band'] = makeMockElement();
            mockElements['wire-crs-summary'] = makeMockElement();
            mockElements['crs-fill'] = makeMockElement();
            mockElements['crs-gauge'] = makeMockElement();
            mockElements['sag-value'] = makeMockElement();
            mockElements['clearance-value'] = makeMockElement();
            mockElements['vni-value'] = makeMockElement();
            mockElements['vni-spectrum'] = makeMockElement();
            mockElements['wire-list'] = makeMockElement();
            mockElements['analysis-metadata'] = makeMockElement();

            const analysis = {
                id: 99,
                status: 'complete',
                risk_score: 44.6,
                max_crs: 100.0,
                crs: 0.446,
                crs_band: 'Moderate',
                ground_clearance_status: 'near',
                sag_m: 1.25,
                ground_clearance_m: 4.5,
                vni: 0.65,
                wires: [{ class_name: 'power_line', confidence: 0.92, crs: 100.0, sag_m: 1.25 }]
            };

            analyzeModule.updateTilesFromJson(analysis);

            assert.match(mockElements['safety-badge'].innerHTML, /Critical/);
            assert.strictEqual(mockElements['safety-risk-summary'].textContent, 'Risk score 44.6 (Moderate)');
            assert.strictEqual(mockElements['crs-value'].textContent, 'Moderate Risk');
            assert.strictEqual(mockElements['crs-band'].textContent, 'Moderate (44.6)');
            assert.strictEqual(mockElements['wire-crs-summary'].textContent, 'Highest wire risk: Critical (100.0)');
            assert.strictEqual(mockElements['crs-fill'].style.width, '44.6%');
        });

        it('429 rate-limit response shows detail message (escaped) and retains queued files and GPS fix for retry', async () => {
            const fakeFile = { name: 'rate_limited.jpg', size: 2048, type: 'image/jpeg' };
            analyzeModule.setCurrentFiles([fakeFile]);
            analyzeModule.setFileThumbURLs(['blob:http://localhost/rate_limited.jpg']);
            analyzeModule.setLastGpsFix('14.6051796', '120.962724', 'device');
            mockElements['input-lat'].value = '14.6051796';
            mockElements['input-lng'].value = '120.962724';
            mockElements['thumbnails'].innerHTML = '<div>thumb_preview</div>';
            mockElements['file-input'].value = 'C:\\fakepath\\rate_limited.jpg';

            const rateLimitDetail = 'Too many uploads. Please wait a few minutes and try again.';
            globalThis.fetch = async () => ({
                ok: false,
                status: 429,
                headers: { get: (k) => (k.toLowerCase() === 'retry-after' ? '895' : null) },
                json: async () => ({ detail: rateLimitDetail })
            });

            await analyzeModule.startAnalysis();

            assert.strictEqual(mockElements['status-reason'].textContent, rateLimitDetail);
            assert.strictEqual(analyzeModule.getCurrentFiles().length, 1);
            assert.strictEqual(analyzeModule.getCurrentFiles()[0].name, 'rate_limited.jpg');
            assert.strictEqual(analyzeModule.getLastGpsFix().lat, '14.6051796');
            assert.strictEqual(analyzeModule.getLastGpsFix().lng, '120.962724');
            assert.strictEqual(analyzeModule.getLastGpsFix().source, 'device');
            assert.strictEqual(revokedUrls.length, 0);
        });

        it('gps_source sends "device" for unedited browser fix, "manual" after editing lat/lng, and maps labels consistently', async () => {
            assert.strictEqual(analyzeModule.formatGpsSourceLabel('device'), 'Device GPS');
            assert.strictEqual(analyzeModule.formatGpsSourceLabel('manual'), 'Manual entry');
            assert.strictEqual(analyzeModule.formatGpsSourceLabel('exif'), 'Photo EXIF');

            // 1. Browser geolocation fix -> unedited -> gps_source="device" (single)
            let latInputHandler = null;
            mockElements['input-lat'] = makeMockElement({
                value: '',
                addEventListener: (ev, cb) => { if (ev === 'input') latInputHandler = cb; }
            });
            mockElements['input-lng'] = makeMockElement({ value: '' });
            mockElements['gps-source-text'] = makeMockElement({ textContent: '' });
            mockElements['btn-gps'] = makeMockElement({ disabled: false });

            analyzeModule.setupGPSHandlers();

            Object.defineProperty(globalThis, 'navigator', {
                value: {
                    geolocation: {
                        getCurrentPosition: (onSuccess) => {
                            onSuccess({ coords: { latitude: 14.605, longitude: 120.962 } });
                        }
                    }
                },
                configurable: true,
                writable: true
            });

            analyzeModule.requestGpsLocation();
            assert.strictEqual(mockElements['gps-source-text'].textContent, 'Source: Device GPS');
            assert.strictEqual(analyzeModule.getLastGpsFix().source, 'device');

            let singleEntries = [];
            analyzeModule.setCurrentFiles([{ name: 'device_single.jpg' }]);
            globalThis.fetch = async (url, opts) => {
                if (url.includes('/api/analysis/upload')) {
                    singleEntries = opts.body.entries;
                    return { ok: true, status: 200, json: async () => ({ analysis_id: 801, batch_id: 81 }) };
                }
                return { ok: true, status: 200, json: async () => ({ status: 'complete' }) };
            };
            await analyzeModule.startAnalysis();
            analyzeModule.stopAllPolling();
            assert.ok(singleEntries.some(([k, v]) => k === 'gps_source' && v === 'device'), 'Unedited browser fix must send gps_source=device');

            // 2. Browser geolocation fix -> user edits lat -> gps_source="manual" (batch)
            analyzeModule.requestGpsLocation();
            assert.strictEqual(analyzeModule.getLastGpsFix().source, 'device');
            mockElements['input-lat'].value = '14.610';
            if (latInputHandler) latInputHandler();
            assert.strictEqual(analyzeModule.getLastGpsFix().source, 'manual');
            assert.strictEqual(mockElements['gps-source-text'].textContent, 'Source: Manual entry');

            let batchEntries = [];
            analyzeModule.setCurrentFiles([{ name: 'b1.jpg' }, { name: 'b2.jpg' }]);
            globalThis.fetch = async (url, opts) => {
                if (url.endsWith('/api/batches')) {
                    batchEntries = opts.body.entries;
                    return { ok: true, status: 200, json: async () => ({ batch_id: 82 }) };
                }
                if (url.includes('/api/batches/82')) {
                    return { ok: true, status: 200, json: async () => ({ analyses: [{ id: 802, status: 'queued' }, { id: 803, status: 'queued' }] }) };
                }
                return { ok: true, status: 200, json: async () => ({ status: 'complete' }) };
            };
            await analyzeModule.startAnalysis();
            analyzeModule.stopAllPolling();
            assert.ok(batchEntries.some(([k, v]) => k === 'gps_source' && v === 'manual'), 'Edited coordinate must send gps_source=manual');
        });
    });

    describe('Demo Flow Batch A Fixes (#3, #6, #8)', () => {
        const analyzeHtmlPath = path.resolve(process.cwd(), 'frontend/pages/analyze.html');
        const analyzeHtml = fs.readFileSync(analyzeHtmlPath, 'utf8');

        it('#3: Batch tabs use /api/analysis/.../original thumbnails and avoid revoked blob URLs', () => {
            assert.match(
                analyzeJs,
                /thumb:\s*`\/api\/analysis\/\$\{data\.analysis_id\}\/original`/,
                'Single upload must point thumb to /api/analysis/${data.analysis_id}/original'
            );
            assert.match(
                analyzeJs,
                /thumb:\s*`\/api\/analysis\/\$\{a\.id\}\/original`/,
                'Batch upload must point thumb to /api/analysis/${a.id}/original'
            );
            assert.doesNotMatch(
                analyzeJs,
                /thumb:\s*fileThumbURLs\[/,
                'Must not assign fileThumbURLs to batchItems thumbs'
            );
        });

        it('#6: Global stepper defines #step-results and calls updateGlobalStepper("results") on terminal state', () => {
            // Check HTML has step-results
            assert.match(analyzeHtml, /id="step-results"/, 'analyze.html must contain #step-results element');
            assert.match(analyzeHtml, /<span[^>]*>Results<\/span>/, 'analyze.html must label step 3 as Results');

            // Check JS calls updateGlobalStepper('results')
            assert.match(
                analyzeJs,
                /updateGlobalStepper\(['"]results['"]\)/,
                'analyze.js must invoke updateGlobalStepper("results") on terminal state'
            );
        });

        it('#8: Numeric formatting uses typeof === "number" and eliminates "!== null ? *.toFixed" pattern', () => {
            // Verify no `!== null ? *.toFixed` pattern remains in analyze.js
            assert.doesNotMatch(
                analyzeJs,
                /!==\s*null\s*\?\s*[a-zA-Z0-9_.]+\.toFixed/,
                'Must eliminate `!== null ? *.toFixed` pattern throughout analyze.js'
            );
            assert.match(
                analyzeJs,
                /typeof\s+w\.crs\s*===\s*['"]number['"]\s*\?\s*w\.crs\.toFixed/,
                'w.crs must check typeof === "number"'
            );
            assert.match(
                analyzeJs,
                /typeof\s+w\.sag_m\s*===\s*['"]number['"]\s*\?\s*`\$\{w\.sag_m\.toFixed/,
                'w.sag_m must check typeof === "number"'
            );
            assert.match(
                analyzeJs,
                /typeof\s+analysis\.vni\s*===\s*['"]number['"]\s*\?\s*analysis\.vni\.toFixed/,
                'analysis.vni must check typeof === "number"'
            );
        });

        it('CRS tile tick labels in analyze.html are 0, 33, 66, 85, 100', () => {
            const crsTileMatch = analyzeHtml.match(/id="tile-crs"[\s\S]*?<!-- Wire Sag & Clearance -->/);
            assert.ok(crsTileMatch, 'analyze.html must contain #tile-crs section');
            const tickMatches = [...crsTileMatch[0].matchAll(/<span>\s*([0-9.]+)\s*<\/span>/g)].map(m => m[1]);
            assert.deepStrictEqual(tickMatches, ['0', '33', '66', '85', '100']);
        });
    });
});
