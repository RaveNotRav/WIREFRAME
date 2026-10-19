import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { formatRelativeTime, getRiskBand, getRiskScorePill, getHeatmapRiskClass, DEFAULT_RISK_BANDS } from './lib/formatters.js';

describe('Admin Dashboard Polish Tests', () => {
    const dashboardHtmlPath = path.resolve(process.cwd(), 'frontend/pages/dashboard.html');
    const dashboardJsPath = path.resolve(process.cwd(), 'frontend/assets/js/dashboard.js');
    const dashboardHtml = fs.readFileSync(dashboardHtmlPath, 'utf8');
    const dashboardJs = fs.readFileSync(dashboardJsPath, 'utf8');

    describe('Container Alignment & Grid Structure', () => {
        it('KPI row, middle row, and Trend Analysis share the unified bento-grid', () => {
            // Verify unified bento-grid container in overview section
            assert.match(
                dashboardHtml,
                /<div\s+class="bento-grid"\s+id="dashboard-overview-grid"/,
                'dashboard.html must contain unified #dashboard-overview-grid bento-grid'
            );

            // Verify KPI tiles are bento-s (span 3 each -> 4 * 3 = 12 columns)
            const kpiTiles = dashboardHtml.match(/class="[^"]*kpi-card[^"]*bento-s[^"]*"/g);
            assert.ok(kpiTiles && kpiTiles.length === 4, 'All 4 KPI cards must have bento-s (span 3)');

            // Verify Trend tile is bento-xxl (span 12)
            assert.match(
                dashboardHtml,
                /<div\s+class="[^"]*bento-tile[^"]*bento-xxl[^"]*"\s+id="trend-tile"/,
                'Trend Analysis tile must have bento-xxl (span 12)'
            );

            // Verify .bento-grid has padding: 0 when inside #dashboard-container to prevent inset misalignment
            assert.match(
                dashboardHtml,
                /#dashboard-container\s+\.bento-grid\s*\{[^}]*padding\s*:\s*0/i,
                '#dashboard-container .bento-grid must have padding: 0 to align with outer container'
            );
        });

        it('Middle row has 3 equal columns (span 4 each) including recent analyses tile', () => {
            // 3 tiles in middle row must use bento-m (span 4 -> 3 * 4 = 12 columns)
            assert.match(dashboardHtml, /id="heatmap-tile"[^>]*class="[^"]*bento-m[^"]*"|class="[^"]*bento-m[^"]*"[^>]*id="heatmap-tile"/);
            assert.match(dashboardHtml, /id="distribution-tile"[^>]*class="[^"]*bento-m[^"]*"|class="[^"]*bento-m[^"]*"[^>]*id="distribution-tile"/);
            assert.match(dashboardHtml, /id="recent-analyses-tile"[^>]*class="[^"]*bento-m[^"]*"|class="[^"]*bento-m[^"]*"[^>]*id="recent-analyses-tile"/);

            // Must contain recent analyses list container
            assert.match(dashboardHtml, /id="recent-analyses-container"/);
        });
    });

    describe('KPI Tiles & Inline SVG Icons', () => {
        it('Each KPI tile contains an inline SVG icon using currentColor and var(--heading-color)', () => {
            // Check all 4 KPI cards for icon SVGs
            assert.match(dashboardHtml, /id="kpi-card-total"[\s\S]*?<polygon points="12 2 2 7 12 12 22 7 12 2"/, 'Total Analyses must have layers/stack SVG');
            assert.match(dashboardHtml, /id="kpi-card-critical"[\s\S]*?<path d="m21\.73 18-8-14a2 2 0 0 0-3\.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1\.73-3Z"/, 'High/Critical must have alert-triangle SVG');
            assert.match(dashboardHtml, /id="kpi-card-avg-risk"[\s\S]*?<path d="M3\.34 19a10 10 0 1 1 17\.32 0"/, 'Avg Risk Score must have gauge SVG');
            assert.match(dashboardHtml, /id="kpi-card-today"[\s\S]*?<rect width="18" height="18" x="3" y="4" rx="2" ry="2"/, 'Processed Today must have calendar SVG');

            // Verify icon styling uses currentColor and var(--heading-color)
            assert.match(dashboardHtml, /\.kpi-icon\s*\{[^}]*color\s*:\s*var\(--heading-color\)/i, '.kpi-icon must use var(--heading-color)');
            assert.match(dashboardHtml, /\.kpi-icon\s+svg\s*\{[^}]*stroke\s*:\s*currentColor/i, '.kpi-icon svg must stroke currentColor');
        });

        it('KPI critical card receives risk accent when HIGH/CRITICAL > 0', () => {
            // CSS rule for has-critical-risk
            assert.match(
                dashboardHtml,
                /\.kpi-card\.has-critical-risk\s*\{[^}]*border-(top|left)\s*:\s*3?4?px\s+solid\s+var\(--risk-critical\)/i,
                'dashboard.html must define .kpi-card.has-critical-risk with var(--risk-critical) accent'
            );

            // JS logic toggling has-critical-risk
            assert.match(
                dashboardJs,
                /criticalCard\.classList\.add\(['"]has-critical-risk['"]\)/,
                'dashboard.js must add has-critical-risk when critical count > 0'
            );
            assert.match(
                dashboardJs,
                /criticalCard\.classList\.remove\(['"]has-critical-risk['"]\)/,
                'dashboard.js must remove has-critical-risk when count is 0'
            );
        });
    });

    describe('Empty States (0 analyses)', () => {
        it('Heatmap, Wire Distribution, Recent Analyses, and Trend Analysis show "No analyses yet"', () => {
            // Check that all 4 render functions have "No analyses yet"
            const noAnalysesMatches = dashboardJs.match(/No analyses yet/g);
            assert.ok(noAnalysesMatches && noAnalysesMatches.length >= 4, 'All 4 overview tiles must contain "No analyses yet" in dashboard.js');

            // Check that Recent Analyses empty state contains "Go to Analyze" link to /analyze.html
            assert.match(
                dashboardJs,
                /<a\s+href="\/analyze\.html"[^>]*class="neu-button[^"]*"[^>]*>Go to Analyze<\/a>/,
                'Recent Analyses empty state must contain Go to Analyze link to /analyze.html with neu-button class'
            );
        });
    });

    describe('Recent Analyses Rows & formatRelativeTime', () => {
        it('Recent list uses escapeHtml, DocumentFragment, lazy/async images, and detail link', () => {
            assert.match(dashboardJs, /function\s+renderRecentAnalyses\(/, 'renderRecentAnalyses must be defined');
            assert.match(dashboardJs, /createDocumentFragment\(\)/, 'renderRecentAnalyses must use DocumentFragment');
            assert.match(dashboardJs, /escapeHtml\(/, 'renderRecentAnalyses must use escapeHtml');
            assert.match(dashboardJs, /loading="lazy"\s+decoding="async"|decoding="async"\s+loading="lazy"/, 'thumbnails must have lazy loading and async decoding');
            assert.match(dashboardJs, /href\s*=\s*[`"']\/analyze\.html\?id=/, 'row must link to /analyze.html?id=');
        });

        it('loadAllAnalyses awaits res.json() before accessing data.data', () => {
            // Static check that assignment to data via await res.json() occurs before data.data access
            const loadAllAnalysesMatch = dashboardJs.match(/async\s+function\s+loadAllAnalyses\(\)\s*\{([\s\S]*?)\n\s*\}/);
            assert.ok(loadAllAnalysesMatch, 'loadAllAnalyses function must be present');
            const fnBody = loadAllAnalysesMatch[1];
            const jsonIdx = fnBody.indexOf('await res.json()');
            const dataAccessIdx = fnBody.indexOf('data.data');
            assert.ok(jsonIdx !== -1, 'loadAllAnalyses must await res.json()');
            assert.ok(dataAccessIdx !== -1, 'loadAllAnalyses must access data.data');
            assert.ok(jsonIdx < dataAccessIdx, 'await res.json() must precede data.data access to avoid ReferenceError');
        });

        it('Reuses canonical getRiskScorePill, verifies thumbnail filename condition, and avoids global leaks', () => {
            // No custom getRecentRiskPill duplicate
            assert.ok(!dashboardJs.includes('function getRecentRiskPill'), 'getRecentRiskPill should not be defined; reuse getRiskScorePill');
            assert.match(dashboardJs, /getRiskScorePill\(item\.risk_score\)/, 'renderRecentAnalyses must reuse getRiskScorePill(item.risk_score)');

            // Thumbnail checks item.filename like History does
            assert.match(dashboardJs, /item\.filename\s*\?\s*[`"']\/api\/analysis\/\${item\.id}\/original[`"']\s*:\s*['"]{2}/, 'Thumbnail source must check item.filename');

            // No window.renderRecentAnalyses global leak
            assert.ok(!dashboardJs.includes('window.renderRecentAnalyses'), 'window.renderRecentAnalyses should not be leaked on window');
        });

        it('formatRelativeTime formats durations correctly', () => {
            const now = new Date('2026-10-18T12:00:00Z').getTime();

            // Null or invalid
            assert.equal(formatRelativeTime(null, now), '--');
            assert.equal(formatRelativeTime('', now), '--');

            // Just now (< 1m)
            assert.equal(formatRelativeTime('2026-10-18T11:59:30Z', now), 'just now');

            // Minutes ago (< 60m)
            assert.equal(formatRelativeTime('2026-10-18T11:45:00Z', now), '15m ago');

            // Hours ago (< 24h)
            assert.equal(formatRelativeTime('2026-10-18T10:00:00Z', now), '2h ago');

            // Days ago (< 30d)
            assert.equal(formatRelativeTime('2026-10-15T12:00:00Z', now), '3d ago');

            // Months ago (< 12mo)
            assert.equal(formatRelativeTime('2026-08-18T12:00:00Z', now), '2mo ago');

            // Years ago
            assert.equal(formatRelativeTime('2024-10-18T12:00:00Z', now), '2y ago');
        });
    });

    describe('KPI Alignment, Heatmap Empty State & Typography Polish', () => {
        it('All 4 KPI tiles share same min-height and center content vertically and horizontally', () => {
            // Check all 4 KPI tiles have min-height and vertical/horizontal centering
            assert.match(dashboardHtml, /\.kpi-card\s*\{[^}]*min-height\s*:\s*120px/i, '.kpi-card must have min-height: 120px');
            assert.match(dashboardHtml, /\.kpi-card\s*\{[^}]*justify-content\s*:\s*center/i, '.kpi-card must use justify-content: center');
            assert.match(dashboardHtml, /\.kpi-card\s*\{[^}]*align-items\s*:\s*center/i, '.kpi-card must use align-items: center');
        });

        it('No sparklines in dashboard.html or dashboard.js', () => {
            assert.ok(!dashboardHtml.toLowerCase().includes('sparkline'), 'dashboard.html must not contain any sparkline references');
            assert.ok(!dashboardJs.toLowerCase().includes('sparkline'), 'dashboard.js must not contain any sparkline references');
            assert.ok(!dashboardHtml.includes('has-sparklines'), 'dashboard.html must not contain has-sparklines');
            assert.ok(!dashboardJs.includes('has-sparklines'), 'dashboard.js must not contain has-sparklines');
        });

        it('KPI label uses --font-size-sm and icon is 20px', () => {
            assert.match(dashboardHtml, /\.kpi-label\s*\{[^}]*font-size\s*:\s*var\(--font-size-sm\)/i, '.kpi-label must use var(--font-size-sm)');
            assert.match(dashboardHtml, /\.kpi-icon\s+svg\s*\{[^}]*width\s*:\s*20px/i, '.kpi-icon svg must have width: 20px');
            assert.match(dashboardHtml, /\.kpi-icon\s+svg\s*\{[^}]*height\s*:\s*20px/i, '.kpi-icon svg must have height: 20px');
        });

        it('Heatmap empty state spans full grid (grid-column: 1 / -1 or non-grid container)', () => {
            assert.match(
                dashboardHtml,
                /\.heatmap-grid\s+\.dashboard-empty-state\s*\{[^}]*grid-column\s*:\s*1\s*\/\s*-1/i,
                'Heatmap empty state must have grid-column: 1 / -1'
            );
            assert.match(
                dashboardHtml,
                /\.heatmap-grid\.(is-empty|empty)|:has\(\.dashboard-empty-state\)/,
                'Heatmap grid must support empty state container styling'
            );
        });

        it('Empty state icons are 32px and text is on one line across all 4 overview tiles', () => {
            // Check that Heatmap, Wire Distribution, Recent Analyses, and Trend all use 32px SVG icons in empty states
            const emptyIcons32 = dashboardJs.match(/<svg\s+width="32"\s+height="32"[^>]*>[\s\S]*?<span>No analyses yet<\/span>/g);
            assert.ok(emptyIcons32 && emptyIcons32.length === 4, 'All 4 empty states must use width="32" height="32" SVG icons');

            // Text is white-space nowrap and font-size-sm
            assert.match(dashboardHtml, /\.dashboard-empty-state\s+span\s*\{[^}]*white-space\s*:\s*nowrap/i, 'Empty state text must not wrap');
            assert.match(dashboardHtml, /\.dashboard-empty-state\s+span\s*\{[^}]*font-size\s*:\s*var\(--font-size-sm\)/i, 'Empty state text must use var(--font-size-sm)');
        });

        it('View All link uses --font-size-sm and aligns with tile title baseline', () => {
            assert.match(dashboardHtml, /\.bento-tile-link\s*\{[^}]*font-size\s*:\s*var\(--font-size-sm\)/i, '.bento-tile-link must use var(--font-size-sm)');
            assert.match(dashboardHtml, /\.bento-tile-header\s*\{[^}]*align-items\s*:\s*baseline/i, '.bento-tile-header must align baseline');
        });

        it('Heatmap container uses adaptive CSS variables and does not have unconditional display:flex override', () => {
            // .heatmap-grid uses repeat(var(--hm-cols, 6), var(--hm-cell, 1fr)) and grid-auto-rows
            assert.match(
                dashboardHtml,
                /\.heatmap-grid\s*\{[^}]*grid-template-columns\s*:\s*repeat\(var\(--hm-cols,\s*6\),\s*var\(--hm-cell,\s*1fr\)\)/i,
                '.heatmap-grid must define adaptive columns with var(--hm-cols, 6)'
            );
            assert.match(
                dashboardHtml,
                /\.heatmap-grid\s*\{[^}]*grid-auto-rows\s*:\s*var\(--hm-cell,\s*auto\)/i,
                '.heatmap-grid must define grid-auto-rows with var(--hm-cell, auto)'
            );
            assert.match(
                dashboardHtml,
                /\.heatmap-grid\s*\{[^}]*align-content\s*:\s*start/i,
                '.heatmap-grid must have align-content: start'
            );
            assert.match(
                dashboardHtml,
                /\.heatmap-cell\s*\{[^}]*width\s*:\s*var\(--hm-cell\)/i,
                '.heatmap-cell must have width: var(--hm-cell)'
            );

            // Verify no rule sets display: flex on #heatmap-container except when qualified with is-empty or empty state
            const styleBlocks = dashboardHtml.match(/([^{}]+)\{([^{}]+)\}/g) || [];
            for (const block of styleBlocks) {
                const [selectorPart, rulePart] = block.split('{');
                if (selectorPart.includes('#heatmap-container') && /display\s*:\s*flex/i.test(rulePart)) {
                    assert.ok(
                        selectorPart.includes('.is-empty') || selectorPart.includes(':has('),
                        `#heatmap-container must only have display: flex when empty, found in: ${selectorPart}`
                    );
                }
            }

            // Verify #heatmap-container by itself only sets flex: 1 without display: flex
            assert.match(
                dashboardHtml,
                /#heatmap-container\s*\{[^}]*flex\s*:\s*1\s*;?\s*\}/i,
                '#heatmap-container should only set flex: 1 without display: flex'
            );
        });

        it('dashboard.js computes adaptive layout, sets --hm-cols and --hm-cell, and sets up ResizeObserver', () => {
            assert.match(dashboardJs, /import\s*\{\s*computeHeatmapLayout\s*\}\s*from\s*['"]\.\/lib\/heatmap_layout\.js['"]/, 'dashboard.js must import computeHeatmapLayout');
            assert.match(dashboardJs, /container\.style\.setProperty\(['"]--hm-cols['"]/, 'dashboard.js must set --hm-cols');
            assert.match(dashboardJs, /container\.style\.setProperty\(['"]--hm-cell['"]/, 'dashboard.js must set --hm-cell');
            assert.match(dashboardJs, /new\s+ResizeObserver\(/, 'dashboard.js must create ResizeObserver');
            assert.match(dashboardJs, /requestAnimationFrame\(/, 'ResizeObserver must be throttled with requestAnimationFrame');
        });
    });

    describe('Risk Band Consistency & Query Parity', () => {
        it('Recent Analyses and dashboard stats query verified inspector records only (&is_public=0&status=complete)', () => {
            // Verify loadAllAnalyses queries is_public=0 and status=complete
            assert.match(
                dashboardJs,
                /\/api\/analysis\?limit=100&is_public=0&status=complete/,
                'loadAllAnalyses must include is_public=0 and status=complete'
            );
            // Verify realCheckRes queries is_public=0 and status=complete
            assert.match(
                dashboardJs,
                /\/api\/analysis\?is_mock=0&is_public=0&status=complete/,
                'realCheckRes must check verified complete records only'
            );
        });

        it('Same score maps to the same risk band in both getRiskScorePill and getHeatmapRiskClass', () => {
            const testScores = [
                { score: 0.0, expectedBand: 'Low', expectedPillClass: 'neu-pill-safety-safe', expectedHeatmapClass: 'risk-Low' },
                { score: 15.0, expectedBand: 'Low', expectedPillClass: 'neu-pill-safety-safe', expectedHeatmapClass: 'risk-Low' },
                { score: 27.0, expectedBand: 'Low', expectedPillClass: 'neu-pill-safety-safe', expectedHeatmapClass: 'risk-Low' },
                { score: 28.6, expectedBand: 'Low', expectedPillClass: 'neu-pill-safety-safe', expectedHeatmapClass: 'risk-Low' },
                { score: 33.0, expectedBand: 'Low', expectedPillClass: 'neu-pill-safety-safe', expectedHeatmapClass: 'risk-Low' },
                { score: 33.1, expectedBand: 'Moderate', expectedPillClass: 'neu-pill-safety-caution', expectedHeatmapClass: 'risk-Moderate' },
                { score: 50.0, expectedBand: 'Moderate', expectedPillClass: 'neu-pill-safety-caution', expectedHeatmapClass: 'risk-Moderate' },
                { score: 66.0, expectedBand: 'Moderate', expectedPillClass: 'neu-pill-safety-caution', expectedHeatmapClass: 'risk-Moderate' },
                { score: 66.1, expectedBand: 'High', expectedPillClass: 'neu-pill-safety-unsafe', expectedHeatmapClass: 'risk-High' },
                { score: 75.0, expectedBand: 'High', expectedPillClass: 'neu-pill-safety-unsafe', expectedHeatmapClass: 'risk-High' },
                { score: 85.0, expectedBand: 'High', expectedPillClass: 'neu-pill-safety-unsafe', expectedHeatmapClass: 'risk-High' },
                { score: 85.1, expectedBand: 'Critical', expectedPillClass: 'neu-pill-safety-critical', expectedHeatmapClass: 'risk-Critical' },
                { score: 100.0, expectedBand: 'Critical', expectedPillClass: 'neu-pill-safety-critical', expectedHeatmapClass: 'risk-Critical' },
            ];

            for (const { score, expectedBand, expectedPillClass, expectedHeatmapClass } of testScores) {
                const band = getRiskBand(score, DEFAULT_RISK_BANDS);
                assert.equal(band.label, expectedBand, `Score ${score} must map to band ${expectedBand}`);

                const pillHtml = getRiskScorePill(score, DEFAULT_RISK_BANDS);
                assert.ok(pillHtml.includes(expectedBand), `Score ${score} pill must include label ${expectedBand}`);
                assert.ok(pillHtml.includes(expectedPillClass), `Score ${score} pill must include class ${expectedPillClass}`);

                const heatmapClass = getHeatmapRiskClass(expectedBand);
                assert.equal(heatmapClass, expectedHeatmapClass, `Band ${expectedBand} must map to heatmap class ${expectedHeatmapClass}`);
            }
        });

        it('Does not use hardcoded 25 / 50 / 75 thresholds in getRiskScorePill', () => {
            assert.ok(!dashboardJs.includes('score >= 75'), 'Must not have hardcoded score >= 75 in dashboard.js');
            assert.ok(!dashboardJs.includes('score >= 50'), 'Must not have hardcoded score >= 50 in dashboard.js');
            assert.ok(!dashboardJs.includes('score >= 25'), 'Must not have hardcoded score >= 25 in dashboard.js');
        });

        it('getRiskBand handles small scores (0-1) on 0-100 scale without ambiguity and window.getHeatmapRiskClass is not leaked', () => {
            const scoresToTest = [
                { score: 0, expected: 'Low' },
                { score: 0.8, expected: 'Low' },
                { score: 1.0, expected: 'Low' },
                { score: 28.6, expected: 'Low' },
                { score: 33.5, expected: 'Moderate' },
                { score: 66.0, expected: 'Moderate' },
                { score: 66.5, expected: 'High' },
                { score: 85.0, expected: 'High' },
                { score: 90, expected: 'Critical' },
                { score: 100, expected: 'Critical' },
            ];

            for (const { score, expected } of scoresToTest) {
                const band = getRiskBand(score);
                assert.ok(band, `Expected band for score ${score}`);
                assert.equal(band.label, expected, `Score ${score} must map to ${expected}`);
            }

            assert.ok(!dashboardJs.includes('window.getHeatmapRiskClass ='), 'dashboard.js must not assign to window.getHeatmapRiskClass');
        });
    });
});
