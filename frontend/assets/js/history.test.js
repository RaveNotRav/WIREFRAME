import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { resolveHistoryPublicFilter, buildHistoryBadgesHtml, PUBLIC_SUBMISSION_BADGE_TEXT } from './history.js';

describe('Public Submissions & Detected Wires Layout Tests', () => {
    const historyHtml = fs.readFileSync(path.resolve(process.cwd(), 'frontend/pages/history.html'), 'utf8');
    const analyzeHtml = fs.readFileSync(path.resolve(process.cwd(), 'frontend/pages/analyze.html'), 'utf8');
    const analyzeCss = fs.readFileSync(path.resolve(process.cwd(), 'frontend/assets/css/analyze.css'), 'utf8');
    const analyzeJs = fs.readFileSync(path.resolve(process.cwd(), 'frontend/assets/js/analyze.js'), 'utf8');

    it('history.js defaults to showing all submissions (verified + Public Submission)', () => {
        assert.equal(resolveHistoryPublicFilter('all'), undefined);
        assert.equal(resolveHistoryPublicFilter(), undefined);
        assert.equal(resolveHistoryPublicFilter('verified'), 0);
        assert.equal(resolveHistoryPublicFilter('public'), 1);

        const pubBadge = buildHistoryBadgesHtml({ is_public: 1, is_mock: 0 });
        assert.ok(pubBadge.includes(PUBLIC_SUBMISSION_BADGE_TEXT), 'Must include Public Submission badge');
        assert.ok(!pubBadge.includes('MOCK DATA'));

        const bothBadge = buildHistoryBadgesHtml({ is_public: 1, is_mock: 1 });
        assert.ok(bothBadge.includes('MOCK DATA'));
        assert.ok(bothBadge.includes(PUBLIC_SUBMISSION_BADGE_TEXT));

        // history.html defaults #tab-all to neu-pressed so Public Submissions appear immediately
        assert.match(historyHtml, /class="neu-button neu-pressed"[^>]*id="tab-all"/);
    });

    it('analyze.css & analyze.html prevent squished #wire-list and enforce flex-shrink: 0 on wire cards', () => {
        assert.match(analyzeHtml, /href="\/assets\/css\/analyze\.css/);
        assert.match(analyzeCss, /#wire-list\s*\{[\s\S]*?min-height\s*:\s*240px/i);
        assert.match(analyzeCss, /#wire-list\s*\{[\s\S]*?max-height\s*:\s*460px/i);
        assert.match(analyzeCss, /#wire-list\s*\{[\s\S]*?overflow-y\s*:\s*auto/i);
        assert.match(analyzeCss, /#wire-list\s*\{[\s\S]*?display\s*:\s*flex/i);
        assert.match(analyzeCss, /#wire-list\s*\{[\s\S]*?flex-direction\s*:\s*column/i);
        assert.match(analyzeCss, /#wire-list\s*\{[\s\S]*?gap\s*:\s*0\.65rem/i);
        assert.match(analyzeCss, /flex-shrink\s*:\s*0/);
        assert.match(analyzeJs, /div\.style\.flexShrink\s*=\s*['"]0['"]/);
    });
});
