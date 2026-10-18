// performance.test.js
import test, { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

describe('Performance & Paint Optimization Tests', () => {
    const pagesDir = path.resolve(process.cwd(), 'frontend/pages');
    const cssDir = path.resolve(process.cwd(), 'frontend/assets/css');
    const jsDir = path.resolve(process.cwd(), 'frontend/assets/js');

    const htmlFiles = fs.readdirSync(pagesDir).filter(f => f.endsWith('.html'));
    const cssFiles = fs.readdirSync(cssDir).filter(f => f.endsWith('.css'));

    it('All pages have inline theme init script in <head> before stylesheets to prevent theme flash', () => {
        for (const file of htmlFiles) {
            const content = fs.readFileSync(path.join(pagesDir, file), 'utf8');
            const headMatch = content.match(/<head[\s\S]*?<\/head>/i);
            assert.ok(headMatch, `${file} must contain a <head> tag`);
            const head = headMatch[0];

            // Must have inline script reading wireframe-theme and setting data-theme
            const scriptMatch = head.match(/<script>[\s\S]*?localStorage\.getItem\(['"]wireframe-theme['"]\)[\s\S]*?document\.documentElement\.setAttribute\(['"]data-theme['"][\s\S]*?<\/script>/i);
            assert.ok(scriptMatch, `${file} must contain inline theme script setting data-theme from wireframe-theme`);

            // The inline script must appear before any <link rel="stylesheet">
            const scriptIndex = head.indexOf(scriptMatch[0]);
            const firstStylesheetIndex = head.search(/<link[^>]+rel=["']stylesheet["']/i);
            assert.ok(firstStylesheetIndex > -1, `${file} must contain a stylesheet link`);
            assert.ok(
                scriptIndex < firstStylesheetIndex,
                `In ${file}, inline theme script must appear before the first stylesheet link`
            );
        }
    });

    it('All pages inline theme script matches theme.js defaults and edge cases exactly', () => {
        for (const file of htmlFiles) {
            const content = fs.readFileSync(path.join(pagesDir, file), 'utf8');
            assert.match(
                content,
                /\/\/\s*Must stay in sync with initTheme\(\) in frontend\/assets\/js\/lib\/theme\.js/,
                `${file} must include comment noting sync with theme.js`
            );
            assert.match(
                content,
                /localStorage\.getItem\(['"]wireframe-theme['"]\)/,
                `${file} must use 'wireframe-theme' storage key`
            );
            assert.match(
                content,
                /if\s*\(\s*!saved\s*\|\|\s*saved\s*===\s*['"]auto['"]\s*\)/,
                `${file} must handle unset or 'auto' using system theme`
            );
            assert.match(
                content,
                /var\s+theme\s*=\s*saved\s*===\s*['"]dark['"]\s*\?\s*['"]dark['"]\s*:\s*['"]light['"]/,
                `${file} must clamp theme to 'dark' or 'light' exactly matching theme.js setTheme`
            );
        }
    });

    it('All pages have class="preload" on <html> and remove it on first frame', () => {
        for (const file of htmlFiles) {
            const content = fs.readFileSync(path.join(pagesDir, file), 'utf8');
            assert.match(
                content,
                /<html[^>]+class=["'][^"']*preload[^"']*["']/i,
                `${file} must have class="preload" on <html>`
            );
            assert.match(
                content,
                /requestAnimationFrame[\s\S]*?document\.documentElement\.classList\.remove\(['"]preload['"]\)/i,
                `${file} inline script must remove preload class via requestAnimationFrame`
            );
        }
    });

    it('All pages have Google Fonts preconnect links in <head>', () => {
        for (const file of htmlFiles) {
            const content = fs.readFileSync(path.join(pagesDir, file), 'utf8');
            assert.match(
                content,
                /<link[^>]+rel=["']preconnect["'][^>]+href=["']https:\/\/fonts\.googleapis\.com["']/i,
                `${file} must preconnect to fonts.googleapis.com`
            );
            assert.match(
                content,
                /<link[^>]+rel=["']preconnect["'][^>]+href=["']https:\/\/fonts\.gstatic\.com["'][^>]*crossorigin/i,
                `${file} must preconnect to fonts.gstatic.com with crossorigin`
            );
        }
    });

    it('Zero transition: all in all CSS and HTML files', () => {
        const transitionAllRegex = /transition\s*:\s*all\b/i;

        for (const file of cssFiles) {
            const content = fs.readFileSync(path.join(cssDir, file), 'utf8');
            assert.doesNotMatch(
                content,
                transitionAllRegex,
                `CSS file ${file} must not contain "transition: all"`
            );
        }

        for (const file of htmlFiles) {
            const content = fs.readFileSync(path.join(pagesDir, file), 'utf8');
            assert.doesNotMatch(
                content,
                transitionAllRegex,
                `HTML file ${file} must not contain "transition: all"`
            );
        }
    });

    it('neumorphism.css suppresses transitions during initial preload', () => {
        const neuCss = fs.readFileSync(path.join(cssDir, 'neumorphism.css'), 'utf8');
        assert.match(
            neuCss,
            /html\.preload\s+\*[\s\S]*?transition\s*:\s*none\s*!important/i,
            'neumorphism.css must suppress transitions for html.preload'
        );
    });

    it('Images in analyze.html and history.html use async decoding and lazy loading', () => {
        const analyzeHtml = fs.readFileSync(path.join(pagesDir, 'analyze.html'), 'utf8');
        assert.match(analyzeHtml, /id="img-annotated"[^>]*decoding="async"|decoding="async"[^>]*id="img-annotated"/, 'img-annotated must have decoding="async"');
        assert.match(analyzeHtml, /id="vni-spectrum"[^>]*decoding="async"|decoding="async"[^>]*id="vni-spectrum"/, 'vni-spectrum must have decoding="async"');

        const historyHtml = fs.readFileSync(path.join(pagesDir, 'history.html'), 'utf8');
        assert.match(historyHtml, /loading="lazy"/i, 'history.html thumbnails must have loading="lazy"');
        assert.match(historyHtml, /decoding="async"/i, 'history.html thumbnails must have decoding="async"');
    });

    it('analyze.js init cannot skip setupUploadHandlers on a rejected fetch (allSettled or try/catch present)', () => {
        const analyzeJs = fs.readFileSync(path.join(jsDir, 'analyze.js'), 'utf8');
        assert.match(
            analyzeJs,
            /Promise\.allSettled\(\s*\[\s*loadUser\(\)\s*,\s*loadSettings\(\)\s*\]\s*\)/,
            'analyze.js must use Promise.allSettled so setupUploadHandlers and init handlers always run even if fetch fails'
        );
        assert.match(analyzeJs, /createDocumentFragment/i, 'analyze.js must use DocumentFragment for batching DOM appends');
    });

    it('history.html awaits fetchUser before loadHistory so role-dependent UI is correct on first render', () => {
        const historyHtml = fs.readFileSync(path.join(pagesDir, 'history.html'), 'utf8');
        assert.match(
            historyHtml,
            /await\s+fetchUser\(\);[\s\S]*?loadHistory\(\s*1\s*\);/,
            'history.html must await fetchUser before loadHistory to prevent role check race'
        );
        assert.match(historyHtml, /createDocumentFragment/i, 'history.html must use DocumentFragment for batching DOM appends');
    });

    it('dashboard.js batches DOM writes and throttles ResizeObserver with rAF', () => {
        const dashboardJs = fs.readFileSync(path.join(jsDir, 'dashboard.js'), 'utf8');
        assert.match(dashboardJs, /requestAnimationFrame\(/, 'dashboard.js must throttle ResizeObserver with requestAnimationFrame');
        assert.match(dashboardJs, /createDocumentFragment/i, 'dashboard.js must use DocumentFragment for batching DOM appends');
    });
});
