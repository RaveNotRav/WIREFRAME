// responsive.test.js
import test, { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

describe('Mobile Responsiveness CSS & HTML Breakpoints', () => {
    const cssDir = path.resolve(process.cwd(), 'frontend/assets/css');
    const pagesDir = path.resolve(process.cwd(), 'frontend/pages');

    it('neumorphism.css has no body overflow-x: hidden band-aid', () => {
        const neuCss = fs.readFileSync(path.join(cssDir, 'neumorphism.css'), 'utf8');
        assert.doesNotMatch(
            neuCss,
            /body[^{]*\{[^}]*overflow-x\s*:\s*hidden/i,
            'body { overflow-x: hidden } band-aid must be removed'
        );
    });

    it('global-header.css contains mobile breakpoint (<768px) with menu toggle and drawer', () => {
        const headerCss = fs.readFileSync(path.join(cssDir, 'global-header.css'), 'utf8');
        assert.match(headerCss, /@media\s*\(\s*max-width\s*:\s*767px\s*\)/i, 'Must define max-width: 767px breakpoint');
        assert.match(headerCss, /\.neu-menu-toggle\s*\{[^}]*display\s*:\s*inline-flex/i, 'Menu toggle must be visible on mobile');
        assert.match(headerCss, /\.neu-mobile-menu/i, 'Mobile menu drawer styles must be present');
        assert.match(headerCss, /\.desktop-nav-link\s*\{[^}]*display\s*:\s*none/i, 'Desktop links must be hidden on mobile');
    });

    it('bento.css contains mobile breakpoint (<768px) with 1-column grid', () => {
        const bentoCss = fs.readFileSync(path.join(cssDir, 'bento.css'), 'utf8');
        assert.match(bentoCss, /@media\s*\(\s*max-width\s*:\s*767px\s*\)/i, 'Must define max-width: 767px breakpoint');
        assert.match(bentoCss, /grid-template-columns\s*:\s*(1fr|minmax\(0,\s*1fr\))/i, 'Must collapse bento grid to 1 column on mobile');
    });

    it('components.css contains mobile breakpoint (<768px) for modals and inputs', () => {
        const compCss = fs.readFileSync(path.join(cssDir, 'components.css'), 'utf8');
        assert.match(compCss, /@media\s*\(\s*max-width\s*:\s*767px\s*\)/i, 'Must define max-width: 767px breakpoint');
        assert.match(compCss, /\.neu-modal\s*\{[^}]*width\s*:\s*calc/i, 'Modal must fit within mobile viewport');
        assert.match(compCss, /font-size\s*:\s*16px/i, 'Form controls must have >= 16px font-size to prevent iOS Safari auto-zoom');
        assert.match(compCss, /\.desktop-only-notice/i, 'Desktop-only notice styles must be present');
    });

    it('index.html contains mobile menu markup, button, and responsive hero', () => {
        const indexHtml = fs.readFileSync(path.join(pagesDir, 'index.html'), 'utf8');
        assert.match(indexHtml, /id="menu-toggle"/, 'index.html must have #menu-toggle');
        assert.match(indexHtml, /id="mobile-menu"/, 'index.html must have #mobile-menu');
        assert.match(indexHtml, /aria-expanded="false"/, '#menu-toggle must have aria-expanded attribute');
        assert.match(indexHtml, /@media\s*\(\s*max-width\s*:\s*768px\s*\)/, 'index.html must have responsive media query');
    });

    it('analyze.html contains mobile menu, responsive tiles, and popup markup', () => {
        const analyzeHtml = fs.readFileSync(path.join(pagesDir, 'analyze.html'), 'utf8');
        assert.match(analyzeHtml, /id="menu-toggle"/, 'analyze.html must have #menu-toggle');
        assert.match(analyzeHtml, /id="mobile-menu"/, 'analyze.html must have #mobile-menu');
        assert.match(analyzeHtml, /id="public-confirmation-modal"/, 'analyze.html must have #public-confirmation-modal');
        assert.match(analyzeHtml, /@media\s*\(\s*max-width\s*:\s*767px\s*\)/, 'analyze.html must have responsive media query');
        assert.match(analyzeHtml, /#btn-download-pdf/, 'analyze.html must style #btn-download-pdf');
        assert.match(analyzeHtml, /class="[^"]*guest-only-nav[^"]*"[^>]*id="nav-login"|id="nav-login"[^>]*class="[^"]*guest-only-nav/, '#nav-login must have guest-only-nav class');
        assert.match(analyzeHtml, /id="mobile-btn-logout"/, '#mobile-menu must contain #mobile-btn-logout');
    });

    it('bento.css restores 6-col for max-width 1024px and scopes 2-col tablet rules to Analyze page', () => {
        const bentoCss = fs.readFileSync(path.join(cssDir, 'bento.css'), 'utf8');
        assert.match(bentoCss, /@media\s*\(\s*max-width\s*:\s*1024px\s*\)\s*\{[^}]*grid-template-columns\s*:\s*repeat\(6,/i, 'Must retain 6-column grid for general bento-grid on max-width: 1024px');
        assert.match(bentoCss, /\.analyze-page|\.analyze-grid/, '2-column tablet layout must be scoped to Analyze page');
    });

    it('login.html notice uses normal flow and does not fixed-cover login card', () => {
        const loginHtml = fs.readFileSync(path.join(pagesDir, 'login.html'), 'utf8');
        assert.doesNotMatch(loginHtml, /id="desktop-notice"[^>]*style="[^"]*position:\s*fixed/i, 'login notice must not use position: fixed');
    });

    it('admin pages (login, dashboard, history) contain desktop-only notice markup and dismiss', () => {
        for (const page of ['login.html', 'dashboard.html', 'history.html']) {
            const html = fs.readFileSync(path.join(pagesDir, page), 'utf8');
            assert.match(
                html,
                /class="desktop-only-notice"/,
                `${page} must include .desktop-only-notice`
            );
            assert.match(
                html,
                /class="notice-dismiss-btn"/,
                `${page} must include .notice-dismiss-btn`
            );
            assert.match(
                html,
                /This page is designed for desktop\. For the best experience, use a computer\./,
                `${page} must include the exact desktop notice text`
            );
        }
    });
});
