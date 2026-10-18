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

    it('index.html landing page has Login on desktop and none on phones (<768px)', () => {
        const indexHtml = fs.readFileSync(path.join(pagesDir, 'index.html'), 'utf8');
        const headerCss = fs.readFileSync(path.join(cssDir, 'global-header.css'), 'utf8');
        // Landing has Login link with desktop-nav-link
        assert.match(indexHtml, /class="[^"]*desktop-nav-link[^"]*"[^>]*id="nav-login"|id="nav-login"[^>]*class="[^"]*desktop-nav-link/, 'index.html must have #nav-login with desktop-nav-link');
        // Landing has no hamburger menu on phones
        assert.doesNotMatch(indexHtml, /id="menu-toggle"/, 'index.html must not contain hamburger #menu-toggle');
        assert.doesNotMatch(indexHtml, /id="mobile-menu"/, 'index.html must not contain #mobile-menu');
        // CSS hides desktop-nav-link on <768px
        assert.match(headerCss, /@media\s*\(\s*max-width\s*:\s*767px\s*\)[\s\S]*\.desktop-nav-link\s*\{[^}]*display\s*:\s*none/i, 'global-header.css must hide desktop-nav-link on <768px');
    });

    it('index.html landing CTA (.neu-button-lg) is full width and centered on phones (<768px)', () => {
        const indexHtml = fs.readFileSync(path.join(pagesDir, 'index.html'), 'utf8');
        assert.match(
            indexHtml,
            /@media\s*\(\s*max-width\s*:\s*768px\s*\)[\s\S]*?\.neu-button-lg\s*\{[^}]*width\s*:\s*100%[^}]*max-width\s*:\s*none[^}]*min-height\s*:\s*48px/i,
            'Landing CTA .neu-button-lg must be width: 100%, max-width: none, and min-height: 48px on phones'
        );
        assert.match(
            indexHtml,
            /@media\s*\(\s*max-width\s*:\s*768px\s*\)[\s\S]*?\.neu-button-lg\s*\{[^}]*text-align\s*:\s*center/i,
            'Landing CTA .neu-button-lg must center text inside the button on phones'
        );
    });

    it('index.html landing page uses flex-column/svh on phones (<768px) and avoids overflowing 100vh + header height', () => {
        const indexHtml = fs.readFileSync(path.join(pagesDir, 'index.html'), 'utf8');
        const phoneMatch = indexHtml.match(/@media\s*\(\s*max-width\s*:\s*768px\s*\)\s*\{([\s\S]*?)\n\s*\}\s*@media/);
        assert.ok(phoneMatch, 'Must define @media (max-width: 768px) section');
        const phoneCss = phoneMatch[1];

        // Uses flex-column layout with min-height svh/vh
        assert.match(phoneCss, /display\s*:\s*flex;\s*flex-direction\s*:\s*column/i, 'Phone layout must use flex-direction: column');
        assert.match(phoneCss, /min-height\s*:\s*100svh/i, 'Phone layout must declare min-height: 100svh');
        assert.match(phoneCss, /\.hero-content\s*\{[^}]*flex\s*:\s*1/i, 'Hero content must take flex: 1 to fill remaining space');

        // Background covers the screen with fixed positioning
        assert.match(phoneCss, /(\.hero-background|\.hero-overlay)[^{]*\{[^}]*position\s*:\s*fixed/i, 'Hero background must use position: fixed on phones to prevent bottom cutoff');

        // Does NOT combine height: 100vh or min-height: calc(100vh - ...) with header causing overflow
        assert.doesNotMatch(phoneCss, /min-height\s*:\s*calc\([^)]*100vh/i, 'Must not use min-height: calc(... 100vh ...) which overflows with header');
    });

    it('index.html defines entrance animations (wf-fade-up, wf-bg-zoom, wf-cta-glow, wf-fade-in) and handles reduced motion and light overlay', () => {
        const indexHtml = fs.readFileSync(path.join(pagesDir, 'index.html'), 'utf8');

        // Four keyframes exist
        assert.match(indexHtml, /@keyframes\s+wf-fade-up\s*\{[^}]*translateY/i, 'Must define @keyframes wf-fade-up');
        assert.match(indexHtml, /@keyframes\s+wf-bg-zoom\s*\{[^}]*scale/i, 'Must define @keyframes wf-bg-zoom');
        assert.match(indexHtml, /@keyframes\s+wf-cta-glow\s*\{[^}]*box-shadow/i, 'Must define @keyframes wf-cta-glow');
        assert.match(indexHtml, /@keyframes\s+wf-fade-in\s*\{[^}]*opacity/i, 'Must define @keyframes wf-fade-in');

        // Hero elements use wf-fade-up
        assert.match(indexHtml, /\.hero-title\s*\{[^}]*animation\s*:\s*[^;]*wf-fade-up/i, '.hero-title must use wf-fade-up');
        assert.match(indexHtml, /\.hero-tagline\s*\{[^}]*animation\s*:\s*[^;]*wf-fade-up/i, '.hero-tagline must use wf-fade-up');
        assert.match(indexHtml, /\.neu-button-lg\s*\{[^}]*animation\s*:\s*[^;]*wf-fade-up/i, 'CTA (.neu-button-lg) must use wf-fade-up');

        // Background zoom on .hero-background
        assert.match(indexHtml, /\.hero-background\s*\{[^}]*animation\s*:\s*[^;]*wf-bg-zoom/i, '.hero-background must use wf-bg-zoom');

        // CTA subtle glow pulse
        assert.match(indexHtml, /\.neu-button-lg\s*\{[^}]*animation\s*:\s*[^;]*wf-cta-glow/i, 'CTA (.neu-button-lg) must use wf-cta-glow');
        assert.match(indexHtml, /\.neu-button-lg:(hover|focus-visible)[^{]*\{[^}]*animation-play-state\s*:\s*paused/i, 'CTA must pause animation on hover/focus');

        // Light theme overlay uses linear-gradient
        assert.match(indexHtml, /\[data-theme=["']light["']\]\s+\.hero-overlay\s*\{[^}]*background\s*:\s*linear-gradient/i, 'Light theme hero overlay must use linear-gradient');

        // Accessibility: prefers-reduced-motion uses wf-fade-in for title/tagline/CTA and disables movement/pulse
        const reducedMotionMatch = indexHtml.match(/@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)\s*\{([\s\S]*?)\n\s*<\/style>/i);
        assert.ok(reducedMotionMatch, 'Must define @media (prefers-reduced-motion: reduce)');
        const reducedMotionCss = reducedMotionMatch[1];

        // No blanket * animation: none
        assert.doesNotMatch(reducedMotionCss, /\*\s*\{[^}]*animation\s*:\s*none/i, 'Must not have blanket * animation: none');

        // Uses wf-fade-in for text/CTA
        assert.match(reducedMotionCss, /\.hero-title[^{]*\{[^}]*wf-fade-in/i, 'prefers-reduced-motion must use wf-fade-in for hero-title');
        assert.match(reducedMotionCss, /\.hero-tagline[^{]*\{[^}]*wf-fade-in/i, 'prefers-reduced-motion must use wf-fade-in for hero-tagline');
        assert.match(reducedMotionCss, /\.neu-button-lg[^{]*\{[^}]*wf-fade-in/i, 'prefers-reduced-motion must use wf-fade-in for CTA');

        // Disables background zoom and CTA glow pulse
        assert.match(reducedMotionCss, /\.hero-background\s*\{[^}]*animation\s*:\s*none/i, 'prefers-reduced-motion must disable background zoom');
        assert.doesNotMatch(reducedMotionCss, /wf-cta-glow/i, 'prefers-reduced-motion must disable CTA glow pulse');

        // Removes transforms
        assert.match(reducedMotionCss, /transform\s*:\s*none/i, 'prefers-reduced-motion must remove transforms');
    });

    it('analyze.html has no guest login, hides hamburger for guests, and keeps logged-in menu', () => {
        const analyzeHtml = fs.readFileSync(path.join(pagesDir, 'analyze.html'), 'utf8');
        const analyzeJs = fs.readFileSync(path.join(process.cwd(), 'frontend/assets/js/analyze.js'), 'utf8');
        
        // No guest login link on analyze.html
        assert.doesNotMatch(analyzeHtml, /id="nav-login"/, 'analyze.html must not have guest #nav-login');
        assert.doesNotMatch(analyzeHtml, /id="mobile-nav-login"/, 'analyze.html must not have #mobile-nav-login');
        assert.doesNotMatch(analyzeHtml, /id="mobile-nav-home"/, 'analyze.html must not have #mobile-nav-home');
        assert.doesNotMatch(analyzeHtml, /id="mobile-nav-analyze"/, 'analyze.html must not have #mobile-nav-analyze');

        // Hamburger and mobile menu are auth-only (hidden for guests by default)
        assert.match(analyzeHtml, /class="[^"]*auth-only-nav[^"]*"[^>]*id="menu-toggle"|id="menu-toggle"[^>]*class="[^"]*auth-only-nav/, '#menu-toggle must have auth-only-nav class');
        assert.match(analyzeHtml, /id="menu-toggle"[^>]*style="[^"]*display:\s*none|style="[^"]*display:\s*none[^"]*"[^>]*id="menu-toggle/, '#menu-toggle must start hidden');
        assert.match(analyzeHtml, /class="[^"]*auth-only-nav[^"]*"[^>]*id="mobile-menu"|id="mobile-menu"[^>]*class="[^"]*auth-only-nav/, '#mobile-menu must have auth-only-nav class');
        assert.match(analyzeHtml, /id="mobile-btn-logout"/, '#mobile-menu must contain #mobile-btn-logout');

        // analyze.js hides auth-only-nav for guests and shows for authenticated users
        assert.match(analyzeJs, /document\.querySelectorAll\('\.auth-only-nav'\)\.forEach\(el => el\.style\.display = 'none'\)/, 'analyze.js must hide auth-only-nav for guests');
        assert.match(analyzeJs, /document\.querySelectorAll\('\.auth-only-nav'\)\.forEach\(el => el\.style\.display = ''\)/, 'analyze.js must show auth-only-nav for logged-in users');
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

    it('index.html styles #nav-login and the theme toggle with accent gradient, white text, and hover rule', () => {
        const indexHtml = fs.readFileSync(path.join(pagesDir, 'index.html'), 'utf8');
        assert.match(
            indexHtml,
            /#theme-toggle[^{]*\{[^}]*background[^:]*:\s*var\(--color-accent-gradient\)[^}]*color\s*:\s*(#fff|#ffffff|white)/i,
            'Landing theme toggle must be styled with var(--color-accent-gradient) and white text'
        );
        assert.match(
            indexHtml,
            /#nav-login[^{]*\{[^}]*background[^:]*:\s*var\(--color-accent-gradient\)[^}]*color\s*:\s*(#fff|#ffffff|white)/i,
            'Landing #nav-login must be styled with var(--color-accent-gradient) and white text'
        );
        assert.match(
            indexHtml,
            /(#nav-login|#theme-toggle):hover[^{]*\{[^}]*var\(--color-accent-gradient-hover\)/i,
            'Landing header buttons must have a hover rule with var(--color-accent-gradient-hover)'
        );
    });
});
