import { test, describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { initTheme, setTheme, toggleTheme, updateThemeToggleUI } from './theme.js';

describe('theme.js unit tests', () => {
    let mockElement;
    let mockButtons;
    let mockStorage;
    let storageThrows = false;

    beforeEach(() => {
        storageThrows = false;
        mockStorage = {};

        mockElement = {
            attributes: {},
            setAttribute(name, val) {
                this.attributes[name] = String(val);
            },
            getAttribute(name) {
                return this.attributes[name] || null;
            },
            removeAttribute(name) {
                delete this.attributes[name];
            },
        };

        mockButtons = [
            {
                attributes: {},
                innerHTML: '',
                setAttribute(name, val) {
                    this.attributes[name] = String(val);
                },
                getAttribute(name) {
                    return this.attributes[name] || null;
                },
            },
        ];

        globalThis.document = {
            documentElement: mockElement,
            querySelectorAll(selector) {
                if (selector.includes('theme-toggle')) {
                    return mockButtons;
                }
                return [];
            },
        };

        globalThis.localStorage = {
            getItem(key) {
                if (storageThrows) throw new Error('SecurityError: localStorage is disabled');
                return mockStorage[key] ?? null;
            },
            setItem(key, val) {
                if (storageThrows) throw new Error('SecurityError: localStorage is disabled');
                mockStorage[key] = String(val);
            },
            removeItem(key) {
                if (storageThrows) throw new Error('SecurityError: localStorage is disabled');
                delete mockStorage[key];
            },
        };

        globalThis.window = {
            matchMedia(query) {
                return {
                    matches: query.includes('dark'),
                };
            },
        };
    });

    describe('initTheme', () => {
        it('restores saved light theme from localStorage', () => {
            mockStorage['wireframe-theme'] = 'light';
            const theme = initTheme();
            assert.equal(theme, 'light');
            assert.equal(mockElement.getAttribute('data-theme'), 'light');
            assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to dark theme');
        });

        it('restores saved dark theme from localStorage', () => {
            mockStorage['wireframe-theme'] = 'dark';
            const theme = initTheme();
            assert.equal(theme, 'dark');
            assert.equal(mockElement.getAttribute('data-theme'), 'dark');
            assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to light theme');
        });

        it('falls back to matchMedia when no preference is saved', () => {
            const theme = initTheme();
            assert.equal(theme, 'dark'); // mock matchMedia returns true for dark
            assert.equal(mockElement.getAttribute('data-theme'), 'dark');
            assert.equal(mockStorage['wireframe-theme'], 'dark');
        });

        it('handles localStorage throwing and applies fallback theme gracefully', () => {
            storageThrows = true;
            assert.doesNotThrow(() => {
                const theme = initTheme();
                assert.equal(theme, 'dark');
                assert.equal(mockElement.getAttribute('data-theme'), 'dark');
            });
        });
    });

    describe('toggleTheme', () => {
        it('flips value from light to dark and dark to light, persisting choice', () => {
            setTheme('light');
            assert.equal(mockElement.getAttribute('data-theme'), 'light');
            assert.equal(mockStorage['wireframe-theme'], 'light');

            const next1 = toggleTheme();
            assert.equal(next1, 'dark');
            assert.equal(mockElement.getAttribute('data-theme'), 'dark');
            assert.equal(mockStorage['wireframe-theme'], 'dark');
            assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to light theme');
            assert.ok(mockButtons[0].innerHTML.includes('<circle'), 'Sun SVG rendered in dark mode');

            const next2 = toggleTheme();
            assert.equal(next2, 'light');
            assert.equal(mockElement.getAttribute('data-theme'), 'light');
            assert.equal(mockStorage['wireframe-theme'], 'light');
            assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to dark theme');
            assert.ok(mockButtons[0].innerHTML.includes('<path'), 'Moon SVG rendered in light mode');
        });

        it('handles localStorage throwing and still toggles document attribute and UI', () => {
            setTheme('light');
            storageThrows = true;

            assert.doesNotThrow(() => {
                const next = toggleTheme();
                assert.equal(next, 'dark');
                assert.equal(mockElement.getAttribute('data-theme'), 'dark');
                assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to light theme');
            });

            assert.doesNotThrow(() => {
                const next2 = toggleTheme();
                assert.equal(next2, 'light');
                assert.equal(mockElement.getAttribute('data-theme'), 'light');
                assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to dark theme');
            });
        });
    });

    describe('button aria-label and icon synchronization', () => {
        it('sets moon icon and "Switch to dark theme" for light mode', () => {
            updateThemeToggleUI('light');
            assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to dark theme');
            assert.ok(mockButtons[0].innerHTML.includes('M21 12.79A9 9 0 1 1 11.21 3'));
        });

        it('sets sun icon and "Switch to light theme" for dark mode', () => {
            updateThemeToggleUI('dark');
            assert.equal(mockButtons[0].getAttribute('aria-label'), 'Switch to light theme');
            assert.ok(mockButtons[0].innerHTML.includes('<circle cx="12" cy="12" r="5">'));
        });
    });

    describe('tokens.css theme token symmetry and presence', () => {
        it('verifies that all token names exist in both [data-theme="light"] and [data-theme="dark"]', async () => {
            const fs = await import('node:fs');
            const path = await import('node:path');
            const tokensPath = path.resolve(process.cwd(), 'frontend/assets/css/tokens.css');
            assert.ok(fs.existsSync(tokensPath), 'tokens.css must exist');
            const css = fs.readFileSync(tokensPath, 'utf8');

            function extractBlock(marker, content) {
                const idx = content.indexOf(marker);
                if (idx === -1) return '';
                const openBrace = content.indexOf('{', idx);
                let depth = 1;
                let pos = openBrace + 1;
                while (depth > 0 && pos < content.length) {
                    if (content[pos] === '{') depth++;
                    else if (content[pos] === '}') depth--;
                    pos++;
                }
                return content.slice(openBrace + 1, pos - 1);
            }

            function extractProps(block) {
                const s = new Set();
                for (const match of block.matchAll(/--([a-zA-Z0-9_-]+)\s*:/g)) {
                    s.add('--' + match[1]);
                }
                return s;
            }

            const lightProps = extractProps(extractBlock('data-theme="light"', css));
            const darkProps = extractProps(extractBlock('data-theme="dark"', css));

            assert.ok(lightProps.size > 0, 'Light theme must declare tokens');
            assert.ok(darkProps.size > 0, 'Dark theme must declare tokens');

            // Find missing tokens in either theme
            const missingInDark = [...lightProps].filter(token => !darkProps.has(token));
            const missingInLight = [...darkProps].filter(token => !lightProps.has(token));

            assert.deepEqual(missingInDark, [], 'All tokens declared in light theme must exist in dark theme');
            assert.deepEqual(missingInLight, [], 'All tokens declared in dark theme must exist in light theme');

            // Also verify that every var(--token) used across all CSS files exists in both themes
            const cssDir = path.resolve(process.cwd(), 'frontend/assets/css');
            const cssFiles = fs.readdirSync(cssDir).filter(f => f.endsWith('.css'));
            for (const file of cssFiles) {
                const fileContent = fs.readFileSync(path.join(cssDir, file), 'utf8');
                for (const match of fileContent.matchAll(/var\(--([a-zA-Z0-9_-]+)/g)) {
                    const tokenName = '--' + match[1];
                    assert.ok(lightProps.has(tokenName), `Token ${tokenName} used in ${file} must exist in [data-theme="light"]`);
                    assert.ok(darkProps.has(tokenName), `Token ${tokenName} used in ${file} must exist in [data-theme="dark"]`);
                }
            }

            // Verify specific accent and header tokens
            assert.ok(lightProps.has('--color-accent-gradient'), '--color-accent-gradient in light theme');
            assert.ok(darkProps.has('--color-accent-gradient'), '--color-accent-gradient in dark theme');
            assert.ok(lightProps.has('--color-accent-gradient-hover'), '--color-accent-gradient-hover in light theme');
            assert.ok(darkProps.has('--color-accent-gradient-hover'), '--color-accent-gradient-hover in dark theme');
            assert.ok(lightProps.has('--color-accent-glow'), '--color-accent-glow in light theme');
            assert.ok(darkProps.has('--color-accent-glow'), '--color-accent-glow in dark theme');
            assert.ok(lightProps.has('--header-btn-bg'), '--header-btn-bg in light theme');
            assert.ok(darkProps.has('--header-btn-bg'), '--header-btn-bg in dark theme');
            assert.ok(lightProps.has('--header-btn-text'), '--header-btn-text in light theme');
            assert.ok(darkProps.has('--header-btn-text'), '--header-btn-text in dark theme');
        });
    });
});

