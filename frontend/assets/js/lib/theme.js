// theme.js - Manage data-theme and localStorage

function getSystemTheme() {
    try {
        if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
    } catch (_) {}
    return 'light';
}

export function updateThemeToggleUI(theme) {
    if (typeof document === 'undefined' || typeof document.querySelectorAll !== 'function') return;
    const isDark = theme === 'dark';
    const toggles = document.querySelectorAll('#theme-toggle, .neu-theme-toggle');
    toggles.forEach(btn => {
        btn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
        btn.innerHTML = isDark
            ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`
            : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
    });
}

export function initTheme() {
    let theme = null;
    try {
        if (typeof localStorage !== 'undefined') {
            theme = localStorage.getItem('wireframe-theme');
        }
    } catch (e) {
        console.warn('Could not access localStorage for theme preference');
    }

    if (!theme || theme === 'auto') {
        theme = getSystemTheme();
    }
    setTheme(theme);
    return theme;
}

export function setTheme(theme) {
    const validTheme = theme === 'dark' ? 'dark' : 'light';

    if (typeof document !== 'undefined' && document.documentElement) {
        document.documentElement.setAttribute('data-theme', validTheme);
    }

    try {
        if (typeof localStorage !== 'undefined') {
            localStorage.setItem('wireframe-theme', validTheme);
        }
    } catch (e) {
        console.warn('Could not save theme preference to localStorage');
    }

    updateThemeToggleUI(validTheme);
    return validTheme;
}

export function toggleTheme() {
    let currentTheme = null;
    try {
        if (typeof localStorage !== 'undefined') {
            currentTheme = localStorage.getItem('wireframe-theme');
        }
    } catch (_) {}

    if (!currentTheme && typeof document !== 'undefined' && document.documentElement) {
        currentTheme = document.documentElement.getAttribute('data-theme');
    }

    if (!currentTheme || currentTheme === 'auto') {
        currentTheme = getSystemTheme();
    }

    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    return nextTheme;
}
