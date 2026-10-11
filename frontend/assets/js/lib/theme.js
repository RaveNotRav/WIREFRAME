// theme.js - Manage data-theme and localStorage

export function initTheme() {
    try {
        let theme = localStorage.getItem('wireframe-theme');
        if (!theme) {
            theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        setTheme(theme);
    } catch (e) {
        console.warn('Could not access localStorage for theme preference');
    }
}

export function setTheme(theme) {
    if (theme === 'auto') {
        document.documentElement.removeAttribute('data-theme');
    } else {
        document.documentElement.setAttribute('data-theme', theme);
    }
    
    try {
        localStorage.setItem('wireframe-theme', theme);
    } catch (e) {
        console.warn('Could not save theme preference to localStorage');
    }
}

export function toggleTheme() {
    try {
        let currentTheme = localStorage.getItem('wireframe-theme');
        if (!currentTheme || currentTheme === 'auto') {
            currentTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    } catch(e) {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    }
}
