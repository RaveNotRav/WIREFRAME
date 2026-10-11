// contrast.js - Script to compute WCAG contrast ratios of CSS variables

function hexToRgb(hex) {
    // Basic hex parsing, can handle rgba if we strip alpha, but let's assume hex
    if (hex.startsWith('rgba')) {
        const parts = hex.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/);
        return [parseInt(parts[1]), parseInt(parts[2]), parseInt(parts[3])];
    }
    hex = hex.replace(/^#/, '');
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const bigint = parseInt(hex, 16);
    return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
}

function luminance(r, g, b) {
    const a = [r, g, b].map(v => {
        v /= 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
}

function contrastRatio(color1, color2) {
    const l1 = luminance(...hexToRgb(color1));
    const l2 = luminance(...hexToRgb(color2));
    const lightest = Math.max(l1, l2);
    const darkest = Math.min(l1, l2);
    return (lightest + 0.05) / (darkest + 0.05);
}

const themes = {
    light: {
        bg: '#E4E9F0',
        primary: '#1F2C38',
        secondary: '#40505E',
        muted: '#576574',
        disabled: '#7c8594',
        low: '#145A32',
        moderate: '#935116',
        high: '#A04000',
        critical: '#A93226',
        border: '#7c8594',
        focus: '#2980b9',
        badge_text_on_color: '#FFFFFF',
    },
    dark: {
        bg: '#1e1e24',
        primary: '#f0f0f0',
        secondary: '#c0c0c0',
        muted: '#9e9e9e',
        disabled: '#6e6e73',
        low: '#2ecc71',
        moderate: '#f1c40f',
        high: '#e67e22',
        critical: '#FF6B6B',
        border: '#6e6e73',
        focus: '#5dade2',
        badge_text_on_color: '#000000',
    }
};

let failed = false;

function check(name, c1, c2, minRatio) {
    const ratio = contrastRatio(c1, c2);
    const passed = ratio >= minRatio;
    console.log(`${passed ? '<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><polyline points='20 6 9 17 4 12'></polyline></svg>' : '<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><line x1='18' y1='6' x2='6' y2='18'></line><line x1='6' y1='6' x2='18' y2='18'></line></svg>'} ${name.padEnd(50)} ${ratio.toFixed(2).padStart(5)}:1 (Min: ${minRatio})`);
    if (!passed) failed = true;
}

for (const [theme, c] of Object.entries(themes)) {
    console.log(`\n=== Theme: ${theme.toUpperCase()} ===`);
    
    const surfaces = [
        { name: 'flat bg', color: c.bg },
        { name: 'raised bg', color: c.bg }, // In this neumorphism, background colors don't change, only shadow does
        { name: 'inset bg', color: c.bg },
        { name: 'tile bg', color: c.bg }
    ];

    console.log('\n-- Text vs Surfaces --');
    for (const surface of surfaces) {
        check(`primary vs ${surface.name}`, c.primary, surface.color, 4.5);
        check(`secondary vs ${surface.name}`, c.secondary, surface.color, 4.5);
        check(`muted vs ${surface.name}`, c.muted, surface.color, 4.5);
    }
    
    console.log('\n-- Inputs --');
    check('input text (primary) vs inset bg', c.primary, c.bg, 4.5);
    check('placeholder (muted) vs inset bg', c.muted, c.bg, 4.5);
    check('disabled text vs surface', c.disabled, c.bg, 3.0); // Disabled text is exempt from 4.5, can be 3.0 or even lower, but we'll enforce 3.0
    
    console.log('\n-- Risk Colors as Text vs Tile Surface --');
    check('low text vs tile bg', c.low, c.bg, 4.5);
    check('moderate text vs tile bg', c.moderate, c.bg, 4.5);
    check('high text vs tile bg', c.high, c.bg, 4.5);
    check('critical text vs tile bg', c.critical, c.bg, 4.5);

    console.log('\n-- Risk Colors as Badge Backgrounds --');
    // If the badge has solid risk background, the text must contrast against it
    // Wait, the badges in components.css use border-left or just a colored dot, they don't have solid background.
    // If they did have a solid background, we check text against the risk color.
    check('badge text vs low bg', c.badge_text_on_color, c.low, 4.5);
    check('badge text vs moderate bg', c.badge_text_on_color, c.moderate, 4.5);
    check('badge text vs high bg', c.badge_text_on_color, c.high, 4.5);
    check('badge text vs critical bg', c.badge_text_on_color, c.critical, 4.5);

    console.log('\n-- UI Boundaries vs Surface --');
    check('border vs bg', c.border, c.bg, 3.0);
    check('focus vs bg', c.focus, c.bg, 3.0);
    
    console.log('\n-- Heatmap Cells (Non-text) vs Surface --');
    check('heatmap low vs bg', c.low, c.bg, 3.0);
    check('heatmap moderate vs bg', c.moderate, c.bg, 3.0);
    check('heatmap high vs bg', c.high, c.bg, 3.0);
    check('heatmap critical vs bg', c.critical, c.bg, 3.0);
}

if (failed) {
    console.error('\nContrast checks failed.');
    process.exit(1);
} else {
    console.log('\nAll contrast checks passed!');
}
