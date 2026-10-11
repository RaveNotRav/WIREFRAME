// formatters.js - Pure logic for formatting and mapping

export function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

export function formatConfidence(conf) {
    if (typeof conf !== 'number') return '0%';
    return (conf * 100).toFixed(0) + '%';
}

export function getWireClassColor(className) {
    const mapping = {
        'power_line': 'var(--wire-power)',
        'telecom_cable': 'var(--wire-telecom)',
        'fiber_optic': 'var(--wire-fiber)',
        'dead_abandoned': 'var(--wire-dead)'
    };
    return mapping[className] || 'var(--text-muted)';
}


export function getSafetyPill(level) {
    if (!level) return '';
    const l = level.toLowerCase();
    return `<span class="neu-pill neu-pill-safety-${l}">${escapeHtml(level)}</span>`;
}

export function parseDbTimestamp(str) {
    if (!str) return null;
    const iso = str.includes('T') ? str : str.replace(' ', 'T');
    const utcIso = (iso.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(iso)) ? iso : iso + 'Z';
    const d = new Date(utcIso);
    return isNaN(d.getTime()) ? null : d;
}

const phtDateTimeFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Manila',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
});

const phtDateFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Manila',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric'
});

export function formatDbTimestamp(str, { includeTime = true } = {}) {
    const d = parseDbTimestamp(str);
    if (!d) return '--';
    return includeTime ? phtDateTimeFormatter.format(d) : phtDateFormatter.format(d);
}
