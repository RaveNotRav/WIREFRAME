// formatters.js - Pure logic for formatting and mapping

export function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
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

export function formatWireClass(className) {
    if (!className) return 'Unknown';
    const known = {
        'power_line': 'Power Line',
        'telecom_cable': 'Telecom Cable',
        'fiber_optic': 'Fiber Optic',
        'dead_abandoned': 'Dead / Abandoned',
        'pole': 'Utility Pole'
    };
    if (known[className]) return known[className];
    return String(className)
        .replace(/[_-]+/g, ' ')
        .split(' ')
        .filter(Boolean)
        .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
        .join(' ');
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

export function formatRelativeTime(dateInput, nowMs = Date.now()) {
    if (!dateInput) return '--';
    const date = typeof dateInput === 'string' ? parseDbTimestamp(dateInput) : dateInput;
    if (!date || isNaN(date.getTime())) return '--';
    const diffMs = nowMs - date.getTime();
    if (diffMs < 0) return 'just now';
    const sec = Math.floor(diffMs / 1000);
    if (sec < 60) return 'just now';
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ago`;
    const days = Math.floor(hr / 24);
    if (days < 30) return `${days}d ago`;
    const months = Math.floor(days / 30);
    if (months < 12) return `${months}mo ago`;
    const years = Math.floor(months / 12);
    return `${years}y ago`;
}

export const DEFAULT_RISK_BANDS = [
    { min: 0.00, max: 0.33, label: 'Low', safetyClass: 'safe' },
    { min: 0.34, max: 0.66, label: 'Moderate', safetyClass: 'caution' },
    { min: 0.67, max: 0.85, label: 'High', safetyClass: 'unsafe' },
    { min: 0.86, max: 1.00, label: 'Critical', safetyClass: 'critical' },
];

export function getRiskBand(score, bands = DEFAULT_RISK_BANDS) {
    if (score === null || score === undefined) {
        return null;
    }
    const num = typeof score === 'number' ? score : Number(score);
    if (isNaN(num)) {
        return null;
    }
    const val = Math.max(0.0, Math.min(1.0, num / 100.0));
    const bandList = Array.isArray(bands) && bands.length > 0 ? bands : DEFAULT_RISK_BANDS;
    for (const b of bandList) {
        if (val <= b.max) {
            return b;
        }
    }
    return bandList[bandList.length - 1];
}

export function getHeatmapRiskClass(band) {
    if (!band) return 'risk-Unknown';
    const validBands = ['Low', 'Moderate', 'High', 'Critical'];
    if (validBands.includes(band)) return `risk-${band}`;
    return 'risk-Unknown';
}

export function getRiskScorePill(score, bands = DEFAULT_RISK_BANDS) {
    if (score === null || score === undefined || typeof score !== 'number' || isNaN(score)) {
        return '-';
    }
    const band = getRiskBand(score, bands);
    if (!band) return '-';
    
    const label = band.label || 'Unknown';
    const classMap = {
        'Low': 'safe',
        'Moderate': 'caution',
        'High': 'unsafe',
        'Critical': 'critical'
    };
    const pillClass = band.safetyClass || classMap[label] || label.toLowerCase();
    return `<span class="neu-pill neu-pill-safety-${pillClass}">${escapeHtml(label)}</span> <span style="font-size:0.85rem; color:var(--text-secondary); margin-left:6px;">(${score.toFixed(1)})</span>`;
}

