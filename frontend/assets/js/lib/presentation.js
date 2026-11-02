import { getRiskBand } from './formatters.js';

export function formatRejectionReason(code) {
    if (!code) return '';
    const map = {
        'resolution_too_low': 'The photo is too small. Use the original photo from your camera, not one sent through Telegram or Messenger.',
        'too_dark': 'The photo is too dark. Retake it in better lighting or daytime.',
        'too_blurry': 'The photo is too blurry. Hold the camera steady and tap to focus on the wires.',
        'corrupt_file': 'The file is corrupt or unreadable. Please choose a valid JPG or PNG image.'
    };
    return map[code] || 'The photo could not be analyzed. Please retake it.';
}

export function statusPresentation(statusOrAnalysis, analysis) {
    let status = statusOrAnalysis;
    let data = analysis;
    if (typeof statusOrAnalysis === 'object' && statusOrAnalysis !== null) {
        status = statusOrAnalysis.status;
        data = statusOrAnalysis;
    }

    const s = (status || '').toLowerCase();
    
    let label = 'Unknown';
    let text = 'Unknown';
    let style = 'neutral';
    let risk = 'neutral';
    let icon = 'help';
    let cssClass = 'badge-unknown';
    let reason = data?.summary || null;

    if (s === 'queued') {
        label = 'Queued'; text = 'Queued'; style = 'neutral'; risk = 'neutral'; icon = 'hourglass'; cssClass = 'badge-queued';
    } else if (s === 'processing') {
        label = 'Processing'; text = 'Processing'; style = 'neutral'; risk = 'neutral'; icon = 'spinner'; cssClass = 'badge-processing';
    } else if (s === 'complete') {
        label = 'Complete'; text = 'Complete'; style = 'success'; risk = 'low'; icon = 'check'; cssClass = 'badge-complete';
    } else if (s === 'rejected') {
        const sum = data?.summary || '';
        reason = formatRejectionReason(sum);
        label = 'Rejected: ' + sum; text = 'Rejected'; style = 'danger'; risk = 'critical'; icon = 'x'; cssClass = 'badge-rejected';
    } else if (s === 'no_wires') {
        label = 'No Wires'; text = 'No Wires'; style = 'warning'; risk = 'neutral'; icon = 'alert'; cssClass = 'badge-no-wires'; reason = 'No wires were detected in the image.';
    } else if (s === 'error') {
        label = 'Error'; text = 'Error'; style = 'danger'; risk = 'critical'; icon = 'alert'; cssClass = 'badge-error'; reason = 'An error occurred during analysis.';
    } else if (s === 'interrupted') {
        label = 'Interrupted'; text = 'Interrupted'; style = 'neutral'; risk = 'neutral'; icon = 'pause'; cssClass = 'badge-interrupted'; reason = 'Analysis was cancelled.';
    }

    return { label, text, style, risk, icon, cssClass, reason };
}

export function safetyPresentation(arg1, arg2, arg3) {
    let analysis = {};
    if (typeof arg1 === 'object' && arg1 !== null) {
        analysis = arg1;
    } else if (typeof arg1 === 'string' && arg2 === undefined && arg3 === undefined) {
        // Single status string legacy support
        const s = arg1.toLowerCase();
        if (s === 'safe') {
            analysis = { ground_clearance_status: 'above', crs_band: 'Low' };
        } else if (s === 'critical' || s === 'danger') {
            analysis = { ground_clearance_status: 'below', crs_band: 'Low' };
        } else if (s === 'warning' || s === 'moderate' || s === 'caution') {
            analysis = { ground_clearance_status: 'near', crs_band: 'Low' };
        } else {
            analysis = { ground_clearance_status: 'unavailable', crs_band: 'unavailable' };
        }
    } else {
        // 3 positional arguments (safetyLevel, clearanceStatus, crsBand)
        analysis = {
            safety_level: arg1,
            ground_clearance_status: arg2,
            crs_band: arg3
        };
    }

    const clrStatus = analysis.ground_clearance_status || analysis.clearance_status || null;
    const hasNumericRiskScore = typeof analysis.risk_score === 'number' && !Number.isNaN(analysis.risk_score);
    const derivedBand = hasNumericRiskScore
        ? (getRiskBand(analysis.risk_score)?.label || null)
        : null;
    const crsBand = derivedBand || analysis.crs_band || null;

    // Resolve per-wire max CRS and band for wire-based floor escalation
    let maxWireCrs = null;
    if (typeof analysis.max_wire_crs === 'number' && !Number.isNaN(analysis.max_wire_crs)) {
        maxWireCrs = analysis.max_wire_crs;
    } else if (typeof analysis.max_crs === 'number' && !Number.isNaN(analysis.max_crs)) {
        maxWireCrs = analysis.max_crs;
    } else if (Array.isArray(analysis.wires) && analysis.wires.length > 0) {
        const wireScores = analysis.wires
            .map(w => (w && typeof w.crs === 'number' && !Number.isNaN(w.crs) ? w.crs : null))
            .filter(v => v !== null);
        if (wireScores.length > 0) {
            maxWireCrs = Math.max(...wireScores);
        }
    }

    let maxWireBand = analysis.max_wire_band || null;
    if (!maxWireBand || maxWireBand === 'unavailable') {
        if (maxWireCrs !== null) {
            maxWireBand = getRiskBand(maxWireCrs)?.label || null;
        } else if (Array.isArray(analysis.wires) && analysis.wires.length > 0) {
            const bandRank = { Low: 1, Moderate: 2, High: 3, Critical: 4 };
            let bestBand = null;
            let bestRank = 0;
            for (const w of analysis.wires) {
                const wb = w?.band || w?.crs_band || (typeof w?.crs === 'number' ? getRiskBand(w.crs)?.label : null);
                if (wb && (bandRank[wb] || 0) > bestRank) {
                    bestRank = bandRank[wb];
                    bestBand = wb;
                }
            }
            maxWireBand = bestBand;
        }
    }
    let reasons = Array.isArray(analysis.safety_reasons) ? [...analysis.safety_reasons] : [];
    if (Array.isArray(analysis.reasons) && reasons.length === 0) {
        reasons = [...analysis.reasons];
    }

    let text = 'Unknown';
    let risk = 'neutral';
    let crsText = 'Unavailable';
    let crsColor = 'var(--text-muted)';

    // 1. Evaluate clearance status
    if (clrStatus === 'below') {
        text = 'Critical';
        risk = 'critical';
        reasons.push('Ground clearance is below legal limit');
    } else if (clrStatus === 'near') {
        text = 'Caution';
        risk = 'high';
        reasons.push('Ground clearance is near legal limit');
    } else if (clrStatus === 'above') {
        text = 'Safe';
        risk = 'low';
        reasons.push('Ground clearance is safe');
    } else if (clrStatus === 'no_pole') {
        reasons.push('The whole pole and the ground were not visible, so ground clearance could not be checked. Retake the photo showing the full pole down to the ground.');
    } else if (clrStatus === 'no_wires') {
        reasons.push('No wires were detected in this photo. Retake it with the overhead wires clearly in view.');
    }

    // 2. Evaluate CRS band (from risk_score when present, else explicit crs_band)
    if (crsBand === 'Critical') {
        text = 'Critical';
        risk = 'critical';
        if (hasNumericRiskScore) {
            reasons.push(`Risk score ${analysis.risk_score.toFixed(1)} (Critical)`);
        }
        reasons.push('CRS band is Critical');
        crsText = 'Critical Risk';
        crsColor = 'var(--risk-critical)';
    } else if (crsBand === 'High') {
        if (risk !== 'critical') {
            text = 'Unsafe';
            risk = 'high';
        }
        if (hasNumericRiskScore) {
            reasons.push(`Risk score ${analysis.risk_score.toFixed(1)} (High)`);
        }
        reasons.push('CRS band is High');
        crsText = 'High Risk';
        crsColor = 'var(--risk-high)';
    } else if (crsBand === 'Moderate') {
        if (risk !== 'critical' && risk !== 'high') {
            text = 'Caution';
            risk = 'medium';
        }
        if (hasNumericRiskScore) {
            reasons.push(`Risk score ${analysis.risk_score.toFixed(1)} (Moderate)`);
        }
        reasons.push('Moderate vegetation risk detected');
        crsText = 'Moderate Risk';
        crsColor = 'var(--risk-medium)';
    } else if (crsBand === 'Low') {
        if (hasNumericRiskScore) {
            reasons.push(`Risk score ${analysis.risk_score.toFixed(1)} (Low)`);
        }
        crsText = 'Low Risk';
        crsColor = 'var(--risk-low)';
    } else {
        crsText = 'Unavailable';
        crsColor = 'var(--text-muted)';
    }

    // 3. Overall badge reconciliation per decisions:
    // When ground_clearance_status is "no_pole" or "no_wires":
    // The badge must never read "Safe": use "Incomplete" unless the CRS band is High or Critical,
    // in which case keep the CRS-based text AND show the message in the reasons.
    if (clrStatus === 'no_pole' || clrStatus === 'no_wires') {
        if (crsBand === 'Critical') {
            text = 'Critical';
            risk = 'critical';
        } else if (crsBand === 'High') {
            text = 'Unsafe';
            risk = 'high';
        } else {
            text = 'Incomplete';
            risk = 'neutral';
        }
    } else if (clrStatus === 'unavailable' || !clrStatus) {
        // When clearance is unavailable: never produce "Safe" from CRS alone.
        if (crsBand === 'Critical') {
            text = 'Critical';
            risk = 'critical';
        } else if (crsBand === 'High') {
            text = 'Unsafe';
            risk = 'high';
        } else if (crsBand === 'Moderate') {
            text = 'Caution';
            risk = 'medium';
        } else {
            text = 'Unknown';
            risk = 'neutral';
        }
        if (!reasons.includes('Data unavailable for complete safety assessment')) {
            reasons.push('Data unavailable for complete safety assessment');
        }
    } else if (clrStatus === 'above') {
        if (crsBand === 'unavailable' || !crsBand) {
            text = 'Unknown';
            risk = 'neutral';
            if (!reasons.includes('Data unavailable for complete safety assessment')) {
                reasons.push('Data unavailable for complete safety assessment');
            }
        }
    }

    // 4. Wire-based worst-case floor from maxWireBand:
    // Critical wire -> Critical; High wire -> at least Unsafe; Moderate/Low -> no escalation.
    if (maxWireBand === 'Critical') {
        text = 'Critical';
        risk = 'critical';
        const wireReason = maxWireCrs !== null
            ? `Highest wire risk is Critical (${maxWireCrs.toFixed(1)})`
            : 'Highest wire risk is Critical';
        if (!reasons.includes(wireReason)) {
            reasons.push(wireReason);
        }
    } else if (maxWireBand === 'High') {
        if (text !== 'Critical') {
            text = 'Unsafe';
            risk = 'high';
        }
        const wireReason = maxWireCrs !== null
            ? `Highest wire risk is High (${maxWireCrs.toFixed(1)})`
            : 'Highest wire risk is High';
        if (!reasons.includes(wireReason)) {
            reasons.push(wireReason);
        }
    }

    if (text === 'Unknown' && reasons.length === 0) {
        reasons.push('Unknown safety status');
    }

    const styleMap = {
        low: 'success',
        medium: 'warning',
        high: 'warning',
        critical: 'danger',
        neutral: 'neutral'
    };
    const iconMap = {
        Safe: 'check',
        Caution: 'alert',
        Unsafe: 'x',
        Critical: 'alert',
        Incomplete: 'help',
        Unknown: 'help'
    };
    const cssSafetyMap = {
        Safe: 'badge-safe',
        Caution: 'badge-warning',
        Unsafe: 'badge-critical',
        Critical: 'badge-critical',
        Incomplete: 'badge-unknown',
        Unknown: 'badge-unknown'
    };

    return {
        text,
        label: text,
        risk,
        style: styleMap[risk] || 'neutral',
        icon: iconMap[text] || 'help',
        cssClass: cssSafetyMap[text] || 'badge-unknown',
        reasons,
        crsBand,
        maxWireBand,
        crsText,
        crsColor
    };
}

export function shouldShowPublicConfirmation(analysis, user = null) {
    if (user) return false;
    if (!analysis || analysis.status !== 'complete') return false;
    const wireCount = Array.isArray(analysis.wires)
        ? analysis.wires.length
        : (typeof analysis.wire_count === 'number' ? analysis.wire_count : 0);
    return wireCount > 0;
}

export const TERMINAL_STATUSES = ['complete', 'rejected', 'no_wires', 'error', 'interrupted'];

export function isTerminalStatus(status) {
    if (!status || typeof status !== 'string') return false;
    return TERMINAL_STATUSES.includes(status.toLowerCase());
}
