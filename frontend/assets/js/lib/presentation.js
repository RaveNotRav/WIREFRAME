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
        const sum = data?.summary || 'Invalid';
        label = 'Rejected: ' + sum; text = 'Rejected'; style = 'danger'; risk = 'critical'; icon = 'x'; cssClass = 'badge-rejected'; reason = sum;
    } else if (s === 'no_wires') {
        label = 'No Wires'; text = 'No Wires'; style = 'success'; risk = 'low'; icon = 'check'; cssClass = 'badge-no-wires'; reason = 'No wires were detected in the image.';
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
    const crsBand = analysis.crs_band || null;
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

    // 2. Evaluate CRS band
    if (crsBand === 'Critical') {
        text = 'Critical';
        risk = 'critical';
        reasons.push('CRS band is Critical');
        crsText = 'Critical Risk';
        crsColor = 'var(--risk-critical)';
    } else if (crsBand === 'High') {
        if (risk !== 'critical') {
            text = 'Unsafe';
            risk = 'high';
        }
        reasons.push('CRS band is High');
        crsText = 'High Risk';
        crsColor = 'var(--risk-high)';
    } else if (crsBand === 'Moderate') {
        if (risk !== 'critical' && risk !== 'high') {
            text = 'Caution';
            risk = 'medium';
        }
        reasons.push('Moderate vegetation risk detected');
        crsText = 'Moderate Risk';
        crsColor = 'var(--risk-medium)';
    } else if (crsBand === 'Low') {
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
        crsText,
        crsColor
    };
}
