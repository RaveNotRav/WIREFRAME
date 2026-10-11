/**
 * Returns an object with the display text and CSS class for a given status.
 */
function statusPresentation(status) {
    if (!status || status === 'unavailable' || status === 'Unknown') {
        return { text: 'Unknown', cssClass: 'badge-unknown' };
    }
    const s = status.toLowerCase();
    if (s === 'complete') return { text: 'Complete', cssClass: 'badge-complete' };
    if (s === 'rejected') return { text: 'Rejected', cssClass: 'badge-rejected' };
    if (s === 'no_wires') return { text: 'No Wires', cssClass: 'badge-no-wires' };
    if (s === 'error') return { text: 'Error', cssClass: 'badge-error' };
    return { text: 'Unknown', cssClass: 'badge-unknown' };
}

/**
 * Returns an object with the display text and CSS class for safety status.
 */
function safetyPresentation(safety) {
    if (!safety || safety === 'unavailable' || safety === 'Unknown') {
        return { text: 'Unknown', cssClass: 'badge-unknown' };
    }
    const s = safety.toLowerCase();
    if (s === 'safe') return { text: 'Safe', cssClass: 'badge-safe' };
    if (s === 'critical' || s === 'danger') return { text: 'Critical', cssClass: 'badge-critical' };
    if (s === 'warning' || s === 'moderate') return { text: 'Warning', cssClass: 'badge-warning' };
    return { text: 'Unknown', cssClass: 'badge-unknown' };
}

module.exports = { statusPresentation, safetyPresentation };
