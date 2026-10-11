export function statusPresentation(status, analysis) {
    if (status === 'queued') return { label: 'Queued', style: 'neutral', icon: 'hourglass' };
    if (status === 'processing') return { label: 'Processing', style: 'neutral', icon: 'spinner' };
    if (status === 'complete') return { label: 'Complete', style: 'success', icon: 'check' };
    if (status === 'rejected') return { label: 'Rejected: ' + (analysis?.summary || 'Invalid'), style: 'danger', icon: 'x' };
    if (status === 'no_wires') return { label: 'No Wires', style: 'success', icon: 'check' };
    if (status === 'error') return { label: 'Error', style: 'danger', icon: 'alert' };
    if (status === 'interrupted') return { label: 'Interrupted', style: 'neutral', icon: 'pause' };
    return { label: 'Unknown', style: 'neutral', icon: 'help' };
}

export function safetyPresentation(safetyLevel, clearanceStatus, crsBand) {
    if (clearanceStatus === 'unavailable' || crsBand === 'unavailable' || !safetyLevel || safetyLevel === 'Unknown') {
        return { label: 'Unknown', style: 'neutral', icon: 'help' };
    }
    if (safetyLevel === 'Safe') return { label: 'Safe', style: 'success', icon: 'check' };
    if (safetyLevel === 'Caution') return { label: 'Caution', style: 'warning', icon: 'alert' };
    if (safetyLevel === 'Unsafe') return { label: 'Unsafe', style: 'danger', icon: 'x' };
    if (safetyLevel === 'Critical') return { label: 'Critical', style: 'danger', icon: 'alert' };
    
    return { label: 'Unknown', style: 'neutral', icon: 'help' };
}
