// history.js - Shared helpers for Analysis History view (including Public Submissions by default)

import { escapeHtml } from './lib/formatters.js';

export const PUBLIC_SUBMISSION_BADGE_TEXT = 'Public Submission';

/**
 * Maps active tab ('all' | 'verified' | 'public') to the `is_public` API filter.
 * Default ('all') returns `undefined` so both Verified Inspector and Public Submissions appear.
 */
export function resolveHistoryPublicFilter(activeTab = 'all') {
    if (activeTab === 'verified') return 0;
    if (activeTab === 'public') return 1;
    return undefined;
}

/**
 * Builds watermark / submission badge HTML for a history gallery item.
 */
export function buildHistoryBadgesHtml(item = {}) {
    let badgeHtml = item.is_mock ? '<div class="mock-watermark">MOCK DATA</div>' : '';
    if (Number(item.is_public) === 1) {
        const tipTop = item.is_mock ? '50px' : '20px';
        badgeHtml += `<div class="mock-watermark public-submission-badge" style="background: var(--color-warning); top: ${tipTop}; opacity: 1;">${escapeHtml(PUBLIC_SUBMISSION_BADGE_TEXT)}</div>`;
    }
    return badgeHtml;
}
