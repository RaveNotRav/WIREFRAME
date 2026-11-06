import { fetchApi } from './main.js';

/**
 * Loads history from the API
 * @param {Object} filters
 * @returns {Promise<{items: Array, total: number, limit: number}>}
 */
export async function loadHistoryData(filters = {}) {
    const { page = 1, limit = 12, is_mock = true, is_public, company, barangay, street, include_invalid } = filters;
    let url = `/api/analysis?page=${page}&limit=${limit}`;
    if (!is_mock) url += `&is_mock=0`;
    if (is_public !== undefined) url += `&is_public=${is_public}`;
    
    if (company) url += `&company=${encodeURIComponent(company)}`;
    if (barangay) url += `&barangay=${encodeURIComponent(barangay)}`;
    if (street) url += `&street=${encodeURIComponent(street)}`;
    if (include_invalid) url += `&include_invalid=true`;

    const res = await fetchApi(url);
    const data = await res.json();

    // A secondary fetch that doesn't block the list
    let finalItems = data.data || [];
    try {
        // Mock secondary fetch
        await fetchApi('/api/sites').catch(() => {});
    } catch (e) {
        // Secondary fetch failure should not block returning items
    }

    return {
        items: finalItems,
        total: data.total || 0,
        limit: data.limit || limit
    };
}

export const NO_VERIFIED_RECORDS_MSG = "No verified records to export. Public tips and mock data are excluded from official exports. Check 'Include Mock' to export test data (marked as unverified).";

export function resolveExportSelection(checkedIds = [], loadedItems = [], options = {}) {
    const includeMock = !!options.includeMock;
    const canIncludeInvalid = !!options.canIncludeInvalid;
    const itemMap = new Map((loadedItems || []).map(it => [Number(it.id), it]));

    let candidateIds = Array.isArray(checkedIds) && checkedIds.length > 0
        ? checkedIds.map(id => Number(id)).filter(n => !Number.isNaN(n))
        : (loadedItems || [])
            .filter(it => it && (it.status === 'complete' || it.status === 'no_wires' || (canIncludeInvalid && it.status === 'rejected')))
            .map(it => Number(it.id));

    if (!includeMock && itemMap.size > 0) {
        candidateIds = candidateIds.filter(id => {
            const it = itemMap.get(id);
            if (!it) return true;
            return !it.is_mock && !it.is_public;
        });
    }

    return candidateIds;
}

export function formatSkippedNoGpsMessage(count) {
    const n = Number(count) || 0;
    return n > 0 ? `${n} record(s) skipped (no GPS).` : '';
}

export async function requestHistoryExport(endpoint, payload, fetchFn = fetchApi) {
    const res = await fetchFn(endpoint, {
        method: 'POST',
        preserveErrorDetail: true,
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            analysis_ids: payload.analysis_ids || [],
            include_mock: !!payload.include_mock,
            include_invalid: !!payload.include_invalid
        })
    });
    return res;
}
