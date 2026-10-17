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
