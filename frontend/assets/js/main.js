// Fetch wrapper for authenticated requests
// shared constants and functions
const API_BASE = '/api';

export function unwrapList(data) {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    if (Array.isArray(data.items)) return data.items;
    if (Array.isArray(data.data)) return data.data; // in case of {data: [...], total: N}
    return [];
}

export async function fetchApi(url, options = {}) {
    options.credentials = 'same-origin';
    
    if (options.method && options.method.toUpperCase() !== 'GET') {
        options.headers = options.headers || {};
        options.headers['X-Requested-With'] = 'XMLHttpRequest';
    }

    const response = await fetch(url, options);

    if (response.status === 401 && !options.allowAnon) {
        window.location.href = '/login.html';
        throw new Error("Unauthorized - Redirecting to login");
    }

    if (response.status === 403) {
        throw new Error("Forbidden: You do not have permission to access this resource.");
    }

    if (!response.ok && !(options.allowAnon && response.status === 401) && !(options.allowAnon && response.status === 429)) {
        let msg = `HTTP ${response.status}`;
        try {
            const data = await response.json();
            if (data.detail) {
                msg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
            } else if (data.error) {
                msg = data.error;
                if (data.error_id) {
                    msg += ` (Error ID: ${data.error_id})`;
                }
            }
        } catch(e) {}
        throw new Error(msg);
    }

    return response;
}


export function showToast(msg, type='info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = 'neu-toast neu-raised';
    toast.setAttribute('role', 'status');
    
    let icon = 'ℹ️';
    if (type === 'error') icon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
    if (type === 'success') icon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    if (type === 'warning') icon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';
    
    const escapeHtml = (unsafe) => {
        return (unsafe || '').toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    };
    
    toast.innerHTML = `<span>${icon}</span> <span>${escapeHtml(msg)}</span>`;
    container.appendChild(toast);
    
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
