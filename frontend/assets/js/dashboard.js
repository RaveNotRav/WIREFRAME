import { initTheme, toggleTheme } from './lib/theme.js';
import { getSafetyPill, formatDbTimestamp, parseDbTimestamp, escapeHtml, formatWireClass, formatRelativeTime, getRiskScorePill as sharedGetRiskScorePill, getHeatmapRiskClass as sharedGetHeatmapRiskClass, getRiskBand, DEFAULT_RISK_BANDS } from './lib/formatters.js';
import { computeHeatmapLayout } from './lib/heatmap_layout.js';

function getRiskScorePill(score) {
    return sharedGetRiskScorePill(score, (typeof state !== 'undefined' && state.riskBands) ? state.riskBands : DEFAULT_RISK_BANDS);
}





document.addEventListener('DOMContentLoaded', async () => {
    initTheme();
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }
    // Auth Check
    let userRole = null;
    try {
        const res = await fetch('/api/auth/me');
        if (!res.ok) throw new Error('Not logged in');
        const userData = await res.json();
        userRole = userData.role;
        
        if (userRole === 'field_inspector') {
            document.body.innerHTML = '<h1>403 Forbidden - Access Denied</h1>';
            return;
        }

        document.getElementById('user-name').textContent = userData.username;
        document.getElementById('user-role').textContent = userData.role;
    } catch (e) {
        window.location.href = '/login.html';
        return;
    }
    
    document.getElementById('btn-logout').addEventListener('click', async () => {
        await fetch('/api/auth/logout', { method: 'POST' });
        window.location.href = '/login.html';
    });

    // Sub-nav handling
    const subNavBtns = document.querySelectorAll('.sub-nav-btn');
    const viewSections = document.querySelectorAll('.view-section');

    subNavBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            subNavBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const targetId = btn.getAttribute('data-target');
            viewSections.forEach(sec => {
                if (sec.id === targetId) sec.classList.add('active');
                else sec.classList.remove('active');
            });
        });
    });

    // Dashboard State
    let state = {
        includeMock: false,
        trendDays: 30,
        heatmapData: null,
        heatmapResizeObserver: null,
        riskBands: DEFAULT_RISK_BANDS
    };

    const mockToggleContainer = document.getElementById('mock-toggle-container') || document.getElementById('include-mock-toggle-container');
    const mockToggle = document.getElementById('include-mock-toggle');
    const mockKpiBanner = document.getElementById('mock-kpi-banner');
    const trendDaysSelect = document.getElementById('trend-days');

    function updateMockLabel() {
        if (mockKpiBanner) {
            mockKpiBanner.style.display = state.includeMock ? 'block' : 'none';
        }
    }

    mockToggle.addEventListener('change', (e) => {
        state.includeMock = e.target.checked;
        updateMockLabel();
        loadDashboardStats();
        loadAllAnalyses();
    });

    trendDaysSelect.addEventListener('change', (e) => {
        state.trendDays = parseInt(e.target.value, 10);
        loadDashboardStats();
    });

    // Determine if mock data exists to show toggle, and auto-enable if no real data
    try {
        const cacheBuster = `&_t=${Date.now()}`;
        
        // Fetch public config for risk_bands if available
        try {
            const configRes = await fetch(`/api/config/public?${cacheBuster.slice(1)}`);
            if (configRes.ok) {
                const configData = await configRes.json();
                if (Array.isArray(configData.risk_bands) && configData.risk_bands.length > 0) {
                    state.riskBands = configData.risk_bands;
                }
            }
        } catch (ce) {
            // Fallback to DEFAULT_RISK_BANDS
        }

        const mockCheckRes = await fetch(`/api/analysis?is_mock=1&limit=1${cacheBuster}`);
        if (mockCheckRes.ok) {
            const mockData = await mockCheckRes.json();
            const hasMock = (mockData.total || 0) > 0;
            if (hasMock) {
                if (mockToggleContainer) mockToggleContainer.style.display = 'block';

                // Check if any verified non-mock complete analyses exist
                const realCheckRes = await fetch(`/api/analysis?is_mock=0&is_public=0&status=complete&limit=1${cacheBuster}`);
                if (realCheckRes.ok) {
                    const realData = await realCheckRes.json();
                    const hasReal = (realData.total || 0) > 0;
                    if (!hasReal) {
                        state.includeMock = true;
                        if (mockToggle) mockToggle.checked = true;
                    }
                }
            }
        }
    } catch (e) {
        console.error('Failed to check mock/real data status:', e);
    }

    updateMockLabel();

    // Load initial data
    loadDashboardStats();
    loadAllAnalyses();

    async function loadDashboardStats() {
        try {
            const url = `/api/stats/dashboard?include_mock=${state.includeMock}&days=${state.trendDays}`;
            const res = await fetch(url);
            if (!res.ok) throw new Error('Failed to load stats');
            const data = await res.json();
            
            renderKPIs(data.kpis);
            renderHeatmap(data.heatmap);
            renderDistribution(data.distribution);
            renderTrendSVG(data.trend);
            if (data.kpis && data.kpis.total_analyses === 0) {
                renderRecentAnalyses([]);
            }
        } catch (e) {
            console.error(e);
        }
    }



    function renderKPIs(kpis) {
        if (!kpis) return;
        const els = ['kpi-total', 'kpi-high-critical', 'kpi-avg-risk', 'kpi-today'];
        els.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('neu-skeleton');
        });
        
        const total = kpis.total_analyses ?? 0;
        const highCrit = kpis.high_critical_count ?? 0;
        const avgRisk = typeof kpis.average_risk_score === 'number' ? kpis.average_risk_score : 0;
        const today = kpis.processed_today ?? 0;

        const totalEl = document.getElementById('kpi-total');
        const criticalEl = document.getElementById('kpi-high-critical');
        const avgRiskEl = document.getElementById('kpi-avg-risk');
        const todayEl = document.getElementById('kpi-today');

        if (totalEl) totalEl.textContent = total;
        if (criticalEl) criticalEl.textContent = highCrit;
        if (avgRiskEl) avgRiskEl.textContent = avgRisk.toFixed(1);
        if (todayEl) todayEl.textContent = today;

        const criticalCard = document.getElementById('kpi-card-critical');
        if (criticalCard) {
            if (highCrit > 0) {
                criticalCard.classList.add('has-critical-risk');
            } else {
                criticalCard.classList.remove('has-critical-risk');
            }
        }
    }

    function getHeatmapRiskClass(band) {
        return sharedGetHeatmapRiskClass(band);
    }

    function renderHeatmapCells(container, items) {
        container.innerHTML = '';
        const fragment = document.createDocumentFragment();
        
        items.forEach(item => {
            const cell = document.createElement('div');
            cell.className = `heatmap-cell ${getHeatmapRiskClass(item.risk_band)}`;
            cell.setAttribute('tabindex', '0');
            cell.setAttribute('role', 'gridcell');
            
            const riskLabel = item.risk_score !== null ? item.risk_score.toFixed(1) : 'N/A';
            const tooltipText = `${item.site_name} | Score: ${riskLabel}`;
            
            cell.setAttribute('data-tooltip', tooltipText);
            cell.setAttribute('aria-label', tooltipText);
            
            const destUrl = item.site_id ? `/history.html?site_id=${item.site_id}` : `/history.html?analysis_id=${item.analysis_id}`;

            cell.addEventListener('click', () => {
                window.location.href = destUrl;
            });
            cell.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    window.location.href = destUrl;
                }
            });
            
            fragment.appendChild(cell);
        });
        container.appendChild(fragment);
    }

    function applyHeatmapLayout(container, heatmapData) {
        if (!container || !heatmapData || heatmapData.length === 0) return;
        const W = container.clientWidth || 360;
        const H = container.clientHeight > 50 ? container.clientHeight : 200;

        let gap = 8;
        if (typeof window !== 'undefined' && window.getComputedStyle) {
            const cs = window.getComputedStyle(container);
            const gapStr = cs.getPropertyValue('--spacing-sm') || cs.gap || '8px';
            const parsed = parseFloat(gapStr);
            if (!isNaN(parsed) && parsed >= 0) gap = parsed;
        }

        const layout = computeHeatmapLayout(heatmapData.length, W, H, gap);
        container.style.setProperty('--hm-cols', layout.cols);
        container.style.setProperty('--hm-cell', `${layout.cellSize}px`);

        if (layout.capped && layout.maxVisible < heatmapData.length) {
            const visibleData = heatmapData.slice(0, layout.maxVisible);
            renderHeatmapCells(container, visibleData);
        }
    }

    function renderHeatmap(heatmapData) {
        state.heatmapData = heatmapData;
        const container = document.getElementById('heatmap-container');
        const legend = document.getElementById('heatmap-legend');
        if (!container) return;
        container.innerHTML = '';
        
        if (!heatmapData || heatmapData.length === 0) {
            if (legend) legend.style.display = 'none';
            container.classList.add('is-empty');
            container.innerHTML = `
                <div class="dashboard-empty-state">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
                    <span>No analyses yet</span>
                </div>
            `;
            return;
        }

        container.classList.remove('is-empty');
        if (legend) legend.style.display = 'flex';

        // Render cells
        renderHeatmapCells(container, heatmapData);

        // After rendering cells, compute layout and apply CSS variables
        applyHeatmapLayout(container, heatmapData);

        // Setup rAF-throttled ResizeObserver once
        if (!state.heatmapResizeObserver && typeof ResizeObserver !== 'undefined') {
            let rAfId = null;
            state.heatmapResizeObserver = new ResizeObserver(() => {
                if (rAfId) cancelAnimationFrame(rAfId);
                rAfId = requestAnimationFrame(() => {
                    if (state.heatmapData && state.heatmapData.length > 0) {
                        applyHeatmapLayout(container, state.heatmapData);
                    }
                });
            });
            state.heatmapResizeObserver.observe(container);
        }
    }

    function calculateDistributionPercentages(dist) {
        const total = Object.values(dist).reduce((sum, val) => sum + val, 0);
        if (total === 0) return dist; // Avoid division by zero
        
        let result = {};
        for (const [key, val] of Object.entries(dist)) {
            result[key] = (val / total) * 100;
        }
        return result;
    }
    // Expose for node testing
    if (typeof window !== 'undefined') {
        window.calculateDistributionPercentages = calculateDistributionPercentages;
    }

    function renderDistribution(dist) {
        const container = document.getElementById('distribution-container');
        if (!container) return;
        container.innerHTML = '';
        
        const totalWires = Object.values(dist || {}).reduce((sum, val) => sum + (Number(val) || 0), 0);
        if (totalWires === 0) {
            container.innerHTML = `
                <div class="dashboard-empty-state">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                    <span>No analyses yet</span>
                </div>
            `;
            return;
        }

        const defaultKeys = ['power_line', 'telecom_cable', 'fiber_optic', 'dead_abandoned'];
        const allKeys = Array.from(new Set([...defaultKeys, ...Object.keys(dist || {})]));
        
        const percentages = calculateDistributionPercentages(dist);
        
        for (const key of allKeys) {
            const val = (dist && dist[key]) || 0;
            const pct = percentages[key] || 0;
            const labelText = formatWireClass(key);
            
            const html = `
                <div class="dist-item">
                    <div class="dist-label-row">
                        <span>${escapeHtml(labelText)}</span>
                        <span>${val} (${pct.toFixed(1)}%)</span>
                    </div>
                    <div class="dist-track">
                        <div class="dist-fill" style="width: 0; background-color: var(--wire-${key}, var(--wire-power, #3b82f6));" data-width="${pct}%"></div>
                    </div>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', html);
        }
        
        // Trigger animation
        setTimeout(() => {
            container.querySelectorAll('.dist-fill').forEach(el => {
                el.style.width = el.getAttribute('data-width');
            });
        }, 50);
    }

    function renderRecentAnalyses(analyses) {
        const container = document.getElementById('recent-analyses-container');
        if (!container) return;
        container.innerHTML = '';
        
        if (!analyses || analyses.length === 0) {
            container.innerHTML = `
                <div class="dashboard-empty-state">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 14 14"/></svg>
                    <span>No analyses yet</span>
                    <a href="/analyze.html" class="neu-button neu-raised-sm" style="margin-top: var(--spacing-xs); text-decoration: none; font-size: var(--font-size-sm); padding: var(--spacing-xs) var(--spacing-md);">Go to Analyze</a>
                </div>
            `;
            return;
        }

        const fallbackSvg = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiM0NDQiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZmlsbD0iI2FhYSIgZGltaW5hbnQtYmFzZWxpbmU9Im1pZGRsZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+Tm8gVGh1bWJuYWlsPC90ZXh0Pjwvc3ZnPg==";
        const frag = document.createDocumentFragment();

        analyses.forEach(item => {
            const row = document.createElement('a');
            row.className = 'recent-analysis-row';
            row.href = `/analyze.html?id=${encodeURIComponent(item.id)}`;
            
            const thumbUrl = item.filename ? `/api/analysis/${item.id}/original` : '';
            const displayName = item.filename || `Analysis #${item.id}`;
            const timeAgo = formatRelativeTime(item.created_at);
            const pillHtml = getRiskScorePill(item.risk_score);
            
            row.innerHTML = `
                <img src="${thumbUrl ? escapeHtml(thumbUrl) : fallbackSvg}" loading="lazy" decoding="async" class="recent-thumb" alt="${escapeHtml(displayName)}" onerror="this.src='${fallbackSvg}'">
                <div class="recent-info">
                    <span class="recent-filename" title="${escapeHtml(displayName)}">${escapeHtml(displayName)}</span>
                    <span class="recent-time">${escapeHtml(timeAgo)}</span>
                </div>
                ${pillHtml}
            `;
            
            frag.appendChild(row);
        });
        container.appendChild(frag);
    }

    function renderTrendSVG(trendData) {
        if (trendData) {
            state.trendData = trendData;
        } else {
            trendData = state.trendData || [];
        }

        const container = document.getElementById('trend-svg-container');
        const tbody = document.getElementById('trend-table-body');
        
        // Only clear tbody if we are explicitly passing new trendData (not on resize)
        if (arguments.length > 0) {
            tbody.innerHTML = '';
            
            // Populate Accessible Table
            const frag = document.createDocumentFragment();
            trendData.forEach(d => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${escapeHtml(d.date)}</td><td>${escapeHtml(d.count)}</td><td>${escapeHtml(d.avg_risk)}</td>`;
                frag.appendChild(tr);
            });
            tbody.appendChild(frag);
        }
        
        container.innerHTML = '';
        if (trendData.length === 0) {
            container.innerHTML = `
                <div class="dashboard-empty-state" style="min-height: 220px;">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    <span>No analyses yet</span>
                </div>
            `;
            return;
        }

        // Draw SVG
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 300;
        
        // Setup ResizeObserver to re-render if dimensions change (e.g. 0 initially then layout happens)
        if (!state.trendResizeObserver && typeof ResizeObserver !== 'undefined') {
            let resizeRaf = null;
            state.trendResizeObserver = new ResizeObserver(entries => {
                if (resizeRaf) cancelAnimationFrame(resizeRaf);
                resizeRaf = requestAnimationFrame(() => {
                    renderTrendSVG();
                });
            });
            state.trendResizeObserver.observe(container);
        }
        const padding = { top: 20, right: 20, bottom: 40, left: 40 };
        
        const innerWidth = width - padding.left - padding.right;
        const innerHeight = height - padding.top - padding.bottom;
        
        // Scale functions
        const maxRisk = 100; // max risk is 100
        const minDateStr = trendData[0].date;
        const maxDateStr = trendData[trendData.length - 1].date;
        const minDate = parseDbTimestamp(minDateStr + ' 00:00:00').getTime();
        const maxDate = parseDbTimestamp(maxDateStr + ' 00:00:00').getTime();
        
        const getX = (dateStr) => {
            if (minDate === maxDate) return padding.left + innerWidth / 2;
            const time = parseDbTimestamp(dateStr + ' 00:00:00').getTime();
            return padding.left + ((time - minDate) / (maxDate - minDate)) * innerWidth;
        };
        
        const getY = (val) => {
            return padding.top + innerHeight - ((val / maxRisk) * innerHeight);
        };
        
        let pathD = "";
        trendData.forEach((d, i) => {
            const x = getX(d.date);
            const y = getY(d.avg_risk);
            if (i === 0) pathD += `M ${x} ${y} `;
            else pathD += `L ${x} ${y} `;
        });

        let pointsHtml = "";
        trendData.forEach(d => {
            const x = getX(d.date);
            const y = getY(d.avg_risk);
            pointsHtml += `<circle class="svg-trend-point" cx="${x}" cy="${y}" r="4" data-tooltip="${escapeHtml(d.date)}: ${escapeHtml(d.avg_risk)}" />`;
        });
        
        const xTicksCount = 5;
        let xAxisHtml = "";
        for (let i=0; i<xTicksCount; i++) {
            const fraction = i / (xTicksCount - 1);
            const time = minDate + fraction * (maxDate - minDate);
            const dateStr = new Date(time + 8 * 3600 * 1000).toISOString().slice(0, 10);
            const x = padding.left + fraction * innerWidth;
            xAxisHtml += `<text class="svg-axis-text" x="${x}" y="${height - 10}" text-anchor="middle">${escapeHtml(dateStr.substring(5))}</text>`;
            xAxisHtml += `<line x1="${x}" y1="${padding.top + innerHeight}" x2="${x}" y2="${padding.top + innerHeight + 5}" stroke="var(--text-secondary)" />`;
        }
        
        const yTicksCount = 5;
        let yAxisHtml = "";
        for (let i=0; i<yTicksCount; i++) {
            const val = i * (100 / (yTicksCount - 1));
            const y = getY(val);
            yAxisHtml += `<text class="svg-axis-text" x="${padding.left - 10}" y="${y + 4}" text-anchor="end">${escapeHtml(Math.round(val))}</text>`;
            yAxisHtml += `<line x1="${padding.left - 5}" y1="${y}" x2="${padding.left}" y2="${y}" stroke="var(--text-secondary)" />`;
        }

        const svgHtml = `
            <svg width="100%" height="100%" viewBox="0 0 ${width} ${height}" style="overflow: visible;">
                <!-- Axes -->
                <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="var(--text-secondary)" />
                <line x1="${padding.left}" y1="${height - padding.bottom}" x2="${width - padding.right}" y2="${height - padding.bottom}" stroke="var(--text-secondary)" />
                
                ${xAxisHtml}
                ${yAxisHtml}
                
                <!-- Path and Points -->
                <path class="svg-trend-line" d="${pathD}" />
                ${pointsHtml}
            </svg>
        `;
        
        container.innerHTML = svgHtml;
    }

    // --- Table Sorting Logic ---
    function sortAnalyses(analyses, column, direction) {
        return [...analyses].sort((a, b) => {
            let valA = a[column];
            let valB = b[column];
            
            if (valA === null || valA === undefined) valA = '';
            if (valB === null || valB === undefined) valB = '';
            
            if (typeof valA === 'string') valA = valA.toLowerCase();
            if (typeof valB === 'string') valB = valB.toLowerCase();
            
            if (valA < valB) return direction === 'asc' ? -1 : 1;
            if (valA > valB) return direction === 'asc' ? 1 : -1;
            return 0;
        });
    }
    if (typeof window !== 'undefined') {
        window.sortAnalyses = sortAnalyses;
    }

    async function loadAllAnalyses() {
        const tbody = document.getElementById('analyses-table-body');
        try {
            // Note: In a real app we'd handle pagination, 
            // but for dashboard view we'll load the first 100 
            // or rely on server-side sort if supported.
            const mockQuery = state.includeMock ? '' : '&is_mock=0';
            const cacheBuster = `&_t=${Date.now()}`;
            const res = await fetch(`/api/analysis?limit=100&is_public=0&status=complete${mockQuery}${cacheBuster}`);
            if (!res.ok) throw new Error('Failed to fetch analyses');
            const data = await res.json();
            const allAnalyses = data.data || [];
            renderRecentAnalyses(allAnalyses.slice(0, 5));
            
            tbody.innerHTML = '';
            const frag = document.createDocumentFragment();
            allAnalyses.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${escapeHtml(item.id)}</td>
                    <td>${formatDbTimestamp(item.created_at, { includeTime: false })}</td>
                    <td>${escapeHtml(item.company || '-')}</td>
                    <td>${escapeHtml(item.site_name || item.street || '-')}</td>
                    <td>${escapeHtml(item.barangay || '-')}</td>
                    <td>${escapeHtml(item.status)}</td>
                    <td>${getRiskScorePill(item.risk_score)}</td>
                `;
                frag.appendChild(tr);
            });
            tbody.appendChild(frag);
            
        } catch (e) {
            renderRecentAnalyses([]);
            tbody.innerHTML = `<tr><td colspan="7" style="color:var(--risk-critical);">${escapeHtml(e.message)}</td></tr>`;
        }
    }
});
