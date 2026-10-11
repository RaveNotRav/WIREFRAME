import { initTheme, toggleTheme } from './lib/theme.js';
import { getSafetyPill } from './lib/formatters.js';

function getRiskScorePill(score) {
    if (score === null || score === undefined) return '-';
    let level = 'Safe';
    if (score >= 75) level = 'Critical';
    else if (score >= 50) level = 'Unsafe';
    else if (score >= 25) level = 'Caution';
    return getSafetyPill(level) + ` <span style="font-size:0.85rem; color:var(--text-secondary); margin-left:6px;">(${score.toFixed(1)})</span>`;
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
        trendDays: 30
    };

    const mockToggleContainer = document.getElementById('mock-toggle-container');
    const mockToggle = document.getElementById('include-mock-toggle');
    const trendDaysSelect = document.getElementById('trend-days');

    mockToggle.addEventListener('change', (e) => {
        state.includeMock = e.target.checked;
        loadDashboardStats();
        loadAllAnalyses();
    });

    trendDaysSelect.addEventListener('change', (e) => {
        state.trendDays = parseInt(e.target.value, 10);
        loadDashboardStats();
    });

    // Determine if mock data exists to show toggle
    try {
        const mockCheckRes = await fetch('/api/analysis?is_mock=1&limit=1');
        if (mockCheckRes.ok) {
            const mockData = await mockCheckRes.json();
            if (mockData.total > 0) {
                mockToggleContainer.style.display = 'block';
            }
        }
    } catch (e) {}

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
        } catch (e) {
            console.error(e);
        }
    }

        
    function renderSparkline(containerId, data, dataKey, color) {
        const container = document.getElementById(containerId);
        if (!container || !data || data.length === 0) return;
        
        const width = container.clientWidth || 100;
        const height = container.clientHeight || 30;
        
        const values = data.map(d => d[dataKey]);
        const max = Math.max(...values, 1);
        const min = Math.min(...values, 0);
        const range = max - min || 1;
        
        const pts = values.map((val, i) => {
            const x = (i / (values.length - 1 || 1)) * width;
            const y = height - ((val - min) / range) * height;
            return `${x},${y}`;
        }).join(' ');
        
        const svg = `<svg width="100%" height="100%" preserveAspectRatio="none">
            <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>`;
        container.innerHTML = svg;
    }

    function renderKPIs(kpis) {
        const els = ['kpi-total', 'kpi-high-critical', 'kpi-avg-risk', 'kpi-today'];
        els.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('neu-skeleton');
        });
        
        document.getElementById('kpi-total').textContent = kpis.total_analyses;
        document.getElementById('kpi-high-critical').textContent = kpis.high_critical_count;
        document.getElementById('kpi-avg-risk').textContent = kpis.average_risk_score.toFixed(1);
        document.getElementById('kpi-today').textContent = kpis.processed_today;
    }

    function getHeatmapRiskClass(band) {
        if (!band) return 'risk-Unknown';
        const validBands = ['Low', 'Moderate', 'High', 'Critical'];
        if (validBands.includes(band)) return `risk-${band}`;
        return 'risk-Unknown';
    }
    if (typeof window !== 'undefined') window.getHeatmapRiskClass = getHeatmapRiskClass;

    function renderHeatmap(heatmapData) {
        const container = document.getElementById('heatmap-container');
        container.innerHTML = '';
        
        heatmapData.forEach(item => {
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
            
            container.appendChild(cell);
        });
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
        container.innerHTML = '';
        
        const labels = {
            'power_line': 'Power Line',
            'telecom_cable': 'Telecom Cable',
            'fiber_optic': 'Fiber Optic',
            'dead_abandoned': 'Dead / Abandoned'
        };
        
        const percentages = calculateDistributionPercentages(dist);
        
        for (const key of Object.keys(labels)) {
            const val = dist[key] || 0;
            const pct = percentages[key] || 0;
            
            const html = `
                <div class="dist-item">
                    <div class="dist-label-row">
                        <span>${labels[key]}</span>
                        <span>${val} (${pct.toFixed(1)}%)</span>
                    </div>
                    <div class="dist-track">
                        <div class="dist-fill" style="width: 0; background-color: var(--wire-${key});" data-width="${pct}%"></div>
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
            trendData.forEach(d => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${d.date}</td><td>${d.count}</td><td>${d.avg_risk}</td>`;
                tbody.appendChild(tr);
            });
        }
        
        container.innerHTML = '';
        if (trendData.length === 0) {
            container.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100%;color:var(--text-secondary);">No trend data available</div>';
            return;
        }

        // Draw SVG
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 300;
        
        // Setup ResizeObserver to re-render if dimensions change (e.g. 0 initially then layout happens)
        if (!state.trendResizeObserver && typeof ResizeObserver !== 'undefined') {
            state.trendResizeObserver = new ResizeObserver(entries => {
                // debounce slightly or just re-render
                renderTrendSVG();
                renderSparkline('sparkline-total', state.trendData, 'count', 'var(--color-primary)');
                renderSparkline('sparkline-risk', state.trendData, 'avg_risk', 'var(--color-unsafe)'); 
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
        const minDate = new Date(minDateStr).getTime();
        const maxDate = new Date(maxDateStr).getTime();
        
        const getX = (dateStr) => {
            if (minDate === maxDate) return padding.left + innerWidth / 2;
            const time = new Date(dateStr).getTime();
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
            pointsHtml += `<circle class="svg-trend-point" cx="${x}" cy="${y}" r="4" data-tooltip="${d.date}: ${d.avg_risk}" />`;
        });
        
        const xTicksCount = 5;
        let xAxisHtml = "";
        for (let i=0; i<xTicksCount; i++) {
            const fraction = i / (xTicksCount - 1);
            const time = minDate + fraction * (maxDate - minDate);
            const dateStr = new Date(time).toISOString().split('T')[0];
            const x = padding.left + fraction * innerWidth;
            xAxisHtml += `<text class="svg-axis-text" x="${x}" y="${height - 10}" text-anchor="middle">${dateStr.substring(5)}</text>`;
            xAxisHtml += `<line x1="${x}" y1="${padding.top + innerHeight}" x2="${x}" y2="${padding.top + innerHeight + 5}" stroke="var(--text-secondary)" />`;
        }
        
        const yTicksCount = 5;
        let yAxisHtml = "";
        for (let i=0; i<yTicksCount; i++) {
            const val = i * (100 / (yTicksCount - 1));
            const y = getY(val);
            yAxisHtml += `<text class="svg-axis-text" x="${padding.left - 10}" y="${y + 4}" text-anchor="end">${Math.round(val)}</text>`;
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
            const res = await fetch(`/api/analysis?limit=100${mockQuery}${cacheBuster}`);
            if (!res.ok) throw new Error('Failed to fetch analyses');
            const data = await res.json();
            
            tbody.innerHTML = '';
            (data.data || []).forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${item.id}</td>
                    <td>${new Date(item.created_at).toLocaleDateString()}</td>
                    <td>${item.company || '-'}</td>
                    <td>${item.site_name || item.street || '-'}</td>
                    <td>${item.barangay || '-'}</td>
                    <td>${item.status}</td>
                    <td>${getRiskScorePill(item.risk_score)}</td>
                `;
                tbody.appendChild(tr);
            });
            
        } catch (e) {
            tbody.innerHTML = `<tr><td colspan="7" style="color:var(--risk-critical);">${e.message}</td></tr>`;
        }
    }
});
