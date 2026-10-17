// analyze.js

import { fetchApi, showToast } from './main.js';
import { initTheme, toggleTheme } from './lib/theme.js';
import { getNaturalCoordinates } from './lib/coordinates.js';
import { escapeHtml, formatConfidence, getWireClassColor } from './lib/formatters.js';
import { statusPresentation, safetyPresentation, shouldShowPublicConfirmation, isTerminalStatus } from './lib/presentation.js';

let currentUser = null;
let currentAnalysis = null;
let currentFiles = [];
let fileThumbURLs = [];
let batchItems = []; // { analysisId, fileObj, status }
let currentActiveItemIndex = 0;
let pollingTimers = {}; // id -> timer
let actionMapping = {};
let calPoints = []; // manual calibration points {x, y}
let currentBatchId = null;

function updateCalibrationButtonVisibility() {
    const calBtn = document.getElementById('btn-calibrate');
    if (!calBtn) return;
    calBtn.style.display = (currentUser && currentAnalysis && currentAnalysis.status === 'complete') ? 'inline-block' : 'none';
}

const MAX_POLLING_CONCURRENCY = 2; // don't overwhelm server
let activePollsCount = 0;

// ---- INIT ----
document.addEventListener('DOMContentLoaded', async () => {
    initTheme();
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
    document.getElementById('btn-logout')?.addEventListener('click', handleLogout);
    
    await loadUser();
    await loadSettings();
    
    setupUploadHandlers();
    setupGPSHandlers();
    setupCalibrationTool();
    setupDecimalValidation();
    updateGlobalStepper('upload');
    // Check if we are viewing an existing analysis
    const params = new URLSearchParams(window.location.search);
    const viewId = params.get('id');
    if (viewId) {
        document.getElementById('tile-batch').style.display = 'block';
        batchItems = [{
            id: parseInt(viewId, 10),
            status: 'queued',
            thumb: '', 
            name: `Analysis #${viewId}`
        }];
        renderBatchTabs();
        switchTab(0);
        updateGlobalStepper('processing');
        startPolling(parseInt(viewId, 10));
    }
});

function setupDecimalValidation() {
    const decimalInputs = document.querySelectorAll('input[type="number"][step="any"], #cal-x, #cal-y, #cal-length');
    decimalInputs.forEach(input => {
        input.addEventListener('input', (e) => {
            // Remove any characters that are not digits, dot, or minus sign
            let val = e.target.value.replace(/[^0-9.-]/g, '');
            
            // Prevent multiple dots
            const parts = val.split('.');
            if (parts.length > 2) {
                val = parts[0] + '.' + parts.slice(1).join('');
            }
            
            // Prevent minus sign anywhere except the beginning
            if (val.indexOf('-') > 0) {
                val = val.replace(/-/g, '');
                val = '-' + val; // move to front, though this is naive, let's just strip interior dashes
            }
            
            // better dash handler:
            val = val.replace(/(?!^)-/g, '');

            if (e.target.value !== val) {
                e.target.value = val;
            }
        });
    });
}

async function loadSettings() {
    try {
        const res = await fetchApi('/api/config/public');
        const data = await res.json();
        if (data.action_mapping) {
            actionMapping = data.action_mapping;
        }
    } catch (e) {
        console.warn('Failed to load settings', e);
    }
}

// ---- AUTH & USER ----
async function loadUser() {
    try {
        const res = await fetchApi('/api/auth/me', { allowAnon: true });
        if (res.status === 401) {
            // Guest mode (links are already hidden by default in HTML)
            currentUser = null;
            document.getElementById('user-name').textContent = 'Public User';
            document.getElementById('user-role').textContent = 'Guest';
            return;
        }
        currentUser = await res.json();
        document.getElementById('user-name').textContent = currentUser.username;
        document.getElementById('user-role').textContent = currentUser.role;
        // Show auth-only links
        if (document.getElementById('btn-logout')) {
            document.getElementById('btn-logout').style.display = '';
        }
        document.querySelectorAll('.auth-only-nav').forEach(el => el.style.display = '');
    } catch (e) {
        console.warn(e.message);
        currentUser = null;
        document.getElementById('user-name').textContent = 'Guest';
        document.getElementById('user-role').textContent = 'Unknown';
    } finally {
        updateCalibrationButtonVisibility();
    }
}

async function handleLogout() {
    try {
        await fetchApi('/api/auth/logout', { method: 'POST' });
    } catch (e) {}
    window.location.href = '/login.html';
}

// ---- UPLOAD & FILES ----
const MAX_FILES = 10;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png'];


function updateGlobalStepper(stage) {
    const sUpload = document.getElementById('step-upload');
    const sProcessing = document.getElementById('step-processing');
    const sResults = document.getElementById('step-results');
    
    const setStepActive = (el) => {
        el.style.opacity = '1';
        const circle = el.querySelector('.step-circle');
        circle.style.background = 'var(--color-primary)';
        circle.style.border = 'none';
        circle.style.color = 'white';
        circle.style.boxShadow = 'var(--neu-raised-sm)';
        el.querySelector('span').style.color = 'var(--text-primary)';
    };
    
    const setStepInactive = (el) => {
        el.style.opacity = '0.5';
        const circle = el.querySelector('.step-circle');
        circle.style.background = 'var(--surface-bg)';
        circle.style.border = '2px solid var(--border-subtle)';
        circle.style.color = 'var(--text-secondary)';
        circle.style.boxShadow = 'none';
        el.querySelector('span').style.color = 'var(--text-secondary)';
    };

    if (stage === 'upload') {
        if (sUpload) setStepActive(sUpload);
        if (sProcessing) setStepInactive(sProcessing);
        if (sResults) setStepInactive(sResults);
    } else if (stage === 'processing') {
        if (sUpload) setStepActive(sUpload);
        if (sProcessing) setStepActive(sProcessing);
        if (sResults) setStepInactive(sResults);
    } else if (stage === 'results') {
        if (sUpload) setStepActive(sUpload);
        if (sProcessing) setStepActive(sProcessing);
        if (sResults) setStepActive(sResults);
    }
}

function setupUploadHandlers() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const btnAnalyze = document.getElementById('btn-analyze');

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    
    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
    
    btnAnalyze.addEventListener('click', startAnalysis);
}

function handleFiles(files) {
    const newFiles = Array.from(files).filter(f => {
        if (!ALLOWED_TYPES.includes(f.type)) {
            showToast(`${f.name} is not a supported format (JPG/PNG only).`, 'error');
            return false;
        }
        if (f.size > MAX_FILE_SIZE) {
            showToast(`${f.name} exceeds the 10MB limit.`, 'error');
            return false;
        }
        return true;
    });

    if (currentFiles.length + newFiles.length > MAX_FILES) {
        showToast(`Maximum of ${MAX_FILES} files allowed.`, 'error');
        return;
    }

    newFiles.forEach(f => {
        currentFiles.push(f);
        const url = URL.createObjectURL(f);
        fileThumbURLs.push(url);
    });

    renderThumbnails();
}

function renderThumbnails() {
    const container = document.getElementById('thumbnails');
    container.innerHTML = '';
    fileThumbURLs.forEach((url, i) => {
        const wrap = document.createElement('div');
        wrap.style.position = 'relative';
        
        const img = document.createElement('img');
        img.src = url;
        img.style.width = '80px';
        img.style.height = '80px';
        img.style.objectFit = 'cover';
        img.style.borderRadius = '8px';
        
        const remove = document.createElement('button');
        remove.setAttribute('aria-label', 'Remove image');
        remove.style.position = 'absolute';
        remove.style.top = '-12px';
        remove.style.right = '-12px';
        remove.style.width = '44px';
        remove.style.height = '44px';
        remove.style.minWidth = '44px';
        remove.style.minHeight = '44px';
        remove.style.background = 'transparent';
        remove.style.border = 'none';
        remove.style.display = 'flex';
        remove.style.alignItems = 'center';
        remove.style.justifyContent = 'center';
        remove.style.padding = '0';
        remove.style.cursor = 'pointer';
        remove.innerHTML = '<span style="width:22px;height:22px;border-radius:50%;background:var(--color-error);color:white;display:flex;align-items:center;justify-content:center;font-size:14px;line-height:1;box-shadow:var(--neu-raised-sm);">×</span>';
        
        remove.onclick = (e) => {
            e.stopPropagation();
            URL.revokeObjectURL(url);
            currentFiles.splice(i, 1);
            fileThumbURLs.splice(i, 1);
            renderThumbnails();
        };

        wrap.appendChild(img);
        wrap.appendChild(remove);
        container.appendChild(wrap);
    });
}

// ---- GPS ----
function setupGPSHandlers() {
    document.getElementById('btn-gps').addEventListener('click', () => {
        if (!navigator.geolocation) {
            showToast('Geolocation is not supported by your browser.', 'error');
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                document.getElementById('input-lat').value = pos.coords.latitude;
                document.getElementById('input-lng').value = pos.coords.longitude;
                document.getElementById('gps-source-text').textContent = 'Source: Browser (Device)';
                validateGPS(pos.coords.latitude, pos.coords.longitude);
            },
            (err) => {
                showToast('Geolocation permission denied or failed.', 'error');
                document.getElementById('gps-source-text').textContent = 'Source: Unknown / Denied';
            }
        );
    });
}

function validateGPS(lat, lng) {
    // Basic Philippines bounding box roughly 4 to 21 Lat, 116 to 127 Lng
    if (lat < 4 || lat > 21 || lng < 116 || lng > 127) {
        showToast('Warning: Location appears to be outside the Philippines.', 'warning');
    }
}

// ---- PIPELINE EXECUTION ----
async function startAnalysis() {
    if (currentFiles.length === 0) {
        showToast('Please select at least one image.', 'error');
        return;
    }

    const company = document.getElementById('input-company').value.trim();
    const barangay = document.getElementById('input-barangay').value.trim();
    const street = document.getElementById('input-street').value.trim();
    const notes = document.getElementById('input-notes').value.trim();
    
    if (!company || !barangay || !street) {
        showToast('Please fill in Company, Barangay, and Street.', 'error');
        return;
    }

    const lat = document.getElementById('input-lat').value;
    const lng = document.getElementById('input-lng').value;
    
    if (lat && lng) validateGPS(parseFloat(lat), parseFloat(lng));
    
    stopAllPolling();
    resetResultTiles();
    const reasonEl = document.getElementById('status-reason');
    if (reasonEl) reasonEl.textContent = '';
    
    document.getElementById('tile-batch').style.display = 'block';
    
    try {
        if (currentFiles.length === 1) {
            const formData = new FormData();
            formData.append('file', currentFiles[0]);
            formData.append('company', company);
            formData.append('barangay', barangay);
            formData.append('street', street);
            if (notes) formData.append('notes', notes);
            if (lat) formData.append('gps_lat', lat);
            if (lng) formData.append('gps_lng', lng);
            
            const res = await fetchApi('/api/analysis/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            history.replaceState(null, '', `?id=${data.analysis_id}`);
            batchItems = [{ id: data.analysis_id, status: 'queued', thumb: fileThumbURLs[0], name: currentFiles[0].name }];
            renderBatchTabs();
            switchTab(0);
            updateGlobalStepper('processing');
            startPolling(data.analysis_id);
            
        } else {
            const formData = new FormData();
            currentFiles.forEach(f => formData.append('files', f));
            formData.append('company', company);
            formData.append('barangay', barangay);
            formData.append('street', street);
            if (notes) formData.append('notes', notes);
            
            const res = await fetchApi('/api/batches', {
                method: 'POST',
                body: formData,
                allowAnon: true
            });
            const data = await res.json();
            
            const batchRes = await fetchApi(`/api/batches/${data.batch_id}`);
            const batchData = await batchRes.json();
            
            currentBatchId = data.batch_id; // Added for cancel
            if (batchData.analyses && batchData.analyses.length > 0) {
                history.replaceState(null, '', `?id=${batchData.analyses[0].id}`);
            }
            
            batchItems = batchData.analyses.map((a, i) => ({
                id: a.id,
                status: a.status,
                thumb: fileThumbURLs[i] || '',
                name: currentFiles[i] ? currentFiles[i].name : `Item ${i+1}`
            }));
            
            renderBatchTabs();
            switchTab(0);
            updateGlobalStepper('processing');
            batchItems.forEach(item => startPolling(item.id));
        }
    } catch (e) {
        console.error("Upload error:", e);
        showToast(e.message, 'error');
    }
}

// ---- BATCH SWITCHER ----
function renderBatchTabs() {
    const tabsContainer = document.getElementById('batch-tabs');
    tabsContainer.innerHTML = '';
    
    batchItems.forEach((item, i) => {
        const btn = document.createElement('button');
        btn.className = `neu-button ${i === currentActiveItemIndex ? 'neu-pressed' : 'neu-flat'}`;
        btn.style.display = 'flex';
        btn.style.alignItems = 'center';
        btn.style.gap = '8px';
        btn.style.minWidth = '150px';
        btn.setAttribute('role', 'tab');
        btn.setAttribute('aria-selected', i === currentActiveItemIndex ? 'true' : 'false');
        btn.tabIndex = i === currentActiveItemIndex ? 0 : -1;
        
        let indicatorColor = 'var(--text-muted)';
        if (item.status === 'complete') indicatorColor = 'var(--risk-low)';
        if (['error', 'rejected', 'interrupted'].includes(item.status)) indicatorColor = 'var(--risk-critical)';
        
        btn.innerHTML = `
            <img src="${escapeHtml(item.thumb)}" style="width:24px; height:24px; object-fit:cover; border-radius:4px;" alt="Thumbnail">
            <span style="flex:1; text-align:left; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(item.name)}</span>
            <span style="width:10px; height:10px; border-radius:50%; background-color:${indicatorColor};" title="${escapeHtml(item.status)}"></span>
        `;
        
        btn.addEventListener('click', () => switchTab(i));
        
        btn.addEventListener('keydown', (e) => {
            let nextIndex = null;
            if (e.key === 'ArrowRight') nextIndex = (i + 1) % batchItems.length;
            if (e.key === 'ArrowLeft') nextIndex = (i - 1 + batchItems.length) % batchItems.length;
            
            if (nextIndex !== null) {
                switchTab(nextIndex);
                tabsContainer.children[nextIndex].focus();
            }
        });
        
        tabsContainer.appendChild(btn);
    });
}

async function switchTab(index) {
    currentActiveItemIndex = index;
    renderBatchTabs(); // update selected state
    
    const activeItem = batchItems[index];
    
    // Clear UI before repopulating
    resetResultTiles();
    document.getElementById('tile-progress').style.display = 'block';
    document.getElementById('stepper').innerHTML = '';
    const reasonEl = document.getElementById('status-reason');
    if (reasonEl) reasonEl.textContent = '';
    
    // Fetch latest instantly
    try {
        const [eventsRes, analysisRes] = await Promise.all([
            fetchApi(`/api/analysis/${activeItem.id}/events`),
            fetchApi(`/api/analysis/${activeItem.id}/result`)
        ]);
        const events = await eventsRes.json();
        const analysis = await analysisRes.json();
        
        renderEvents(events);
        const pres = statusPresentation(analysis);
        updateStatusBadge(pres.text, pres.risk, pres.reason);
        updateCancelButton(activeItem.status);
        
        const terminalStatuses = ['complete', 'rejected', 'no_wires', 'error', 'interrupted'];
        if (terminalStatuses.includes(analysis.status)) {
            populateResultTiles(analysis);
        }

        const pdfBtn = document.getElementById('btn-download-pdf');
        if (analysis.status === 'complete' || analysis.status === 'no_wires') {
            pdfBtn.href = `/api/reports/${analysis.id}.pdf`;
            pdfBtn.style.display = 'inline-block';
        } else {
            pdfBtn.style.display = 'none';
        }
        
        const watermark = document.getElementById('mock-watermark');
        if (watermark) {
            watermark.style.display = analysis.is_mock ? 'block' : 'none';
        }
    } catch (e) {
        showToast('Failed to load item: ' + e.message, 'error');
    }
}

// ---- POLLING ----
function startPolling(analysisId) {
    if (pollingTimers[analysisId]) return; // already polling
    pollItem(analysisId, 0);
}

document.getElementById('btn-cancel')?.addEventListener('click', async () => {
    const activeId = batchItems[currentActiveItemIndex]?.id;
    if (!activeId && !currentBatchId) return;
    
    // We try to cancel the batch if batch, or the analysis. 
    // Wait, the API specifies POST /api/batches/{id}/cancel
    const batchToCancel = currentBatchId || activeId; // If single upload, it created a batch under the hood. Let's just use currentBatchId.
    if (!currentBatchId) {
        showToast("No batch ID available to cancel.", "warning");
        return;
    }
    
    try {
        await fetchApi(`/api/batches/${currentBatchId}/cancel`, { method: 'POST' });
        showToast('Cancel requested', 'info');
        // Let polling pick up the interrupted state
    } catch (e) {
        showToast('Failed to cancel: ' + e.message, 'error');
    }
});

function stopAllPolling() {
    Object.values(pollingTimers).forEach(clearTimeout);
    pollingTimers = {};
}

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopAllPolling();
    } else {
        batchItems.forEach(item => {
            if (!['complete', 'rejected', 'no_wires', 'error', 'interrupted'].includes(item.status)) {
                startPolling(item.id);
            }
        });
    }
});

async function pollItem(analysisId, errorCount) {
    if (activePollsCount >= MAX_POLLING_CONCURRENCY) {
        pollingTimers[analysisId] = setTimeout(() => pollItem(analysisId, errorCount), 1000);
        return;
    }

    activePollsCount++;
    try {
        const [eventsRes, analysisRes] = await Promise.all([
            fetchApi(`/api/analysis/${analysisId}/events`),
            fetchApi(`/api/analysis/${analysisId}/result`)
        ]);
        
        const events = await eventsRes.json();
        const analysis = await analysisRes.json();
        
        // Update item status in batchItems
        const itemObj = batchItems.find(b => b.id === analysisId);
        if (itemObj && itemObj.status !== analysis.status) {
            itemObj.status = analysis.status;
            renderBatchTabs();
        }

        // Only update DOM if this is the active tab
        if (batchItems[currentActiveItemIndex]?.id === analysisId) {
            renderEvents(events);
            const pres = statusPresentation(analysis);
            updateStatusBadge(pres.text, pres.risk, pres.reason);
            updateCancelButton(analysis.status);

            const pdfBtn = document.getElementById('btn-download-pdf');
            if (analysis.status === 'complete' || analysis.status === 'no_wires') {
                pdfBtn.href = `/api/reports/${analysis.id}.pdf`;
                pdfBtn.style.display = 'inline-block';
            } else {
                pdfBtn.style.display = 'none';
            }
            
            const watermark = document.getElementById('mock-watermark');
            if (watermark) {
                watermark.style.display = analysis.is_mock ? 'block' : 'none';
            }
        }
        
        if (isTerminalStatus(analysis.status)) {
            delete pollingTimers[analysisId];
            if (batchItems[currentActiveItemIndex]?.id === analysisId) {
                try {
                    const finalEventsRes = await fetchApi(`/api/analysis/${analysisId}/events`);
                    const finalEvents = await finalEventsRes.json();
                    renderEvents(finalEvents);
                } catch (e) {
                    console.warn('Failed to fetch final events:', e);
                }
                populateResultTiles(analysis);
                if (shouldShowPublicConfirmation(analysis, currentUser)) {
                    setTimeout(() => {
                        const modal = document.getElementById('public-confirmation-modal');
                        if (modal) modal.classList.add('open');
                    }, 1500); // 1.5 second delay to let them see results first
                }
            }
        } else {
            pollingTimers[analysisId] = setTimeout(() => pollItem(analysisId, 0), 2000); // 2s backoff
        }
    } catch (e) {
        errorCount++;
        if (errorCount > 3) {
            delete pollingTimers[analysisId];
            if (batchItems[currentActiveItemIndex]?.id === analysisId) {
                updateStatusBadge('ERROR', 'Critical', 'Failed to poll server: ' + e.message);
            }
        } else {
            const backoff = Math.min(10000, 2000 * Math.pow(2, errorCount));
            pollingTimers[analysisId] = setTimeout(() => pollItem(analysisId, errorCount), backoff);
        }
    } finally {
        activePollsCount--;
    }
}

function renderEvents(events) {
    const stepper = document.getElementById('stepper');
    stepper.innerHTML = '';
    
    events.forEach((ev, i) => {
        const isLast = i === events.length - 1;
        const div = document.createElement('div');
        div.className = `neu-step ${isLast ? 'active' : 'completed'}`;
        
        const iconSvg = isLast 
            ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>` 
            : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
            
        div.innerHTML = `
            <div class="neu-step-icon" style="display: flex; align-items: center; justify-content: center;">${iconSvg}</div>
            <span>${escapeHtml(ev.state)} - <small>${escapeHtml(ev.detail)}</small></span>
        `;
        stepper.appendChild(div);
    });
}

function updateStatusBadge(text, risk, reason) {
    const badge = document.getElementById('status-badge');
    badge.innerHTML = `<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="5"/></svg> ${escapeHtml(text)}`;
    badge.setAttribute('data-risk', risk);
    
    const reasonEl = document.getElementById('status-reason');
    if (reasonEl) reasonEl.textContent = reason || '';
}

function updateCancelButton(status) {
    const btn = document.getElementById('btn-cancel');
    if (!btn) return;
    const terminalStatuses = ['complete', 'rejected', 'no_wires', 'error', 'interrupted'];
    if (terminalStatuses.includes(status)) {
        btn.style.display = 'none';
    } else {
        btn.style.display = 'inline-block';
    }
}

// ---- RENDERING RESULTS ----
function resetResultTiles() {
    currentAnalysis = null;
    const reasonEl = document.getElementById('status-reason');
    if (reasonEl) reasonEl.textContent = '';
    ['tile-image', 'tile-safety', 'tile-action', 'tile-crs', 'tile-sag', 'tile-vni', 'tile-wires', 'tile-disclosure'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
    updateCalibrationButtonVisibility();
}

function populateResultTiles(analysis) {
    if (analysis.status === 'rejected' || analysis.status === 'error') return;

    currentAnalysis = analysis;

    // Show image regardless of wires (no_wires needs image)
    document.getElementById('tile-image').style.display = 'block';

    updateCalibrationButtonVisibility();
    
    // Add timestamp query param to force reload if re-processed (e.g. calibration)
    const ts = new Date().getTime();
    document.getElementById('img-annotated').src = `/api/analysis/${analysis.id}/overlay?t=${ts}`;
    
    document.getElementById('tile-disclosure').style.display = 'block';

    if (analysis.status === 'no_wires') {
        document.getElementById('tile-vni').style.display = 'block';
        document.getElementById('vni-value').textContent = (analysis.vni !== null && analysis.vni !== undefined) ? analysis.vni.toFixed(2) : '--';
        document.getElementById('vni-spectrum').src = `/api/analysis/${analysis.id}/spectrum?t=${ts}`;
        
        document.getElementById('tile-action').style.display = 'block';
        document.getElementById('action-text').textContent = actionMapping['no_wires'] || 'No action needed.';
    }

    // Results
    document.getElementById('tile-safety').style.display = 'block';
    document.getElementById('tile-action').style.display = 'block';
    document.getElementById('tile-vni').style.display = 'block';

    if (analysis.status !== 'no_wires') {
        document.getElementById('tile-crs').style.display = 'block';
        document.getElementById('tile-sag').style.display = 'block';
        document.getElementById('tile-wires').style.display = 'block';
    }

    updateTilesFromJson(analysis);
}

function updateTilesFromJson(analysis) {
    const pres = safetyPresentation(analysis);

    // Safety
    
    const safetyBadge = document.getElementById('safety-badge');
    
    // Metadata
    if (analysis.company || analysis.barangay || analysis.street || analysis.notes) {
        document.getElementById('analysis-metadata').style.display = 'block';
        document.getElementById('meta-company').textContent = analysis.company || '--';
        document.getElementById('meta-location').textContent = (analysis.barangay || '--') + ' / ' + (analysis.street || '--');
        if (analysis.notes) {
            document.getElementById('meta-notes-container').style.display = 'block';
            document.getElementById('meta-notes').textContent = analysis.notes;
        } else {
            document.getElementById('meta-notes-container').style.display = 'none';
        }
    } else {
        document.getElementById('analysis-metadata').style.display = 'none';
    }

    safetyBadge.innerHTML = `<svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="5"/></svg> ${escapeHtml(pres.text)}`;
    safetyBadge.setAttribute('data-risk', pres.risk);
    
    const reasonsList = document.getElementById('safety-reasons');
    reasonsList.innerHTML = '';
    pres.reasons.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        reasonsList.appendChild(li);
    });

    // Action
    let actionText = actionMapping[pres.risk] || 'No action defined.';
    if (analysis.status === 'no_wires') {
        actionText = actionMapping['no_wires'] || 'No action needed.';
    } else if (pres.risk === 'critical' && analysis.ground_clearance_status === 'below' && analysis.crs_band !== 'Critical') {
        actionText = 'Immediate dispatch required. Ground clearance below PEC minimum.';
    } else if (pres.risk === 'high' && analysis.ground_clearance_status === 'near' && analysis.crs_band !== 'High' && analysis.crs_band !== 'Critical') {
        actionText = 'Schedule maintenance soon. Ground clearance nearing PEC minimum.';
    }
    
    document.getElementById('action-text').textContent = actionText;

    // CRS
    document.getElementById('crs-value').textContent = pres.crsText;
    document.getElementById('crs-band').textContent = analysis.crs_band || '--';
    const gauge = document.getElementById('crs-fill');
    
    let crsPercent = 0;
    if (analysis.crs !== null && analysis.crs !== undefined && analysis.crs_band !== 'unavailable') {
        crsPercent = analysis.crs * 100;
    }
    gauge.style.width = `${crsPercent}%`;
    gauge.style.backgroundColor = pres.crsColor;
    
    const gaugeContainer = document.getElementById('crs-gauge');
    gaugeContainer.setAttribute('aria-valuenow', analysis.crs !== null ? analysis.crs : 0);
    gaugeContainer.setAttribute('aria-valuetext', pres.crsText);

    // Sag & Clearance
    const sag = analysis.sag_m !== null ? analysis.sag_m : '--';
    const cl = analysis.ground_clearance_m !== null ? analysis.ground_clearance_m : '--';
    document.getElementById('sag-value').textContent = `${typeof sag === 'number' ? sag.toFixed(2) : sag} m`;
    document.getElementById('clearance-value').textContent = `${typeof cl === 'number' ? cl.toFixed(2) : cl} m`;

    // VNI
    document.getElementById('vni-value').textContent = (analysis.vni !== null && analysis.vni !== undefined) ? analysis.vni.toFixed(2) : '--';
    const ts = new Date().getTime();
    document.getElementById('vni-spectrum').src = `/api/analysis/${analysis.id}/spectrum?t=${ts}`;
    
    // Wire List
    const wireList = document.getElementById('wire-list');
    wireList.innerHTML = '';
    if (analysis.wires) {
        analysis.wires.forEach((w, i) => {
            const div = document.createElement('div');
            div.style.padding = '10px';
            div.style.border = 'var(--border-subtle)';
            div.style.borderRadius = '8px';
            div.style.display = 'flex';
            div.style.justifyContent = 'space-between';
            
            const color = getWireClassColor(w.class_name);
            div.innerHTML = `
                <div>
                    <span style="display:inline-block;width:12px;height:12px;background:${color};border-radius:50%;margin-right:8px;"></span>
                    <strong>Wire ${i+1} (${escapeHtml(w.class_name)})</strong>
                    <div style="font-size:0.85rem;color:var(--text-muted);">Confidence: ${formatConfidence(w.confidence)}</div>
                </div>
                <div style="text-align:right;">
                    <div>CRS: ${w.crs !== null ? w.crs.toFixed(2) : '--'}</div>
                    <div style="font-size:0.85rem;color:var(--text-muted);">Sag: ${w.sag_m !== null ? w.sag_m.toFixed(2) : '--'}m</div>
                </div>
            `;
            wireList.appendChild(div);
        });
    }
}


// ---- MANUAL CALIBRATION TOOL ----
function setupCalibrationTool() {
    const btnOpen = document.getElementById('btn-calibrate');
    const modal = document.getElementById('calibration-modal');
    const btnCancel = document.getElementById('btn-cal-cancel');
    const btnSubmit = document.getElementById('btn-cal-submit');
    const canvas = document.getElementById('calibration-canvas');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('calibration-img');
    const ptsList = document.getElementById('cal-points-list');

    // Focus Trap
    const focusableElementsString = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
    
    btnOpen.addEventListener('click', () => {
        if (!currentUser) {
            showToast('Authentication required for calibration.', 'error');
            return;
        }
        modal.classList.add('open');
        canvas.focus();
        
        const activeId = batchItems[currentActiveItemIndex]?.id;
        if (activeId) {
            img.src = `/api/analysis/${activeId}/original?t=${new Date().getTime()}`;
        }
        
        img.onload = () => {
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            drawPoints();
        };
        calPoints = [];
        drawPoints();
    });

    btnCancel.addEventListener('click', () => {
        modal.classList.remove('open');
        calPoints = [];
        btnOpen.focus();
    });
    
    document.getElementById('btn-cal-add-pt')?.addEventListener('click', () => {
        const vx = parseFloat(document.getElementById('cal-x').value);
        const vy = parseFloat(document.getElementById('cal-y').value);
        if (isNaN(vx) || isNaN(vy)) {
            showToast('Invalid X/Y', 'error');
            return;
        }
        if (calPoints.length >= 2) return;
        
        calPoints.push({ x: vx, y: vy });
        drawPoints();
    });
    
    // Focus Trap logic for modal
    modal.addEventListener('keydown', (e) => {
        const focusableElements = modal.querySelectorAll(focusableElementsString);
        const first = focusableElements[0];
        const last = focusableElements[focusableElements.length - 1];
        
        if (e.key === 'Tab') {
            if (e.shiftKey) {
                if (document.activeElement === first) {
                    e.preventDefault();
                    last.focus();
                }
            } else {
                if (document.activeElement === last) {
                    e.preventDefault();
                    first.focus();
                }
            }
        }
    });

    canvas.addEventListener('click', (e) => {
        if (calPoints.length >= 2) return;
        
        const coords = getNaturalCoordinates(e, img, img.naturalWidth, img.naturalHeight, 'contain');
        calPoints.push(coords);
        drawPoints();
    });
    
    let crosshairPos = { x: canvas.clientWidth / 2, y: canvas.clientHeight / 2 };
    
    canvas.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            modal.classList.remove('open');
            btnOpen.focus();
            return;
        }
        
        const step = e.shiftKey ? 10 : 2;
        let moved = false;
        
        if (e.key === 'ArrowUp') { crosshairPos.y -= step; moved = true; }
        if (e.key === 'ArrowDown') { crosshairPos.y += step; moved = true; }
        if (e.key === 'ArrowLeft') { crosshairPos.x -= step; moved = true; }
        if (e.key === 'ArrowRight') { crosshairPos.x += step; moved = true; }
        
        if (moved) {
            e.preventDefault();
            // clamp
            crosshairPos.x = Math.max(0, Math.min(canvas.clientWidth, crosshairPos.x));
            crosshairPos.y = Math.max(0, Math.min(canvas.clientHeight, crosshairPos.y));
            drawPoints();
        }
        
        if (e.key === 'Enter') {
            e.preventDefault();
            if (calPoints.length >= 2) return;
            // simulate click at crosshair
            const coords = getNaturalCoordinates({ clientX: crosshairPos.x + canvas.getBoundingClientRect().left, clientY: crosshairPos.y + canvas.getBoundingClientRect().top }, img, img.naturalWidth, img.naturalHeight, 'contain');
            calPoints.push(coords);
            drawPoints();
        }
    });

    function drawPoints() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const rect = img.getBoundingClientRect();
        const scaleX = rect.width / img.naturalWidth;
        const scaleY = rect.height / img.naturalHeight;
        const scale = Math.min(scaleX, scaleY);
        const offsetX = (rect.width - img.naturalWidth * scale) / 2;
        const offsetY = (rect.height - img.naturalHeight * scale) / 2;
        
        // draw crosshair if focused
        if (document.activeElement === canvas) {
            ctx.beginPath();
            ctx.moveTo(crosshairPos.x - 10, crosshairPos.y);
            ctx.lineTo(crosshairPos.x + 10, crosshairPos.y);
            ctx.moveTo(crosshairPos.x, crosshairPos.y - 10);
            ctx.lineTo(crosshairPos.x, crosshairPos.y + 10);
            ctx.strokeStyle = '#fff';
            ctx.stroke();
        }

        calPoints.forEach((p) => {
            const drawX = p.x * scale + offsetX;
            const drawY = p.y * scale + offsetY;
            
            ctx.beginPath();
            ctx.arc(drawX, drawY, 5, 0, 2 * Math.PI);
            ctx.fillStyle = '#e74c3c';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.stroke();
        });
        
        if (calPoints.length === 2) {
            ptsList.textContent = 'Scale reference points selected (2). Enter real length (meters) and click Calibrate.';
        } else {
            ptsList.textContent = `Points selected: ${calPoints.length} / 2`;
        }
    }

    btnSubmit.addEventListener('click', async () => {
        const len = parseFloat(document.getElementById('cal-length').value);
        if (calPoints.length !== 2) {
            showToast('Please select exactly 2 points.', 'error');
            return;
        }
        if (isNaN(len) || len < 0.1 || len > 100) {
            showToast('Please enter a valid length between 0.1 and 100 meters.', 'error');
            return;
        }

        const payload = {
            points: calPoints.slice(0, 2),
            real_length_m: len
        };

        const activeId = batchItems[currentActiveItemIndex]?.id;
        if (!activeId) return;

        try {
            const res = await fetchApi(`/api/analysis/${activeId}/calibrate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            showToast('Calibration applied.', 'success');
            modal.classList.remove('open');
            btnOpen.focus();
            
            // Re-fetch analysis to get updated tiles
            const analysisRes = await fetchApi(`/api/analysis/${activeId}/result`);
            const analysis = await analysisRes.json();
            populateResultTiles(analysis);

            const ts = new Date().getTime();
            const imgAnnotated = document.getElementById('img-annotated');
            if (imgAnnotated) {
                imgAnnotated.src = `/api/analysis/${activeId}/overlay?t=${ts}`;
            }
            
            // Aria-live announcement
            const reasonEl = document.getElementById('status-reason');
            if (reasonEl) reasonEl.textContent = 'Calibration updated successfully. Results refreshed.';
        } catch (e) {
            showToast('Calibration failed: ' + e.message, 'error');
        }
    });
}

// ---- TOASTS ----
