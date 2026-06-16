let currentData = null;
let macroEvents = [];

const workspaceState = {
    selectedEventId: null,
    selectedCategory: '',
    searchQuery: '',
    sortOrder: 'desc'
};

/**
 * Escapes HTML entities to prevent XSS when inserting untrusted data.
 * @param {string} str - Raw string from server data
 * @returns {string} Escaped string safe for innerHTML
 */
function sanitizeHTML(str) {
    if (str === null || str === undefined) return '';
    const text = String(str);
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

document.addEventListener('DOMContentLoaded', () => {
    if (document.documentElement.classList.contains('dark-mode')) {
        document.body.classList.add('dark-mode');
    }
    setupThemeToggle();
    fetchHistoryIndex();
    fetchData();
    setupNavigation();
    setupModal();
    setupClickableCards();
    setupMacroControls();
    setupDelegatedClicks();
});


function setupMacroControls() {
    const searchInput = document.getElementById('macro-search');
    const categorySelect = document.getElementById('macro-category-filter');
    const sortSelect = document.getElementById('macro-sort');

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            workspaceState.searchQuery = e.target.value.toLowerCase();
            renderMacroTimeline();
        });
    }

    if (categorySelect) {
        categorySelect.addEventListener('change', (e) => {
            workspaceState.selectedCategory = e.target.value;
            renderMacroTimeline();
        });
    }

    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            workspaceState.sortOrder = e.target.value;
            renderMacroTimeline();
        });
    }

    // Keyboard navigation for timeline
    document.addEventListener('keydown', (e) => {
        const macroTab = document.getElementById('macro');
        if (!macroTab || macroTab.style.display === 'none') return;
        
        // Skip if typing in search
        if (e.target.id === 'macro-search') return;
        
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            const cards = Array.from(document.querySelectorAll('.macro-event-card'));
            if (cards.length === 0) return;
            
            e.preventDefault();
            const currentIndex = cards.findIndex(c => c.dataset.eventId === workspaceState.selectedEventId);
            
            if (e.key === 'ArrowDown') {
                const nextIndex = currentIndex < cards.length - 1 ? currentIndex + 1 : 0;
                cards[nextIndex].focus();
                cards[nextIndex].click();
                cards[nextIndex].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else if (e.key === 'ArrowUp') {
                const prevIndex = currentIndex > 0 ? currentIndex - 1 : Math.max(0, cards.length - 1);
                cards[prevIndex].focus();
                cards[prevIndex].click();
                cards[prevIndex].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    });
}

function setupDelegatedClicks() {
    const detailContent = document.getElementById('macro-detail-content');
    if (detailContent) {
        detailContent.addEventListener('click', (e) => {
            const trigger = e.target.closest('.related-event-trigger');
            if (trigger) {
                e.preventDefault();
                const eventId = trigger.dataset.eventId;
                selectMacroEvent(eventId);
            }
        });
    }

    const modal = document.getElementById('premium-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            // Delegated click for completion tabs
            const tabBtn = e.target.closest('.completion-tab-trigger');
            if (tabBtn) {
                const tabName = tabBtn.dataset.tab;
                switchCompletionTab(tabName);
                return;
            }
            
            // Delegated click for completion group toggle
            const rowTrigger = e.target.closest('.completion-row-trigger');
            if (rowTrigger) {
                const targetId = rowTrigger.dataset.target;
                const el = document.getElementById(targetId);
                if (el) {
                    el.style.display = el.style.display === 'none' ? 'block' : 'none';
                }
                return;
            }
        });

        // Hover delegation for confidence cells in the modal
        modal.addEventListener('mouseover', (e) => {
            const cell = e.target.closest('.confidence-cell');
            if (cell) {
                showConfidenceTip(cell);
            }
        });

        modal.addEventListener('mouseout', (e) => {
            const cell = e.target.closest('.confidence-cell');
            if (cell) {
                hideConfidenceTip(cell);
            }
        });
    }

    const backtestTab = document.getElementById('backtest');
    if (backtestTab) {
        backtestTab.addEventListener('click', (e) => {
            const card = e.target.closest('#backtest-completions-card');
            if (card) {
                openCompletionsDrilldown();
            }
        });
    }
}

function setupThemeToggle() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    
    btn.addEventListener('click', () => {
        const isDark = document.body.classList.contains('dark-mode');
        if (isDark) {
            document.body.classList.remove('dark-mode');
            document.documentElement.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        } else {
            document.body.classList.add('dark-mode');
            document.documentElement.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
        }
    });
}

async function fetchHistoryIndex() {
    try {
        const response = await fetch('./history/index.json');
        if (response.ok) {
            const historyDates = await response.json();
            const selector = document.getElementById('history-selector');
            if (historyDates && historyDates.length > 0) {
                historyDates.forEach(date => {
                    const opt = document.createElement('option');
                    opt.value = date;
                    opt.textContent = date;
                    selector.appendChild(opt);
                });
                selector.classList.remove('hidden');
                
                selector.addEventListener('change', (e) => {
                    const selectedDate = e.target.value;
                    fetchData(selectedDate);
                });
            }
        }
    } catch (e) {
        console.log("No history index found.");
    }
}

function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault(); // prevent default anchor scroll
            // Remove active from all
            navItems.forEach(nav => nav.classList.remove('active'));
            // Add active to clicked
            item.classList.add('active');
            
            // Switch tabs
            const targetId = item.getAttribute('href').replace('#', '');
            switchTab(targetId);
            
            // Update URL hash without jumping
            history.pushState(null, null, item.getAttribute('href'));
        });
    });
    
    // Check initial hash
    if (window.location.hash) {
        const hash = window.location.hash.replace('#', '');
        const activeNav = document.querySelector(`.nav-item[href="#${hash}"]`);
        if (activeNav) {
            activeNav.click();
        }
    }
}

function switchTab(tabId) {
    const overview = document.getElementById('overview');
    const consensus = document.getElementById('consensus');
    const eigen = document.getElementById('eigen');
    const macro = document.getElementById('macro');
    const analytics = document.getElementById('analytics');
    const ete = document.getElementById('ete');
    const alerts = document.getElementById('alerts-container'); 
    const backtest = document.getElementById('backtest');

    const sections = [overview, consensus, eigen, macro, analytics, ete, backtest];
    sections.forEach(s => { if (s) s.style.display = 'none'; });

    const mainPanels = document.querySelector('.main-panels');
    const sidePanels = document.querySelector('.side-panels');

    if (tabId === 'overview') {
        if (mainPanels) mainPanels.classList.remove('single-panel');
        if (sidePanels) sidePanels.style.display = 'flex';
        
        overview.style.display = 'grid';
        consensus.style.display = 'flex';
        eigen.style.display = 'flex';
        if (alerts) alerts.style.display = 'flex';
    } else {
        if (mainPanels) mainPanels.classList.add('single-panel');
        
        if (tabId === 'eigen') {
            if (sidePanels) sidePanels.style.display = 'flex';
            eigen.style.display = 'flex';
            if (alerts) alerts.style.display = 'none';
        } else {
            if (sidePanels) sidePanels.style.display = 'none';
            
            if (tabId === 'consensus') {
                consensus.style.display = 'flex';
                if (alerts) alerts.style.display = 'none';
            } else if (tabId === 'macro') {
                macro.style.display = 'flex';
                if (alerts) alerts.style.display = 'none';
            } else if (tabId === 'analytics') {
                if (analytics) analytics.style.display = 'block';
                if (alerts) alerts.style.display = 'none';
            } else if (tabId === 'ete') {
                overview.style.display = 'none';
                consensus.style.display = 'none';
                eigen.style.display = 'none';
                if (macro) macro.style.display = 'none';
                if (ete) {
                    ete.style.display = 'flex';
                    if (!window.eteLoaded) {
                        const selectedDate = document.getElementById('history-selector')?.value || '';
                        fetchETEManifest(selectedDate);
                        window.eteLoaded = true;
                    }
                }
                if (alerts) alerts.style.display = 'none';
                const backtest = document.getElementById('backtest');
                if (backtest) backtest.style.display = 'none';
            } else if (tabId === 'backtest') {
                overview.style.display = 'none';
                consensus.style.display = 'none';
                eigen.style.display = 'none';
                if (macro) macro.style.display = 'none';
                if (ete) ete.style.display = 'none';
                if (alerts) alerts.style.display = 'none';
                const backtest = document.getElementById('backtest');
                if (backtest) {
                    backtest.style.display = 'flex';
                    if (!window.backtestLoaded) {
                        fetchBacktestMetrics();
                        window.backtestLoaded = true;
                    }
                }
            }
        }
    }
}

async function fetchData(date = '') {
    try {
        const dataUrl = date ? `./history/data_${date}.json` : './data.json';
        const analyticsUrl = date ? `./history/analytics_${date}.json` : './analytics.json';
        const manifestUrl = date ? `./history/manifest_${date}.json` : './manifest.json';
        
        const [dataRes, analyticsRes, manifestRes] = await Promise.all([
            fetch(dataUrl),
            fetch(analyticsUrl).catch(() => null),
            fetch(manifestUrl).catch(() => null)
        ]);
        
        if (!dataRes.ok) {
            throw new Error(`HTTP error! status: ${dataRes.status}`);
        }
        
        const data = await dataRes.json();
        const analytics = analyticsRes && analyticsRes.ok ? await analyticsRes.json() : null;
        const manifest = manifestRes && manifestRes.ok ? await manifestRes.json() : null;
        
        if (analytics) {
            data.analytics = analytics;
        }
        if (manifest) {
            data.manifest = manifest;
        }

        currentData = data;
        renderDashboard(data);
        
        if (analytics && analytics.analytics) {
            renderAnalyticsWorkspace(analytics.analytics);
        }
        
        const generatedAt = manifest ? manifest.generated_at : data.generated_at;
        if (generatedAt) {
            checkStaleData(generatedAt);
        }
        
        // Reset ETE and Backtest load states so they reload when switched to
        window.eteLoaded = false;
        window.backtestLoaded = false;
        
        // If currently on ETE tab, reload immediately
        const activeTab = document.querySelector('.nav-item.active')?.getAttribute('href')?.replace('#', '') || 'overview';
        if (activeTab === 'ete') {
            fetchETEManifest(date);
            window.eteLoaded = true;
        }
        
        document.getElementById('error-boundary').classList.add('hidden');
        document.getElementById('dashboard-content').style.display = 'flex';
        
    } catch (error) {
        console.error('Failed to fetch or parse dashboard data:', error);
        showError(error.message);
    }
}

function renderAnalyticsWorkspace(analytics) {
    const content = document.getElementById('analytics-content');
    if (!content) return;
    
    const biz = analytics.business || {};
    const ai = analytics.ai || {};
    const ops = analytics.operational || {};
    const qual = analytics.quality || {};
    
    content.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
            <!-- Business Metrics -->
            <div class="glass-panel" style="padding: 20px; border-radius: 8px;">
                <h3 style="margin-top:0; color:var(--primary);">Business Health</h3>
                <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">${analytics.total_events}</div>
                <div style="color:var(--text-muted); font-size:0.9rem;">Total Processed Events</div>
                
                <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin: 15px 0;">
                
                <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                    <span style="color:var(--text-muted);">High Priority:</span>
                    <span class="premium-badge badge-strong">${biz.high_priority_circulars || 0}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--text-muted);">Upcoming Effective:</span>
                    <span class="premium-badge badge-gap-up">${biz.upcoming_effective_dates || 0}</span>
                </div>
            </div>
            
            <!-- AI Performance -->
            <div class="glass-panel" style="padding: 20px; border-radius: 8px;">
                <h3 style="margin-top:0; color:var(--accent-color);">AI Enrichment</h3>
                <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">${(ai.processing_success_rate * 100).toFixed(1)}%</div>
                <div style="color:var(--text-muted); font-size:0.9rem;">AI Generation Success</div>
                
                <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin: 15px 0;">
                
                <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                    <span style="color:var(--text-muted);">Avg Latency:</span>
                    <span class="premium-badge badge-label">${(ai.avg_latency_ms || 0).toFixed(0)} ms</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--text-muted);">Theme Coverage:</span>
                    <span class="premium-badge badge-label">${(qual.ai_enrichment_coverage * 100).toFixed(1)}%</span>
                </div>
            </div>
            
            <!-- Quality & Completeness -->
            <div class="glass-panel" style="padding: 20px; border-radius: 8px;">
                <h3 style="margin-top:0; color:#3b82f6;">Data Quality</h3>
                <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">${(qual.avg_quality_score || 0).toFixed(1)} / 100</div>
                <div style="color:var(--text-muted); font-size:0.9rem;">Average Data Quality</div>
                
                <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin: 15px 0;">
                
                <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                    <span style="color:var(--text-muted);">Missing Attachments:</span>
                    <span class="premium-badge badge-gap-down">${qual.events_missing_attachment || 0}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:var(--text-muted);">Validation Failures:</span>
                    <span class="premium-badge ${qual.validation_failures > 0 ? 'badge-gap-down' : 'badge-label'}">${qual.validation_failures || 0}</span>
                </div>
            </div>
        </div>
    `;
}

function renderDashboard(data) {
    // Render Timestamp
    const lastUpdatedEl = document.getElementById('last-updated-time');
    const dateObj = new Date(data.generated_at);
    lastUpdatedEl.textContent = dateObj.toLocaleString();

    // Render Stats
    document.getElementById('stat-extraction').textContent = data.stage_statistics.extraction_count || 0;
    document.getElementById('stat-vsa').textContent = data.stage_statistics.vsa || 0;
    document.getElementById('stat-trending').textContent = data.stage_statistics.trending || 0;
    document.getElementById('stat-eigen').textContent = data.stage_statistics.eigen_filter || 0;
    
    if (document.getElementById('stat-anomaly')) {
        document.getElementById('stat-anomaly').textContent = data.anomaly_list?.length || 0;
    }
    if (document.getElementById('stat-triggers')) {
        document.getElementById('stat-triggers').textContent = data.triggers_list?.length || 0;
    }

    // Render Consensus
    const tbody = document.getElementById('consensus-body');
    tbody.innerHTML = '';
    
    let totalConsensus = 0;
    let bullishConsensus = 0;
    let bearishConsensus = 0;
    let neutralConsensus = 0;

    if (data.consensus && data.consensus.length > 0) {
        totalConsensus = data.consensus.length;
        data.consensus.forEach(item => {
            const lower = (item.daily_sentiment || '').toLowerCase();
            if (lower === 'bullish') bullishConsensus++;
            else if (lower === 'bearish') bearishConsensus++;
            else neutralConsensus++;
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="symbol-cell">${sanitizeHTML(item.symbol)}</td>
                <td>${item.score_pct.toFixed(1)}% <br><small style="color:var(--text-muted)">${'⭐'.repeat(item.stars)}</small></td>
                <td><span class="rating-badge ${getRatingClass(item.label)}">${sanitizeHTML(item.label)}</span></td>
                <td class="sentiment ${getSentimentClass(item.daily_sentiment)}">${sanitizeHTML(item.daily_sentiment)}</td>
                <td class="sentiment ${getSentimentClass(item.weekly_sentiment)}">${sanitizeHTML(item.weekly_sentiment)}</td>
                <td class="sentiment ${getSentimentClass(item.monthly_sentiment)}">${sanitizeHTML(item.monthly_sentiment)}</td>
            `;
            tbody.appendChild(tr);
        });
    } else {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted)">No consensus picks found.</td></tr>`;
    }

    const consensusHeader = document.querySelector('#consensus .panel-header');
    if (consensusHeader) {
        consensusHeader.innerHTML = `
            <h2>Top Consensus Picks 
                <span class="premium-badge badge-label">Total: ${totalConsensus}</span> 
                ${bullishConsensus > 0 ? `<span class="premium-badge badge-gap-up">Bullish: ${bullishConsensus}</span>` : ''}
                ${bearishConsensus > 0 ? `<span class="premium-badge badge-gap-down">Bearish: ${bearishConsensus}</span>` : ''}
                ${neutralConsensus > 0 ? `<span class="premium-badge badge-label">Neutral: ${neutralConsensus}</span>` : ''}
            </h2>
        `;
    }

    // Render Eigen Stats
    document.getElementById('eigen-daily-count').textContent = data.eigen_filters?.daily?.length || 0;
    document.getElementById('eigen-weekly-count').textContent = data.eigen_filters?.weekly?.length || 0;
    document.getElementById('eigen-monthly-count').textContent = data.eigen_filters?.monthly?.length || 0;
    
    renderEigenTables(data);

    // Render Alerts
    const alertsList = document.getElementById('alerts-list');
    alertsList.innerHTML = '';
    let totalAlerts = 0;
    if (data.ticker_alerts && data.ticker_alerts.length > 0) {
        totalAlerts = data.ticker_alerts.length;
        data.ticker_alerts.forEach(alert => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${sanitizeHTML(alert.symbol)}</strong>: ${sanitizeHTML(alert.pattern)} <br><small style="color:var(--text-muted); margin-top:4px; display:block;">${sanitizeHTML(alert.description)}</small>`;
            alertsList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = "No active alerts.";
        li.style.background = 'transparent';
        li.style.borderLeft = 'none';
        li.style.color = 'var(--text-muted)';
        alertsList.appendChild(li);
    }
    
    const alertsHeader = document.querySelector('#alerts-container .panel-header');
    if (alertsHeader) {
        alertsHeader.innerHTML = `
            <h2>Ticker Alerts 
                <span class="premium-badge badge-strong">Total: ${totalAlerts}</span>
            </h2>
        `;
    }

    // Render Macro Intelligence
    const macroCount = document.getElementById('macro-count');
    
    if (macroCount) {
        if (data.macro_intelligence) {
            macroCount.textContent = data.macro_intelligence.total_events || 0;
            macroEvents = data.macro_intelligence.recent_events || [];
            renderMacroTimeline();
        } else {
            macroCount.textContent = '0';
            macroEvents = [];
            renderMacroTimeline();
        }
    }
}

function renderMacroTimeline() {
    const timeline = document.getElementById('macro-timeline');
    if (!timeline) return;
    
    if (!macroEvents || macroEvents.length === 0) {
        timeline.innerHTML = `<p style="color:var(--text-muted); text-align:center; padding: 20px;">No macro intelligence data.</p>`;
        return;
    }
    
    const searchStr = workspaceState.searchQuery;
    const categoryFilter = workspaceState.selectedCategory;
    const sortOrder = workspaceState.sortOrder;
    
    let filtered = macroEvents.filter(evt => {
        // Handle both old schema (flat) and new schema (nested)
        const title = (evt.official_data?.title || evt.title || '').toLowerCase();
        const content = (evt.derived_data?.ai_summary || evt.official_data?.content || evt.summary || '').toLowerCase();
        const eventId = (evt.event_id || '').toLowerCase();
        const category = evt.official_data?.category || evt.category || '';
        
        if (searchStr && !title.includes(searchStr) && !content.includes(searchStr) && !eventId.includes(searchStr)) {
            return false;
        }
        if (categoryFilter && category !== categoryFilter) {
            return false;
        }
        return true;
    });
    
    filtered.sort((a, b) => {
        const dateA = new Date(a.official_data?.publication_date || a.published_at || a.published || 0).getTime();
        const dateB = new Date(b.official_data?.publication_date || b.published_at || b.published || 0).getTime();
        return sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
    });
    
    timeline.innerHTML = '';
    
    // Reset workspace selection state on re-render to ensure safety
    // Usually in React we'd keep selection unless it's filtered out, but for simplicity:
    const currentlySelectedExists = filtered.some(e => e.event_id === workspaceState.selectedEventId);
    if (!currentlySelectedExists) {
        workspaceState.selectedEventId = null;
    }
    renderMacroWorkspace();
    
    if (filtered.length === 0) {
        timeline.innerHTML = `<p style="color:var(--text-muted); text-align:center; padding: 20px;">No events match your criteria.</p>`;
        return;
    }
    
    filtered.forEach(event => {
        const card = document.createElement('div');
        card.className = 'macro-event-card glass-panel clickable-card';
        card.setAttribute('role', 'listitem');
        card.tabIndex = 0; // Make focusable
        
        if (event.event_id === workspaceState.selectedEventId) {
            card.classList.add('selected');
            card.setAttribute('aria-selected', 'true');
        } else {
            card.setAttribute('aria-selected', 'false');
        }
        card.dataset.eventId = event.event_id;
        
        const title = event.official_data?.title || event.title || 'Untitled Event';
        const pubDateStr = event.official_data?.publication_date || event.published_at || event.published;
        const pubDate = pubDateStr ? new Date(pubDateStr).toLocaleDateString() : 'Unknown Date';
        const summary = event.derived_data?.ai_summary || event.official_data?.content || event.summary || '';
        const category = event.official_data?.category || event.category || 'Uncategorized';
        
        const isNewHtml = (event.metadata?.lifecycle_status === 'NEW' || event.processing_state === 'NEW' || event.is_new_since_last_session) 
            ? `<span class="premium-badge badge-strong" style="background:var(--accent-color);color:#000;font-weight:bold;animation: pulse 2s infinite;">NEW</span>` 
            : '';

        card.innerHTML = `
            <div class="macro-event-header" style="display:flex; flex-wrap:wrap; gap:8px;">
                <span class="macro-date">${sanitizeHTML(pubDate)}</span>
                <span class="premium-badge badge-label">${sanitizeHTML(category)}</span>
                <span class="premium-badge badge-strong" style="font-family:monospace; font-size:0.8em; opacity:0.7;">ID: ${sanitizeHTML(event.event_id || '---')}</span>
                ${isNewHtml}
            </div>
            <h3>${sanitizeHTML(title)}</h3>
            <p>${sanitizeHTML(summary.substring(0, 150))}${summary.length > 150 ? '...' : ''}</p>
        `;
        
        card.addEventListener('click', () => {
            selectMacroEvent(event.event_id);
        });
        
        timeline.appendChild(card);
    });
}

// --- Workspace State Actions ---

function selectMacroEvent(eventId) {
    workspaceState.selectedEventId = eventId;
    
    // Update timeline visual selection
    document.querySelectorAll('.macro-event-card').forEach(c => {
        if (c.dataset.eventId === eventId) {
            c.classList.add('selected');
            c.setAttribute('aria-selected', 'true');
        } else {
            c.classList.remove('selected');
            c.setAttribute('aria-selected', 'false');
        }
    });
    
    renderMacroWorkspace();
}

// --- Independent Widget Renderers ---

function renderMacroWorkspace() {
    const detailEmpty = document.getElementById('macro-detail-empty');
    const detailContent = document.getElementById('macro-detail-content');
    
    if (!detailEmpty || !detailContent) return;
    
    if (!workspaceState.selectedEventId) {
        detailEmpty.style.display = 'flex';
        detailContent.style.display = 'none';
        detailContent.innerHTML = '';
        return;
    }
    
    const event = macroEvents.find(e => e.event_id === workspaceState.selectedEventId);
    if (!event) return;
    
    detailEmpty.style.display = 'none';
    detailContent.style.display = 'flex';
    
    detailContent.innerHTML = `
        ${renderHeaderWidget(event)}
        <div style="flex:1; overflow-y:auto; padding-right:10px;">
            ${renderAIInsightsWidget(event)}
            ${renderOverviewWidget(event)}
            ${renderAttachmentsWidget(event)}
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                ${renderMetadataWidget(event)}
                ${renderRelatedEventsWidget(event)}
            </div>
        </div>
    `;
}

function renderHeaderWidget(event) {
    const title = event.official_data?.title || event.title || 'Untitled Event';
    const pubDateStr = event.official_data?.publication_date || event.published_at || event.published;
    const pubDate = pubDateStr ? new Date(pubDateStr).toLocaleDateString() : 'Unknown Date';
    const url = event.official_data?.official_url || event.url || '#';
    const source = event.official_data?.source || event.source || 'Unknown';
    const category = event.official_data?.category || event.category || 'Uncategorized';
    
    return `
        <div class="macro-detail-header">
            <h2>${sanitizeHTML(title)}</h2>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <span class="premium-badge badge-label"><strong>Published:</strong> ${sanitizeHTML(pubDate)}</span>
                <span class="premium-badge badge-label"><strong>Source:</strong> <a href="${sanitizeHTML(url)}" target="_blank" style="color:white; text-decoration:underline;">${sanitizeHTML(source)}</a></span>
                <span class="premium-badge badge-label"><strong>Category:</strong> ${sanitizeHTML(category)}</span>
                <span class="premium-badge badge-strong"><strong>ID:</strong> ${sanitizeHTML(event.event_id)}</span>
            </div>
        </div>
    `;
}

function renderOverviewWidget(event) {
    const summary = (event.official_data?.content || event.summary || 'No official content available.');
    return `
        <div style="margin-bottom: 20px; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid var(--glass-border);">
            <h4 style="margin-top: 0; color:var(--text-muted); margin-bottom:10px;">OFFICIAL CONTENT / ABSTRACT</h4>
            <p style="line-height:1.6; color:var(--text-main);">${sanitizeHTML(summary).replace(/\n/g, '<br>')}</p>
        </div>
    `;
}

function renderAIInsightsWidget(event) {
    if (!event.derived_data || !event.derived_data.ai_summary) return '';
    const themesHtml = event.derived_data.themes ? event.derived_data.themes.map(t => `<span class="premium-badge badge-label">${sanitizeHTML(t)}</span>`).join(' ') : '';
    return `
        <div style="margin-bottom: 20px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid var(--accent-color, var(--primary));">
            <h4 style="margin-top: 0; color:var(--accent-color, var(--primary)); margin-bottom:10px;">✨ AI Enriched Summary</h4>
            <p style="line-height:1.6; color:var(--text-main);">${sanitizeHTML(event.derived_data.ai_summary).replace(/\n/g, '<br>')}</p>
            ${event.derived_data.themes ? `<p style="margin-top:15px;"><strong style="color:var(--text-muted);">Themes:</strong> ${themesHtml}</p>` : ''}
        </div>
    `;
}

function renderAttachmentsWidget(event) {
    if (!event.official_data?.attachments || event.official_data.attachments.length === 0) return '';
    return `
        <div style="margin-bottom: 20px; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid var(--glass-border);">
            <h4 style="margin-top: 0; color:var(--text-muted); margin-bottom:10px;">ATTACHMENTS</h4>
            <ul style="margin: 0; padding-left: 20px; color:var(--text-main);">
                ${event.official_data.attachments.map(a => `<li><a href="${sanitizeHTML(a)}" target="_blank" style="color:var(--primary); text-decoration:none;">${sanitizeHTML(a)}</a></li>`).join('')}
            </ul>
        </div>
    `;
}

function renderMetadataWidget(event) {
    if (!event.metadata) return '';
    const latency = event.metadata.processing_latency_ms ? `${event.metadata.processing_latency_ms} ms` : 'N/A';
    const sourceSystem = event.metadata.source_system || 'Unknown';
    const confidence = event.metadata.confidence_score ? `${(event.metadata.confidence_score * 100).toFixed(1)}%` : '--';

    return `
        <div style="padding: 16px; background: rgba(0,0,0,0.1); border-radius: 8px; font-size: 0.85rem;">
            <h4 style="margin-top: 0; color:var(--text-muted); margin-bottom:10px;">METADATA</h4>
            <table style="width:100%; border-collapse: collapse; color:var(--text-main);">
                <tr><td style="padding:4px 0; color:var(--text-muted);">Source System</td><td style="text-align:right;">${sanitizeHTML(sourceSystem)}</td></tr>
                <tr><td style="padding:4px 0; color:var(--text-muted);">Processing Latency</td><td style="text-align:right;">${sanitizeHTML(latency)}</td></tr>
                <tr><td style="padding:4px 0; color:var(--text-muted);">Confidence</td><td style="text-align:right;">${sanitizeHTML(confidence)}</td></tr>
                <tr><td style="padding:4px 0; color:var(--text-muted);">Lifecycle Status</td><td style="text-align:right;">${sanitizeHTML(event.metadata.lifecycle_status || 'Unknown')}</td></tr>
            </table>
        </div>
    `;
}

function renderRelatedEventsWidget(event) {
    if (!event.derived_data || !event.derived_data.related_events || event.derived_data.related_events.length === 0) {
        return `
            <div style="padding: 16px; background: rgba(0,0,0,0.1); border-radius: 8px; font-size: 0.85rem;">
                <h4 style="margin-top: 0; color:var(--text-muted); margin-bottom:10px;">RELATED EVENTS</h4>
                <p style="color:var(--text-muted); margin:0;">No related events identified.</p>
            </div>
        `;
    }

    return `
        <div style="padding: 16px; background: rgba(0,0,0,0.1); border-radius: 8px; font-size: 0.85rem;">
            <h4 style="margin-top: 0; color:var(--text-muted); margin-bottom:10px;">RELATED EVENTS</h4>
            <ul style="margin:0; padding-left:20px; color:var(--text-main);">
                ${event.derived_data.related_events.map(ev => `
                    <li style="margin-bottom:4px;">
                        <a href="#" class="related-event-trigger" data-event-id="${sanitizeHTML(ev.event_id)}" style="color:var(--primary); text-decoration:none;">${sanitizeHTML(ev.event_id)}</a>
                        <span style="color:var(--text-muted); margin-left:4px;">(${sanitizeHTML(ev.relationship_type)})</span>
                    </li>
                `).join('')}
            </ul>
        </div>
    `;
}

function setupClickableCards() {
    const cardExtraction = document.getElementById('card-extraction');
    const cardVsa = document.getElementById('card-vsa');
    const cardTrending = document.getElementById('card-trending');
    const cardEigen = document.getElementById('card-eigen');
    const cardAnomaly = document.getElementById('card-anomaly');
    const cardTriggers = document.getElementById('card-triggers');

    if (cardExtraction) cardExtraction.addEventListener('click', () => openModal('Extraction Details', currentData?.extraction_list || []));
    if (cardVsa) cardVsa.addEventListener('click', () => openModal('VSA Processed Details', currentData?.vsa_list || []));
    if (cardTrending) cardTrending.addEventListener('click', () => openModal('Trending Details', currentData?.trending_list || []));
    if (cardAnomaly) cardAnomaly.addEventListener('click', () => openModal('Anomaly Details', currentData?.anomaly_list || []));
    if (cardTriggers) cardTriggers.addEventListener('click', () => openModal('Triggers Details', currentData?.triggers_list || []));
    
    if (cardEigen) cardEigen.addEventListener('click', () => {
        const eigenNav = document.querySelector('.nav-item[href="#eigen"]');
        if (eigenNav) eigenNav.click();
        else switchTab('eigen');
    });
}

function setupModal() {
    const modal = document.getElementById('premium-modal');
    const closeBtn = document.getElementById('modal-close');
    
    if (closeBtn && modal) {
        closeBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.add('hidden');
            }
        });
    }
}

function openModal(title, list, customHtml = '') {
    const modal = document.getElementById('premium-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalGrid = document.getElementById('modal-grid');
    
    if (!modal || !modalTitle || !modalGrid) return;
    
    modalTitle.textContent = title;
    modalGrid.innerHTML = customHtml;
    
    if (list && list.length > 0) {
        list.forEach(symbol => {
            const symEl = document.createElement('div');
            symEl.className = 'modal-symbol-item';
            symEl.textContent = symbol;
            modalGrid.appendChild(symEl);
        });
    } else if (!customHtml) {
        modalGrid.innerHTML = `<p style="color:var(--text-muted); grid-column: 1/-1;">No data available.</p>`;
    }
    
    modal.classList.remove('hidden');
}

function checkStaleData(generatedAtString) {
    const generatedDate = new Date(generatedAtString);
    const now = new Date();
    
    const diffTime = now - generatedDate;
    const diffHours = diffTime / (1000 * 60 * 60);
    
    // 26 hour threshold
    if (diffHours > 26) {
        document.getElementById('stale-warning').classList.remove('hidden');
    } else {
        document.getElementById('stale-warning').classList.add('hidden');
    }
}

function showError(message) {
    document.getElementById('dashboard-content').style.display = 'none';
    const errorBoundary = document.getElementById('error-boundary');
    errorBoundary.classList.remove('hidden');
    document.getElementById('error-message').textContent = message || "Unable to load data.";
    
    const statusBadge = document.getElementById('system-status');
    statusBadge.classList.add('error');
    statusBadge.innerHTML = '<span class="status-dot"></span> System Error';
}

function getRatingClass(label) {
    const lower = label.toLowerCase();
    if (lower.includes('buy')) return 'rating-buy';
    if (lower.includes('sell')) return 'rating-sell';
    return 'rating-neutral';
}

function getSentimentClass(sentiment) {
    const lower = sentiment.toLowerCase();
    if (lower === 'bullish') return 'bullish';
    if (lower === 'bearish') return 'bearish';
    return 'neutral';
}

function formatBadgeCell(text, category) {
    if (text === undefined || text === null || text === '' || text === '--') return text || '--';
    
    const lower = String(text).toLowerCase();
    let badgeClass = 'badge-label';
    
    if (category === 'gap') {
        if (lower.includes('up')) badgeClass = 'badge-gap-up';
        else if (lower.includes('down')) badgeClass = 'badge-gap-down';
    } else if (category === 'label' || category === 'band') {
        if (lower.includes('strong')) badgeClass = 'badge-strong';
        else if (lower.includes('weak')) badgeClass = 'badge-weak';
    }
    
    return `<span class="premium-badge ${badgeClass}">${text}</span>`;
}

function renderEigenTables(data) {
    const timeframes = ['daily', 'weekly', 'monthly'];
    
    timeframes.forEach(timeframe => {
        const tbody = document.getElementById(`eigen-body-${timeframe}`);
        if (!tbody) return;
        
        tbody.innerHTML = '';
        const items = data.eigen_filters?.[timeframe] || [];
        
        if (items.length > 0) {
            items.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="symbol-cell">${item.symbol}</td>
                    <td class="sentiment ${getSentimentClass(item.sentiment || '')}">${item.sentiment || 'None'}</td>
                    <td>${formatBadgeCell(item.label, 'label')}</td>
                    <td>${formatBadgeCell(item.gap_dir, 'gap')}</td>
                    <td>${formatBadgeCell(item.close_band, 'band')}</td>
                    <td>${item.vol_delta_pct !== undefined ? item.vol_delta_pct + '%' : '--'}</td>
                    <td>${item.delta_cp !== undefined ? item.delta_cp.toFixed(4) : '--'}</td>
                `;
                tbody.appendChild(tr);
            });
        } else {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-muted)">No matches found.</td></tr>`;
        }
    });
}

function toggleEigenTable(timeframe) {
    const tableContainer = document.getElementById(`eigen-table-${timeframe}`);
    if (tableContainer) {
        if (tableContainer.style.display === 'none') {
            tableContainer.style.display = 'block';
        } else {
            tableContainer.style.display = 'none';
        }
    }
}

// ==========================================
// Eigen Transition Engine (ETE) Integration
// ==========================================

async function fetchETEManifest(date = '') {
    try {
        const manifestUrl = date ? `./history/manifest_${date}.json` : './manifest.json';
        const response = await fetch(manifestUrl);
        if (!response.ok) {
            document.getElementById('ete-loading').textContent = 'No ETE data generated yet.';
            return;
        }
        const manifest = await response.json();
        
        const headerStats = document.getElementById('ete-manifest-stats');
        if (headerStats) {
            headerStats.innerHTML = `
                <span class="premium-badge badge-strong">Active: ${manifest.active_sequences || 0}</span>
                <span class="premium-badge badge-gap-up">Completed: ${manifest.completed_sequences || 0}</span>
                <span class="premium-badge badge-label">Events: ${manifest.research_events || 0}</span>
            `;
        }
        
        if (manifest.files && manifest.files.summary) {
            fetchETESummary(manifest.files.summary);
        } else {
            document.getElementById('ete-loading').textContent = 'No ETE summary found in manifest.';
        }
    } catch (e) {
        console.error('Failed to load ETE manifest', e);
        document.getElementById('ete-loading').textContent = 'Failed to load ETE data.';
    }
}

async function fetchETESummary(summaryFile) {
    try {
        const response = await fetch(`./${summaryFile}`);
        if (!response.ok) throw new Error("Summary fetch failed");
        const summaryData = await response.json();
        
        document.getElementById('ete-loading').style.display = 'none';
        document.getElementById('ete-content').style.display = 'block';
        
        renderETE(summaryData);
    } catch (e) {
        console.error('Failed to load ETE summary', e);
        document.getElementById('ete-loading').textContent = 'Failed to load ETE summary.';
    }
}

function renderETE(summaryData) {
    const allItems = [...(summaryData.active || []), ...(summaryData.completed || [])];
    
    // Reset counts
    document.getElementById('ete-daily-count').textContent = '0';
    document.getElementById('ete-weekly-count').textContent = '0';
    document.getElementById('ete-monthly-count').textContent = '0';
    
    ['daily', 'weekly', 'monthly'].forEach(timeframe => {
        const tbody = document.getElementById(`ete-body-${timeframe}`);
        if (!tbody) return;
        tbody.innerHTML = '';
        
        const items = allItems.filter(i => i.timeframe === timeframe);
        document.getElementById(`ete-${timeframe}-count`).textContent = items.length;
        
        if (items.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted)">No sequences found.</td></tr>`;
            return;
        }
        
        items.forEach(item => {
            const tr = document.createElement('tr');
            
            let stateClass = 'badge-label';
            if (item.state === 'Completed') stateClass = 'badge-gap-up';
            else if (item.state === 'Waiting') stateClass = 'badge-strong';
            else if (item.state === 'Failed') stateClass = 'badge-gap-down';
            else if (item.state === 'Triggered') stateClass = 'badge-label';

            tr.innerHTML = `
                <td class="symbol-cell">${sanitizeHTML(item.symbol)}</td>
                <td><span class="premium-badge ${stateClass}">${sanitizeHTML(item.state)}</span></td>
                <td>${sanitizeHTML(item.current_stage)}</td>
                <td>${item.confidence ? item.confidence.toFixed(1) + '%' : '--'}</td>
                <td><button class="btn-small">View</button></td>
            `;
            
            const btn = tr.querySelector('.btn-small');
            btn.addEventListener('click', () => {
                openSequenceDrilldown(item);
            });
            tbody.appendChild(tr);
        });
    });
}

function toggleEteTable(timeframe) {
    const tableContainer = document.getElementById(`ete-table-${timeframe}`);
    if (tableContainer) {
        if (tableContainer.style.display === 'none') {
            tableContainer.style.display = 'block';
        } else {
            tableContainer.style.display = 'none';
        }
    }
}

// ==========================================
// Backtest & Replay Integration
// ==========================================

async function fetchBacktestMetrics() {
    try {
        const response = await fetch('./backtest_results.json');
        if (!response.ok) {
            document.getElementById('backtest-loading').textContent = 'No backtest metrics found. The backtest engine may not have run yet.';
            return;
        }
        const data = await response.json();
        renderBacktest(data);
    } catch (e) {
        console.error('Failed to load backtest metrics', e);
        document.getElementById('backtest-loading').textContent = 'Failed to load backtest metrics.';
    }
}

function openSequenceDrilldown(item) {
    let html = `<div style="grid-column: 1/-1;">`;
    
    html += `<div style="margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap;">
        <span class="premium-badge badge-strong">Symbol: ${sanitizeHTML(item.symbol)}</span>
        <span class="premium-badge badge-label">Timeframe: ${sanitizeHTML(item.timeframe)}</span>
        <span class="premium-badge badge-gap-up">Confidence: ${item.confidence ? item.confidence.toFixed(1) + '%' : '--'}</span>
    </div>`;

    html += `<div class="sequence-timeline" style="border-left: 2px solid var(--accent-color); padding-left: 20px; margin-left: 10px;">`;
    
    if (item.progress && item.progress.length > 0) {
        item.progress.forEach((evt, idx) => {
            let color = "var(--text-color)";
            if (evt.action === "FAIL" || evt.action === "PAUSE") color = "#ff4c4c";
            if (evt.action === "ADVANCE" || evt.action === "COMPLETED") color = "#4cff4c";
            
            html += `
                <div style="position: relative; margin-bottom: 25px;">
                    <div style="position: absolute; left: -27px; top: 0; width: 12px; height: 12px; border-radius: 50%; background: ${color};"></div>
                    <div style="font-weight: bold; color: ${color};">${sanitizeHTML(evt.action)} <span style="color: var(--text-muted); font-size: 0.9em; font-weight: normal;">(Stage: ${sanitizeHTML(evt.stage)})</span></div>
                    <div style="color: var(--text-muted); font-size: 0.85em; margin-top: 4px;">Date: ${sanitizeHTML(evt.timestamp)}</div>
                    ${evt.failure_reason && evt.failure_reason !== "None" ? `<div style="color: #ff4c4c; font-size: 0.85em; margin-top: 4px;">Reason: ${sanitizeHTML(evt.failure_reason)}</div>` : ''}
                    <div style="font-size: 0.85em; margin-top: 4px; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 4px;">
                        Rule Evaluated: ${sanitizeHTML(evt.rule_evaluated)} <br>
                        Metrics: Vol &Delta; ${evt.metrics?.vol_delta_pct?.toFixed(1) || 0}% | Spread &Delta; ${evt.metrics?.spread_delta_pct?.toFixed(1) || 0}%
                    </div>
                </div>
            `;
        });
    } else {
        html += `<p style="color:var(--text-muted);">No transition events found.</p>`;
    }
    
    html += `</div></div>`;
    openModal(`Sequence Timeline`, [], html);
}

function renderBacktest(data) {
    const container = document.getElementById('backtest-metrics-container');
    container.innerHTML = '';
    
    if (data.overall_metrics) {
        const o = data.overall_metrics;
        container.innerHTML = `
            <div class="stat-card glass-panel" style="text-align: center; padding: 20px;">
                <h3 style="color:var(--text-muted); margin-bottom: 15px; font-size:1em; text-transform:uppercase; letter-spacing:1px;">Total Evaluated</h3>
                <div style="font-size: 2.5em; font-weight: bold;">${o.total_sequences || 0}</div>
            </div>
            <div class="stat-card glass-panel" style="text-align: center; padding: 20px;">
                <h3 style="color:var(--text-muted); margin-bottom: 15px; font-size:1em; text-transform:uppercase; letter-spacing:1px;">Win Rate</h3>
                <div style="font-size: 2.5em; font-weight: bold; color: #4cff4c;">${o.win_rate ? o.win_rate.toFixed(1) + '%' : '0%'}</div>
            </div>
            <div class="stat-card glass-panel clickable-card" id="backtest-completions-card" style="text-align: center; padding: 20px; cursor: pointer;">
                <h3 style="color:var(--text-muted); margin-bottom: 15px; font-size:1em; text-transform:uppercase; letter-spacing:1px;">Completions <span style="font-size: 0.6em; color: var(--accent-color);">(Click for details)</span></h3>
                <div style="font-size: 2.5em; font-weight: bold; color: var(--accent-color);">${o.total_completed || 0}</div>
            </div>
            <div class="stat-card glass-panel" style="text-align: center; padding: 20px;">
                <h3 style="color:var(--text-muted); margin-bottom: 15px; font-size:1em; text-transform:uppercase; letter-spacing:1px;">Failures</h3>
                <div style="font-size: 2.5em; font-weight: bold; color: #ff4c4c;">${o.total_failed || 0}</div>
            </div>
        `;
    } else {
        container.innerHTML = '<p>No metrics available.</p>';
    }
}

async function openCompletionsDrilldown() {
    try {
        const response = await fetch(`./backtest_results.json?t=${new Date().getTime()}`);
        if (!response.ok) {
            console.error('Failed to fetch completions');
            return;
        }
        const data = await response.json();
        
        const daily = data.completions_daily || [];
        const weekly = data.completions_weekly || [];
        const monthly = data.completions_monthly || [];
        
        let html = `
            <div class="tabs-container" style="margin-bottom: 20px; grid-column: 1/-1;">
                <div class="tab-headers" style="display: flex; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                    <button class="tab-btn active completion-tab-trigger" data-tab="daily" id="tab-btn-daily" style="background:transparent; border:none; color:var(--text-color); cursor:pointer; font-weight:bold; padding: 8px 16px; border-radius: 4px; background: rgba(255,255,255,0.1);">Daily (${daily.length})</button>
                    <button class="tab-btn completion-tab-trigger" data-tab="weekly" id="tab-btn-weekly" style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-weight:bold; padding: 8px 16px; border-radius: 4px;">Weekly (${weekly.length})</button>
                    <button class="tab-btn completion-tab-trigger" data-tab="monthly" id="tab-btn-monthly" style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-weight:bold; padding: 8px 16px; border-radius: 4px;">Monthly (${monthly.length})</button>
                </div>
            </div>
            <div class="tab-content" id="tab-content-daily" style="display:block; grid-column: 1/-1;">
                ${renderCompletionTable(daily)}
            </div>
            <div class="tab-content" id="tab-content-weekly" style="display:none; grid-column: 1/-1;">
                ${renderCompletionTable(weekly)}
            </div>
            <div class="tab-content" id="tab-content-monthly" style="display:none; grid-column: 1/-1;">
                ${renderCompletionTable(monthly)}
            </div>
        `;
        
        openModal('Backtest Completions', [], html);
    } catch (e) {
        console.error(e);
    }
}

function switchCompletionTab(tabName) {
    ['daily', 'weekly', 'monthly'].forEach(t => {
        const btn = document.getElementById(`tab-btn-${t}`);
        const content = document.getElementById(`tab-content-${t}`);
        if (btn && content) {
            if (t === tabName) {
                btn.style.color = 'var(--text-color)';
                btn.style.background = 'rgba(255,255,255,0.1)';
                content.style.display = 'block';
            } else {
                btn.style.color = 'var(--text-muted)';
                btn.style.background = 'transparent';
                content.style.display = 'none';
            }
        }
    });
}

function renderCompletionTable(items) {
    if (!items || items.length === 0) {
        return `<p style="color:var(--text-muted); text-align:center; padding: 20px;">No completions found for this timeframe.</p>`;
    }
    
    // Group items by symbol
    const groups = {};
    items.forEach(item => {
        if (!groups[item.symbol]) groups[item.symbol] = [];
        groups[item.symbol].push(item);
    });
    
    // Sort symbols by the latest completion date descending
    const sortedSymbols = Object.keys(groups).sort((a, b) => {
        const latestA = Math.max(...groups[a].map(i => new Date(i.completion_date).getTime() || 0));
        const latestB = Math.max(...groups[b].map(i => new Date(i.completion_date).getTime() || 0));
        return latestB - latestA;
    });
    
    let html = `<div class="table-container" style="max-height: 60vh; overflow-y: auto;">`;
    
    sortedSymbols.forEach((symbol, index) => {
        const groupItems = groups[symbol];
        
        // Sort items within the group descending by completion date
        groupItems.sort((a, b) => new Date(b.completion_date).getTime() - new Date(a.completion_date).getTime());
        
        const latestDate = groupItems[0].completion_date || '--';
        const toggleId = `comp-group-${Math.random().toString(36).substring(2, 9)}`;
        const displayState = index === 0 ? 'block' : 'none'; // Auto-expand the very first symbol
        
        // Accordion Row Header
        html += `
            <div class="eigen-row completion-row-trigger" data-target="${toggleId}" style="cursor:pointer; margin-bottom: 5px; background: rgba(255,255,255,0.03);">
                <span>
                    <strong style="color: var(--text-color); font-size: 1.1em;">${sanitizeHTML(symbol)}</strong> 
                    <span style="color:var(--text-muted); font-size: 0.9em; margin-left: 10px;">Latest: ${sanitizeHTML(latestDate)}</span>
                </span>
                <span class="badge" style="background: rgba(255,255,255,0.1); color: var(--text-color);">${groupItems.length}</span>
            </div>
            
            <div class="eigen-table-container" id="${toggleId}" style="display: ${displayState}; margin-bottom: 15px; border-left: 2px solid rgba(255,255,255,0.1); padding-left: 10px;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Trigger Date</th>
                            <th>Completion Date</th>
                            <th>Pattern</th>
                            <th>Sentiment</th>
                            <th>Vol Surge</th>
                            <th>Confidence</th>
                            <th>Max Fwd Rtn (5b)</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        groupItems.forEach(item => {
            const sentimentClass = item.sentiment === 'Bullish' ? 'badge-gap-up' : 'badge-gap-down';
            const returnColor = item.fwd_return_5b > 0 ? '#4cff4c' : '#ff4c4c';
            
            let confidenceHtml = '—';
            if (item.confidence !== null && item.confidence !== undefined && item.confidence !== 0) {
                let badgeClass = 'badge-weak';
                if (item.confidence >= 70.0) {
                    badgeClass = 'badge-gap-up';
                } else if (item.confidence < 40.0) {
                    badgeClass = 'badge-gap-down';
                }
                confidenceHtml = `
                    <span class="confidence-cell" 
                          data-symbol="${sanitizeHTML(item.symbol)}"
                          data-date="${sanitizeHTML(item.start_date)}"
                          data-confidence="${item.confidence.toFixed(1)}"
                          data-vol="${item.vol_score !== undefined ? item.vol_score : 0.0}"
                          data-close="${item.close_score !== undefined ? item.close_score : 0.0}"
                          data-drift="${item.drift_score !== undefined ? item.drift_score : 0.0}"
                          data-pattern="${sanitizeHTML(item.trigger_pattern)}"
                          data-sentiment="${sanitizeHTML(item.sentiment)}">
                      <span class="premium-badge ${badgeClass}">${item.confidence.toFixed(1)}%</span>
                    </span>
                `;
            }
            
            html += `
                <tr>
                    <td>${sanitizeHTML(item.start_date || '--')}</td>
                    <td>${sanitizeHTML(item.completion_date || '--')}</td>
                    <td><span class="premium-badge badge-label">${sanitizeHTML(item.trigger_pattern)}</span></td>
                    <td><span class="premium-badge ${sentimentClass}">${sanitizeHTML(item.sentiment)}</span></td>
                    <td>${item.vol_surge_pct !== undefined ? item.vol_surge_pct.toFixed(1) + '%' : '--'}</td>
                    <td>${confidenceHtml}</td>
                    <td style="color: ${returnColor}; font-weight: bold;">${item.fwd_return_5b !== null ? item.fwd_return_5b.toFixed(2) + '%' : '--'}</td>
                </tr>
            `;
        });
        
        html += `</tbody></table></div>`;
    });
    
    html += `</div>`;
    return html;
}

// Initialize Cache and Global Tooltip state
if (!window.tooltipCache) window.tooltipCache = {};

function getGlobalTooltip() {
    let tip = document.getElementById('global-confidence-tooltip');
    if (!tip) {
        tip = document.createElement('div');
        tip.id = 'global-confidence-tooltip';
        tip.style.position = 'fixed';
        tip.style.display = 'none';
        document.body.appendChild(tip);
        
        // Prevent mouseleave when user moves mouse inside the tooltip itself
        // So that they can click the input field, type, and save the key
        tip.addEventListener('mouseenter', () => {
            if (window.tooltipHideTimeout) {
                clearTimeout(window.tooltipHideTimeout);
                window.tooltipHideTimeout = null;
            }
        });
        
        tip.addEventListener('mouseleave', () => {
            hideConfidenceTip(null, true);
        });
    }
    return tip;
}

window.tooltipHideTimeout = null;

async function showConfidenceTip(cell) {
    if (window.tooltipHideTimeout) {
        clearTimeout(window.tooltipHideTimeout);
        window.tooltipHideTimeout = null;
    }

    const tip = getGlobalTooltip();
    window.globalTooltipActiveCell = cell;

    const { symbol, date, confidence, vol, close, drift, pattern, sentiment } = cell.dataset;
    const volPct   = (parseFloat(vol)   * 100).toFixed(1);
    const closePct = (parseFloat(close) * 100).toFixed(1);
    const driftPct = (parseFloat(drift) * 100).toFixed(1);

    // Initial render of sub-scores skeleton
    tip.innerHTML = `
      <div class="tip-header">Confidence breakdown</div>
      <div class="tip-scores">
        <div class="tip-score-item"><span class="tip-score-val">${volPct}%</span>Volume</div>
        <div class="tip-score-item"><span class="tip-score-val">${closePct}%</span>Close</div>
        <div class="tip-score-item"><span class="tip-score-val">${driftPct}%</span>Drift</div>
      </div>
      <div class="tip-summary tip-loading">Generating summary…</div>
    `;

    tip.style.display = 'block';

    // Position calculation (Fixed Viewport-relative)
    const rect = cell.getBoundingClientRect();
    const TOOLTIP_W = 280;
    const TOOLTIP_H = 180;

    const spaceAbove = rect.top;
    const spaceBelow = window.innerHeight - rect.bottom;

    let top = spaceAbove >= TOOLTIP_H || spaceAbove >= spaceBelow
      ? rect.top - TOOLTIP_H - 8        // above
      : rect.bottom + 8;                 // flip below

    let left = rect.left + rect.width / 2 - TOOLTIP_W / 2;
    left = Math.max(10, Math.min(left, window.innerWidth - TOOLTIP_W - 10));

    tip.style.top  = top + 'px';
    tip.style.left = left + 'px';

    // Cache lookup key
    const cacheKey = `${symbol}_${date}_${pattern}`;
    
    // Check cache
    if (window.tooltipCache[cacheKey]) {
        renderSummaryText(tip, window.tooltipCache[cacheKey], cell);
        return;
    }

    // Check localStorage API key
    const apiKey = localStorage.getItem('gemini_api_key') || '';
    if (!apiKey) {
        // Display setup prompt for key
        renderKeySetupPrompt(tip, cell, () => {
            // Callback after saving key - retry summary generation
            showConfidenceTip(cell);
        });
        
        // Render fallback summary below key setup prompt so there is zero blank state
        const fallback = generateLocalSummary(confidence, pattern, sentiment, volPct, closePct, driftPct);
        renderSummaryText(tip, fallback, cell);
        return;
    }

    // Call Gemini API
    const prompt = `You are a quantitative trading analyst. Summarize in 2 sentences what a confidence score of ${confidence}% means for a "${pattern}" (${sentiment}) trade signal. Sub-scores: Volume surge ${volPct}% (40% weight), Close position ${closePct}% (40% weight), Close drift ${driftPct}% (20% weight). Be direct and specific — state the dominant factor and what it implies for trade conviction.`;

    try {
        const summary = await fetchGeminiSummary(prompt, apiKey);
        if (window.globalTooltipActiveCell === cell) {
            window.tooltipCache[cacheKey] = summary;
            renderSummaryText(tip, summary, cell);
        }
    } catch (err) {
        console.warn('Gemini API call failed, falling back to local analyst:', err);
        const fallback = generateLocalSummary(confidence, pattern, sentiment, volPct, closePct, driftPct);
        if (window.globalTooltipActiveCell === cell) {
            renderSummaryText(tip, fallback, cell);
        }
    }
}

function hideConfidenceTip(cell, immediate = false) {
    if (immediate) {
        const tip = document.getElementById('global-confidence-tooltip');
        if (tip) tip.style.display = 'none';
        window.globalTooltipActiveCell = null;
    } else {
        // Delay hide slightly so user can move mouse into tooltip
        window.tooltipHideTimeout = setTimeout(() => {
            const tip = document.getElementById('global-confidence-tooltip');
            if (tip) tip.style.display = 'none';
            window.globalTooltipActiveCell = null;
        }, 300);
    }
}

function renderSummaryText(tip, text, cell) {
    if (window.globalTooltipActiveCell !== cell) return;
    
    let summaryDiv = tip.querySelector('.tip-summary');
    if (!summaryDiv) {
        summaryDiv = document.createElement('div');
        summaryDiv.className = 'tip-summary';
        tip.appendChild(summaryDiv);
    }
    summaryDiv.classList.remove('tip-loading');
    summaryDiv.textContent = text;
}

function renderKeySetupPrompt(tip, cell, onSaveCallback) {
    if (window.globalTooltipActiveCell !== cell) return;
    
    // Remove existing if any
    const existing = tip.querySelector('.key-setup-container');
    if (existing) existing.remove();
    
    const container = document.createElement('div');
    container.className = 'key-setup-container';
    container.innerHTML = `
        <div class="key-setup-title">Enter Gemini API key for AI summaries:</div>
        <div class="key-setup-row">
            <input type="password" class="key-input" placeholder="AIzaSy...">
            <button class="key-save-btn">Save</button>
        </div>
    `;
    
    // Wire up save button
    const btn = container.querySelector('.key-save-btn');
    const input = container.querySelector('.key-input');
    
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const key = input.value.trim();
        if (key) {
            localStorage.setItem('gemini_api_key', key);
            if (onSaveCallback) onSaveCallback();
        }
    });
    
    // Append before the summary section if possible, otherwise at the end
    const summary = tip.querySelector('.tip-summary');
    if (summary) {
        tip.insertBefore(container, summary);
    } else {
        tip.appendChild(container);
    }
}

async function fetchGeminiSummary(prompt, apiKey) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            contents: [{
                parts: [{
                    text: prompt
                }]
            }]
        })
    });
    if (!response.ok) throw new Error(`API error ${response.status}`);
    const data = await response.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
    if (text) return text.trim();
    throw new Error('Empty response');
}

function generateLocalSummary(confidence, pattern, sentiment, volPct, closePct, driftPct) {
  const conf = parseFloat(confidence);
  const volVal = parseFloat(volPct);
  const closeVal = parseFloat(closePct);
  const driftVal = parseFloat(driftPct);
  
  const dominant = volVal >= closeVal && volVal >= driftVal ? 'volume'
                 : closeVal >= driftVal ? 'close position' : 'drift';

  const tier = conf >= 70 ? 'high' : conf >= 40 ? 'moderate' : 'low';

  const domPhrase = {
    volume:         `Volume surge of ${volPct}% vs the 20-period average is the primary driver.`,
    'close position': `A close position score of ${closePct}% indicates price closed ${closeVal >= 50 ? 'near the favourable' : 'away from the favourable'} extreme.`,
    drift:          `The drift score of ${driftPct}% reflects ${driftVal >= 50 ? 'improving' : 'deteriorating'} close position versus the prior candle.`
  }[dominant];

  const convPhrase = tier === 'high'   ? `Overall conviction for this ${sentiment} ${pattern} signal is strong.`
                   : tier === 'moderate' ? `Overall conviction is moderate — treat as a qualifying signal requiring confirmation.`
                   :                       `Overall conviction is low — this signal carries elevated uncertainty.`;

  return `${domPhrase} ${convPhrase}`;
}

