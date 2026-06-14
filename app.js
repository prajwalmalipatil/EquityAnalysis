let currentData = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchHistoryIndex();
    fetchData();
    setupNavigation();
    setupModal();
    setupClickableCards();
});

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
    const ete = document.getElementById('ete');
    const alerts = document.getElementById('alerts-container'); 

    if (tabId === 'overview') {
        overview.style.display = 'grid';
        consensus.style.display = 'flex';
        eigen.style.display = 'flex';
        if (macro) macro.style.display = 'none';
        if (ete) ete.style.display = 'none';
        if (alerts) alerts.style.display = 'flex';
        const backtest = document.getElementById('backtest');
        if (backtest) backtest.style.display = 'none';
    } else if (tabId === 'consensus') {
        overview.style.display = 'none';
        consensus.style.display = 'flex';
        eigen.style.display = 'none';
        if (macro) macro.style.display = 'none';
        if (ete) ete.style.display = 'none';
        if (alerts) alerts.style.display = 'none';
        const backtest = document.getElementById('backtest');
        if (backtest) backtest.style.display = 'none';
    } else if (tabId === 'eigen') {
        overview.style.display = 'none';
        consensus.style.display = 'none';
        eigen.style.display = 'flex';
        if (macro) macro.style.display = 'none';
        if (ete) ete.style.display = 'none';
        if (alerts) alerts.style.display = 'none';
        const backtest = document.getElementById('backtest');
        if (backtest) backtest.style.display = 'none';
    } else if (tabId === 'macro') {
        overview.style.display = 'none';
        consensus.style.display = 'none';
        eigen.style.display = 'none';
        if (macro) macro.style.display = 'flex';
        if (ete) ete.style.display = 'none';
        if (alerts) alerts.style.display = 'none';
        const backtest = document.getElementById('backtest');
        if (backtest) backtest.style.display = 'none';
    } else if (tabId === 'ete') {
        overview.style.display = 'none';
        consensus.style.display = 'none';
        eigen.style.display = 'none';
        if (macro) macro.style.display = 'none';
        if (ete) {
            ete.style.display = 'flex';
            if (!window.eteLoaded) {
                fetchETEManifest();
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

async function fetchData(date = '') {
    try {
        const url = date ? `./history/data_${date}.json` : './data.json';
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Validate schema version
        if (!data.schema_version) {
            throw new Error('Invalid data schema: Missing schema_version');
        }

        currentData = data;
        renderDashboard(data);
        checkStaleData(data.generated_at);
        document.getElementById('error-boundary').classList.add('hidden');
        document.getElementById('dashboard-content').style.display = 'flex';
        
    } catch (error) {
        console.error('Failed to fetch or parse dashboard data:', error);
        showError(error.message);
    }
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
                <td class="symbol-cell">${item.symbol}</td>
                <td>${item.score_pct.toFixed(1)}% <br><small style="color:var(--text-muted)">${'⭐'.repeat(item.stars)}</small></td>
                <td><span class="rating-badge ${getRatingClass(item.label)}">${item.label}</span></td>
                <td class="sentiment ${getSentimentClass(item.daily_sentiment)}">${item.daily_sentiment}</td>
                <td class="sentiment ${getSentimentClass(item.weekly_sentiment)}">${item.weekly_sentiment}</td>
                <td class="sentiment ${getSentimentClass(item.monthly_sentiment)}">${item.monthly_sentiment}</td>
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
            li.innerHTML = `<strong>${alert.symbol}</strong>: ${alert.pattern} <br><small style="color:var(--text-muted); margin-top:4px; display:block;">${alert.description}</small>`;
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
    const macroTimeline = document.getElementById('macro-timeline');
    
    if (macroCount && macroTimeline) {
        if (data.macro_intelligence) {
            macroCount.textContent = data.macro_intelligence.total_events || 0;
            macroTimeline.innerHTML = '';
            
            const recentEvents = data.macro_intelligence.recent_events || [];
            if (recentEvents.length > 0) {
                recentEvents.forEach(event => {
                    const eventCard = document.createElement('div');
                    eventCard.className = 'macro-event-card glass-panel clickable-card';
                    
                    const pubDate = new Date(event.published_at).toLocaleDateString();
                    const impact = event.impact || {};
                    
                    let badgeClass = 'badge-label';
                    if (impact.direction === 'Positive') badgeClass = 'badge-gap-up';
                    else if (impact.direction === 'Negative') badgeClass = 'badge-gap-down';
                    
                    const isNewHtml = event.is_new_since_last_session ? `<span class="premium-badge badge-strong" style="background:var(--accent-color);color:#000;font-weight:bold;animation: pulse 2s infinite;">NEW SINCE LAST TRADING SESSION</span>` : '';

                    eventCard.innerHTML = `
                        <div class="macro-event-header" style="display:flex; flex-wrap:wrap; gap:8px;">
                            <span class="macro-date">${pubDate}</span>
                            <span class="premium-badge ${badgeClass}">${impact.direction || 'Unknown'}</span>
                            ${isNewHtml}
                        </div>
                        <h3>${event.title}</h3>
                        <p>${event.summary ? event.summary.substring(0, 150) + '...' : ''}</p>
                    `;
                    
                    eventCard.addEventListener('click', () => {
                        const impactHtml = `
                            <div style="grid-column: 1/-1; text-align: left;">
                                <p><strong>Published:</strong> ${pubDate}</p>
                                <p><strong>Source:</strong> <a href="${event.url}" target="_blank">${event.source}</a></p>
                                <div style="margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                                    <h4 style="margin-top: 0;">Event-to-Market Impact</h4>
                                    <p><strong>Assets:</strong> ${(impact.asset_classes || []).join(', ')}</p>
                                    <p><strong>Sectors:</strong> ${(impact.sectors || []).join(', ')}</p>
                                    <p><strong>Horizon:</strong> ${impact.horizon || '--'}</p>
                                </div>
                                <div style="margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                                    <h4 style="margin-top: 0;">Detailed Summary</h4>
                                    <p>${event.summary || 'No summary available.'}</p>
                                </div>
                            </div>
                        `;
                        openModal(event.title, [], impactHtml);
                    });
                    
                    macroTimeline.appendChild(eventCard);
                });
            } else {
                macroTimeline.innerHTML = `<p style="color:var(--text-muted); text-align:center; padding: 20px;">No macro events found.</p>`;
            }
        } else {
            macroCount.textContent = '0';
            macroTimeline.innerHTML = `<p style="color:var(--text-muted); text-align:center; padding: 20px;">No macro intelligence data.</p>`;
        }
    }
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

async function fetchETEManifest() {
    try {
        const response = await fetch('./manifest.json');
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
                <td class="symbol-cell">${item.symbol}</td>
                <td><span class="premium-badge ${stateClass}">${item.state}</span></td>
                <td>${item.current_stage}</td>
                <td>${item.confidence ? item.confidence.toFixed(1) + '%' : '--'}</td>
                <td><button class="btn-small" onclick="openSequenceDrilldown('${encodeURIComponent(JSON.stringify(item))}')">View</button></td>
            `;
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

function openSequenceDrilldown(sequenceDataStr) {
    const item = JSON.parse(decodeURIComponent(sequenceDataStr));
    let html = `<div style="grid-column: 1/-1;">`;
    
    html += `<div style="margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap;">
        <span class="premium-badge badge-strong">Symbol: ${item.symbol}</span>
        <span class="premium-badge badge-label">Timeframe: ${item.timeframe}</span>
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
                    <div style="font-weight: bold; color: ${color};">${evt.action} <span style="color: var(--text-muted); font-size: 0.9em; font-weight: normal;">(Stage: ${evt.stage})</span></div>
                    <div style="color: var(--text-muted); font-size: 0.85em; margin-top: 4px;">Date: ${evt.timestamp}</div>
                    ${evt.failure_reason && evt.failure_reason !== "None" ? `<div style="color: #ff4c4c; font-size: 0.85em; margin-top: 4px;">Reason: ${evt.failure_reason}</div>` : ''}
                    <div style="font-size: 0.85em; margin-top: 4px; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 4px;">
                        Rule Evaluated: ${evt.rule_evaluated} <br>
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
            <div class="stat-card glass-panel clickable-card" style="text-align: center; padding: 20px; cursor: pointer;" onclick="openCompletionsDrilldown()">
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
                    <button class="tab-btn active" onclick="switchCompletionTab('daily')" id="tab-btn-daily" style="background:transparent; border:none; color:var(--text-color); cursor:pointer; font-weight:bold; padding: 8px 16px; border-radius: 4px; background: rgba(255,255,255,0.1);">Daily (${daily.length})</button>
                    <button class="tab-btn" onclick="switchCompletionTab('weekly')" id="tab-btn-weekly" style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-weight:bold; padding: 8px 16px; border-radius: 4px;">Weekly (${weekly.length})</button>
                    <button class="tab-btn" onclick="switchCompletionTab('monthly')" id="tab-btn-monthly" style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-weight:bold; padding: 8px 16px; border-radius: 4px;">Monthly (${monthly.length})</button>
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
    
    let html = \`
    <div class="table-container" style="max-height: 60vh; overflow-y: auto;">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Trigger Date</th>
                    <th>Completion Date</th>
                    <th>Pattern</th>
                    <th>Sentiment</th>
                    <th>Vol Surge</th>
                    <th>Max Fwd Rtn (5b)</th>
                </tr>
            </thead>
            <tbody>
    \`;
    
    items.forEach(item => {
        const sentimentClass = item.sentiment === 'Bullish' ? 'badge-gap-up' : 'badge-gap-down';
        const returnColor = item.fwd_return_5b > 0 ? '#4cff4c' : '#ff4c4c';
        
        html += \`
            <tr>
                <td class="symbol-cell">\${item.symbol}</td>
                <td>\${item.start_date || '--'}</td>
                <td>\${item.completion_date || '--'}</td>
                <td><span class="premium-badge badge-label">\${item.trigger_pattern}</span></td>
                <td><span class="premium-badge \${sentimentClass}">\${item.sentiment}</span></td>
                <td>\${item.vol_surge_pct !== undefined ? item.vol_surge_pct.toFixed(1) + '%' : '--'}</td>
                <td style="color: \${returnColor}; font-weight: bold;">\${item.fwd_return_5b !== null ? item.fwd_return_5b.toFixed(2) + '%' : '--'}</td>
            </tr>
        \`;
    });
    
    html += \`</tbody></table></div>\`;
    return html;
}

