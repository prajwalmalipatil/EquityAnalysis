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
    const alerts = document.getElementById('alerts-container'); // Need to wrap alerts

    if (tabId === 'overview') {
        overview.style.display = 'grid';
        consensus.style.display = 'flex';
        eigen.style.display = 'flex';
        if (macro) macro.style.display = 'none';
        if (alerts) alerts.style.display = 'flex';
    } else if (tabId === 'consensus') {
        overview.style.display = 'none';
        consensus.style.display = 'flex';
        eigen.style.display = 'none';
        if (macro) macro.style.display = 'none';
        if (alerts) alerts.style.display = 'none';
    } else if (tabId === 'eigen') {
        overview.style.display = 'none';
        consensus.style.display = 'none';
        eigen.style.display = 'flex';
        if (macro) macro.style.display = 'none';
        if (alerts) alerts.style.display = 'none';
    } else if (tabId === 'macro') {
        overview.style.display = 'none';
        consensus.style.display = 'none';
        eigen.style.display = 'none';
        if (macro) macro.style.display = 'flex';
        if (alerts) alerts.style.display = 'none';
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
                    
                    eventCard.innerHTML = `
                        <div class="macro-event-header">
                            <span class="macro-date">${pubDate}</span>
                            <span class="premium-badge ${badgeClass}">${impact.direction || 'Unknown'}</span>
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

    if (cardExtraction) cardExtraction.addEventListener('click', () => openModal('Extraction Details', currentData?.extraction_list || []));
    if (cardVsa) cardVsa.addEventListener('click', () => openModal('VSA Processed Details', currentData?.vsa_list || []));
    if (cardTrending) cardTrending.addEventListener('click', () => openModal('Trending Details', currentData?.trending_list || []));
    
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
