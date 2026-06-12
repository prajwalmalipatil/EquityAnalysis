document.addEventListener('DOMContentLoaded', () => {
    fetchData();
    setupNavigation();
});

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
    const alerts = document.getElementById('alerts-container'); // Need to wrap alerts

    if (tabId === 'overview') {
        overview.style.display = 'grid';
        consensus.style.display = 'flex';
        eigen.style.display = 'flex';
        if (alerts) alerts.style.display = 'flex';
    } else if (tabId === 'consensus') {
        overview.style.display = 'none';
        consensus.style.display = 'flex';
        eigen.style.display = 'none';
        if (alerts) alerts.style.display = 'none';
    } else if (tabId === 'eigen') {
        overview.style.display = 'none';
        consensus.style.display = 'none';
        eigen.style.display = 'flex';
        if (alerts) alerts.style.display = 'none';
    }
}

async function fetchData() {
    try {
        const response = await fetch('./data.json');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Validate schema version
        if (!data.schema_version) {
            throw new Error('Invalid data schema: Missing schema_version');
        }

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
    
    if (data.consensus && data.consensus.length > 0) {
        data.consensus.forEach(item => {
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

    // Render Eigen Stats
    document.getElementById('eigen-daily-count').textContent = data.eigen_filters?.daily?.length || 0;
    document.getElementById('eigen-weekly-count').textContent = data.eigen_filters?.weekly?.length || 0;
    document.getElementById('eigen-monthly-count').textContent = data.eigen_filters?.monthly?.length || 0;
    
    renderEigenTables(data);

    // Render Alerts
    const alertsList = document.getElementById('alerts-list');
    alertsList.innerHTML = '';
    if (data.ticker_alerts && data.ticker_alerts.length > 0) {
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
                    <td>${item.label}</td>
                    <td>${item.gap_dir}</td>
                    <td>${item.close_band}</td>
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
