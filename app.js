document.addEventListener('DOMContentLoaded', () => {
    fetchData();
    setupNavigation();
});

function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            // Remove active from all
            navItems.forEach(nav => nav.classList.remove('active'));
            // Add active to clicked
            item.classList.add('active');
        });
    });
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

    // Render Alerts
    const alertsList = document.getElementById('alerts-list');
    alertsList.innerHTML = '';
    if (data.ticker_alerts && data.ticker_alerts.length > 0) {
        data.ticker_alerts.forEach(alert => {
            const li = document.createElement('li');
            li.textContent = alert;
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
