// Dashboard JavaScript for ExoHabitatAI

// Load dashboard data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
});

async function loadDashboardData() {
    try {
        // Show loading indicator
        showLoadingState();
        
        // Load statistics
        const statsResponse = await fetch('/api/statistics');
        const statsData = await statsResponse.json();
        
        if (statsData.status === 'success') {
            updateStatistics(statsData.statistics, statsData.additional);
        } else {
            console.error('Statistics error:', statsData.error);
        }
        
        // Load rankings to get habitability scores
        const rankingsResponse = await fetch('/api/rankings?top=1000');
        const rankingsData = await rankingsResponse.json();
        
        if (rankingsData.status === 'success') {
            updateStatisticsFromRankings(rankingsData.rankings);
        }
        
        // Load planets data for charts
        const planetsResponse = await fetch('/api/planets?limit=1000');
        const planetsData = await planetsResponse.json();
        
        if (planetsData.status === 'success') {
            createCharts(planetsData.data);
        } else {
            console.error('Planets error:', planetsData.error);
        }
        
        // Hide loading indicator
        hideLoadingState();
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        hideLoadingState();
        showError('Failed to load dashboard data. Please refresh the page.');
    }
}

function updateStatistics(stats, additional) {
    // Update total planets
    document.getElementById('totalPlanets').textContent = additional.total_records || '-';
    
    // Calculate average habitability score if available
    if (stats.combined_habitability_score) {
        const avgScore = stats.combined_habitability_score.mean || 0;
        document.getElementById('avgScore').textContent = avgScore.toFixed(3);
    } else if (stats.predicted_score) {
        const avgScore = stats.predicted_score.mean || 0;
        document.getElementById('avgScore').textContent = avgScore.toFixed(3);
    }
    
    // Update average temperature
    if (stats.surface_temp) {
        const avgTemp = stats.surface_temp.mean || 0;
        document.getElementById('avgTemp').textContent = Math.round(avgTemp);
    }
}

function updateStatisticsFromRankings(rankings) {
    if (!rankings || rankings.length === 0) return;
    
    // Calculate average habitability score from rankings
    const scores = rankings.map(p => p.predicted_score || p.combined_habitability_score || 0).filter(s => s > 0);
    if (scores.length > 0) {
        const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
        document.getElementById('avgScore').textContent = avgScore.toFixed(3);
    }
    
    // Count potentially habitable planets (score > 0.5)
    const habitableCount = rankings.filter(p => {
        const score = p.predicted_score || p.combined_habitability_score || 0;
        return score >= 0.5;
    }).length;
    
    document.getElementById('habitableCount').textContent = habitableCount || 0;
}

function showLoadingState() {
    // Add loading indicators if needed
    const cards = document.querySelectorAll('.card .card-body h3');
    cards.forEach(card => {
        if (card.textContent === '-') {
            card.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';
        }
    });
}

function hideLoadingState() {
    // Remove loading indicators
}

function showError(message) {
    // Create error alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('.container').prepend(alertDiv);
}

function createCharts(data) {
    if (!data || data.length === 0) return;
    
    const df = data;
    
    // Create habitability distribution chart
    createHabitabilityDistribution(df);
    
    // Create star type distribution chart
    createStarTypeDistribution(df);
    
    // Create feature importance chart (placeholder)
    createFeatureImportance();
    
    // Create correlation matrix (simplified)
    createCorrelationMatrix(df);
}

function createHabitabilityDistribution(data) {
    // Count habitability classes
    const classCounts = {};
    data.forEach(row => {
        const className = row.habitability_class || 'Unknown';
        classCounts[className] = (classCounts[className] || 0) + 1;
    });
    
    const trace = {
        values: Object.values(classCounts),
        labels: Object.keys(classCounts),
        type: 'pie',
        marker: {
            colors: ['#dc3545', '#fd7e14', '#ffc107', '#198754']
        }
    };
    
    const layout = {
        title: 'Habitability Class Distribution',
        height: 400
    };
    
    Plotly.newPlot('habitabilityDistribution', [trace], layout);
}

function createStarTypeDistribution(data) {
    // Count star types
    const typeCounts = {};
    data.forEach(row => {
        const starType = row.star_type || 'Unknown';
        typeCounts[starType] = (typeCounts[starType] || 0) + 1;
    });
    
    const trace = {
        x: Object.keys(typeCounts),
        y: Object.values(typeCounts),
        type: 'bar',
        marker: {
            color: '#0d6efd'
        }
    };
    
    const layout = {
        title: 'Star Type Distribution',
        xaxis: { title: 'Star Type' },
        yaxis: { title: 'Count' },
        height: 400
    };
    
    Plotly.newPlot('starTypeDistribution', [trace], layout);
}

function createFeatureImportance() {
    // Placeholder feature importance (would come from model)
    const features = ['Radius', 'Mass', 'Density', 'Surface Temp', 'Orbital Period', 
                      'Distance', 'Star Type', 'Star Temp', 'Metallicity'];
    const importance = [0.15, 0.12, 0.10, 0.18, 0.08, 0.14, 0.09, 0.10, 0.04];
    
    const trace = {
        x: importance,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: '#198754'
        }
    };
    
    const layout = {
        title: 'Feature Importance for Habitability Prediction',
        xaxis: { title: 'Importance' },
        yaxis: { title: 'Feature' },
        height: 400
    };
    
    Plotly.newPlot('featureImportance', [trace], layout);
}

function createCorrelationMatrix(data) {
    // Simplified correlation matrix (would need actual correlation calculation)
    const features = ['Radius', 'Mass', 'Density', 'Temp', 'Period', 'Distance'];
    const correlations = [
        [1.0, 0.7, 0.5, 0.3, 0.2, 0.4],
        [0.7, 1.0, 0.6, 0.4, 0.3, 0.3],
        [0.5, 0.6, 1.0, 0.2, 0.1, 0.2],
        [0.3, 0.4, 0.2, 1.0, 0.5, 0.6],
        [0.2, 0.3, 0.1, 0.5, 1.0, 0.7],
        [0.4, 0.3, 0.2, 0.6, 0.7, 1.0]
    ];
    
    const trace = {
        z: correlations,
        x: features,
        y: features,
        type: 'heatmap',
        colorscale: 'RdYlBu',
        zmid: 0
    };
    
    const layout = {
        title: 'Parameter Correlation Matrix',
        height: 500
    };
    
    Plotly.newPlot('correlationMatrix', [trace], layout);
}

// Export functions
function exportToPDF() {
    // Show loading indicator
    const btn = event.target.closest('button');
    const originalHTML = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    btn.disabled = true;
    
    // Trigger download (top 50 by default for dashboard)
    window.location.href = '/api/export/pdf?top=50';
    
    // Reset button after a delay
    setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.disabled = false;
    }, 2000);
}

function exportToExcel() {
    // Show loading indicator
    const btn = event.target.closest('button');
    const originalHTML = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    btn.disabled = true;
    
    // Trigger download (top 100 by default for dashboard)
    window.location.href = '/api/export/excel?top=100';
    
    // Reset button after a delay
    setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.disabled = false;
    }, 2000);
}


