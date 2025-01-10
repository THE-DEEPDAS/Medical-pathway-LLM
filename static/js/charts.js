function renderCharts(data) {
    // Render heart rate trend
    new Chart(document.getElementById('heartRateChart'), {
        type: 'line',
        data: {
            labels: Array.from({length: 24}, (_, i) => i + 'h'),
            datasets: [{
                label: 'Heart Rate',
                data: data.trends.heart_rate_history,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        }
    });

    // Render stress levels
    new Chart(document.getElementById('stressChart'), {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Stress Level',
                data: data.trends.stress_levels,
                backgroundColor: 'rgb(255, 99, 132)'
            }]
        }
    });
}
