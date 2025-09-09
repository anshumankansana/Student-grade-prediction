class GradePredictorUI {
    constructor() {
        this.gradeChart = null;
        this.importanceChart = null;
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadInitialData();
    }

    setupEventListeners() {
        const form = document.getElementById('prediction-form');
        form.addEventListener('submit', (e) => this.handlePrediction(e));
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadMetrics(),
                this.loadSampleData(),
                this.loadDataDistribution()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    async loadMetrics() {
        try {
            const response = await fetch('/metrics');
            const data = await response.json();
            
            this.displayMetrics(data.model_metrics);
            this.createImportanceChart(data.feature_importance);
        } catch (error) {
            console.error('Error loading metrics:', error);
            document.getElementById('metrics-display').innerHTML = 
                '<div class="error">Unable to load model metrics</div>';
        }
    }

    displayMetrics(metrics) {
        const metricsHtml = `
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value">${metrics.test_r2.toFixed(3)}</div>
                    <div class="metric-label">R² Score</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${metrics.test_rmse.toFixed(2)}</div>
                    <div class="metric-label">RMSE</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${metrics.cv_r2_mean.toFixed(3)}</div>
                    <div class="metric-label">CV R² Mean</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">±${metrics.cv_r2_std.toFixed(3)}</div>
                    <div class="metric-label">CV R² Std</div>
                </div>
            </div>
        `;
        document.getElementById('metrics-display').innerHTML = metricsHtml;
    }

    createImportanceChart(importance) {
        const ctx = document.getElementById('importance-chart').getContext('2d');
        
        if (this.importanceChart) {
            this.importanceChart.destroy();
        }

        const labels = Object.keys(importance).map(key => 
            key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        );
        const values = Object.values(importance);

        this.importanceChart = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance',
                    data: values,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: Math.max(...values) * 1.1
                    }
                }
            }
        });
    }

    async loadDataDistribution() {
        try {
            const response = await fetch('/data-distribution');
            const data = await response.json();
            this.createGradeChart(data.grade_distribution);
        } catch (error) {
            console.error('Error loading data distribution:', error);
        }
    }

    createGradeChart(distribution) {
        const ctx = document.getElementById('grade-chart').getContext('2d');
        
        if (this.gradeChart) {
            this.gradeChart.destroy();
        }

        const labels = Object.keys(distribution).map(range => {
            const [start, end] = range.replace('(', '').replace(']', '').split(', ');
            return `${Math.round(parseFloat(start))}-${Math.round(parseFloat(end))}`;
        });
        const values = Object.values(distribution);

        this.gradeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Number of Students',
                    data: values,
                    backgroundColor: 'rgba(72, 187, 120, 0.8)',
                    borderColor: 'rgba(72, 187, 120, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    async loadSampleData() {
        try {
            const response = await fetch('/sample-data');
            const data = await response.json();
            this.displaySampleData(data);
        } catch (error) {
            console.error('Error loading sample data:', error);
            document.getElementById('sample-data-table').innerHTML = 
                '<div class="error">Unable to load sample data</div>';
        }
    }

    displaySampleData(data) {
        if (!data || data.length === 0) {
            document.getElementById('sample-data-table').innerHTML = 
                '<div class="loading">No sample data available</div>';
            return;
        }

        const headers = Object.keys(data[0]);
        const tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        ${headers.map(header => `<th>${header.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${data.slice(0, 10).map(row => `
                        <tr>
                            ${headers.map(header => `<td>${typeof row[header] === 'number' ? row[header].toFixed(1) : row[header]}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        document.getElementById('sample-data-table').innerHTML = tableHtml;
    }

    async handlePrediction(event) {
        event.preventDefault();
        
        const formData = this.getFormData();
        const resultDiv = document.getElementById('prediction-result');
        
        try {
            resultDiv.style.display = 'block';
            document.getElementById('predicted-grade').textContent = 'Loading...';
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayPrediction(result);
            
        } catch (error) {
            console.error('Prediction error:', error);
            document.getElementById('predicted-grade').textContent = 'Error';
            document.getElementById('grade-range').textContent = 'Unable to predict';
        }
    }

    getFormData() {
        return {
            hours_studied_per_week: parseFloat(document.getElementById('hours_studied').value),
            attendance_percentage: parseFloat(document.getElementById('attendance').value),
            previous_grade: parseFloat(document.getElementById('previous_grade').value),
            sleep_hours_per_night: parseFloat(document.getElementById('sleep_hours').value),
            has_tutor: document.getElementById('has_tutor').checked,
            study_group_participation: document.getElementById('study_group').checked,
            assignments_completion_percentage: parseFloat(document.getElementById('assignments_completion').value),
            extracurricular_hours_per_week: parseFloat(document.getElementById('extracurricular_hours').value)
        };
    }

    displayPrediction(result) {
        const gradeElement = document.getElementById('predicted-grade');
        const rangeElement = document.getElementById('grade-range');
        
        gradeElement.textContent = result.predicted_grade.toFixed(1);
        rangeElement.textContent = `${result.confidence_interval.lower.toFixed(1)} to ${result.confidence_interval.upper.toFixed(1)}`;
        
        // Add grade-based color coding
        const grade = result.predicted_grade;
        if (grade >= 90) {
            gradeElement.style.color = '#38a169';
        } else if (grade >= 80) {
            gradeElement.style.color = '#3182ce';
        } else if (grade >= 70) {
            gradeElement.style.color = '#d69e2e';
        } else {
            gradeElement.style.color = '#e53e3e';
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new GradePredictorUI();
});