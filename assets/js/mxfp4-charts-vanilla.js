// Vanilla JavaScript implementation of MXFP4 charts for Jekyll blog integration

class MXFP4Charts {
    constructor() {
        this.mxfp4Values = [
            { code: '0000', binary: 0, value: 0.0, sign: 0, exp: 0, mantissa: 0 },
            { code: '0001', binary: 1, value: 0.5, sign: 0, exp: 0, mantissa: 1 },
            { code: '0010', binary: 2, value: 1.0, sign: 0, exp: 1, mantissa: 0 },
            { code: '0011', binary: 3, value: 1.5, sign: 0, exp: 1, mantissa: 1 },
            { code: '0100', binary: 4, value: 2.0, sign: 0, exp: 2, mantissa: 0 },
            { code: '0101', binary: 5, value: 3.0, sign: 0, exp: 2, mantissa: 1 },
            { code: '0110', binary: 6, value: 4.0, sign: 0, exp: 3, mantissa: 0 },
            { code: '0111', binary: 7, value: 6.0, sign: 0, exp: 3, mantissa: 1 },
            { code: '1000', binary: 8, value: -0.0, sign: 1, exp: 0, mantissa: 0 },
            { code: '1001', binary: 9, value: -0.5, sign: 1, exp: 0, mantissa: 1 },
            { code: '1010', binary: 10, value: -1.0, sign: 1, exp: 1, mantissa: 0 },
            { code: '1011', binary: 11, value: -1.5, sign: 1, exp: 1, mantissa: 1 },
            { code: '1100', binary: 12, value: -2.0, sign: 1, exp: 2, mantissa: 0 },
            { code: '1101', binary: 13, value: -3.0, sign: 1, exp: 2, mantissa: 1 },
            { code: '1110', binary: 14, value: -4.0, sign: 1, exp: 3, mantissa: 0 },
            { code: '1111', binary: 15, value: -6.0, sign: 1, exp: 3, mantissa: 1 }
        ];

        this.selectedView = 'mapping';
        this.charts = {};
    }

    quantizeToMXFP4(value) {
        let minDiff = Infinity;
        let nearestValue = null;

        this.mxfp4Values.forEach(mxfp4 => {
            const diff = Math.abs(value - mxfp4.value);
            if (diff < minDiff) {
                minDiff = diff;
                nearestValue = mxfp4;
            }
        });

        return nearestValue;
    }

    generateMappingData() {
        const data = [];
        for (let i = -7; i <= 7; i += 0.1) {
            const original = parseFloat(i.toFixed(1));
            const quantized = this.quantizeToMXFP4(original);
            data.push({
                original: original,
                quantized: quantized.value,
                error: original - quantized.value,
                code: quantized.code
            });
        }
        return data;
    }

    generateDistributionData() {
        const samples = 10000;
        const distribution = {};

        // Initialize counts for each MXFP4 value
        this.mxfp4Values.forEach(val => {
            distribution[val.value] = 0;
        });

        // Generate normally distributed values (typical for NN weights)
        for (let i = 0; i < samples; i++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

            // Scale to fit roughly in MXFP4 range
            const scaledValue = normal * 1.5;

            // Clamp to MXFP4 range
            const clampedValue = Math.max(-6, Math.min(6, scaledValue));

            // Quantize and count
            const quantized = this.quantizeToMXFP4(clampedValue);
            distribution[quantized.value]++;
        }

        // Convert to chart format
        return this.mxfp4Values.map(val => ({
            value: val.value,
            code: val.code,
            count: distribution[val.value],
            probability: distribution[val.value] / samples
        })).sort((a, b) => a.value - b.value);
    }

    createMappingChart() {
        const mappingData = this.generateMappingData();

        const ctx = document.getElementById('mappingChart').getContext('2d');

        if (this.charts.mapping) {
            this.charts.mapping.destroy();
        }

        this.charts.mapping = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'y = x (identity)',
                    data: mappingData.map(d => ({ x: d.original, y: d.original })),
                    borderColor: '#3b82f6',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0
                }, {
                    label: 'MXFP4 Quantized',
                    data: mappingData.map(d => ({ x: d.original, y: d.quantized })),
                    borderColor: '#ef4444',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    stepped: 'after'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        min: -7,
                        max: 7,
                        title: {
                            display: true,
                            text: 'Original Value'
                        }
                    },
                    y: {
                        min: -7,
                        max: 7,
                        title: {
                            display: true,
                            text: 'Quantized Value'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true
                    },
                    tooltip: {
                        mode: 'nearest',
                        intersect: false
                    }
                }
            }
        });
    }

    createErrorChart() {
        const mappingData = this.generateMappingData();

        const ctx = document.getElementById('errorChart').getContext('2d');

        if (this.charts.error) {
            this.charts.error.destroy();
        }

        this.charts.error = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Quantization Error',
                    data: mappingData.map(d => ({ x: d.original, y: d.error })),
                    borderColor: '#f59e0b',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        min: -7,
                        max: 7,
                        title: {
                            display: true,
                            text: 'Original Value'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Quantization Error'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'nearest',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                return `Error: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    createDistributionChart() {
        const distributionData = this.generateDistributionData();

        const ctx = document.getElementById('distributionChart').getContext('2d');

        if (this.charts.distribution) {
            this.charts.distribution.destroy();
        }

        this.charts.distribution = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: distributionData.map(d => d.value.toString()),
                datasets: [{
                    label: 'Probability',
                    data: distributionData.map(d => d.probability),
                    backgroundColor: '#3b82f6',
                    borderColor: '#1d4ed8',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'MXFP4 Value'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Probability'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(1) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Probability: ${(context.parsed.y * 100).toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    createValueTable() {
        const tableContainer = document.getElementById('valueTable');

        let tableHTML = `
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <thead class="table-light">
                        <tr>
                            <th>Binary Code</th>
                            <th>Sign</th>
                            <th>Exponent</th>
                            <th>Mantissa</th>
                            <th>Value</th>
                            <th>Decimal Index</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        this.mxfp4Values.forEach((item, index) => {
            tableHTML += `
                <tr>
                    <td class="font-monospace">${item.code}</td>
                    <td class="text-center">${item.sign}</td>
                    <td class="text-center">${item.exp.toString(2).padStart(2, '0')}</td>
                    <td class="text-center">${item.mantissa}</td>
                    <td class="text-center fw-bold">${item.value}</td>
                    <td class="text-center">${item.binary}</td>
                </tr>
            `;
        });

        tableHTML += `
                    </tbody>
                </table>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <h5>Value Statistics</h5>
                    <div class="small">
                        <p><strong>Total values:</strong> 16</p>
                        <p><strong>Range:</strong> [-6.0, 6.0]</p>
                        <p><strong>Zero values:</strong> 2 (0.0, -0.0)</p>
                        <p><strong>Positive values:</strong> 7</p>
                        <p><strong>Negative values:</strong> 7</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <h5>Spacing Analysis</h5>
                    <div class="small">
                        <p><strong>Smallest gap:</strong> 0.5 (between 0.0 and 0.5)</p>
                        <p><strong>Largest gap:</strong> 2.0 (between 4.0 and 6.0)</p>
                        <p><strong>Non-uniform spacing:</strong> Logarithmic-like distribution</p>
                        <p><strong>Dense region:</strong> [-1.5, 1.5] (8 out of 16 values)</p>
                    </div>
                </div>
            </div>
        `;

        tableContainer.innerHTML = tableHTML;
    }

    switchView(view) {
        this.selectedView = view;

        // Update button states
        document.querySelectorAll('.chart-nav-btn').forEach(btn => {
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-outline-primary');
        });
        document.getElementById(`btn-${view}`).classList.remove('btn-outline-primary');
        document.getElementById(`btn-${view}`).classList.add('btn-primary');

        // Hide all views
        document.querySelectorAll('.chart-view').forEach(view => {
            view.style.display = 'none';
        });

        // Show selected view
        document.getElementById(`view-${view}`).style.display = 'block';

        // Create charts for the selected view
        if (view === 'mapping') {
            setTimeout(() => {
                this.createMappingChart();
                this.createErrorChart();
            }, 100);
        } else if (view === 'distribution') {
            setTimeout(() => {
                this.createDistributionChart();
            }, 100);
        } else if (view === 'table') {
            this.createValueTable();
        }
    }

    init() {
        // Set up navigation buttons
        document.getElementById('btn-mapping').addEventListener('click', () => this.switchView('mapping'));
        document.getElementById('btn-distribution').addEventListener('click', () => this.switchView('distribution'));
        document.getElementById('btn-table').addEventListener('click', () => this.switchView('table'));

        // Initialize with mapping view
        this.switchView('mapping');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('mxfp4-charts-container')) {
        const charts = new MXFP4Charts();
        charts.init();
    }
});