/**
 * MXFP4 Deep Dive Visualizations
 * Interactive visualizations for understanding MXFP4 quantization,
 * matrix multiplication, and MoE layers
 */

// MXFP4 Constants and Utilities
const MXFP4_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6];
const MXFP4_LOOKUP = {
    0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0,
    8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
};

class MXFP4Visualizer {
    constructor() {
        this.charts = {};
        this.animations = {};
        this.canvases = new Map();
        this.setupCanvases();
        this.initializeCharts();
        this.setupEventListeners();
    }

    setupCanvases() {
        // Get all canvas elements and set up proper sizing
        const canvasIds = [
            'valueSpaceCanvas',
            'mappingCanvas',
            'errorCanvas',
            'distributionCanvas',
            'matmulCanvas',
            'moeDiagramCanvas',
            'moeMemoryCanvas',
            'memoryChart',
            'throughputChart',
            'trainingThroughputChart',
            'inferenceLatencyChart'
        ];

        canvasIds.forEach(id => {
            const canvas = document.getElementById(id);
            if (canvas) {
                this.setupCanvas(canvas);
                this.canvases.set(id, canvas);
            }
        });
    }

    setupCanvas(canvas) {
        const container = canvas.parentElement;
        const dpr = window.devicePixelRatio || 1;

        // Set display size
        const rect = container.getBoundingClientRect();
        canvas.style.width = '100%';
        canvas.style.height = container.style.height || '400px';

        // Set actual canvas size accounting for device pixel ratio
        const displayWidth = rect.width || container.offsetWidth;
        const displayHeight = parseInt(container.style.height) || 400;

        canvas.width = displayWidth * dpr;
        canvas.height = displayHeight * dpr;

        // Scale the context to match device pixel ratio
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        // Store the display dimensions for drawing calculations
        canvas.displayWidth = displayWidth;
        canvas.displayHeight = displayHeight;
    }

    initializeCharts() {
        // Add a small delay to ensure DOM is fully rendered
        requestAnimationFrame(() => {
            this.createValueSpaceChart(); // Default view
            this.createMatMulVisualization();
            this.createMoEVisualization();
            this.createPerformanceCharts();
        });
    }

    setupEventListeners() {
        // Value space navigation
        this.setupValueSpaceNavigation();

        // Matrix multiplication controls
        const stepButton = document.getElementById('step-button');
        const resetButton = document.getElementById('reset-button');
        const randomizeButton = document.getElementById('randomize-button');
        const simulateButton = document.getElementById('simulate-button');
        const speedSlider = document.getElementById('speed-slider');

        if (stepButton) {
            stepButton.addEventListener('click', () => this.stepMatMul());
        }
        if (resetButton) {
            resetButton.addEventListener('click', () => this.resetMatMul());
        }
        if (randomizeButton) {
            randomizeButton.addEventListener('click', () => this.randomizeMatrices());
        }
        if (simulateButton) {
            simulateButton.addEventListener('click', () => {
                this.toggleSimulation();
            });
        }
        if (speedSlider) {
            speedSlider.addEventListener('input', (e) => this.updateSimulationSpeed(e.target.value));
        }

        // MoE controls
        const animateMoE = document.getElementById('animate-moe');
        const showRouting = document.getElementById('show-routing');
        const highlightQuantization = document.getElementById('highlight-quantization');

        if (animateMoE) {
            animateMoE.addEventListener('click', () => this.animateMoEFlow());
        }
        if (showRouting) {
            showRouting.addEventListener('click', () => this.showRouting());
        }
        if (highlightQuantization) {
            highlightQuantization.addEventListener('click', () => this.highlightQuantization());
        }

        // Sliders for MoE efficiency
        const expertSlider = document.getElementById('num-experts-slider');
        const sizeSlider = document.getElementById('expert-size-slider');

        if (expertSlider) {
            expertSlider.addEventListener('input', (e) => {
                document.getElementById('num-experts-value').textContent = e.target.value;
                this.updateMoEEfficiencyChart();
            });
        }
        if (sizeSlider) {
            sizeSlider.addEventListener('input', (e) => {
                document.getElementById('expert-size-value').textContent = e.target.value;
                this.updateMoEEfficiencyChart();
            });
        }
    }

    setupValueSpaceNavigation() {

        const navButtons = [
            { id: 'btn-value-space', view: 'view-value-space' },
            { id: 'btn-mapping', view: 'view-mapping' },
            { id: 'btn-distribution', view: 'view-distribution' },
            { id: 'btn-table', view: 'view-table' }
        ];

        navButtons.forEach(({ id, view }) => {
            const button = document.getElementById(id);

            if (button) {
                button.addEventListener('click', (e) => {
                    e.preventDefault();

                    // Update button states - more robust approach using querySelectorAll
                    const allButtons = document.querySelectorAll('#mxfp4-format-viz .btn-group .btn');
                    allButtons.forEach(btn => {
                        btn.classList.remove('btn-primary');
                        btn.classList.add('btn-outline-primary');
                    });
                    button.classList.remove('btn-outline-primary');
                    button.classList.add('btn-primary');

                    // Show/hide views
                    navButtons.forEach(({ view: otherView }) => {
                        const otherViewEl = document.getElementById(otherView);
                        if (otherViewEl) {
                            otherViewEl.style.display = 'none';
                        }
                    });

                    const viewEl = document.getElementById(view);

                    if (viewEl) {
                        viewEl.style.display = 'block';

                        // Small delay to ensure canvas is visible before drawing
                        setTimeout(() => {
                            // Trigger appropriate visualization
                            switch (view) {
                                case 'view-value-space':
                                    this.createValueSpaceChart();
                                    break;
                                case 'view-mapping':
                                    this.createMappingChart();
                                    this.createErrorChart();
                                    break;
                                case 'view-distribution':
                                    this.createDistributionChart();
                                    break;
                                case 'view-table':
                                    this.createValueTable();
                                    break;
                            }
                        }, 50);
                    }
                });
            }
        });
    }

    createValueSpaceChart() {
        const canvas = document.getElementById('valueSpaceCanvas');
        if (!canvas) return;

        // Canvas is already set up in setupCanvas
        const ctx = canvas.getContext('2d');
        this.drawValueSpace(ctx, canvas.displayWidth, canvas.displayHeight);
    }

    drawValueSpace(ctx, width, height) {
        const margin = 60;
        const plotWidth = width - 2 * margin;
        const plotHeight = height - 2 * margin;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.stroke();

        // Draw MXFP4 values as discrete points
        const uniqueValues = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6];
        const xScale = plotWidth / (uniqueValues.length - 1);

        ctx.fillStyle = '#e74c3c';
        uniqueValues.forEach((value, i) => {
            const x = margin + i * xScale;
            const y = height - margin - (value + 6) * plotHeight / 12; // Normalize to plot area

            // Draw point
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, 2 * Math.PI);
            ctx.fill();

            // Label
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(value.toString(), x, height - margin + 20);

            // Binary representation
            const code = Object.keys(MXFP4_LOOKUP).find(key => MXFP4_LOOKUP[key] === value);
            if (code !== undefined) {
                const binary = parseInt(code).toString(2).padStart(4, '0');
                ctx.font = '10px monospace';
                ctx.fillText(binary, x, height - margin + 35);
            }
            ctx.fillStyle = '#e74c3c';
        });

        // Add labels
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('MXFP4 Values', width / 2, height - 10);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Value', 0, 0);
        ctx.restore();
    }

    createMappingChart() {
        const canvas = document.getElementById('mappingCanvas');
        if (!canvas) {
            return;
        }

        // Set up canvas if not already done
        if (!canvas.displayWidth) {
            this.setupCanvas(canvas);
        }

        const ctx = canvas.getContext('2d');
        this.drawMappingVisualization(ctx, canvas.displayWidth, canvas.displayHeight);
    }

    drawMappingVisualization(ctx, width, height) {
        const margin = 60;
        const plotWidth = width - 2 * margin;
        const plotHeight = height - 2 * margin;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.stroke();

        // Create continuous input range
        const inputRange = [];
        const mxfp4Values = [];
        const steps = 1000;

        for (let i = 0; i <= steps; i++) {
            const input = -7 + (14 * i) / steps; // Range from -7 to 7
            inputRange.push(input);
            mxfp4Values.push(this.quantizeToMXFP4(input));
        }

        // Draw continuous input line
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i <= steps; i++) {
            const x = margin + (i * plotWidth) / steps;
            const y = height - margin - ((inputRange[i] + 7) * plotHeight) / 14;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw quantized step function
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i <= steps; i++) {
            const x = margin + (i * plotWidth) / steps;
            const y = height - margin - ((mxfp4Values[i] + 7) * plotHeight) / 14;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Add legend with better formatting - top left position
        const legendX = margin + 20;
        const legendY = 40;
        const legendPadding = 15;

        // Legend background
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.fillRect(legendX - legendPadding, legendY - 10, 230, 70);
        ctx.strokeRect(legendX - legendPadding, legendY - 10, 230, 70);

        // Legend title
        ctx.fillStyle = '#333';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Legend', legendX, legendY + 5);

        // Continuous Input legend item
        ctx.fillStyle = '#3498db';
        ctx.fillRect(legendX, legendY + 15, 20, 3);
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.fillText('Continuous Input', legendX + 30, legendY + 20);

        // MXFP4 Quantized legend item
        ctx.fillStyle = '#e74c3c';
        ctx.fillRect(legendX, legendY + 35, 20, 3);
        ctx.fillStyle = '#333';
        ctx.fillText('MXFP4 Quantized', legendX + 30, legendY + 40);

        // Add labels
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Input Value', width / 2, height - 10);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Output Value', 0, 0);
        ctx.restore();
    }

    createDistributionChart() {
        const canvas = document.getElementById('distributionCanvas');
        if (!canvas) {
            return;
        }

        // Set up canvas if not already done
        if (!canvas.displayWidth) {
            this.setupCanvas(canvas);
        }

        const ctx = canvas.getContext('2d');
        this.drawDistributionVisualization(ctx, canvas.displayWidth, canvas.displayHeight);
    }

    drawDistributionVisualization(ctx, width, height) {
        const margin = 60;
        const plotWidth = width - 2 * margin;
        const plotHeight = height - 2 * margin;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Generate normally distributed samples (simulating NN weights)
        const samples = this.generateNormalSamples(10000, 0, 1.5);

        // Count distribution of MXFP4 values
        const mxfp4Counts = {};
        const uniqueValues = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6];

        uniqueValues.forEach(val => mxfp4Counts[val] = 0);

        samples.forEach(sample => {
            const quantized = this.quantizeToMXFP4(sample);
            mxfp4Counts[quantized]++;
        });

        // Draw histogram
        const maxCount = Math.max(...Object.values(mxfp4Counts));
        const barWidth = plotWidth / uniqueValues.length;

        uniqueValues.forEach((value, i) => {
            const count = mxfp4Counts[value];
            const barHeight = (count / maxCount) * plotHeight * 0.8;
            const x = margin + i * barWidth;
            const y = height - margin - barHeight;

            // Bar
            ctx.fillStyle = this.getBarColor(value);
            ctx.fillRect(x + barWidth * 0.1, y, barWidth * 0.8, barHeight);

            // Value label
            ctx.fillStyle = '#333';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(value.toString(), x + barWidth / 2, height - margin + 15);

            // Count label
            ctx.fillText(count.toString(), x + barWidth / 2, y - 5);
        });

        // Axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('MXFP4 Values', width / 2, height - 10);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Frequency', 0, 0);
        ctx.restore();
    }

    createValueTable() {
        const tableContainer = document.getElementById('valueTable');
        if (!tableContainer) return;

        const tableData = [
            { code: 0, binary: '0000', sign: 0, exp: '00', mant: 0, calculation: '+1.0×2⁻¹×1.0', value: 0.0 },
            { code: 1, binary: '0001', sign: 0, exp: '00', mant: 1, calculation: '+1.0×2⁻¹×1.5', value: 0.5 },
            { code: 2, binary: '0010', sign: 0, exp: '01', mant: 0, calculation: '+1.0×2⁰×1.0', value: 1.0 },
            { code: 3, binary: '0011', sign: 0, exp: '01', mant: 1, calculation: '+1.0×2⁰×1.5', value: 1.5 },
            { code: 4, binary: '0100', sign: 0, exp: '10', mant: 0, calculation: '+1.0×2¹×1.0', value: 2.0 },
            { code: 5, binary: '0101', sign: 0, exp: '10', mant: 1, calculation: '+1.0×2¹×1.5', value: 3.0 },
            { code: 6, binary: '0110', sign: 0, exp: '11', mant: 0, calculation: '+1.0×2²×1.0', value: 4.0 },
            { code: 7, binary: '0111', sign: 0, exp: '11', mant: 1, calculation: '+1.0×2²×1.5', value: 6.0 },
            { code: 8, binary: '1000', sign: 1, exp: '00', mant: 0, calculation: '-1.0×2⁻¹×1.0', value: -0.0 },
            { code: 9, binary: '1001', sign: 1, exp: '00', mant: 1, calculation: '-1.0×2⁻¹×1.5', value: -0.5 },
            { code: 10, binary: '1010', sign: 1, exp: '01', mant: 0, calculation: '-1.0×2⁰×1.0', value: -1.0 },
            { code: 11, binary: '1011', sign: 1, exp: '01', mant: 1, calculation: '-1.0×2⁰×1.5', value: -1.5 },
            { code: 12, binary: '1100', sign: 1, exp: '10', mant: 0, calculation: '-1.0×2¹×1.0', value: -2.0 },
            { code: 13, binary: '1101', sign: 1, exp: '10', mant: 1, calculation: '-1.0×2¹×1.5', value: -3.0 },
            { code: 14, binary: '1110', sign: 1, exp: '11', mant: 0, calculation: '-1.0×2²×1.0', value: -4.0 },
            { code: 15, binary: '1111', sign: 1, exp: '11', mant: 1, calculation: '-1.0×2²×1.5', value: -6.0 }
        ];

        const table = document.createElement('table');
        table.className = 'table table-striped table-hover';

        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Code</th>
                <th>Binary</th>
                <th>Sign</th>
                <th>Exp</th>
                <th>Mant</th>
                <th>Calculation</th>
                <th>Value</th>
                <th>Usage</th>
            </tr>
        `;
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        tableData.forEach(row => {
            const tr = document.createElement('tr');
            const usage = this.getValueUsage(row.value);
            tr.innerHTML = `
                <td><code>${row.code}</code></td>
                <td><code>${row.binary}</code></td>
                <td>${row.sign}</td>
                <td><code>${row.exp}</code></td>
                <td>${row.mant}</td>
                <td><code>${row.calculation}</code></td>
                <td><strong>${row.value}</strong></td>
                <td><span class="badge bg-${usage.color}">${usage.text}</span></td>
            `;
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        tableContainer.innerHTML = '';
        tableContainer.appendChild(table);
    }

    // Utility functions
    quantizeToMXFP4(value) {
        const mxfp4Values = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6];

        // Clamp to MXFP4 range
        if (value > 6) return 6;
        if (value < -6) return -6;

        // Find nearest MXFP4 value
        let nearest = mxfp4Values[0];
        let minDist = Math.abs(value - nearest);

        for (const mxfpValue of mxfp4Values) {
            const dist = Math.abs(value - mxfpValue);
            if (dist < minDist) {
                minDist = dist;
                nearest = mxfpValue;
            }
        }

        return nearest;
    }

    generateNormalSamples(count, mean, stddev) {
        const samples = [];
        for (let i = 0; i < count; i++) {
            // Box-Muller transform
            const u1 = Math.random();
            const u2 = Math.random();
            const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            samples.push(mean + stddev * z0);
        }
        return samples;
    }

    getBarColor(value) {
        if (Math.abs(value) <= 1) return '#2ecc71'; // Green for common values
        if (Math.abs(value) <= 2) return '#f39c12'; // Orange for medium values
        return '#e74c3c'; // Red for rare values
    }

    getValueUsage(value) {
        if (Math.abs(value) <= 1) return { color: 'success', text: 'High' };
        if (Math.abs(value) <= 3) return { color: 'warning', text: 'Medium' };
        return { color: 'danger', text: 'Low' };
    }

    createErrorChart() {
        const canvas = document.getElementById('errorCanvas');
        if (!canvas) {
            return;
        }

        // Set up canvas if not already done
        if (!canvas.displayWidth) {
            this.setupCanvas(canvas);
        }

        const ctx = canvas.getContext('2d');
        this.drawErrorVisualization(ctx, canvas.displayWidth, canvas.displayHeight);
    }

    drawErrorVisualization(ctx, width, height) {
        const margin = 60;
        const plotWidth = width - 2 * margin;
        const plotHeight = height - 2 * margin;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Create input range and calculate errors
        const inputRange = [];
        const errors = [];
        const steps = 1000;

        for (let i = 0; i <= steps; i++) {
            const input = -7 + (14 * i) / steps;
            const quantized = this.quantizeToMXFP4(input);
            const error = quantized - input;

            inputRange.push(input);
            errors.push(error);
        }

        // Draw error curve
        ctx.strokeStyle = '#9b59b6';
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i <= steps; i++) {
            const x = margin + (i * plotWidth) / steps;
            const y = height - margin - ((errors[i] + 1) * plotHeight) / 2; // Assuming error range [-1, 1]
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw zero line
        ctx.strokeStyle = '#95a5a6';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        const zeroY = height - margin - plotHeight / 2;
        ctx.beginPath();
        ctx.moveTo(margin, zeroY);
        ctx.lineTo(width - margin, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Input Value', width / 2, height - 10);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Quantization Error', 0, 0);
        ctx.restore();
    }

    createMatMulVisualization() {
        const canvas = document.getElementById('matmulCanvas');
        if (!canvas) return;

        // Initialize state
        this.matmulStep = 0;
        this.maxMatmulSteps = 16; // 4x4 result matrix
        this.isSimulating = false;
        this.simulationSpeed = 5; // Default speed
        this.simulationInterval = null;

        // Initialize matrices for visualization
        this.matrixA = this.generateMXFP4Matrix(4, 4);
        this.matrixB = this.generateMXFP4Matrix(4, 4);
        this.resultMatrix = new Array(4).fill(0).map(() => new Array(4).fill(0));

        // Update UI
        this.updateStepDisplay();
        this.updateSpeedDisplay();
        this.drawMatMulStep();
    }

    generateMXFP4Matrix(rows, cols) {
        return new Array(rows).fill(0).map(() =>
            new Array(cols).fill(0).map(() => {
                const code = Math.floor(Math.random() * 16);
                return {
                    value: MXFP4_LOOKUP[code],
                    code: code
                };
            })
        );
    }

    drawMatMulStep() {
        const canvas = document.getElementById('matmulCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.displayWidth, canvas.displayHeight);

        // Calculate larger cell size for better text visibility
        const cellSize = Math.max(45, Math.min(70, (canvas.displayWidth - 200) / 10));

        // Calculate matrix positions with improved layout
        const matrixWidth = 4 * cellSize;
        const spacing = 100;

        // Position input matrices (A × B) in top row, result matrix below
        const inputWidth = matrixWidth + spacing + matrixWidth;
        const inputStartX = (canvas.displayWidth - inputWidth) / 2;

        // Calculate proper vertical spacing
        const matrixHeight = 4 * cellSize;
        const titleHeight = 25; // Space for title above matrix
        const verticalSpacing = 120; // Space between input and result matrices

        const matrixAStart = { x: inputStartX, y: titleHeight + 60 };
        const matrixBStart = { x: inputStartX + matrixWidth + spacing, y: titleHeight + 60 };
        const resultStart = {
            x: (canvas.displayWidth - matrixWidth) / 2,
            y: matrixAStart.y + matrixHeight + verticalSpacing
        };

        // Draw operation symbols
        this.drawOperationSymbols(ctx, matrixAStart, matrixBStart, resultStart, cellSize);

        // Draw matrices with improved styling
        this.drawMatrix(ctx, this.matrixA, matrixAStart, cellSize, 'Matrix A (4×4 MXFP4)', '#3498db');
        this.drawMatrix(ctx, this.matrixB, matrixBStart, cellSize, 'Matrix B (4×4 MXFP4)', '#2ecc71');
        this.drawMatrix(ctx, this.resultMatrix, resultStart, cellSize, 'Result C = A × B', '#e74c3c', true);

        // Add step information
        this.drawStepInfo(ctx, canvas.displayWidth, cellSize, resultStart);

        // Highlight current computation step
        if (this.matmulStep > 0 && this.matmulStep <= this.maxMatmulSteps) {
            this.highlightMatMulStep(ctx, cellSize, matrixAStart, matrixBStart, resultStart);
        }
    }

    drawMatrix(ctx, matrix, start, cellSize, title, color, isResult = false) {
        const rows = matrix.length;
        const cols = matrix[0].length;
        const matrixWidth = cols * cellSize;

        // Title - centered above matrix
        ctx.fillStyle = '#333';
        ctx.font = 'bold 18px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(title, start.x + matrixWidth / 2, start.y - 20);

        // Matrix cells
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const x = start.x + j * cellSize;
                const y = start.y + i * cellSize;

                // Cell background
                ctx.fillStyle = isResult ? 'rgba(255, 255, 255, 0.9)' : 'rgba(255, 255, 255, 0.9)';
                ctx.fillRect(x, y, cellSize, cellSize);

                // Cell border
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, cellSize, cellSize);

                // Cell content
                ctx.fillStyle = '#333';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                if (isResult) {
                    // Result matrix - larger font for computed values
                    ctx.font = `bold ${Math.max(14, cellSize * 0.35)}px Arial`;
                    ctx.fillText(matrix[i][j].toFixed(1), x + cellSize / 2, y + cellSize / 2);
                } else {
                    // Input matrices - larger MXFP4 values
                    const value = matrix[i][j].value;
                    const valueFontSize = Math.max(12, cellSize * 0.3);
                    ctx.font = `bold ${valueFontSize}px Arial`;
                    ctx.fillText(value.toString(), x + cellSize / 2, y + cellSize / 2 - cellSize * 0.18);

                    // Binary code - more readable size
                    const binaryFontSize = Math.max(9, cellSize * 0.2);
                    ctx.font = `${binaryFontSize}px monospace`;
                    ctx.fillStyle = '#666';
                    const binary = matrix[i][j].code.toString(2).padStart(4, '0');
                    ctx.fillText(binary, x + cellSize / 2, y + cellSize / 2 + cellSize * 0.18);
                }
            }
        }
    }

    highlightMatMulStep(ctx, cellSize, matrixAStart, matrixBStart, resultStart) {
        // Calculate which cell we're computing
        const resultRow = Math.floor((this.matmulStep - 1) / 4);
        const resultCol = (this.matmulStep - 1) % 4;

        // Highlight with animated glow effect
        const glowColor = '#f39c12';

        // Result cell - thick orange border
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = 4;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 8;
        ctx.strokeRect(
            resultStart.x + resultCol * cellSize,
            resultStart.y + resultRow * cellSize,
            cellSize,
            cellSize
        );

        // Row in A - highlighted
        ctx.lineWidth = 3;
        ctx.shadowBlur = 4;
        for (let j = 0; j < 4; j++) {
            ctx.strokeRect(
                matrixAStart.x + j * cellSize,
                matrixAStart.y + resultRow * cellSize,
                cellSize,
                cellSize
            );
        }

        // Column in B - highlighted
        for (let i = 0; i < 4; i++) {
            ctx.strokeRect(
                matrixBStart.x + resultCol * cellSize,
                matrixBStart.y + i * cellSize,
                cellSize,
                cellSize
            );
        }

        // Reset shadow
        ctx.shadowBlur = 0;

        // Compute and update result
        let sum = 0;
        const debugInfo = [];
        for (let k = 0; k < 4; k++) {
            const aVal = this.matrixA[resultRow][k];
            const bVal = this.matrixB[k][resultCol];
            const product = aVal.value * bVal.value;
            sum += product;
            debugInfo.push(`${aVal.value} × ${bVal.value} = ${product}`);
        }
        this.resultMatrix[resultRow][resultCol] = sum;

        // Debug logging for the calculation
        if (resultRow === 3 && resultCol === 3) {
            console.log(`Row 4, Col 4 calculation:`);
            debugInfo.forEach((info, i) => console.log(`  Step ${i+1}: ${info}`));
            console.log(`  Final sum: ${sum}`);
        }
    }

    drawOperationSymbols(ctx, matrixAStart, matrixBStart, resultStart, cellSize) {
        ctx.fillStyle = '#666';
        ctx.font = 'bold 40px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // × symbol between A and B
        const multiplyX = (matrixAStart.x + 4 * cellSize + matrixBStart.x) / 2;
        const multiplyY = matrixAStart.y + (2 * cellSize);
        ctx.fillText('×', multiplyX, multiplyY);

        // = symbol above result matrix (positioned to avoid title overlap)
        const equalsX = resultStart.x + (2 * cellSize);
        const equalsY = resultStart.y - 60; // Moved higher to avoid title overlap
        ctx.fillText('=', equalsX, equalsY);
    }

    drawStepInfo(ctx, width, cellSize, resultStart) {
        if (this.matmulStep === 0) return;

        const resultRow = Math.floor((this.matmulStep - 1) / 4);
        const resultCol = (this.matmulStep - 1) % 4;

        // Position info box below the result matrix
        const matrixHeight = 4 * cellSize;
        const infoBoxY = resultStart.y + matrixHeight + 20; // 20px below result matrix
        const infoBoxHeight = 100; // Increased height for better text display

        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fillRect(10, infoBoxY, width - 20, infoBoxHeight);

        // Add border for better visibility
        ctx.strokeStyle = '#f39c12';
        ctx.lineWidth = 2;
        ctx.strokeRect(10, infoBoxY, width - 20, infoBoxHeight);

        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';

        // First line: operation description
        ctx.fillText(`Computing C[${resultRow}][${resultCol}] = A[${resultRow}][:] · B[:][${resultCol}]`, 20, infoBoxY + 25);

        // Second line: detailed computation (split if too long)
        let computation = '';
        let sum = 0;
        for (let k = 0; k < 4; k++) {
            const aVal = this.matrixA[resultRow][k].value;
            const bVal = this.matrixB[k][resultCol].value;
            sum += aVal * bVal;

            if (k > 0) computation += ' + ';
            computation += `(${aVal.toString()}×${bVal.toString()})`;
        }

        const computationWithResult = `${computation} = ${sum.toFixed(2)}`;

        ctx.fillText(computationWithResult, 20, infoBoxY + 45);
        ctx.fillText(`MXFP4 quantized values (no additional scaling)`, 20, infoBoxY + 65);
        ctx.fillText(`Result: ${sum.toFixed(2)}`, 20, infoBoxY + 85);
    }

    stepMatMul() {
        if (this.matmulStep < this.maxMatmulSteps) {
            this.matmulStep++;
            this.updateStepDisplay();
            this.drawMatMulStep();
        }
    }

    resetMatMul() {
        this.stopSimulation();
        this.matmulStep = 0;
        this.updateStepDisplay();
        this.resultMatrix = new Array(4).fill(0).map(() => new Array(4).fill(0));
        this.drawMatMulStep();
    }

    randomizeMatrices() {
        this.matrixA = this.generateMXFP4Matrix(4, 4);
        this.matrixB = this.generateMXFP4Matrix(4, 4);
        this.resetMatMul();
    }

    toggleSimulation() {
        const button = document.getElementById('simulate-button');

        if (!button) {
            return;
        }

        if (this.isSimulating) {
            this.stopSimulation();
            button.textContent = 'Auto Simulate';
            button.className = 'btn btn-success';
        } else {
            this.startSimulation();
            button.textContent = 'Stop Simulation';
            button.className = 'btn btn-warning';
        }
    }

    startSimulation() {
        if (this.isSimulating) {
            return;
        }

        this.isSimulating = true;
        const delay = 1100 - (this.simulationSpeed * 100); // 1000ms to 100ms

        this.simulationInterval = setInterval(() => {
            if (this.matmulStep >= this.maxMatmulSteps) {
                this.stopSimulation();
                const button = document.getElementById('simulate-button');
                if (button) {
                    button.textContent = 'Auto Simulate';
                    button.className = 'btn btn-success';
                }
                // Ensure final state is rendered
                this.drawMatMulStep();
                return;
            }
            this.stepMatMul();
        }, delay);
    }

    stopSimulation() {
        this.isSimulating = false;
        if (this.simulationInterval) {
            clearInterval(this.simulationInterval);
            this.simulationInterval = null;
        }
    }

    updateSimulationSpeed(value) {
        this.simulationSpeed = parseInt(value);
        this.updateSpeedDisplay();

        // If currently simulating, restart with new speed
        if (this.isSimulating) {
            this.stopSimulation();
            this.isSimulating = false; // Reset flag
            this.startSimulation();
        }
    }

    updateStepDisplay() {
        const currentStepEl = document.getElementById('current-step');
        const totalStepsEl = document.getElementById('total-steps');
        if (currentStepEl) currentStepEl.textContent = this.matmulStep;
        if (totalStepsEl) totalStepsEl.textContent = this.maxMatmulSteps;
    }

    updateSpeedDisplay() {
        const speedValueEl = document.getElementById('speed-value');
        if (speedValueEl) {
            const speeds = ['Very Slow', 'Slow', 'Slow', 'Medium', 'Medium', 'Medium', 'Fast', 'Fast', 'Very Fast', 'Lightning'];
            speedValueEl.textContent = speeds[this.simulationSpeed - 1] || 'Medium';
        }
    }

    createMoEVisualization() {
        const canvas = document.getElementById('moeDiagramCanvas');
        if (!canvas) return;

        this.drawMoEDiagram();
        this.createMoEEfficiencyChart();
    }

    drawMoEDiagram() {
        const canvas = document.getElementById('moeDiagramCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.displayWidth, canvas.displayHeight);

        const centerX = canvas.displayWidth / 2;
        const centerY = canvas.displayHeight / 2;
        const expertRadius = 40;
        const routerRadius = 30;

        // Draw router
        ctx.fillStyle = '#3498db';
        ctx.beginPath();
        ctx.arc(centerX, centerY - 150, routerRadius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Router', centerX, centerY - 145);

        // Draw experts in a circle
        const numExperts = 8;
        const expertCircleRadius = 120;

        for (let i = 0; i < numExperts; i++) {
            const angle = (i * 2 * Math.PI) / numExperts;
            const expertX = centerX + Math.cos(angle) * expertCircleRadius;
            const expertY = centerY + Math.sin(angle) * expertCircleRadius;

            // Expert circle
            ctx.fillStyle = '#e74c3c';
            ctx.beginPath();
            ctx.arc(expertX, expertY, expertRadius, 0, 2 * Math.PI);
            ctx.fill();

            // Expert label
            ctx.fillStyle = '#fff';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`Expert ${i + 1}`, expertX, expertY - 5);
            ctx.fillText('MXFP4', expertX, expertY + 5);

            // Connection to router (dashed for inactive experts)
            ctx.setLineDash(i < 4 ? [] : [5, 5]); // First 4 active
            ctx.strokeStyle = i < 4 ? '#2ecc71' : '#95a5a6';
            ctx.lineWidth = i < 4 ? 3 : 1;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY - 150 + routerRadius);
            ctx.lineTo(expertX, expertY - expertRadius);
            ctx.stroke();
        }

        // Input token
        ctx.fillStyle = '#9b59b6';
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(centerX, centerY - 250, 25, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Token', centerX, centerY - 245);

        // Arrow from token to router
        this.drawArrow(ctx, centerX, centerY - 225, centerX, centerY - 180, '#333');

        // Output
        ctx.fillStyle = '#16a085';
        ctx.beginPath();
        ctx.arc(centerX, centerY + 200, 30, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.fillText('Output', centerX, centerY + 205);

        // Arrows from active experts to output
        for (let i = 0; i < 4; i++) {
            const angle = (i * 2 * Math.PI) / numExperts;
            const expertX = centerX + Math.cos(angle) * expertCircleRadius;
            const expertY = centerY + Math.sin(angle) * expertCircleRadius;

            this.drawArrow(ctx, expertX, expertY + expertRadius, centerX, centerY + 170, '#2ecc71');
        }

        // Add legend
        this.drawMoELegend(ctx, canvas.displayWidth - 150, 50);
    }

    drawArrow(ctx, fromX, fromY, toX, toY, color) {
        const headlen = 10;
        const dx = toX - fromX;
        const dy = toY - fromY;
        const angle = Math.atan2(dy, dx);

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.lineTo(toX - headlen * Math.cos(angle - Math.PI / 6), toY - headlen * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headlen * Math.cos(angle + Math.PI / 6), toY - headlen * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
    }

    drawMoELegend(ctx, x, y) {
        const items = [
            { color: '#2ecc71', text: 'Active Expert', dash: false },
            { color: '#95a5a6', text: 'Inactive Expert', dash: true },
            { color: '#e74c3c', text: 'MXFP4 Weights', dash: false }
        ];

        ctx.font = '12px Arial';
        ctx.textAlign = 'left';

        items.forEach((item, i) => {
            const itemY = y + i * 25;

            // Legend color/line
            ctx.strokeStyle = item.color;
            ctx.setLineDash(item.dash ? [5, 5] : []);
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x, itemY);
            ctx.lineTo(x + 20, itemY);
            ctx.stroke();

            // Legend text
            ctx.fillStyle = '#333';
            ctx.fillText(item.text, x + 25, itemY + 4);
        });

        ctx.setLineDash([]);
    }

    createMoEEfficiencyChart() {
        const canvas = document.getElementById('moeMemoryCanvas');
        if (!canvas) return;

        this.updateMoEEfficiencyChart();
    }

    updateMoEEfficiencyChart() {
        const canvas = document.getElementById('moeMemoryCanvas');
        if (!canvas) return;

        // Reseup canvas to get current dimensions
        this.setupCanvas(canvas);
        const ctx = canvas.getContext('2d');

        const numExperts = parseInt(document.getElementById('num-experts-slider')?.value || 128);
        const expertSizeMB = parseInt(document.getElementById('expert-size-slider')?.value || 100);

        // Calculate memory usage
        const bf16Memory = numExperts * expertSizeMB * 2; // 2 bytes per param
        const mxfp4Memory = numExperts * expertSizeMB * 0.56; // ~3.6x compression

        this.drawBarChart(ctx, canvas.displayWidth, canvas.displayHeight,
            ['BF16', 'MXFP4'],
            [bf16Memory, mxfp4Memory],
            ['#e74c3c', '#2ecc71'],
            'Memory Usage (MB)',
            `${numExperts} Experts, ${expertSizeMB}M params each`);
    }

    drawBarChart(ctx, width, height, labels, values, colors, yLabel, title) {
        const margin = 80;
        const plotWidth = width - 2 * margin;
        const plotHeight = height - 2 * margin;

        ctx.clearRect(0, 0, width, height);

        // Title
        ctx.fillStyle = '#333';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(title, width / 2, 30);

        // Axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.stroke();

        const maxValue = Math.max(...values);
        const barWidth = plotWidth / (labels.length * 2);

        // Draw bars
        labels.forEach((label, i) => {
            const barHeight = (values[i] / maxValue) * plotHeight;
            const x = margin + (i + 0.5) * (plotWidth / labels.length) - barWidth / 2;
            const y = height - margin - barHeight;

            ctx.fillStyle = colors[i];
            ctx.fillRect(x, y, barWidth, barHeight);

            // Value labels
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`${values[i].toFixed(0)} MB`, x + barWidth / 2, y - 5);

            // Bar labels
            ctx.fillText(label, x + barWidth / 2, height - margin + 20);
        });

        // Y-axis label
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();

        // Compression ratio annotation
        const ratio = values[0] / values[1];
        ctx.fillStyle = '#f39c12';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${ratio.toFixed(1)}× Compression`, width / 2, height - 20);
    }

    createPerformanceCharts() {
        this.createMemoryChart();
        this.createThroughputChart();
        this.createTrainingThroughputChart();
        this.createInferenceLatencyChart();
    }

    createMemoryChart() {
        const canvas = document.getElementById('memoryChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        const modelSizes = ['7B', '13B', '30B', '65B', '120B'];
        const bf16Memory = [14, 26, 60, 130, 240];
        const mxfp4Memory = [3.9, 7.2, 16.7, 36.1, 66.7];

        this.drawDoubleBarChart(ctx, canvas.displayWidth, canvas.displayHeight,
            modelSizes, bf16Memory, mxfp4Memory,
            ['BF16', 'MXFP4'], ['#e74c3c', '#2ecc71'],
            'Memory (GB)', 'Model Memory Usage');
    }

    createThroughputChart() {
        const canvas = document.getElementById('throughputChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        const batchSizes = [1, 2, 4, 8, 16];
        const bf16Throughput = [12, 22, 38, 65, 95];
        const mxfp4Throughput = [15, 28, 52, 89, 145];

        this.drawDoubleBarChart(ctx, canvas.displayWidth, canvas.displayHeight,
            batchSizes.map(x => x.toString()), bf16Throughput, mxfp4Throughput,
            ['BF16', 'MXFP4'], ['#e74c3c', '#2ecc71'],
            'Tokens/sec', 'Inference Throughput');
    }

    createTrainingThroughputChart() {
        const canvas = document.getElementById('trainingThroughputChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        const modelSizes = ['7B', '13B', '30B', '65B'];
        const bf16Throughput = [3200, 1800, 750, 320];
        const mxfp4Throughput = [4100, 2400, 1100, 480];

        this.drawDoubleBarChart(ctx, canvas.displayWidth, canvas.displayHeight,
            modelSizes, bf16Throughput, mxfp4Throughput,
            ['BF16', 'MXFP4'], ['#e74c3c', '#2ecc71'],
            'Tokens/sec', 'Training Throughput');
    }

    createInferenceLatencyChart() {
        const canvas = document.getElementById('inferenceLatencyChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        const sequenceLengths = [512, 1024, 2048, 4096];
        const bf16Latency = [45, 92, 185, 375];
        const mxfp4Latency = [32, 65, 128, 260];

        this.drawDoubleBarChart(ctx, canvas.displayWidth, canvas.displayHeight,
            sequenceLengths.map(x => x.toString()), bf16Latency, mxfp4Latency,
            ['BF16', 'MXFP4'], ['#e74c3c', '#2ecc71'],
            'Latency (ms)', 'Inference Latency');
    }

    drawDoubleBarChart(ctx, width, height, labels, values1, values2, seriesLabels, colors, yLabel, title) {
        const margin = 60;
        const plotWidth = width - 2 * margin;
        const plotHeight = height - 2 * margin;

        ctx.clearRect(0, 0, width, height);

        // Title
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(title, width / 2, 25);

        // Axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.stroke();

        const maxValue = Math.max(...values1, ...values2);
        const groupWidth = plotWidth / labels.length;
        const barWidth = groupWidth * 0.35;

        // Draw bars
        labels.forEach((label, i) => {
            const groupX = margin + i * groupWidth + groupWidth * 0.1;

            // First series
            const bar1Height = (values1[i] / maxValue) * plotHeight;
            const bar1Y = height - margin - bar1Height;
            ctx.fillStyle = colors[0];
            ctx.fillRect(groupX, bar1Y, barWidth, bar1Height);

            // Second series
            const bar2Height = (values2[i] / maxValue) * plotHeight;
            const bar2Y = height - margin - bar2Height;
            ctx.fillStyle = colors[1];
            ctx.fillRect(groupX + barWidth + 5, bar2Y, barWidth, bar2Height);

            // Labels
            ctx.fillStyle = '#333';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(label, groupX + barWidth + 2.5, height - margin + 15);
        });

        // Legend
        ctx.fillStyle = colors[0];
        ctx.fillRect(width - 120, 50, 15, 15);
        ctx.fillStyle = colors[1];
        ctx.fillRect(width - 120, 75, 15, 15);

        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(seriesLabels[0], width - 100, 62);
        ctx.fillText(seriesLabels[1], width - 100, 87);

        // Y-axis label
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
    }

    // Animation methods
    animateMoEFlow() {
        // Animate token flow through MoE
        this.animateTokenFlow = true;
        this.tokenAnimationStep = 0;
        this.animateToken();
    }

    animateToken() {
        if (!this.animateTokenFlow) return;

        this.tokenAnimationStep = (this.tokenAnimationStep + 1) % 100;

        // Redraw with animation
        this.drawMoEDiagram();

        if (this.tokenAnimationStep < 50) {
            // Token moving to router
            const progress = this.tokenAnimationStep / 50;
            this.drawAnimatedToken(progress * 0.5);
        } else {
            // Experts processing
            this.highlightActiveExperts();
        }

        if (this.tokenAnimationStep < 99) {
            requestAnimationFrame(() => this.animateToken());
        } else {
            this.animateTokenFlow = false;
        }
    }

    drawAnimatedToken(progress) {
        const canvas = document.getElementById('moeDiagramCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const centerX = canvas.displayWidth / 2;
        const centerY = canvas.displayHeight / 2;
        const startY = centerY - 250;
        const endY = centerY - 180;
        const currentY = startY + (endY - startY) * progress;

        // Draw animated token
        ctx.fillStyle = '#f39c12';
        ctx.beginPath();
        ctx.arc(centerX, currentY, 25, 0, 2 * Math.PI);
        ctx.fill();
    }

    highlightActiveExperts() {
        const canvas = document.getElementById('moeDiagramCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const centerX = canvas.displayWidth / 2;
        const centerY = canvas.displayHeight / 2;

        // Add glow effect to active experts
        for (let i = 0; i < 4; i++) {
            const angle = (i * 2 * Math.PI) / 8;
            const expertX = centerX + Math.cos(angle) * 120;
            const expertY = centerY + Math.sin(angle) * 120;

            ctx.shadowColor = '#2ecc71';
            ctx.shadowBlur = 20;
            ctx.fillStyle = '#2ecc71';
            ctx.beginPath();
            ctx.arc(expertX, expertY, 45, 0, 2 * Math.PI);
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    showRouting() {
        // Highlight routing connections with probabilities
        const canvas = document.getElementById('moeDiagramCanvas');
        if (!canvas) return;

        this.drawMoEDiagram();
        this.drawRoutingProbabilities();
    }

    drawRoutingProbabilities() {
        const canvas = document.getElementById('moeDiagramCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const centerX = canvas.displayWidth / 2;
        const centerY = canvas.displayHeight / 2;
        const routingProbs = [0.35, 0.28, 0.22, 0.15, 0.0, 0.0, 0.0, 0.0];

        // Draw probability labels on connections
        for (let i = 0; i < 8; i++) {
            const angle = (i * 2 * Math.PI) / 8;
            const midX = centerX + Math.cos(angle) * 60;
            const midY = centerY + Math.sin(angle) * 60;

            if (routingProbs[i] > 0) {
                ctx.fillStyle = '#f39c12';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(`${(routingProbs[i] * 100).toFixed(0)}%`, midX, midY);
            }
        }
    }

    highlightQuantization() {
        // Show quantization details in experts
        this.drawMoEDiagram();
        this.showQuantizationDetails();
    }

    showQuantizationDetails() {
        const canvas = document.getElementById('moeDiagramCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        // Add quantization info boxes
        const infoBoxes = [
            { x: 50, y: 50, title: "MXFP4 Benefits", items: ["4× memory reduction", "Native HW support", "Stochastic rounding"] },
            { x: 350, y: 50, title: "Block Structure", items: ["32-element blocks", "Shared scale factor", "E2M1 format"] }
        ];

        infoBoxes.forEach(box => {
            this.drawInfoBox(ctx, box.x, box.y, box.title, box.items);
        });
    }

    drawInfoBox(ctx, x, y, title, items) {
        const boxWidth = 140;
        const boxHeight = 80;

        // Box background
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fillRect(x, y, boxWidth, boxHeight);
        ctx.strokeStyle = '#333';
        ctx.strokeRect(x, y, boxWidth, boxHeight);

        // Title
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(title, x + 5, y + 15);

        // Items
        ctx.font = '10px Arial';
        items.forEach((item, i) => {
            ctx.fillText(`• ${item}`, x + 5, y + 35 + i * 15);
        });
    }
}

// Global instance to avoid multiple initializations
let mxfp4Visualizer = null;
window.mxfp4Visualizer = null;

// Initialize visualizations when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {

    function initializeVisualizations() {
        try {
            // Only initialize once
            if (!mxfp4Visualizer) {
                mxfp4Visualizer = new MXFP4Visualizer();
                window.mxfp4Visualizer = mxfp4Visualizer;
            }
        } catch (error) {
            console.error('Error initializing MXFP4 visualizations:', error);
        }
    }

    // Check if all required elements exist before initializing
    const requiredElements = [
        'valueSpaceCanvas',
        'matmulCanvas',
        'moeDiagramCanvas',
        'moeMemoryCanvas'
    ];

    const checkElements = () => {
        const elementStatus = requiredElements.map(id => ({
            id,
            exists: !!document.getElementById(id)
        }));

        const allExist = elementStatus.every(el => el.exists);
        if (allExist) {
            // Wait a bit for layout to stabilize
            setTimeout(initializeVisualizations, 150);
        } else {
            // Retry after a short delay
            setTimeout(checkElements, 100);
        }
    };

    checkElements();
});

// Handle window resize
window.addEventListener('resize', function () {
    if (mxfp4Visualizer) {
        setTimeout(() => {
            try {
                mxfp4Visualizer.setupCanvases();
                mxfp4Visualizer.initializeCharts();
            } catch (error) {
                console.error('Error reinitializing after resize:', error);
            }
        }, 200);
    }
});

// Manual button setup for navigation - moved from inline scripts
document.addEventListener('DOMContentLoaded', function () {
    // Wait a bit for everything to load
    setTimeout(() => {
        // Manual button click handlers
        const btnValueSpace = document.getElementById('btn-value-space');
        const btnMapping = document.getElementById('btn-mapping');
        const btnDistribution = document.getElementById('btn-distribution');
        const btnTable = document.getElementById('btn-table');

        if (btnValueSpace) {
            btnValueSpace.onclick = function (e) {
                e.preventDefault();
                showView('view-value-space');
                setActiveButton(this);
            };
        }

        if (btnMapping) {
            btnMapping.onclick = function (e) {
                e.preventDefault();
                showView('view-mapping');
                setActiveButton(this);
            };
        }

        if (btnDistribution) {
            btnDistribution.onclick = function (e) {
                e.preventDefault();
                showView('view-distribution');
                setActiveButton(this);
            };
        }

        if (btnTable) {
            btnTable.onclick = function (e) {
                e.preventDefault();
                showView('view-table');
                setActiveButton(this);
            };
        }

        function showView(viewId) {
            // Hide all views
            ['view-value-space', 'view-mapping', 'view-distribution', 'view-table'].forEach(id => {
                const view = document.getElementById(id);
                if (view) {
                    view.style.display = 'none';
                }
            });

            // Show selected view
            const targetView = document.getElementById(viewId);
            if (targetView) {
                targetView.style.display = 'block';

                // Create visualizations based on view
                setTimeout(() => {
                    if (window.mxfp4Visualizer) {
                        switch (viewId) {
                            case 'view-value-space':
                                window.mxfp4Visualizer.createValueSpaceChart();
                                break;
                            case 'view-mapping':
                                window.mxfp4Visualizer.createMappingChart();
                                window.mxfp4Visualizer.createErrorChart();
                                break;
                            case 'view-distribution':
                                window.mxfp4Visualizer.createDistributionChart();
                                break;
                            case 'view-table':
                                window.mxfp4Visualizer.createValueTable();
                                break;
                        }
                    }
                }, 100);
            }
        }

        function setActiveButton(activeBtn) {
            // Get all navigation buttons
            const allButtons = document.querySelectorAll('#mxfp4-format-viz .btn-group .btn');

            // Remove active class from ALL buttons
            allButtons.forEach(btn => {
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-outline-primary');
            });

            // Add active class to clicked button
            if (activeBtn) {
                activeBtn.classList.remove('btn-outline-primary');
                activeBtn.classList.add('btn-primary');
            }
        }

        // Set initial active button (Value Space)
        if (btnValueSpace) {
            setActiveButton(btnValueSpace);
        }
    }, 300);
});