// RoPE Implementation
class RotaryEmbedding {
    constructor(headDim, base = 10000, scalingFactor = 1.0, initialContextLength = 2048) {
        this.headDim = headDim;
        this.base = base;
        this.scalingFactor = scalingFactor;
        this.initialContextLength = initialContextLength;
        this.freqPairs = Math.floor(headDim / 2);

        this.computeFrequencies();
    }

    computeFrequencies() {
        // Compute base inverse frequencies
        const invFreq = [];
        for (let i = 0; i < this.freqPairs; i++) {
            invFreq.push(1.0 / Math.pow(this.base, (2 * i) / this.headDim));
        }

        if (this.scalingFactor === 1.0) {
            this.invFreq = invFreq;
        } else {
            // Apply YaRN scaling
            this.invFreq = this.applyYarnScaling(invFreq);
        }
    }

    applyYarnScaling(baseInvFreq) {
        const scaledInvFreq = [];
        const alpha = this.scalingFactor;

        // Add validation for scaling factor
        if (alpha <= 0 || !isFinite(alpha)) {
            console.warn('Invalid scaling factor, using 1.0');
            return baseInvFreq;
        }

        for (let i = 0; i < baseInvFreq.length; i++) {
            const freq = baseInvFreq[i];

            // Validate base frequency
            if (freq <= 0 || !isFinite(freq)) {
                scaledInvFreq.push(baseInvFreq[i]);
                continue;
            }

            const wavelength = 2 * Math.PI / freq;

            // YaRN ramp function with bounds checking
            let rampFactor;
            if (wavelength < this.initialContextLength) {
                rampFactor = 1.0; // No scaling for high frequencies
            } else if (wavelength > this.initialContextLength * alpha) {
                rampFactor = alpha; // Full scaling for low frequencies
            } else {
                // Smooth interpolation in between
                const denominator = this.initialContextLength * (alpha - 1);
                if (denominator > 0) {
                    const t = (wavelength - this.initialContextLength) / denominator;
                    rampFactor = 1.0 + (alpha - 1.0) * Math.max(0, Math.min(1, t));
                } else {
                    rampFactor = 1.0;
                }
            }

            // Ensure ramp factor is valid
            if (rampFactor <= 0 || !isFinite(rampFactor)) {
                rampFactor = 1.0;
            }

            const scaledFreq = freq / rampFactor;

            // Validate scaled frequency
            if (scaledFreq > 0 && isFinite(scaledFreq)) {
                scaledInvFreq.push(scaledFreq);
            } else {
                scaledInvFreq.push(freq); // Fallback to original
            }
        }

        return scaledInvFreq;
    }

    computeCosSinCache(seqLen) {
        const cosCache = [];
        const sinCache = [];

        for (let pos = 0; pos < seqLen; pos++) {
            const cosRow = [];
            const sinRow = [];

            for (let i = 0; i < this.freqPairs; i++) {
                const angle = pos * this.invFreq[i];
                cosRow.push(Math.cos(angle));
                sinRow.push(Math.sin(angle));
            }

            cosCache.push(cosRow);
            sinCache.push(sinRow);
        }

        return { cosCache, sinCache };
    }

    getWavelengths() {
        return this.invFreq.map(freq => 2 * Math.PI / freq);
    }
}

// Global variables
let currentConfig = {
    scalingFactor: 4.0,
    headDim: 64,
    seqLength: 2048,
    baseFreq: 10000,
    initialContext: 2048
};

// Debounce helper
function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

function updateVisualization() {
    // Update metrics
    updateMetrics();

    // Update all plots without visual feedback to prevent flickering
    Promise.all([
        plotFrequencyAnalysis(),
        plotWavelengthComparison(),
        plotFrequencyScalingRatio(),
        plotYarnRampEffect(),
        plotEmbeddingsHeatmap()
    ]);
}

// Create debounced version
const debouncedUpdate = debounce(updateVisualization, 300);

function updateMetrics() {
    const effectiveContext = Math.floor(currentConfig.initialContext * currentConfig.scalingFactor);
    const freqPairs = Math.floor(currentConfig.headDim / 2);

    // Create standard and YaRN RoPE instances
    const ropeStandard = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, 1.0, currentConfig.initialContext);
    const ropeYarn = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, currentConfig.scalingFactor, currentConfig.initialContext);

    const wavelengthsStandard = ropeStandard.getWavelengths();
    const wavelengthsYarn = ropeYarn.getWavelengths();

    const maxWavelength = Math.floor(Math.max(...wavelengthsYarn));
    const minWavelength = Math.floor(Math.min(...wavelengthsStandard));

    document.getElementById('effectiveContext').textContent = effectiveContext;
    document.getElementById('freqPairs').textContent = freqPairs;
    document.getElementById('maxWavelength').textContent = maxWavelength;
    document.getElementById('minWavelength').textContent = minWavelength;
}

function plotFrequencyAnalysis() {
    return new Promise((resolve) => {
        const ropeStandard = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, 1.0, currentConfig.initialContext);
        const ropeYarn = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, currentConfig.scalingFactor, currentConfig.initialContext);

        const dimIndices = Array.from({length: ropeStandard.freqPairs}, (_, i) => i);

        const trace1 = {
            x: dimIndices,
            y: ropeStandard.invFreq,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Standard RoPE',
            line: { color: '#2E86AB', width: 3 },
            marker: { size: 8, color: 'white', line: { color: '#2E86AB', width: 2 } }
        };

        const trace2 = {
            x: dimIndices,
            y: ropeYarn.invFreq,
            type: 'scatter',
            mode: 'lines+markers',
            name: `YaRN (${currentConfig.scalingFactor}×)`,
            line: { color: '#A23B72', width: 3 },
            marker: { size: 8, color: 'white', line: { color: '#A23B72', width: 2 } }
        };

        const layout = {
            title: {
                text: 'Inverse Frequencies',
                font: { size: 16, family: 'Segoe UI, sans-serif' }
            },
            xaxis: {
                title: 'Dimension Pair Index',
                gridcolor: '#e9ecef',
                gridwidth: 1
            },
            yaxis: {
                title: 'Inverse Frequency',
                type: 'log',
                gridcolor: '#e9ecef',
                gridwidth: 1
            },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: { family: 'Segoe UI, sans-serif' },
            legend: {
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center'
            },
            autosize: true,
            height: 400,
            margin: { l: 60, r: 30, t: 60, b: 80 }
        };

        Plotly.react('frequencyPlot', [trace1, trace2], layout, {responsive: true}).then(() => {
            resolve();
        });
    });
}

function plotWavelengthComparison() {
    return new Promise((resolve) => {
        const ropeStandard = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, 1.0, currentConfig.initialContext);
        const ropeYarn = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, currentConfig.scalingFactor, currentConfig.initialContext);

        const wavelengthsStd = ropeStandard.getWavelengths();
        const wavelengthsYarn = ropeYarn.getWavelengths();
        const dimIndices = Array.from({length: ropeStandard.freqPairs}, (_, i) => i);

        const trace1 = {
            x: dimIndices,
            y: wavelengthsStd,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Standard RoPE',
            line: { color: '#2E86AB', width: 3 },
            marker: { size: 8, color: 'white', line: { color: '#2E86AB', width: 2 } }
        };

        const trace2 = {
            x: dimIndices,
            y: wavelengthsYarn,
            type: 'scatter',
            mode: 'lines+markers',
            name: `YaRN (${currentConfig.scalingFactor}×)`,
            line: { color: '#A23B72', width: 3 },
            marker: { size: 8, color: 'white', line: { color: '#A23B72', width: 2 } }
        };

        const layout = {
            title: {
                text: 'Positional Wavelengths',
                font: { size: 16, family: 'Segoe UI, sans-serif' }
            },
            xaxis: {
                title: 'Dimension Pair Index',
                gridcolor: '#e9ecef',
                gridwidth: 1
            },
            yaxis: {
                title: 'Wavelength (tokens)',
                type: 'log',
                gridcolor: '#e9ecef',
                gridwidth: 1
            },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: { family: 'Segoe UI, sans-serif' },
            legend: {
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center'
            },
            autosize: true,
            height: 400,
            margin: { l: 60, r: 30, t: 60, b: 80 }
        };

        Plotly.react('wavelengthPlot', [trace1, trace2], layout, {responsive: true}).then(() => {
            resolve();
        });
    });
}

function plotFrequencyScalingRatio() {
    return new Promise((resolve, reject) => {
        try {
            const ropeStandard = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, 1.0, currentConfig.initialContext);
            const ropeYarn = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, currentConfig.scalingFactor, currentConfig.initialContext);

            const dimIndices = Array.from({length: ropeStandard.freqPairs}, (_, i) => i);
            const ratio = ropeYarn.invFreq.map((freq, i) => freq / ropeStandard.invFreq[i]);
            const expectedRatio = 1.0 / currentConfig.scalingFactor;

            const data = [
                {
                    x: dimIndices,
                    y: ratio,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Actual Ratio',
                    line: { color: '#F18F01', width: 3 },
                    marker: { size: 6, color: 'white', line: { color: '#F18F01', width: 2 } }
                },
                {
                    x: dimIndices,
                    y: Array(dimIndices.length).fill(expectedRatio),
                    type: 'scatter',
                    mode: 'lines',
                    name: `Expected (1/${currentConfig.scalingFactor.toFixed(1)})`,
                    line: { color: '#6c757d', width: 2, dash: 'dash' }
                }
            ];

            const layout = {
                title: {
                    text: 'YaRN/Standard Frequency Ratio',
                    font: { size: 16, family: 'Segoe UI, sans-serif' }
                },
                xaxis: {
                    title: 'Dimension Pair Index',
                    gridcolor: '#e9ecef',
                    gridwidth: 1
                },
                yaxis: {
                    title: 'Frequency Ratio',
                    gridcolor: '#e9ecef',
                    gridwidth: 1
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: { family: 'Segoe UI, sans-serif' },
                legend: {
                    orientation: 'h',
                    y: -0.2,
                    x: 0.5,
                    xanchor: 'center'
                },
                autosize: true,
                height: 400,
                margin: { l: 60, r: 30, t: 60, b: 80 }
            };

            Plotly.react('ratioPlot', data, layout, {responsive: true}).then(() => {
                resolve();
            });
        } catch (error) {
            console.error('Error in plotFrequencyScalingRatio:', error);
            reject(error);
        }
    });
}

function plotYarnRampEffect() {
    return new Promise((resolve, reject) => {
        try {
            const ropeStandard = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, 1.0, currentConfig.initialContext);
            const ropeYarn = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, currentConfig.scalingFactor, currentConfig.initialContext);

            const wavelengthsStd = ropeStandard.getWavelengths();
            const wavelengthsYarn = ropeYarn.getWavelengths();
            const scalingEffect = wavelengthsYarn.map((w, i) => w / wavelengthsStd[i]);
            const dimIndices = Array.from({length: ropeStandard.freqPairs}, (_, i) => i);

            const data = [
                {
                    x: dimIndices,
                    y: scalingEffect,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Wavelength Scaling',
                    line: { color: '#9C27B0', width: 4 },
                    marker: {
                        size: 8,
                        color: '#9C27B0',
                        line: { color: 'white', width: 2 }
                    }
                },
                {
                    x: dimIndices,
                    y: Array(dimIndices.length).fill(currentConfig.scalingFactor),
                    type: 'scatter',
                    mode: 'lines',
                    name: `Expected (${currentConfig.scalingFactor.toFixed(1)}×)`,
                    line: { color: '#6c757d', width: 2, dash: 'dash' }
                },
                {
                    x: dimIndices,
                    y: Array(dimIndices.length).fill(1.0),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'No Scaling (1×)',
                    line: { color: '#6c757d', width: 2, dash: 'dot' }
                }
            ];

            const layout = {
                title: {
                    text: 'YaRN Ramp Function Effect',
                    font: { size: 16, family: 'Segoe UI, sans-serif' }
                },
                xaxis: {
                    title: 'Dimension Pair Index',
                    gridcolor: '#e9ecef',
                    gridwidth: 1
                },
                yaxis: {
                    title: 'Wavelength Scaling Factor',
                    gridcolor: '#e9ecef',
                    gridwidth: 1
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: { family: 'Segoe UI, sans-serif' },
                legend: {
                    orientation: 'h',
                    y: -0.2,
                    x: 0.5,
                    xanchor: 'center'
                },
                autosize: true,
                height: 400,
                margin: { l: 60, r: 30, t: 60, b: 80 }
            };

            Plotly.react('rampPlot', data, layout, {responsive: true}).then(() => {
                resolve();
            });
        } catch (error) {
            console.error('Error in plotYarnRampEffect:', error);
            reject(error);
        }
    });
}

function plotEmbeddingsHeatmap() {
    return new Promise((resolve, reject) => {
        try {
            const rope = new RotaryEmbedding(currentConfig.headDim, currentConfig.baseFreq, currentConfig.scalingFactor, currentConfig.initialContext);
            const displayLen = Math.min(currentConfig.seqLength, 128); // Reduced for stability
            const { cosCache } = rope.computeCosSinCache(displayLen);

            // Transpose for proper heatmap display
            const heatmapData = [];
            for (let dim = 0; dim < rope.freqPairs; dim++) {
                const row = [];
                for (let pos = 0; pos < displayLen; pos++) {
                    row.push(cosCache[pos][dim]);
                }
                heatmapData.push(row);
            }

            const data = [{
                z: heatmapData,
                type: 'heatmap',
                colorscale: 'RdYlBu',
                reversescale: true,
                showscale: true,
                colorbar: {
                    title: 'Cosine Value',
                    titleside: 'right'
                }
            }];

            const layout = {
                title: {
                    text: `RoPE Cosine Embeddings (${currentConfig.scalingFactor.toFixed(1)}× YaRN)`,
                    font: { size: 16, family: 'Segoe UI, sans-serif' }
                },
                xaxis: {
                    title: 'Position',
                    gridcolor: '#e9ecef',
                    gridwidth: 1
                },
                yaxis: {
                    title: 'Dimension Pair',
                    gridcolor: '#e9ecef',
                    gridwidth: 1
                },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: { family: 'Segoe UI, sans-serif' },
                autosize: true,
                height: 500,
                margin: { l: 60, r: 30, t: 60, b: 50 }
            };

            Plotly.react('heatmapPlot', data, layout, {responsive: true}).then(() => {
                resolve();
            });
        } catch (error) {
            console.error('Error in plotEmbeddingsHeatmap:', error);
            reject(error);
        }
    });
}

// Initialize visualization on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Setting up event listeners...');

    // Scaling factor slider
    const scalingSlider = document.getElementById('scalingFactor');
    if (scalingSlider) {
        scalingSlider.oninput = function(e) {
            console.log('Scaling factor changed to:', e.target.value);
            const value = parseFloat(e.target.value);
            const valueSpan = document.getElementById('scalingFactorValue');
            if (valueSpan) {
                valueSpan.textContent = value.toFixed(1) + '×';
            }
            currentConfig.scalingFactor = value;
            debouncedUpdate();
        };
    }

    // Sequence length slider
    const seqSlider = document.getElementById('seqLength');
    if (seqSlider) {
        seqSlider.oninput = function(e) {
            console.log('Sequence length changed to:', e.target.value);
            const value = parseInt(e.target.value);
            const valueSpan = document.getElementById('seqLengthValue');
            if (valueSpan) {
                valueSpan.textContent = value;
            }
            currentConfig.seqLength = value;
            debouncedUpdate();
        };
    }

    // Head dimension dropdown
    const headDimSelect = document.getElementById('headDim');
    if (headDimSelect) {
        headDimSelect.onchange = function(e) {
            console.log('Head dimension changed to:', e.target.value);
            currentConfig.headDim = parseInt(e.target.value);
            debouncedUpdate();
        };
    }

    // Base frequency input
    const baseFreqInput = document.getElementById('baseFreq');
    if (baseFreqInput) {
        baseFreqInput.oninput = function(e) {
            const value = parseInt(e.target.value);
            if (value > 0) {
                console.log('Base frequency changed to:', value);
                currentConfig.baseFreq = value;
                debouncedUpdate();
            }
        };
    }

    // Initial context input
    const initialContextInput = document.getElementById('initialContext');
    if (initialContextInput) {
        initialContextInput.oninput = function(e) {
            const value = parseInt(e.target.value);
            if (value > 0) {
                console.log('Initial context changed to:', value);
                currentConfig.initialContext = value;
                debouncedUpdate();
            }
        };
    }

    console.log('Event listeners set up, loading initial visualization...');
    // Initial load
    setTimeout(() => {
        updateVisualization();
    }, 200);
});