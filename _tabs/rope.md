---
layout: wide-page
title: RoPE Visualizer
icon: fas fa-compass
order: 4
---

<style>
#rope-visualizer-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
}

#rope-visualizer-container * {
    box-sizing: border-box;
}

#rope-visualizer-container .container {
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
}

@media (max-width: 768px) {
    #rope-visualizer-container .container {
        padding: 20px;
        border-radius: 15px;
        margin: 0 5px;
    }
}

@media (max-width: 500px) {
    #rope-visualizer-container .container {
        padding: 15px;
        border-radius: 10px;
    }
}

#rope-visualizer-container .header {
    text-align: center;
    margin-bottom: 30px;
}

#rope-visualizer-container .header h1 {
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 10px;
    color: #2c3e50;
}

#rope-visualizer-container .header p {
    font-size: 1.2em;
    color: #6c757d;
    margin-bottom: 5px;
}

#rope-visualizer-container .subtitle {
    font-size: 1em;
    color: #adb5bd;
    font-style: italic;
}

#rope-visualizer-container .controls {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 25px;
    border: 2px solid #e9ecef;
}

#rope-visualizer-container .controls h3 {
    margin-bottom: 20px;
    color: #2c3e50;
    font-weight: 600;
}

#rope-visualizer-container .control-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    align-items: end;
}

#rope-visualizer-container .control-group {
    display: flex;
    flex-direction: column;
}

#rope-visualizer-container .control-group label {
    font-weight: 500;
    margin-bottom: 8px;
    color: #34495e;
}

#rope-visualizer-container .control-group input,
#rope-visualizer-container .control-group select {
    padding: 10px;
    border: 2px solid #e0e6ed;
    border-radius: 8px;
    font-size: 14px;
    transition: border-color 0.3s ease;
}

#rope-visualizer-container .control-group input:focus,
#rope-visualizer-container .control-group select:focus {
    outline: none;
    border-color: #667eea;
}

#rope-visualizer-container .visualization-grid {
    display: flex;
    flex-direction: column;
    gap: 25px;
    margin-bottom: 25px;
}

#rope-visualizer-container .viz-panel {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    border: 2px solid #e9ecef;
}

#rope-visualizer-container .viz-panel h4 {
    margin-bottom: 15px;
    color: #2c3e50;
    font-weight: 600;
    text-align: center;
}

#rope-visualizer-container .heatmap-container {
    grid-column: 1 / -1;
}

#rope-visualizer-container .metrics-display {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 25px;
    border: 2px solid #e9ecef;
}

#rope-visualizer-container .metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
}

#rope-visualizer-container .metric-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

#rope-visualizer-container .metric-value {
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 5px;
}

#rope-visualizer-container .metric-label {
    font-size: 0.9em;
    opacity: 0.9;
}

#rope-visualizer-container .info-panel {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    border: 2px solid #e9ecef;
}

#rope-visualizer-container .info-panel h4 {
    color: #2c3e50;
    margin-bottom: 15px;
    font-weight: 600;
}

#rope-visualizer-container .info-panel p {
    line-height: 1.6;
    color: #5a6c7d;
    margin-bottom: 10px;
}


@media (max-width: 768px) {
    #rope-visualizer-container .control-grid {
        grid-template-columns: 1fr;
    }

    #rope-visualizer-container .header h1 {
        font-size: 2em;
    }

    #rope-visualizer-container .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 480px) {
    #rope-visualizer-container .metrics-grid {
        grid-template-columns: 1fr;
    }
}
</style>

<div id="rope-visualizer-container">
    <div class="container">
        <div class="header">
            <h1>🎯 RoPE Embeddings with YaRN</h1>
            <p>Interactive Visualization of Rotary Position Embeddings</p>
            <div class="subtitle">Yet another RoPE extensioN (YaRN) Scaling Analysis</div>
        </div>

        <div class="controls">
            <h3>🎛️ Configuration Parameters</h3>
            <div class="control-grid">
                <div class="control-group">
                    <label for="scalingFactor">YaRN Scaling Factor:</label>
                    <input type="range" id="scalingFactor" min="1" max="8" step="0.5" value="4">
                    <span id="scalingFactorValue">4.0×</span>
                </div>

                <div class="control-group">
                    <label for="headDim">Head Dimension:</label>
                    <select id="headDim">
                        <option value="32">32</option>
                        <option value="64" selected>64</option>
                        <option value="128">128</option>
                    </select>
                </div>

                <div class="control-group">
                    <label for="seqLength">Sequence Length:</label>
                    <input type="range" id="seqLength" min="256" max="8192" step="256" value="2048">
                    <span id="seqLengthValue">2048</span>
                </div>

                <div class="control-group">
                    <label for="baseFreq">Base Frequency:</label>
                    <input type="number" id="baseFreq" value="10000" step="1000">
                </div>

                <div class="control-group">
                    <label for="initialContext">Initial Context Length:</label>
                    <input type="number" id="initialContext" value="2048" step="256">
                </div>
            </div>
        </div>

        <div class="metrics-display">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="effectiveContext">8192</div>
                    <div class="metric-label">Effective Context</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="freqPairs">32</div>
                    <div class="metric-label">Frequency Pairs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="maxWavelength">65536</div>
                    <div class="metric-label">Max Wavelength</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="minWavelength">2</div>
                    <div class="metric-label">Min Wavelength</div>
                </div>
            </div>
        </div>

        <div class="visualization-grid">
            <div class="viz-panel">
                <h4>📊 Frequency Analysis</h4>
                <div id="frequencyPlot"></div>
            </div>

            <div class="viz-panel">
                <h4>📈 Wavelength Comparison</h4>
                <div id="wavelengthPlot"></div>
            </div>

            <div class="viz-panel">
                <h4>📉 Frequency Scaling Ratio</h4>
                <div id="ratioPlot"></div>
            </div>

            <div class="viz-panel">
                <h4>🎯 YaRN Ramp Function Effect</h4>
                <div id="rampPlot"></div>
            </div>
        </div>

        <div class="viz-panel heatmap-container">
            <h4>🔥 RoPE Embeddings Heatmap</h4>
            <div id="heatmapPlot"></div>
        </div>

        <div class="info-panel">
            <h4>ℹ️ About YaRN and RoPE</h4>
            <p><strong>Rotary Position Embedding (RoPE)</strong> encodes positional information by rotating query and key vectors using rotation matrices based on position and dimension.</p>
            <p><strong>YaRN (Yet another RoPE extensioN)</strong> extends the effective context length by intelligently scaling frequency components, allowing models to handle longer sequences than their training context.</p>
            <p><strong>Key Benefits:</strong> YaRN maintains performance on shorter sequences while enabling processing of much longer sequences, making it ideal for long-form text generation and analysis.</p>
        </div>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<script src="{{ '/assets/js/rope-visualizer.js' | relative_url }}"></script>