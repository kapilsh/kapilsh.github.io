---
layout: wide-page
title: MXFP4 Visualizer
icon: fas fa-calculator
order: 3
---

<style>
#mxfp4-visualizer-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
}

#mxfp4-visualizer-container * {
    box-sizing: border-box;
}

#mxfp4-visualizer-container .container {
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
}

@media (max-width: 768px) {
    #mxfp4-visualizer-container .container {
        padding: 20px;
        border-radius: 15px;
        margin: 0 5px;
    }
}

@media (max-width: 500px) {
    #mxfp4-visualizer-container .container {
        padding: 15px;
        border-radius: 10px;
    }
}

#mxfp4-visualizer-container .title {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 30px;
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#mxfp4-visualizer-container .controls {
    display: flex;
    flex-direction: column;
    gap: 20px;
    margin-bottom: 30px;
}

#mxfp4-visualizer-container .input-section {
    background: linear-gradient(135deg, #4CAF50, #45a049);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

#mxfp4-visualizer-container .output-section {
    background: linear-gradient(135deg, #2196F3, #1976D2);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

#mxfp4-visualizer-container .section-title {
    font-size: 1.3rem;
    margin-bottom: 15px;
    text-align: center;
}

#mxfp4-visualizer-container .input-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 15px;
    width: 100%;
    max-width: 100%;
}

@media (max-width: 768px) {
    #mxfp4-visualizer-container .input-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 480px) {
    #mxfp4-visualizer-container .input-grid {
        grid-template-columns: 1fr;
    }
}

#mxfp4-visualizer-container .input-field {
    padding: 8px;
    border: none;
    border-radius: 8px;
    text-align: center;
    font-size: 0.9rem;
    background: rgba(255,255,255,0.9);
    color: #333;
    width: 100%;
    max-width: 100%;
    min-width: 0;
}

#mxfp4-visualizer-container .buttons {
    display: flex;
    gap: 10px;
    justify-content: center;
}

#mxfp4-visualizer-container .btn {
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s;
}

#mxfp4-visualizer-container .btn:hover {
    transform: translateY(-2px);
}

#mxfp4-visualizer-container .btn-primary {
    background: #FF5722;
    color: white;
}

#mxfp4-visualizer-container .btn-secondary {
    background: rgba(255,255,255,0.8);
    color: #333;
}

#mxfp4-visualizer-container .stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 15px;
    font-size: 0.9rem;
}

#mxfp4-visualizer-container .visualizations {
    display: flex;
    flex-direction: column;
    gap: 30px;
    margin-bottom: 30px;
}

#mxfp4-visualizer-container .viz-panel {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    border: 2px solid #e9ecef;
}

#mxfp4-visualizer-container .viz-title {
    text-align: center;
    font-size: 1.2rem;
    margin-bottom: 20px;
    color: #495057;
    font-weight: 600;
}

#mxfp4-visualizer-container .lut-grid {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 8px;
    margin-bottom: 20px;
}

#mxfp4-visualizer-container .lut-cell {
    background: linear-gradient(135deg, #9C27B0, #7B1FA2);
    color: white;
    padding: 12px 8px;
    border-radius: 8px;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    border: 2px solid transparent;
}

#mxfp4-visualizer-container .lut-cell:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 15px rgba(156, 39, 176, 0.4);
}

#mxfp4-visualizer-container .lut-cell.selected {
    border-color: #FFD700;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.6);
}

#mxfp4-visualizer-container .quantization-flow {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}

@media (max-width: 768px) {
    #mxfp4-visualizer-container .quantization-flow {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 480px) {
    #mxfp4-visualizer-container .quantization-flow {
        grid-template-columns: 1fr;
    }
}

#mxfp4-visualizer-container .flow-item {
    text-align: center;
    padding: 15px;
    border-radius: 12px;
    background: white;
    border: 2px solid #dee2e6;
    transition: all 0.3s;
}

#mxfp4-visualizer-container .flow-item:hover {
    border-color: #007bff;
    box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
}

#mxfp4-visualizer-container .flow-value {
    font-size: 1.1rem;
    font-weight: bold;
    color: #007bff;
}

#mxfp4-visualizer-container .flow-arrow {
    font-size: 1.5rem;
    margin: 10px 0;
    color: #28a745;
}

#mxfp4-visualizer-container .flow-result {
    font-size: 0.95rem;
    color: #6c757d;
    margin-top: 5px;
}

#mxfp4-visualizer-container .scale-info {
    background: linear-gradient(135deg, #FF9800, #F57C00);
    color: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
    font-weight: 600;
}

#mxfp4-visualizer-container .error-analysis {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
}

#mxfp4-visualizer-container .error-title {
    font-weight: bold;
    color: #856404;
    margin-bottom: 10px;
}

#mxfp4-visualizer-container .error-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

@media (max-width: 768px) {
    #mxfp4-visualizer-container .error-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 480px) {
    #mxfp4-visualizer-container .error-grid {
        grid-template-columns: 1fr;
    }
}

#mxfp4-visualizer-container .error-item {
    text-align: center;
    padding: 8px;
    background: white;
    border-radius: 6px;
    border: 1px solid #ffeaa7;
}

#mxfp4-visualizer-container .compression-stats {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-top: 30px;
}

#mxfp4-visualizer-container .chart-container {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}
</style>

<div id="mxfp4-visualizer-container">
    <div class="container">
        <h1 class="title">🔬 MXFP4 Quantization Visualizer</h1>

        <div class="controls">
            <div class="input-section">
                <h3 class="section-title">📊 Input BF16 Values</h3>
                <div class="input-grid">
                    <input type="number" class="input-field" id="val0" step="0.1" value="250.5">
                    <input type="number" class="input-field" id="val1" step="0.1" value="-125.25">
                    <input type="number" class="input-field" id="val2" step="0.1" value="75.75">
                    <input type="number" class="input-field" id="val3" step="0.1" value="400.0">
                    <input type="number" class="input-field" id="val4" step="0.1" value="120.2">
                    <input type="number" class="input-field" id="val5" step="0.1" value="-80.8">
                    <input type="number" class="input-field" id="val6" step="0.1" value="310.1">
                    <input type="number" class="input-field" id="val7" step="0.1" value="-230.3">
                    <input type="number" class="input-field" id="val8" step="0.1" value="90.9">
                    <input type="number" class="input-field" id="val9" step="0.1" value="520.2">
                    <input type="number" class="input-field" id="val10" step="0.1" value="-170.7">
                    <input type="number" class="input-field" id="val11" step="0.1" value="280.8">
                    <input type="number" class="input-field" id="val12" step="0.1" value="-40.4">
                    <input type="number" class="input-field" id="val13" step="0.1" value="160.6">
                    <input type="number" class="input-field" id="val14" step="0.1" value="-350.5">
                    <input type="number" class="input-field" id="val15" step="0.1" value="470.7">
                </div>
                <div class="buttons">
                    <button class="btn btn-primary" id="quantizeBtn">🚀 Quantize</button>
                    <button class="btn btn-secondary" id="randomBtn">🎲 Random</button>
                    <button class="btn btn-secondary" id="presetBtn">📋 Preset</button>
                </div>
                <div class="stats">
                    <div>📏 Vec Size: <span id="vecSize">16</span></div>
                </div>
            </div>

            <div class="output-section">
                <h3 class="section-title">🎯 Output Comparison</h3>
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 1.1rem; margin-bottom: 10px; text-align: center;">BF16 Input (16-bit each)</div>
                    <div class="input-grid">
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_0">0100000010100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_1">1011111110100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_2">0011111101100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_3">0100000010000000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_4">0100000010100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_5">1011111110100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_6">0011111101100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_7">0100000010000000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_8">0100000010100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_9">1011111110100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_10">0011111101100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_11">0100000010000000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_12">0100000010100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_13">1011111110100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_14">0011111101100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 6px; border-radius: 6px; text-align: center; color: #333; font-weight: bold; font-size: 0.7rem;" id="bf16_15">0100000010000000</div>
                    </div>
                </div>
                <div>
                    <div style="font-size: 1.1rem; margin-bottom: 10px; text-align: center;">MXFP4 Output (4-bit each)</div>
                    <div class="input-grid">
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code0">0100</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code1">1010</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code2">0010</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code3">0110</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code4">0011</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code5">1001</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code6">0101</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code7">1100</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code8">0010</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code9">0110</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code10">1011</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code11">0101</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code12">1001</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code13">0011</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code14">1110</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px; text-align: center; color: #333; font-weight: bold;" id="code15">0110</div>
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.9); padding: 12px; border-radius: 8px; text-align: center; color: #333; font-weight: bold; margin-bottom: 15px; font-size: 1.1rem;">
                    ⚖️ Scale Factor: <span id="scaleValue">1.0</span>
                </div>
                <div class="stats">
                    <div>💾 Original: <span id="originalBits">256</span> bits</div>
                    <div>📦 Compressed: <span id="compressedBits">96</span> bits</div>
                    <div>🗜️ Ratio: <span id="compressionRatio">2.67</span>:1</div>
                    <div>💰 Saved: <span id="savedPercent">62.5</span>%</div>
                </div>
            </div>
        </div>

        <div class="visualizations">
            <div class="viz-panel">
                <h4 class="viz-title">🎲 MXFP4 E2M1 Lookup Table</h4>
                <div class="lut-grid" id="lutGrid"></div>
                <div style="text-align: center; font-size: 0.9rem; color: #666; margin-top: 10px;">
                    Highlighted values show the quantized output mappings
                </div>
            </div>

            <div class="viz-panel">
                <h4 class="viz-title">⚙️ Quantization Process</h4>
                <div class="scale-info" id="scaleInfo">
                    Scale Factor: 1.0 (Max value: 4.0 → MXFP4 range: ±6.0)
                </div>
                <div class="quantization-flow" id="quantFlow"></div>
            </div>
        </div>

        <div class="error-analysis">
            <div class="error-title">📈 Quantization Error Analysis</div>
            <div class="error-grid" id="errorGrid"></div>
            <div style="margin-top: 15px; text-align: center;">
                <strong>MSE:</strong> <span id="mseValue">0.0938</span> |
                <strong>MAE:</strong> <span id="maeValue">0.250</span> |
                <strong>Max Error:</strong> <span id="maxErrorValue">0.500</span>
            </div>
        </div>

        <div class="compression-stats">
            <h3 style="margin-bottom: 15px;">📊 Memory Analysis</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div>
                    <div style="font-size: 2rem;">📦</div>
                    <div>Original Size</div>
                    <div style="font-size: 1.2rem; font-weight: bold;" id="originalSize">64 bits</div>
                </div>
                <div>
                    <div style="font-size: 2rem;">🗜️</div>
                    <div>MXFP4 Size</div>
                    <div style="font-size: 1.2rem; font-weight: bold;" id="mxfp4Size">48 bits</div>
                </div>
                <div>
                    <div style="font-size: 2rem;">💰</div>
                    <div>Memory Saved</div>
                    <div style="font-size: 1.2rem; font-weight: bold;" id="memorySaved">16 bits</div>
                </div>
            </div>
        </div>
    </div>
</div>

<script src="{{ '/assets/js/mxfp4-visualizer.js' | relative_url }}"></script>