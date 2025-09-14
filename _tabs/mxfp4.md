---
# the default layout is 'page'
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
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 15px;
    justify-content: center;
}

#mxfp4-visualizer-container .input-grid > * {
    flex: 1;
    min-width: 100px;
    max-width: 150px;
}

#mxfp4-visualizer-container .input-field {
    padding: 10px;
    border: none;
    border-radius: 8px;
    text-align: center;
    font-size: 1rem;
    background: rgba(255,255,255,0.9);
    color: #333;
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
    display: flex;
    flex-direction: column;
    gap: 15px;
    margin-bottom: 20px;
}

@media (min-width: 768px) {
    #mxfp4-visualizer-container .quantization-flow {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
    }
}

@media (min-width: 1000px) {
    #mxfp4-visualizer-container .quantization-flow {
        grid-template-columns: repeat(4, 1fr);
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
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
}

#mxfp4-visualizer-container .error-grid .error-item {
    flex: 1;
    min-width: 120px;
    max-width: 200px;
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
                    <input type="number" class="input-field" id="val0" step="0.1" value="2.5">
                    <input type="number" class="input-field" id="val1" step="0.1" value="-1.25">
                    <input type="number" class="input-field" id="val2" step="0.1" value="0.75">
                    <input type="number" class="input-field" id="val3" step="0.1" value="4.0">
                </div>
                <div class="buttons">
                    <button class="btn btn-primary" onclick="quantize()">🚀 Quantize</button>
                    <button class="btn btn-secondary" onclick="generateRandom()">🎲 Random</button>
                    <button class="btn btn-secondary" onclick="loadPreset()">📋 Preset</button>
                </div>
                <div class="stats">
                    <div>📏 Vec Size: <span id="vecSize">4</span></div>
                    <div>⚖️ Scale: <span id="scaleValue">1.0</span></div>
                </div>
            </div>

            <div class="output-section">
                <h3 class="section-title">🎯 Output Comparison</h3>
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 1.1rem; margin-bottom: 10px; text-align: center;">BF16 Input (16-bit each)</div>
                    <div class="input-grid">
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 8px; text-align: center; color: #333; font-weight: bold; font-size: 0.8rem;" id="bf16_0">0100000010100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 8px; text-align: center; color: #333; font-weight: bold; font-size: 0.8rem;" id="bf16_1">1011111110100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 8px; text-align: center; color: #333; font-weight: bold; font-size: 0.8rem;" id="bf16_2">0011111101100000</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 8px; text-align: center; color: #333; font-weight: bold; font-size: 0.8rem;" id="bf16_3">0100000010000000</div>
                    </div>
                </div>
                <div>
                    <div style="font-size: 1.1rem; margin-bottom: 10px; text-align: center;">MXFP4 Output (4-bit each)</div>
                    <div class="input-grid">
                        <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; text-align: center; color: #333; font-weight: bold;" id="code0">0100</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; text-align: center; color: #333; font-weight: bold;" id="code1">1010</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; text-align: center; color: #333; font-weight: bold;" id="code2">0010</div>
                        <div style="background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; text-align: center; color: #333; font-weight: bold;" id="code3">0110</div>
                    </div>
                </div>
                <div class="stats">
                    <div>💾 Original: <span id="originalBits">64</span> bits</div>
                    <div>📦 Compressed: <span id="compressedBits">48</span> bits</div>
                    <div>🗜️ Ratio: <span id="compressionRatio">1.33</span>:1</div>
                    <div>💰 Saved: <span id="savedPercent">25.0</span>%</div>
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

<script>
        // MXFP4 E2M1 lookup table
        const MXFP4_LUT = {
            0b0000: 0.0,  0b0001: 0.5,  0b0010: 1.0,  0b0011: 1.5,
            0b0100: 2.0,  0b0101: 3.0,  0b0110: 4.0,  0b0111: 6.0,
            0b1000: -0.0, 0b1001: -0.5, 0b1010: -1.0, 0b1011: -1.5,
            0b1100: -2.0, 0b1101: -3.0, 0b1110: -4.0, 0b1111: -6.0
        };

        const VALUE_TO_CODE = {};
        for (let [code, value] of Object.entries(MXFP4_LUT)) {
            VALUE_TO_CODE[value] = parseInt(code);
        }

        let selectedLUTCells = new Set();

        function floatToBF16Binary(value) {
            // Convert float to Float32Array to get the binary representation
            const buffer = new ArrayBuffer(4);
            const view = new DataView(buffer);
            view.setFloat32(0, value);

            // Get the 32-bit representation
            const bits32 = view.getUint32(0);

            // Convert to BF16 (truncate mantissa: keep sign + 8-bit exp + 7-bit mantissa)
            // BF16 is the upper 16 bits of Float32
            const bf16 = (bits32 >>> 16) & 0xFFFF;

            return bf16.toString(2).padStart(16, '0');
        }

        function initializeLUT() {
            const lutGrid = document.getElementById('lutGrid');
            lutGrid.innerHTML = '';

            for (let code = 0; code <= 15; code++) {
                const value = MXFP4_LUT[code];
                const cell = document.createElement('div');
                cell.className = 'lut-cell';
                cell.innerHTML = `
                    <div style="font-family: monospace; font-size: 0.7rem;">${code.toString(2).padStart(4, '0')}</div>
                    <div style="font-weight: bold;">${value}</div>
                `;
                cell.onclick = () => toggleLUTCell(cell, code);
                lutGrid.appendChild(cell);
            }
        }

        function toggleLUTCell(cell, code) {
            if (selectedLUTCells.has(code)) {
                selectedLUTCells.delete(code);
                cell.classList.remove('selected');
            } else {
                selectedLUTCells.add(code);
                cell.classList.add('selected');
            }
        }

        function findNearestMXFP4(value) {
            let bestCode = 0;
            let bestError = Math.abs(value - MXFP4_LUT[0]);

            for (let code = 0; code <= 15; code++) {
                const error = Math.abs(value - MXFP4_LUT[code]);
                if (error < bestError) {
                    bestError = error;
                    bestCode = code;
                }
            }

            return { code: bestCode, value: MXFP4_LUT[bestCode], error: bestError };
        }

        function computeScaleFactor(values) {
            const maxAbs = Math.max(...values.map(Math.abs));
            if (maxAbs === 0) return 1.0;

            const maxMXFP4 = 6.0;
            const rawScale = maxAbs / maxMXFP4;

            // Use power-of-2 scaling
            const scaleExp = Math.ceil(Math.log2(rawScale));
            return Math.pow(2, Math.max(0, scaleExp));
        }

        function quantize() {
            // Get input values
            const values = [
                parseFloat(document.getElementById('val0').value) || 0,
                parseFloat(document.getElementById('val1').value) || 0,
                parseFloat(document.getElementById('val2').value) || 0,
                parseFloat(document.getElementById('val3').value) || 0
            ];

            // Compute scale factor
            const scale = computeScaleFactor(values);
            document.getElementById('scaleValue').textContent = scale.toFixed(3);

            // Scale values and quantize
            const scaledValues = values.map(v => v / scale);
            const quantResults = scaledValues.map(v => findNearestMXFP4(v));

            // Update BF16 input binary and MXFP4 output codes
            values.forEach((value, i) => {
                // Update BF16 binary representation
                const bf16Element = document.getElementById(`bf16_${i}`);
                bf16Element.textContent = floatToBF16Binary(value);

                // Update MXFP4 output code
                const codeElement = document.getElementById(`code${i}`);
                codeElement.textContent = quantResults[i].code.toString(2).padStart(4, '0');
            });

            // Highlight corresponding LUT cells for all quantized values
            const lutCells = document.querySelectorAll('.lut-cell');
            lutCells.forEach((cell, idx) => {
                cell.classList.remove('selected');
            });

            quantResults.forEach((result) => {
                if (result.code < lutCells.length) {
                    lutCells[result.code].classList.add('selected');
                }
            });

            // Update scale info
            const maxVal = Math.max(...values.map(Math.abs));
            document.getElementById('scaleInfo').innerHTML = `
                Scale Factor: ${scale.toFixed(3)} (Max value: ${maxVal.toFixed(3)} → MXFP4 range: ±6.0)
            `;

            // Update quantization flow
            updateQuantizationFlow(values, scaledValues, quantResults);

            // Calculate compression stats
            updateCompressionStats(values.length);

            // Calculate and display errors
            updateErrorAnalysis(values, quantResults, scale);
        }

        function updateQuantizationFlow(original, scaled, quantResults) {
            const flowContainer = document.getElementById('quantFlow');
            flowContainer.innerHTML = '';

            for (let i = 0; i < original.length; i++) {
                const flowItem = document.createElement('div');
                flowItem.className = 'flow-item';
                flowItem.innerHTML = `
                    <div class="flow-value">${original[i].toFixed(2)}</div>
                    <div class="flow-arrow">↓</div>
                    <div style="font-size: 0.9rem; color: #666;">÷ ${document.getElementById('scaleValue').textContent}</div>
                    <div class="flow-arrow">↓</div>
                    <div style="font-size: 0.9rem; color: #666;">${scaled[i].toFixed(2)}</div>
                    <div class="flow-arrow">↓</div>
                    <div class="flow-result">${quantResults[i].value.toFixed(1)}</div>
                    <div style="font-size: 0.8rem; color: #999; margin-top: 5px;">${quantResults[i].code.toString(2).padStart(4, '0')}</div>
                `;
                flowContainer.appendChild(flowItem);
            }
        }

        function updateCompressionStats(numValues) {
            const originalBits = numValues * 16; // bf16
            const dataBits = numValues * 4; // 4-bit codes
            const scaleBits = 32; // one float32 scale
            const compressedBits = dataBits + scaleBits;
            const ratio = originalBits / compressedBits;
            const savedPercent = ((originalBits - compressedBits) / originalBits) * 100;

            document.getElementById('originalBits').textContent = originalBits;
            document.getElementById('compressedBits').textContent = compressedBits;
            document.getElementById('compressionRatio').textContent = ratio.toFixed(2);
            document.getElementById('savedPercent').textContent = savedPercent.toFixed(1);

            document.getElementById('originalSize').textContent = `${originalBits} bits`;
            document.getElementById('mxfp4Size').textContent = `${compressedBits} bits`;
            document.getElementById('memorySaved').textContent = `${originalBits - compressedBits} bits`;
        }

        function updateErrorAnalysis(originalValues, quantResults, scale) {
            const errorGrid = document.getElementById('errorGrid');
            errorGrid.innerHTML = '';

            const errors = [];

            for (let i = 0; i < originalValues.length; i++) {
                const reconstructed = quantResults[i].value * scale;
                const error = originalValues[i] - reconstructed;
                errors.push(Math.abs(error));

                const errorItem = document.createElement('div');
                errorItem.className = 'error-item';
                errorItem.innerHTML = `
                    <div style="font-weight: bold; color: ${error >= 0 ? '#e74c3c' : '#27ae60'};">
                        ${error >= 0 ? '+' : ''}${error.toFixed(3)}
                    </div>
                    <div style="font-size: 0.8rem; color: #666;">
                        ${originalValues[i].toFixed(2)} → ${reconstructed.toFixed(2)}
                    </div>
                `;
                errorGrid.appendChild(errorItem);
            }

            // Calculate error metrics
            const mse = errors.reduce((sum, e) => sum + e * e, 0) / errors.length;
            const mae = errors.reduce((sum, e) => sum + e, 0) / errors.length;
            const maxError = Math.max(...errors);

            document.getElementById('mseValue').textContent = mse.toFixed(4);
            document.getElementById('maeValue').textContent = mae.toFixed(3);
            document.getElementById('maxErrorValue').textContent = maxError.toFixed(3);
        }

        function generateRandom() {
            for (let i = 0; i < 4; i++) {
                const randomValue = (Math.random() - 0.5) * 8; // Range: -4 to 4
                document.getElementById(`val${i}`).value = randomValue.toFixed(2);
            }
            quantize();
        }

        function loadPreset() {
            // Demonstrate different scale factors for different blocks
            const examples = [
                { values: [0.1, 0.2, 0.3, 0.15], desc: "Small values (scale ≈ 0.05)" },
                { values: [5.0, -4.8, 3.2, -5.9], desc: "Large values (scale ≈ 1.0)" },
                { values: [1.0, 2.0, 3.0, -1.5], desc: "Medium values (scale ≈ 0.5)" }
            ];

            const example = examples[Math.floor(Date.now() / 3000) % examples.length];
            example.values.forEach((val, i) => {
                document.getElementById(`val${i}`).value = val;
            });

            // Update description to show this is about different scale factors
            const scaleInfo = document.getElementById('scaleInfo');
            setTimeout(() => {
                scaleInfo.innerHTML += `<br><small style="opacity:0.8;">${example.desc}</small>`;
            }, 100);

            quantize();
        }

        // Initialize the visualizer
        document.addEventListener('DOMContentLoaded', function() {
            initializeLUT();
            quantize();

            // Add event listeners to input fields
            for (let i = 0; i < 4; i++) {
                document.getElementById(`val${i}`).addEventListener('input', quantize);
            }
        });
</script>