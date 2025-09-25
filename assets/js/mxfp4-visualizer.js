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
        cell.addEventListener('click', () => toggleLUTCell(cell, code));
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
    const values = [];
    for (let i = 0; i < 16; i++) {
        values.push(parseFloat(document.getElementById(`val${i}`).value) || 0);
    }

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
        codeElement.innerHTML = `
            <div style="font-size: 0.8rem; font-family: monospace;">${quantResults[i].code.toString(2).padStart(4, '0')}</div>
            <div style="font-size: 0.7rem; margin-top: 2px;">${quantResults[i].value.toFixed(1)}</div>
        `;
    });

    // Highlight corresponding LUT cells for all quantized values
    const lutCells = document.querySelectorAll('.lut-cell');
    lutCells.forEach((cell) => {
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
    updateQuantizationFlow(values, scaledValues, quantResults, scale);

    // Calculate compression stats
    updateCompressionStats(values.length);

    // Calculate and display errors
    updateErrorAnalysis(values, quantResults, scale);
}

function updateQuantizationFlow(original, scaled, quantResults, scale) {
    const flowContainer = document.getElementById('quantFlow');
    flowContainer.innerHTML = '';

    // Show first 8 values for visualization (to keep it manageable)
    const displayCount = Math.min(8, original.length);
    for (let i = 0; i < displayCount; i++) {
        const flowItem = document.createElement('div');
        flowItem.className = 'flow-item';
        flowItem.innerHTML = `
            <div class="flow-value">${original[i].toFixed(2)}</div>
            <div class="flow-arrow">↓</div>
            <div style="font-size: 0.9rem; color: #666;">÷ ${scale.toFixed(3)}</div>
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
    for (let i = 0; i < 16; i++) {
        const randomValue = (Math.random() - 0.5) * 2000; // Range: -1000 to 1000
        document.getElementById(`val${i}`).value = randomValue.toFixed(2);
    }
    quantize();
}

function loadPreset() {
    // Demonstrate different scale factors for different blocks
    const examples = [
        {
            values: [10.1, 20.2, 15.15, 30.3, -12.12, 25.25, -8.08, 18.18, 22.22, -14.14, 19.19, 11.11, -16.16, 28.28, 13.13, -9.09],
            desc: "Small values (fine scale)"
        },
        {
            values: [500.0, -480.8, 320.2, -590.9, 410.1, -370.7, 620.2, -450.5, 380.8, 510.1, -420.2, 390.9, -530.3, 460.6, -310.1, 570.7],
            desc: "Large values (coarse scale)"
        },
        {
            values: [100.0, 200.0, -150.5, 230.3, -80.8, 170.7, -120.2, 210.1, 140.4, -180.8, 250.5, -110.1, 190.9, -220.2, 160.6, -90.9],
            desc: "Medium values (medium scale)"
        }
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
    for (let i = 0; i < 16; i++) {
        document.getElementById(`val${i}`).addEventListener('input', quantize);
    }

    // Add event listeners to buttons
    document.getElementById('quantizeBtn').addEventListener('click', quantize);
    document.getElementById('randomBtn').addEventListener('click', generateRandom);
    document.getElementById('presetBtn').addEventListener('click', loadPreset);
});