// Simplified JavaScript for the embedded MXFP4 visualizer
const EMBED_MXFP4_LUT = {
    0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0,
    8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
};

function findNearestEmbedMXFP4(value) {
    let bestCode = 0;
    let bestError = Math.abs(value - EMBED_MXFP4_LUT[0]);

    for (let code = 0; code <= 15; code++) {
        const error = Math.abs(value - EMBED_MXFP4_LUT[code]);
        if (error < bestError) {
            bestError = error;
            bestCode = code;
        }
    }

    return { code: bestCode, value: EMBED_MXFP4_LUT[bestCode] };
}

function computeEmbedScaleFactor(values) {
    const maxAbs = Math.max(...values.map(Math.abs));
    if (maxAbs === 0) return 1.0;

    const maxMXFP4 = 6.0;
    const rawScale = maxAbs / maxMXFP4;

    // Use power-of-2 scaling (same as full visualizer)
    const scaleExp = Math.ceil(Math.log2(rawScale));
    return Math.pow(2, Math.max(0, scaleExp));
}

function embedQuantize() {
    const values = [
        parseFloat(document.getElementById('embed-val0').value) || 0,
        parseFloat(document.getElementById('embed-val1').value) || 0,
        parseFloat(document.getElementById('embed-val2').value) || 0,
        parseFloat(document.getElementById('embed-val3').value) || 0
    ];

    // Compute dynamic scale factor from input BF16 values
    const scale = computeEmbedScaleFactor(values);

    const scaledValues = values.map(v => v / scale);
    const quantResults = scaledValues.map(v => findNearestEmbedMXFP4(v));

    quantResults.forEach((result, i) => {
        document.getElementById(`embed-code${i}`).textContent = result.code.toString(2).padStart(4, '0');
    });

    document.getElementById('embed-scale').textContent = scale.toFixed(2);

    const originalBits = 64;
    const compressedBits = 48;
    document.getElementById('embed-ratio').textContent = (originalBits / compressedBits).toFixed(2);
}

// Initialize on page load and add event listeners for real-time updates
document.addEventListener('DOMContentLoaded', function () {
    embedQuantize();

    // Add event listeners to all input fields
    for (let i = 0; i < 4; i++) {
        const input = document.getElementById(`embed-val${i}`);
        if (input) {
            input.addEventListener('input', embedQuantize);
            input.addEventListener('change', embedQuantize);
        }
    }
});