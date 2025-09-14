---
title: Understanding MXFP4 Quantization
description: >-
  Visualizer for MXFP4 quantization
date: 2025-09-14
categories: [Tool]
tags: [AI, Machine Learning, LLM, Quantization]
pin: true
math: false
author: ks
---

# Understanding MXFP4 Quantization

MXFP4 (Microscaling 4-bit Floating Point) is quantization format that enables significant memory compression for neural networks while maintaining reasonable accuracy and was introduced with GPT OSS model versions from OpenAI.

## Key Concepts

- **Block-wise scaling**: Each group of 32 elements shares a scale factor
- **E2M1 format**: 1 sign bit + 2 exponent bits + 1 mantissa bit
- **Dynamic range**: Adapts to local statistics of each block

## Interactive Visualizer

Try the interactive visualizer below to understand how MXFP4 quantization works:

<div id="mxfp4-visualizer-embed">
<style>
/* Scoped CSS for the embedded visualizer */
#mxfp4-visualizer-embed {
    margin: 20px 0;
    padding: 0;
}

#mxfp4-visualizer-embed * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

#mxfp4-visualizer-embed .visualizer-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    color: white;
}

#mxfp4-visualizer-embed .visualizer-inner {
    background: white;
    border-radius: 10px;
    padding: 20px;
    color: #333;
}

/* Add more scoped styles here... */
</style>

<div class="visualizer-container">
    <div class="visualizer-inner">
        <h3 style="text-align: center; margin-bottom: 20px; color: #667eea;">🔬 MXFP4 Quantization Visualizer</h3>
        
        <!-- Simplified version of the visualizer for embedding -->
        <div style="display: flex; flex-direction: column; gap: 20px; margin-bottom: 20px;">
            <div style="background: #4CAF50; padding: 15px; border-radius: 8px; color: white;">
                <h4>Input BF16 Values</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 10px 0;">
                    <input type="number" id="embed-val0" value="2.5" style="padding: 8px; border: none; border-radius: 4px; text-align: center; background: #E8F5E8; color: #333;" step="0.1">
                    <input type="number" id="embed-val1" value="-1.25" style="padding: 8px; border: none; border-radius: 4px; text-align: center; background: #E8F5E8; color: #333;" step="0.1">
                    <input type="number" id="embed-val2" value="0.75" style="padding: 8px; border: none; border-radius: 4px; text-align: center; background: #E8F5E8; color: #333;" step="0.1">
                    <input type="number" id="embed-val3" value="4.0" style="padding: 8px; border: none; border-radius: 4px; text-align: center; background: #E8F5E8; color: #333;" step="0.1">
                </div>
            </div>
            
            <div style="background: #2196F3; padding: 15px; border-radius: 8px; color: white;">
                <h4>MXFP4 Output</h4>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 10px 0;">
                    <div id="embed-code0" style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 4px; text-align: center; color: #333;">0100</div>
                    <div id="embed-code1" style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 4px; text-align: center; color: #333;">1010</div>
                    <div id="embed-code2" style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 4px; text-align: center; color: #333;">0010</div>
                    <div id="embed-code3" style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 4px; text-align: center; color: #333;">0110</div>
                </div>
                <div style="text-align: center; margin-top: 10px;">
                    Scale: <span id="embed-scale">1.0</span> | Compression: <span id="embed-ratio">1.33</span>:1
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 15px;">
            <a href="/mxfp4/" style="background: #9C27B0; color: white; padding: 12px 24px; border-radius: 6px; text-decoration: none; font-weight: bold;">🚀 Open Full Visualizer</a>
        </div>
    </div>
</div>

<script>
// Simplified JavaScript for the embedded version
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

function embedQuantize() {
    const values = [
        parseFloat(document.getElementById('embed-val0').value) || 0,
        parseFloat(document.getElementById('embed-val1').value) || 0,
        parseFloat(document.getElementById('embed-val2').value) || 0,
        parseFloat(document.getElementById('embed-val3').value) || 0
    ];
    
    const maxAbs = Math.max(...values.map(Math.abs));
    const scale = Math.max(1.0, Math.pow(2, Math.ceil(Math.log2(maxAbs / 6.0))));
    
    const scaledValues = values.map(v => v / scale);
    const quantResults = scaledValues.map(v => findNearestEmbedMXFP4(v));
    
    quantResults.forEach((result, i) => {
        document.getElementById(`embed-code${i}`).textContent = result.code.toString(2).padStart(4, '0');
    });
    
    document.getElementById('embed-scale').textContent = scale.toFixed(2);
    
    const originalBits = 64;
    const compressedBits = 48;
    document.getElementById('embed-ratio').textContent = (originalBits/compressedBits).toFixed(2);
}

// Initialize on page load and add event listeners for real-time updates
document.addEventListener('DOMContentLoaded', function() {
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
</script>
</div>

## How It Works

1. **Input Processing**: The algorithm takes your BF16 values and groups them into blocks
2. **Scale Computation**: Each block gets a scale factor based on its maximum absolute value
3. **Quantization**: Values are scaled and mapped to the nearest MXFP4 representable value
4. **Storage**: The result is 4-bit codes plus scale factors

## Benefits

- **Memory Efficiency**: ~75% memory reduction compared to BF16
- **Hardware Friendly**: Optimized for modern GPU tensor cores
- **Adaptive Precision**: Each block optimizes its own quantization range

Try different values in the visualizer above to see how the quantization adapts!'
