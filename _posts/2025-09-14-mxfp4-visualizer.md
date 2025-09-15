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

MXFP4 (Microscaling 4-bit Floating Point) is a quantization format that enables significant memory compression for neural networks while maintaining reasonable accuracy and was introduced with GPT OSS model versions from OpenAI.

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

<script src="{{ '/assets/js/mxfp4-embed.js' | relative_url }}"></script>
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
