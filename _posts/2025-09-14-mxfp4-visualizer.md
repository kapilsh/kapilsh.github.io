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

As LLMs continue to grow in size, quantized versions of these models provide a good way to locally experiment and post-train them. MXFP4 (Microscaling 4-bit Floating Point) quantization was recently used in open source releases of gpt-oss. In this post, we'll dive deep into how MXFP4 works and explore its compression benefits through practical examples.

## What is MXFP4 Quantization?

MXFP4 is a quantization format that compresses neural network parameters using three key principles:

- **4 bits per element**: Each quantized value uses only 4 bits
- **Shared scaling factors**: Elements within a block share a common scale factor
- **Block-wise quantization**: Typically groups 32-128 elements per block

This approach balances compression efficiency with precision by recognizing that nearby parameters in neural networks often have similar magnitudes.

## MXFP4 Format Structure

Each MXFP4 value uses 4 bits arranged as:

- **1 bit**: Sign
- **2 bits**: Exponent  
- **1 bit**: Mantissa

Here's the complete mapping of 4-bit codes to values:

| Binary | Sign | Exp | Mantissa | Value |
|--------|------|-----|----------|-------|
| 0000   | 0    | 00  | 0        | 0.0   |
| 0001   | 0    | 00  | 1        | 0.5   |
| 0010   | 0    | 01  | 0        | 1.0   |
| 0011   | 0    | 01  | 1        | 1.5   |
| 0100   | 0    | 10  | 0        | 2.0   |
| 0101   | 0    | 10  | 1        | 3.0   |
| 0110   | 0    | 11  | 0        | 4.0   |
| 0111   | 0    | 11  | 1        | 6.0   |
| 1000   | 1    | 00  | 0        | -0.0  |
| 1001   | 1    | 00  | 1        | -0.5  |
| 1010   | 1    | 01  | 0        | -1.0  |
| 1011   | 1    | 01  | 1        | -1.5  |
| 1100   | 1    | 10  | 0        | -2.0  |
| 1101   | 1    | 10  | 1        | -3.0  |
| 1110   | 1    | 11  | 0        | -4.0  |
| 1111   | 1    | 11  | 1        | -6.0  |

The MXFP4 format can represent values in the range [-6, 6] before applying the shared scale factor.

## Step-by-Step Example: Quantizing a BF16 Tensor

Let's walk through quantizing a simple 4-element bf16 tensor:

```python
# Original bf16 values
original = [2.5, -1.25, 0.75, 4.0]
```

### Step 1: Calculate the Shared Scale Factor

```python
# Find the maximum absolute value
max_abs = max(abs(x) for x in original)  # max_abs = 4.0

# Calculate scale factor to fit in MXFP4 range [-6, 6]
scale_factor = max_abs / 6.0  # scale_factor = 4.0/6.0 ≈ 0.6667

# For efficiency, use power-of-2 scaling
import math
scale_exp = math.ceil(math.log2(max_abs / 6.0))  # scale_exp = 0
scale_factor = 2 ** scale_exp  # scale_factor = 1.0
```

### Step 2: Scale the Values

```python
scaled_values = [x / scale_factor for x in original]
# scaled_values = [2.5, -1.25, 0.75, 4.0]
```

### Step 3: Quantize to MXFP4

Find the nearest MXFP4 representation for each scaled value:

```python
# Quantization mapping:
# 2.5  → 2.0  (code: 0b0100)
# -1.25 → -1.0 (code: 0b1010)  
# 0.75 → 1.0  (code: 0b0010)
# 4.0  → 4.0  (code: 0b0110)

mxfp4_values = [2.0, -1.0, 1.0, 4.0]
mxfp4_codes = [0b0100, 0b1010, 0b0010, 0b0110]
```

### Step 4: Storage Format

```python
# Metadata
scale_factor = 1.0
scale_exponent = 0  # Since scale_factor = 2^0

# Quantized 4-bit values (packed into 2 bytes)
packed_data = 0b0100101000100110  # 01001010 00100110
```

### Step 5: Dequantization

```python
# Unpack 4-bit codes
codes = [0b0100, 0b1010, 0b0010, 0b0110]

# Map codes to MXFP4 values  
mxfp4_values = [2.0, -1.0, 1.0, 4.0]

# Apply scale factor
dequantized = [val * scale_factor for val in mxfp4_values]
# dequantized = [2.0, -1.0, 1.0, 4.0]
```

### Results Comparison

```
Original:     [2.5, -1.25, 0.75, 4.0]
Dequantized:  [2.0, -1.0,  1.0,  4.0] 
Error:        [0.5, 0.25, -0.25, 0.0]
```

## Storage Efficiency Analysis

### Small Tensor (4 elements)
- **Original bf16**: 4 × 16 bits = 64 bits
- **MXFP4**: 4 × 4 bits + 16 bits (scale) = 32 bits  
- **Compression ratio**: 2:1

### Large Tensor (1 Million elements)

The compression ratio improves dramatically with larger tensors because the scale factor overhead becomes negligible. Using 32-element blocks:

- **Total elements**: 1,000,000
- **Number of blocks**: 1,000,000 ÷ 32 = 31,250 blocks

**Original bf16**: 1,000,000 × 16 bits = 16,000,000 bits

**MXFP4**:
- Quantized values: 1,000,000 × 4 bits = 4,000,000 bits
- Scale factors: 31,250 × 16 bits = 500,000 bits  
- **Total**: 4,500,000 bits

**Compression ratio**: 16,000,000 ÷ 4,500,000 ≈ **3.56:1**

### Block Size Impact

| Block Size | Num Blocks | Scale Overhead | Total MXFP4 Bits | Compression Ratio |
|------------|------------|----------------|-------------------|-------------------|
| 16         | 62,500     | 1,000,000      | 5,000,000         | 3.20:1           |
| 32         | 31,250     | 500,000        | 4,500,000         | 3.56:1           |
| 64         | 15,625     | 250,000        | 4,250,000         | 3.76:1           |
| 128        | 7,813      | 125,008        | 4,125,008         | 3.88:1           |
| 256        | 3,907      | 62,512         | 4,062,512         | 3.94:1           |

## Key Insights

### Theoretical Maximum
As block size approaches infinity, the compression ratio approaches:
```
Limit = original_bits_per_element / quantized_bits_per_element = 16 / 4 = 4:1
```

### Block Size Trade-off
- **Larger blocks**: Better compression ratio
- **Smaller blocks**: Better quantization accuracy (values in each block are more similar)

### Real-World Impact
For a 1M element tensor:
- **Original bf16**: 2.0 MB
- **MXFP4 (32-elem blocks)**: 0.56 MB  
- **Memory saved**: 1.44 MB (72% reduction)

## MXFP4 Analysis Charts

Explore how MXFP4 quantization behaves across different scenarios:

<div id="mxfp4-charts-container">
    <!-- Navigation -->
    <div class="mb-3">
        <div class="btn-group" role="group">
            <button type="button" class="btn btn-primary chart-nav-btn" id="btn-mapping">Value Mapping</button>
            <button type="button" class="btn btn-outline-primary chart-nav-btn" id="btn-distribution">Distribution</button>
            <button type="button" class="btn btn-outline-primary chart-nav-btn" id="btn-table">Value Table</button>
        </div>
    </div>

    <!-- Mapping View -->
    <div id="view-mapping" class="chart-view">
        <div class="mb-4">
            <h4>Continuous to Discrete Mapping</h4>
            <p class="text-muted small">
                Shows how continuous input values (blue line) map to discrete MXFP4 values (red line).
                The step function illustrates the quantization process.
            </p>
            <div style="height: 400px;">
                <canvas id="mappingChart"></canvas>
            </div>
        </div>

        <div class="mb-4">
            <h4>Quantization Error</h4>
            <p class="text-muted small">
                Shows the error (difference) between original and quantized values.
                Notice larger errors in sparse regions of the MXFP4 value space.
            </p>
            <div style="height: 300px;">
                <canvas id="errorChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Distribution View -->
    <div id="view-distribution" class="chart-view" style="display: none;">
        <h4>Value Distribution (Simulated Neural Network Weights)</h4>
        <p class="text-muted small">
            Shows how normally distributed values (typical of neural network weights)
            map to the 16 discrete MXFP4 values. Values cluster around zero due to the normal distribution.
        </p>
        <div style="height: 500px;">
            <canvas id="distributionChart"></canvas>
        </div>

        <div class="mt-4 text-muted small">
            <p><strong>Key Observations:</strong></p>
            <ul>
                <li>Values near zero (0.0, ±0.5, ±1.0) have highest probability</li>
                <li>Extreme values (±4.0, ±6.0) are rarely used</li>
                <li>The distribution reflects typical neural network weight patterns</li>
                <li>Non-uniform spacing in MXFP4 values affects representation efficiency</li>
            </ul>
        </div>
    </div>

    <!-- Table View -->
    <div id="view-table" class="chart-view" style="display: none;">
        <h4>Complete MXFP4 Value Table</h4>
        <div id="valueTable">
            <!-- Table will be populated by JavaScript -->
        </div>
    </div>
</div>

<!-- Load Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="{{ '/assets/js/mxfp4-charts-vanilla.js' | relative_url }}"></script>

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

Try different values in the visualizer above to see how the quantization adapts!
