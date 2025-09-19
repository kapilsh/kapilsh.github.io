---
title: Triton Kernels - RMS Norm
description: >-
  RMS Normalization Triton kernel implementation for LLMs
date: 2025-09-17
categories: [Blog]
tags: [AI, Machine Learning, LLM, Triton, GPU, Kernels]
pin: true
math: true
author: ks
---

RMSNorm is a crucial component in modern transformer architectures. Most modern LLMs now user RMSNorm by default compared to the original versions where LayerNorm was more popular. Unlike LayerNorm, RMSNorm simplifies the computation by removing the mean centering step, making it more efficient for GPU implementation with Triton kernels.

## What is RMS Normalization?

RMS Normalization normalizes inputs using only the root mean square of the values, without subtracting the mean. The formula is:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma
$$

Where:
- $x$ is the input vector
- $n$ is the dimension of the input
- $\epsilon$ is a small constant for numerical stability
- $\gamma$ (weight) is a learnable scaling parameter

## RMSNorm Triton Kernel

In this post, we will walk through how a triton kernel for RMSNorm might look like, to simplify we will just consider forward only kernel for now. RMSNorm uses a combination of row-wise reductions (variance calculation) and point-wise ops (division, mult). Hence, BLOCKs will need to be processed such that we process each row at a time. (Just like `softmax`). 

Let's walk through a concrete example showing how RMS Normalization works in a Triton kernel with block-level processing:

### Example Setup

Consider a **3×8 input matrix** where each row needs independent RMS normalization with **BLOCK_SIZE=4**:

```
Row 0: [ 2.0, -1.0,  3.0,  0.5, -0.5,  1.5, -2.0,  1.0]
Row 1: [ 4.0, -3.0,  2.5,  1.0, -1.5,  0.0, -0.5,  2.0]
Row 2: [-1.0,  3.5, -2.5,  1.5,  0.0, -3.0,  2.5, -0.5]
```

**Learnable Weights γ:**
```
γ = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

### Step 1: Program ID Mapping

Each Triton program (CUDA block) processes **one row** independently:

```python
row = tl.program_id(0)  # Each CUDA block processes one row
Y += row * stride       # Point to output row
X += row * stride       # Point to input row
```

**Block Assignment:**
- **Program 0**: Processes Row 0
- **Program 1**: Processes Row 1
- **Program 2**: Processes Row 2

### Step 2: Block-wise Sum of Squares Calculation

With **BLOCK_SIZE=4**, each row is processed in 2 blocks:


#### Row 0 Processing (Program ID 0):

```python
# Triton kernel code for Row 0
_sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for off in range(0, N, BLOCK_SIZE):  # off = 0, then off = 4
    cols = off + tl.arange(0, BLOCK_SIZE)  # [0,1,2,3] then [4,5,6,7]
    a = tl.load(X + cols, mask=cols < N, other=0.)
    _sum_sq += a * a  # Accumulate squared values
```


| Block | Columns | Values | Squared Values | Block Sum |
|-------|---------|--------|----------------|-----------|
| **Block 1** | 0-3 | `[2.0, -1.0, 3.0, 0.5]` | `[4.0, 1.0, 9.0, 0.25]` | **14.25** |
| **Block 2** | 4-7 | `[-0.5, 1.5, -2.0, 1.0]` | `[0.25, 2.25, 4.0, 1.0]` | **7.50** |
| | | | **Total Sum of Squares** | **21.75** |


#### All Rows Summary:

| Row | Sum of Squares | Mean Square (÷8) | RMS | Reciprocal RMS |
|-----|----------------|------------------|-----|----------------|
| **Row 0** | 21.75 | 2.719 | **1.65** | **0.606** |
| **Row 1** | 44.25 | 5.531 | **2.35** | **0.425** |
| **Row 2** | 50.00 | 6.250 | **2.50** | **0.400** |

### Step 3: RMS Calculation

```python
sum_sq = tl.sum(_sum_sq, axis=0)      # Total sum of squares for the row
rms = tl.sqrt(sum_sq / N)             # Root Mean Square
rrms = 1 / (rms + eps)                # Reciprocal RMS (for efficiency)
```

### Step 4: Normalization and Weight Application

Now we apply the transformation **y = γ × (x / RMS)** in blocks:

```python
# Triton kernel code for normalization
for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)      # Load weights γ
    x = tl.load(X + cols, mask=mask, other=0.)
    x_norm = x * rrms                     # Normalize: x / RMS(x)
    y = x_norm * w                        # Scale: γ * x_norm
    tl.store(Y + cols, y, mask=mask)      # Store result
```

#### Row 0 Normalization (RMS = 1.65, rrms = 0.606):

| Block | Input Values | x × rrms | γ × (x × rrms) | Output |
|-------|--------------|----------|----------------|--------|
| **Block 1** | `[2.0, -1.0, 3.0, 0.5]` | `[1.21, -0.61, 1.82, 0.30]` | `[1.21, -0.61, 1.82, 0.30]` | **[1.21, -0.61, 1.82, 0.30]** |
| **Block 2** | `[-0.5, 1.5, -2.0, 1.0]` | `[-0.30, 0.91, -1.21, 0.61]` | `[-0.30, 0.91, -1.21, 0.61]` | **[-0.30, 0.91, -1.21, 0.61]** |



### Complete Output Matrix

After processing all rows:

```
Output Matrix Y (3×8):
Row 0: [ 1.21, -0.61,  1.82,  0.30, -0.30,  0.91, -1.21,  0.61]
Row 1: [ 1.70, -1.28,  1.06,  0.43, -0.64,  0.00, -0.21,  0.85]
Row 2: [-0.40,  1.40, -1.00,  0.60,  0.00, -1.20,  1.00, -0.20]
```

## Full Kernel

```python
@triton.jit
def rms_norm_forward(
    input_ptr,
    output_ptr,
    weight_ptr,
    rstd_ptr,
    row_stride,
    feature_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of input and output tensors to compute
    row_idx = tl.program_id(0)
    output_ptr += row_idx * row_stride
    input_ptr += row_idx * row_stride

    # ==== REDUCTION PART ====
    # Compute variance (mean of squared values for RMS)
    sum_of_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        input_values = tl.load(
            input_ptr + col_indices, mask=col_indices < feature_dim, other=0.0
        ).to(tl.float32)
        sum_of_squares += input_values * input_values

    variance = tl.sum(sum_of_squares, axis=0) / feature_dim
    reciprocal_std = 1 / tl.sqrt(variance + eps)

    # Store reciprocal standard deviation for backward pass
    tl.store(rstd_ptr + row_idx, reciprocal_std)

    # === POINTWISE OPS ====
    # Normalize input and apply weight transformation
    for block_offset in range(0, feature_dim, BLOCK_SIZE):
        col_indices = block_offset + tl.arange(0, BLOCK_SIZE)
        valid_mask = col_indices < feature_dim

        weight_values = tl.load(weight_ptr + col_indices, mask=valid_mask)
        input_values = tl.load(input_ptr + col_indices, mask=valid_mask, other=0.0).to(
            tl.float32
        )

        normalized_values = input_values * reciprocal_std
        output_values = normalized_values * weight_values

        # Write final output
        tl.store(output_ptr + col_indices, output_values, mask=valid_mask)
```

## Let's see it in action

Below we have a visualization for `2 x 8` Tensor with `BLOCK_SIZE = 2`:

![RMS Norm](/assets/rms_norm.gif)

## Interactive RMS Norm Visualizer

Explore how RMS Normalization works in Triton kernels with this interactive visualization:

<div id="rms-norm-visualizer-container">
    <style>
        #rms-norm-visualizer-container {
            font-family: 'Monaco', 'Menlo', monospace;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            border-radius: 15px;
        }

        #rms-norm-visualizer-container .container {
            max-width: 100%;
            margin: 0 auto;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            overflow-x: auto;
        }

        #rms-norm-visualizer-container h1 {
            text-align: center;
            color: #64b5f6;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        #rms-norm-visualizer-container .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
        }

        #rms-norm-visualizer-container .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }

        #rms-norm-visualizer-container label {
            font-size: 12px;
            color: #b3e5fc;
            font-weight: bold;
        }

        #rms-norm-visualizer-container input,
        #rms-norm-visualizer-container button,
        #rms-norm-visualizer-container select {
            padding: 8px 12px;
            border: none;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.4);
            color: white;
            font-family: inherit;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(100, 181, 246, 0.3);
        }

        #rms-norm-visualizer-container select option {
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border: none;
        }

        #rms-norm-visualizer-container button {
            background: linear-gradient(45deg, #4fc3f7, #29b6f6);
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(79, 195, 247, 0.3);
        }

        #rms-norm-visualizer-container button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 195, 247, 0.4);
        }

        #rms-norm-visualizer-container .visualization {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-bottom: 25px;
        }

        #rms-norm-visualizer-container .tensor-display {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 15px;
            border: 2px solid rgba(100, 181, 246, 0.3);
            min-width: 0;
            overflow: hidden;
        }

        #rms-norm-visualizer-container .tensor-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #81c784;
            text-align: center;
        }

        #rms-norm-visualizer-container .tensor-grid {
            display: grid;
            gap: 3px;
            margin-bottom: 15px;
        }

        #rms-norm-visualizer-container .tensor-cell {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 6px;
            text-align: center;
            font-size: 10px;
            border-radius: 4px;
            transition: all 0.3s ease;
            min-height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            word-break: break-all;
            overflow: hidden;
        }

        @media (max-width: 768px) {
            #rms-norm-visualizer-container .tensor-cell {
                padding: 4px;
                font-size: 8px;
                min-height: 16px;
            }
        }

        #rms-norm-visualizer-container .tensor-cell.processing {
            background: rgba(255, 193, 7, 0.6);
            border-color: #ffc107;
            animation: pulse 1s infinite;
        }

        #rms-norm-visualizer-container .tensor-cell.processed {
            background: rgba(76, 175, 80, 0.6);
            border-color: #4caf50;
        }

        #rms-norm-visualizer-container .tensor-cell.current-row {
            border-color: #ff9800;
            background: rgba(255, 152, 0, 0.3);
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        #rms-norm-visualizer-container .steps-display {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 15px;
            border: 2px solid rgba(100, 181, 246, 0.3);
            min-width: 0;
            overflow: hidden;
        }

        #rms-norm-visualizer-container .step {
            margin-bottom: 15px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            border-left: 4px solid #64b5f6;
        }

        #rms-norm-visualizer-container .step-title {
            font-weight: bold;
            color: #64b5f6;
            margin-bottom: 8px;
        }

        #rms-norm-visualizer-container .step-content {
            font-size: 13px;
            line-height: 1.4;
        }

        #rms-norm-visualizer-container .formula {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            margin: 8px 0;
            border: 1px solid rgba(100, 181, 246, 0.3);
        }

        #rms-norm-visualizer-container .block-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
            font-size: 12px;
            color: #b3e5fc;
        }

        #rms-norm-visualizer-container .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }

        #rms-norm-visualizer-container .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #4fc3f7, #29b6f6);
            width: 0%;
            transition: width 0.3s ease;
        }

        #rms-norm-visualizer-container .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }

        #rms-norm-visualizer-container .stat {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }

        #rms-norm-visualizer-container .stat-label {
            font-size: 11px;
            color: #b3e5fc;
            margin-bottom: 5px;
        }

        #rms-norm-visualizer-container .stat-value {
            font-size: 14px;
            font-weight: bold;
            color: #81c784;
        }

        #rms-norm-visualizer-container .row-label {
            font-size: 10px;
            color: #b3e5fc;
            margin-bottom: 5px;
            text-align: left;
        }

        #rms-norm-visualizer-container .calculations-display {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 15px;
            border: 2px solid rgba(100, 181, 246, 0.3);
            min-width: 0;
            overflow: hidden;
        }

        #rms-norm-visualizer-container .calculations-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #ffb74d;
            text-align: center;
        }

        #rms-norm-visualizer-container .calculations-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }

        #rms-norm-visualizer-container .calculations-table th,
        #rms-norm-visualizer-container .calculations-table td {
            padding: 8px 12px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 11px;
        }

        #rms-norm-visualizer-container .calculations-table th {
            background: rgba(255, 183, 77, 0.3);
            color: #ffb74d;
            font-weight: bold;
        }

        #rms-norm-visualizer-container .calculations-table td {
            background: rgba(255, 255, 255, 0.05);
        }

        #rms-norm-visualizer-container .calculations-table .current-processing {
            background: rgba(255, 193, 7, 0.6);
            border-color: #ffc107;
            animation: pulse 1s infinite;
        }

        #rms-norm-visualizer-container .calculations-table .completed {
            background: rgba(76, 175, 80, 0.6);
            border-color: #4caf50;
        }
    </style>

    <div class="container">
        <h1>🚀 RMS Normalization Triton Kernel Visualization</h1>

        <div class="controls">
            <div class="control-group">
                <label>Rows (Batch Size)</label>
                <select id="numRows">
                    <option value="2" selected>2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                </select>
            </div>

            <div class="control-group">
                <label>Feature Dimension</label>
                <select id="featureDim">
                    <option value="2">2</option>
                    <option value="4">4</option>
                    <option value="8" selected>8</option>
                </select>
            </div>

            <div class="control-group">
                <label>Block Size</label>
                <select id="blockSize">
                    <!-- Options will be dynamically generated -->
                </select>
            </div>

            <div class="control-group">
                <label>Epsilon</label>
                <input type="number" id="epsilon" value="1e-6" step="1e-7" min="1e-10">
            </div>

            <button onclick="generateRandomInput()">🎲 Random Input</button>
            <button onclick="runVisualization()">▶️ Run Kernel</button>
            <button onclick="reset()">🔄 Reset</button>
        </div>

        <div class="visualization">
            <div class="tensor-display">
                <div class="tensor-title">Input Tensor</div>
                <div id="inputTensor" class="tensor-grid"></div>
                <div class="block-info">
                    <span>Current Row: <span id="currentRow">-</span></span>
                    <span>Block Range: <span id="blockRange">-</span></span>
                </div>
            </div>

            <div class="calculations-display">
                <div class="calculations-title">Per-Row Calculations</div>
                <table class="calculations-table">
                    <thead>
                        <tr>
                            <th>Row</th>
                            <th>Sum of Squares</th>
                            <th>Mean Square</th>
                            <th>RMS</th>
                            <th>Rstd (1/√(MS+ε))</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="calculationsTableBody">
                        <!-- Rows will be populated by JavaScript -->
                    </tbody>
                </table>
                <div class="block-info">
                    <span>Processing: Block size = <span id="currentBlockSize">4</span></span>
                    <span>Epsilon = <span id="currentEpsilon">1e-6</span></span>
                </div>
            </div>

            <div class="tensor-display">
                <div class="tensor-title">Output Tensor</div>
                <div id="outputTensor" class="tensor-grid"></div>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">Current Row RMS</div>
                        <div class="stat-value" id="currentRMS">-</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Current Row Rstd</div>
                        <div class="stat-value" id="currentRstd">-</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script src="{{ '/assets/js/rms-norm-visualizer.js' | relative_url }}"></script>

## Benchmarks

Let's see triton benchmarks with our kernel implementation and compare it w.r.t. Torch and Torch Compiled versions. I have an RTX 4090 24gb, which has a memory bandwidth of 1,008 GB/s. Based on the benchmark, we are able to hit 80%is of the memory bandwidth which is typical -- so seems like the kernel is performing really well.

![RMS Norm benchmark](/assets/rms_norm_benchmark.png)

## Code

**[🔗 Triton Kernels Implementation](https://github.com/kapilsh/gpt-oss-scratch/tree/main/kernels)**

