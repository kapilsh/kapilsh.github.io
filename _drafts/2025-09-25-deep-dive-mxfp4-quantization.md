---
title: Deep Dive into MXFP4 Quantization - From Theory to Matrix Operations and MoE Layers
description: >-
  Complete technical exploration of MXFP4 quantization, including matrix multiplication and Mixture of Experts implementation with interactive visualizations
date: 2025-09-25
categories: [Deep Learning]
tags: [AI, Machine Learning, LLM, Quantization,]
pin: true
math: true
author: ks
---

MXFP4 quantization has emerged as a revolutionary approach to neural network compression, enabling massive models like GPT-OSS-120B to run on a single H100 GPU. This comprehensive guide explores MXFP4 from fundamental principles through practical matrix operations to complex Mixture of Experts (MoE) implementations.

## What Makes MXFP4 Special?

MXFP4 (Microscaling 4-bit Floating Point) represents a paradigm shift in neural network quantization. Unlike traditional quantization methods that struggle with the diverse dynamic ranges in neural networks, MXFP4 uses **block-wise shared scaling** to achieve both high compression ratios and acceptable precision.

### Core Innovation: Microscaling

The "microscaling" in MXFP4 refers to its block-based approach:
- **Block size**: Typically 32-128 elements
- **Shared scale factor**: One 8-bit exponent per block
- **Individual mantissas**: 4 bits per element

This design leverages the observation that nearby parameters in neural networks often have similar magnitudes, making shared scaling highly effective.

## MXFP4 Format Deep Dive

### Bit Structure (E2M1)

Each MXFP4 value uses exactly 4 bits:
- **1 sign bit**: Positive (0) or negative (1)
- **2 exponent bits**: Powers of 2 from 2⁰ to 2²
- **1 mantissa bit**: Either 1.0 or 1.5 multiplier

### Mathematical Representation

For a 4-bit code `SEMM` (Sign, Exponent, Mantissa):

```
value = (-1)^S × mantissa × 2^(exponent_bias + E)
```

Where:
- `mantissa = 1.0 + M × 0.5` (giving us 1.0 or 1.5)
- `exponent_bias = -1` (so exponents range from 2⁻¹ to 2²)

### Complete Value Mapping

<div id="mxfp4-format-viz" class="my-4">
  <h4>MXFP4 Value Space Analysis</h4>

  <!-- Navigation -->
  <div class="mb-3">
    <div class="btn-group" role="group">
      <button type="button" class="btn btn-outline-primary chart-nav-btn" id="btn-value-space">Value Space</button>
      <button type="button" class="btn btn-outline-primary chart-nav-btn" id="btn-mapping">Continuous Mapping</button>
      <button type="button" class="btn btn-outline-primary chart-nav-btn" id="btn-distribution">Distribution</button>
      <button type="button" class="btn btn-outline-primary chart-nav-btn" id="btn-table">Value Table</button>
    </div>
  </div>

  <!-- Value Space View -->
  <div id="view-value-space" class="chart-view">
    <p class="text-muted small">
      The 16 discrete MXFP4 values with their binary representations and bit structure.
    </p>
    <div id="value-space-chart" style="height: 400px; width: 100%; position: relative;">
      <canvas id="valueSpaceCanvas" style="width: 100%; height: 100%; display: block;"></canvas>
    </div>
  </div>

  <!-- Mapping View -->
  <div id="view-mapping" class="chart-view" style="display: none;">
    <div class="mb-4">
      <h5>Continuous to Discrete Mapping</h5>
      <p class="text-muted small">
        Shows how continuous input values map to discrete MXFP4 values.
        The step function illustrates the quantization process and quantization error.
      </p>
      <div style="height: 400px; width: 100%; position: relative;">
        <canvas id="mappingCanvas" style="width: 100%; height: 100%; display: block;"></canvas>
      </div>
    </div>

    <div class="mb-4">
      <h5>Quantization Error Analysis</h5>
      <p class="text-muted small">
        Error between original and quantized values. Notice larger errors in sparse regions.
      </p>
      <div style="height: 300px; width: 100%; position: relative;">
        <canvas id="errorCanvas" style="width: 100%; height: 100%; display: block;"></canvas>
      </div>
    </div>
  </div>

  <!-- Distribution View -->
  <div id="view-distribution" class="chart-view" style="display: none;">
    <h5>Neural Network Weight Distribution</h5>
    <p class="text-muted small">
      How normally distributed values (typical of neural network weights) map to MXFP4 values.
    </p>
    <div style="height: 500px; width: 100%; position: relative;">
      <canvas id="distributionCanvas" style="width: 100%; height: 100%; display: block;"></canvas>
    </div>

    <div class="mt-4 text-muted small">
      <p><strong>Key Observations:</strong></p>
      <ul>
        <li>Values near zero (0.0, ±0.5, ±1.0) have highest probability</li>
        <li>Extreme values (±4.0, ±6.0) are rarely used in practice</li>
        <li>Non-uniform spacing affects representation efficiency</li>
        <li>Distribution matches typical neural network weight patterns</li>
      </ul>
    </div>
  </div>

  <!-- Table View -->
  <div id="view-table" class="chart-view" style="display: none;">
    <h5>Complete MXFP4 Value Table</h5>
    <div id="valueTable" class="table-responsive">
      <!-- Table will be populated by JavaScript -->
    </div>
  </div>
</div>

## Advanced Quantization Process

### 1. Block Formation and Analysis

```python
def analyze_block(values, block_size=32):
    """Analyze a block for optimal MXFP4 quantization"""
    # Reshape into blocks
    blocks = values.reshape(-1, block_size)

    # Per-block statistics
    max_abs_per_block = np.max(np.abs(blocks), axis=1)
    mean_abs_per_block = np.mean(np.abs(blocks), axis=1)

    # Optimal scale factors (power of 2 for efficiency)
    optimal_scales = np.power(2, np.ceil(np.log2(max_abs_per_block / 6.0)))

    return blocks, max_abs_per_block, optimal_scales
```

### 2. Stochastic Rounding Implementation

One key innovation in MXFP4 is **stochastic rounding** to minimize quantization bias:

```python
def stochastic_quantize_mxfp4(value, scale_factor, use_stochastic=True):
    """Quantize a single value to MXFP4 with optional stochastic rounding"""
    scaled_value = value / scale_factor

    # Find the two nearest MXFP4 values
    mxfp4_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6]

    if abs(scaled_value) > 6:
        # Clamp to maximum representable value
        return 6.0 if scaled_value > 0 else -6.0

    # Find closest values
    distances = [(abs(scaled_value - mv), mv) for mv in mxfp4_values]
    distances.sort()

    closest_val = distances[0][1]

    if not use_stochastic or len(distances) < 2:
        return closest_val

    # Stochastic rounding between two closest values
    second_val = distances[1][1]
    closest_dist = distances[0][0]
    second_dist = distances[1][0]

    total_dist = closest_dist + second_dist
    if total_dist > 0:
        prob_second = closest_dist / total_dist
        return second_val if np.random.random() < prob_second else closest_val

    return closest_val
```

### 3. Hardware-Optimized Layout Conversion

Modern implementations convert MXFP4 to specialized hardware layouts:

```python
def convert_to_hopper_mx_layout(mxfp4_codes, block_size=32):
    """Convert MXFP4 codes to NVIDIA Hopper MX layout"""
    # Pack 4-bit codes into bytes (2 codes per byte)
    packed_codes = []
    for i in range(0, len(mxfp4_codes), 2):
        byte_val = (mxfp4_codes[i] << 4) | mxfp4_codes[i + 1]
        packed_codes.append(byte_val)

    # Organize into blocks with scale factors
    num_blocks = len(mxfp4_codes) // block_size
    blocks = np.array(packed_codes).reshape(num_blocks, block_size // 2)

    return blocks
```

## MXFP4 in Matrix Multiplication

Matrix multiplication is the computational backbone of neural networks. Let's explore how MXFP4 quantization is efficiently applied during matrix operations.

### Mixed Precision GEMM

The key insight is that we can perform matrix multiplication directly on quantized data:

```
C = (A_quantized × scale_A) @ (B_quantized × scale_B)
C = (A_quantized @ B_quantized) × (scale_A × scale_B)
```

### Block-wise Matrix Multiplication

<div id="matmul-visualization-wrapper" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; margin: 30px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
  <div id="matmul-visualization" style="background: white; border-radius: 10px; padding: 25px;">
    <h4 style="color: #333; margin-bottom: 20px; text-align: center;">🧮 MXFP4 Matrix Multiplication Visualization</h4>
    <div class="controls mb-3">
      <div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
        <button id="step-button" class="btn btn-primary">Step Forward</button>
        <button id="reset-button" class="btn btn-secondary">Reset</button>
        <button id="randomize-button" class="btn btn-info">Randomize Matrices</button>
        <button id="simulate-button" class="btn btn-success">Auto Simulate</button>
        <span style="margin-left: 15px; font-weight: 500;">
          Step: <span id="current-step">0</span> / <span id="total-steps">16</span>
        </span>
        <span style="margin-left: 15px; font-size: 12px; color: #666;">
          Speed: <input type="range" id="speed-slider" min="1" max="10" value="5" style="width: 80px;">
          <span id="speed-value">Medium</span>
        </span>
      </div>
    </div>
    <div id="matmul-canvas-container" style="height: 800px; width: 100%; position: relative;">
      <canvas id="matmulCanvas" style="width: 100%; height: 100%; display: block;"></canvas>
    </div>
  </div>
</div>

### Optimized Implementation

Here's how matrix multiplication works with MXFP4 tensors:

```python
def mxfp4_matrix_multiply(A_quantized, A_scales, B_quantized, B_scales):
    """
    Perform matrix multiplication on MXFP4 quantized tensors

    Args:
        A_quantized: MXFP4 codes for matrix A [M, K]
        A_scales: Scale factors for A [M // block_size]
        B_quantized: MXFP4 codes for matrix B [K, N]
        B_scales: Scale factors for B [K // block_size]
    """
    # Convert MXFP4 codes to actual values
    A_dequant = dequantize_mxfp4(A_quantized, A_scales)
    B_dequant = dequantize_mxfp4(B_quantized, B_scales)

    # Perform matrix multiplication
    C = A_dequant @ B_dequant

    return C

def dequantize_mxfp4(quantized_codes, scale_factors, block_size=32):
    """Dequantize MXFP4 codes back to floating point values"""
    mxfp4_lookup = {
        0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0,
        8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
    }

    # Map codes to MXFP4 values
    mxfp4_values = np.array([mxfp4_lookup[code] for code in quantized_codes.flatten()])
    mxfp4_values = mxfp4_values.reshape(quantized_codes.shape)

    # Apply scale factors block-wise
    result = np.zeros_like(mxfp4_values, dtype=np.float32)
    for i, scale in enumerate(scale_factors):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, mxfp4_values.shape[0])
        result[start_idx:end_idx] = mxfp4_values[start_idx:end_idx] * scale

    return result
```

### Performance Analysis

The computational benefits of MXFP4 matrix multiplication:

1. **Memory Bandwidth**: 4× reduction in data transfer
2. **Cache Efficiency**: More data fits in cache levels
3. **Specialized Hardware**: Native support in modern accelerators

<div id="performance-comparison" class="my-4">
  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 1; min-width: 300px;">
      <h5>Memory Usage Comparison</h5>
      <div style="height: 300px; width: 100%; position: relative;">
        <canvas id="memoryChart" style="width: 100%; height: 100%; display: block;"></canvas>
      </div>
    </div>
    <div style="flex: 1; min-width: 300px;">
      <h5>Throughput Analysis</h5>
      <div style="height: 300px; width: 100%; position: relative;">
        <canvas id="throughputChart" style="width: 100%; height: 100%; display: block;"></canvas>
      </div>
    </div>
  </div>
</div>

## MXFP4 in Mixture of Experts (MoE) Layers

MoE architectures benefit tremendously from MXFP4 quantization due to their massive parameter counts and sparse activation patterns.

### MoE Architecture Overview

A typical MoE layer consists of:
- **Router Network**: Selects which experts to activate
- **Expert Networks**: Specialized feedforward networks
- **Combining Logic**: Aggregates expert outputs

### MXFP4 MoE Implementation

Based on the GPT-OSS implementation, here's how MXFP4 is applied:

```python
class MXFP4_MoE_Layer:
    def __init__(self, config):
        self.num_experts = config.num_experts  # e.g., 128
        self.experts_per_token = config.experts_per_token  # e.g., 4
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Quantized expert weights
        self.expert_weights_quantized = {}
        self.expert_scales = {}

        self._initialize_experts()

    def _initialize_experts(self):
        """Initialize and quantize all expert weights"""
        for expert_id in range(self.num_experts):
            # Create expert weights (typically much larger than hidden_size)
            w1 = torch.randn(self.hidden_size, self.intermediate_size * 2)
            w2 = torch.randn(self.intermediate_size, self.hidden_size)

            # Quantize to MXFP4
            w1_quant, w1_scales = self.quantize_mx4(w1)
            w2_quant, w2_scales = self.quantize_mx4(w2)

            self.expert_weights_quantized[expert_id] = {
                'w1': w1_quant, 'w2': w2_quant
            }
            self.expert_scales[expert_id] = {
                'w1': w1_scales, 'w2': w2_scales
            }

    def quantize_mx4(self, weight_tensor, block_size=32):
        """Quantize weights using MXFP4 format"""
        # Transpose for optimal memory layout (as in GPT-OSS)
        w = weight_tensor.mT.contiguous()

        # Convert to bfloat16 first
        w = w.to(torch.bfloat16)

        # Apply microscaling quantization
        w_quantized, w_scales = self.downcast_to_mxfp(w, block_size)

        # Convert to hardware-optimized layout
        w_quantized = self.convert_to_mx_layout(w_quantized)

        return w_quantized, w_scales

    def forward(self, x, router_weights):
        """Forward pass through MoE layer"""
        batch_size, seq_len, hidden_size = x.shape

        # Router determines expert selection
        expert_indices, expert_weights = self.route_tokens(x, router_weights)

        # Process through selected experts
        outputs = []
        for expert_id in expert_indices:
            # Get quantized weights for this expert
            w1_quant = self.expert_weights_quantized[expert_id]['w1']
            w2_quant = self.expert_weights_quantized[expert_id]['w2']
            w1_scales = self.expert_scales[expert_id]['w1']
            w2_scales = self.expert_scales[expert_id]['w2']

            # Dequantize and compute (or use fused kernel)
            expert_output = self.expert_forward(x, w1_quant, w1_scales, w2_quant, w2_scales)
            outputs.append(expert_output)

        # Combine expert outputs
        final_output = self.combine_expert_outputs(outputs, expert_weights)
        return final_output

    def route_tokens(self, x, router_weights):
        """Token routing logic"""
        # Compute routing probabilities
        routing_logits = x @ router_weights  # [batch, seq, num_experts]
        routing_probs = torch.softmax(routing_logits, dim=-1)

        # Select top-k experts per token
        expert_weights, expert_indices = torch.topk(routing_probs, self.experts_per_token, dim=-1)

        return expert_indices, expert_weights
```

### MoE Memory Efficiency

The memory savings in MoE layers with MXFP4 are dramatic:

<div id="moe-efficiency" class="my-4">
  <h4>MoE Memory Usage: BF16 vs MXFP4</h4>
  <div class="controls mb-3">
    <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;">
      <div>
        <label for="num-experts-slider">Number of Experts: <span id="num-experts-value">128</span></label>
        <input type="range" id="num-experts-slider" min="8" max="512" value="128" style="width: 150px;">
      </div>
      <div>
        <label for="expert-size-slider">Expert Size (M params): <span id="expert-size-value">100</span></label>
        <input type="range" id="expert-size-slider" min="10" max="1000" value="100" style="width: 150px;">
      </div>
    </div>
  </div>
  <div id="moe-memory-chart" style="height: 400px; width: 100%; position: relative;">
    <canvas id="moeMemoryCanvas" style="width: 100%; height: 100%; display: block;"></canvas>
  </div>
</div>

### Real-World Example: GPT-OSS-120B

The GPT-OSS-120B model demonstrates MXFP4's power in MoE architectures:

- **Total Parameters**: 120B
- **Active Parameters per Token**: ~6B (due to MoE sparsity)
- **Memory with BF16**: ~240GB
- **Memory with MXFP4**: ~67GB (3.6× reduction)
- **Single H100 Capability**: Enabled by MXFP4 quantization

### MoE Visualization

<div id="moe-architecture-viz" class="my-4">
  <h4>Interactive MoE Layer Visualization</h4>
  <div class="controls mb-3">
    <button id="animate-moe" class="btn btn-success">Animate Token Flow</button>
    <button id="show-routing" class="btn btn-info">Show Routing</button>
    <button id="highlight-quantization" class="btn btn-warning">Highlight Quantization</button>
  </div>
  <div id="moe-diagram" style="height: 600px; width: 100%; position: relative;">
    <canvas id="moeDiagramCanvas" style="width: 100%; height: 100%; display: block;"></canvas>
  </div>
</div>

## Advanced Topics

### Triton Kernels for MXFP4

Modern implementations use Triton kernels for optimal GPU performance:

```python
import triton
import triton.language as tl

@triton.jit
def mxfp4_gemm_kernel(
    # Input pointers
    a_ptr, b_ptr, c_ptr,
    # Scale factor pointers
    scale_a_ptr, scale_b_ptr,
    # Matrix dimensions
    M, N, K,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Triton kernel for MXFP4 matrix multiplication"""
    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main computation loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A block (MXFP4)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=a_mask)

        # Load B block (MXFP4)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=b_mask)

        # Load scale factors
        scale_a = tl.load(scale_a_ptr + k)
        scale_b = tl.load(scale_b_ptr + k)

        # Dequantize and accumulate
        a_fp = dequantize_mxfp4(a) * scale_a
        b_fp = dequantize_mxfp4(b) * scale_b
        accumulator += tl.dot(a_fp, b_fp)

        # Update offsets for next iteration
        offs_k += BLOCK_SIZE_K

    # Store result
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], accumulator, mask=c_mask)

@triton.jit
def dequantize_mxfp4(codes):
    """Dequantize MXFP4 codes to floating point"""
    # MXFP4 lookup table
    values = tl.constexpr([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

    # Map codes to values (simplified - actual implementation more complex)
    return tl.gather(values, codes)
```

### Error Analysis and Mitigation

MXFP4 quantization introduces errors that can accumulate during training:

1. **Quantization Error**: Difference between original and quantized values
2. **Accumulation Error**: Errors that compound through network layers
3. **Gradient Mismatch**: Forward/backward pass precision differences

Mitigation strategies:
- **Stochastic Rounding**: Reduces quantization bias
- **Error Feedback**: Accumulates and corrects rounding errors
- **Mixed Precision**: Uses higher precision for critical computations

## Performance Benchmarks

### Memory Comparison

| Model Size | BF16 Memory | MXFP4 Memory | Reduction |
|------------|-------------|--------------|-----------|
| 7B params  | 14 GB       | 3.9 GB       | 3.6×      |
| 70B params | 140 GB      | 39 GB        | 3.6×      |
| 120B params| 240 GB      | 67 GB        | 3.6×      |

### Throughput Analysis

<div id="performance-metrics" class="my-4">
  <div style="display: flex; flex-wrap: wrap; gap: 20px;">
    <div style="flex: 1; min-width: 300px;">
      <h5>Training Throughput</h5>
      <div style="height: 300px; width: 100%; position: relative;">
        <canvas id="trainingThroughputChart" style="width: 100%; height: 100%; display: block;"></canvas>
      </div>
    </div>
    <div style="flex: 1; min-width: 300px;">
      <h5>Inference Latency</h5>
      <div style="height: 300px; width: 100%; position: relative;">
        <canvas id="inferenceLatencyChart" style="width: 100%; height: 100%; display: block;"></canvas>
      </div>
    </div>
  </div>
</div>

## Future Directions

### Hardware Support

- **NVIDIA Blackwell**: Native MXFP4 support
- **AMD RDNA4**: Expected MX format acceleration
- **Custom ASICs**: Optimized for microscaling formats

### Algorithm Improvements

1. **Adaptive Block Sizes**: Dynamic block sizing based on weight distributions
2. **Hierarchical Scaling**: Multi-level scaling for better precision
3. **Learned Quantization**: Neural networks that learn optimal quantization strategies

### Applications Beyond LLMs

- **Computer Vision**: ConvNets with MXFP4 weights
- **Recommendation Systems**: Large embedding tables
- **Scientific Computing**: High-performance linear algebra

## Conclusion

MXFP4 quantization represents a fundamental advancement in neural network compression. Its block-wise microscaling approach elegantly balances compression efficiency with computational accuracy, enabling the democratization of large-scale AI models.

Key takeaways:
- **3.6× memory reduction** with acceptable accuracy loss
- **Native hardware support** in modern accelerators
- **Particularly effective** for MoE architectures
- **Production-ready** with implementations in major frameworks

The combination of clever format design, stochastic rounding, and hardware optimization makes MXFP4 a compelling choice for the next generation of efficient AI systems.

---

*Try the interactive visualizations above to gain deeper insights into MXFP4's behavior across different scenarios. The format's non-uniform value distribution and block-wise scaling create fascinating trade-offs that are best understood through hands-on exploration.*

<!-- Custom styles for visualizations -->
<style>
.btn-group {
  display: flex;
  width: 100%;
  gap: 8px;
}

.btn {
  display: block;
  flex: 1;
  padding: 12px 8px;
  margin: 0;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.4;
  text-align: center;
  text-decoration: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
  cursor: pointer;
  border: 1px solid transparent;
  border-radius: 6px;
  transition: all 0.15s ease-in-out;
  user-select: none;
}

.btn-primary {
  color: #fff;
  background-color: #007bff;
  border-color: #007bff;
}

.btn-primary:hover {
  color: #fff;
  background-color: #0056b3;
  border-color: #004085;
}

.btn-outline-primary {
  color: #007bff;
  background-color: transparent;
  border-color: #007bff;
}

.btn-outline-primary:hover {
  color: #fff;
  background-color: #007bff;
  border-color: #007bff;
}

.btn-secondary {
  color: #fff;
  background-color: #6c757d;
  border-color: #6c757d;
}

.btn-secondary:hover {
  color: #fff;
  background-color: #545b62;
  border-color: #4e555b;
}

.btn-info {
  color: #fff;
  background-color: #17a2b8;
  border-color: #17a2b8;
}

.btn-info:hover {
  color: #fff;
  background-color: #138496;
  border-color: #117a8b;
}

.btn-success {
  color: #fff;
  background-color: #28a745;
  border-color: #28a745;
}

.btn-success:hover {
  color: #fff;
  background-color: #218838;
  border-color: #1e7e34;
}

.btn-warning {
  color: #212529;
  background-color: #ffc107;
  border-color: #ffc107;
}

.btn-warning:hover {
  color: #212529;
  background-color: #e0a800;
  border-color: #d39e00;
}

.chart-view {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
  margin-top: 15px;
}

.table-responsive {
  max-height: 500px;
  overflow-y: auto;
}

.badge {
  font-size: 0.7em;
}

.bg-success { background-color: #28a745 !important; }
.bg-warning { background-color: #ffc107 !important; color: #000 !important; }
.bg-danger { background-color: #dc3545 !important; }

@media (max-width: 768px) {
  .btn-group {
    flex-direction: column;
    gap: 4px;
  }

  .btn {
    font-size: 13px;
    padding: 10px 8px;
  }
}

@media (max-width: 480px) {
  .btn {
    font-size: 12px;
    padding: 8px 6px;
  }
}
</style>

<!-- Load visualization scripts -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


<script src="{{ '/assets/js/mxfp4-deep-dive-viz.js' | relative_url }}"></script>

