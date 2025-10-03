---
title: Understanding Grouped GEMM
description: >-
  Grouped GEMM and MoE architecture exploration with gpt-oss models
date: 2025-09-29
categories: [Blog]
tags: [AI, Machine Learning, Triton, GPU, Kernels]
pin: true
math: false
author: ks
---


Mixture of Experts (MoE) architectures have become a practical approach to scaling neural networks. These models route different inputs to different expert networks, allowing for larger total model capacity while keeping per-token computation manageable. However, implementing MoE efficiently requires solving a specific computational problem: performing multiple matrix multiplications where each multiplication has different dimensions.

This post examines Group GEMM (Grouped General Matrix Multiply) and its role in efficient MoE inference, with particular attention to how quantization techniques like MXFP4 reduce memory requirements. We'll analyze actual implementations from [OpenAI's GPT-OSS repository](https://github.com/openai/gpt-oss) and the [Triton Group GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html).

## The MoE Routing Problem

### Basic Architecture

In a standard MoE layer, instead of a single feed-forward network, you have multiple expert networks. For example, with 128 experts where each token is routed to the top 4:

```python
class MLPBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        self.num_experts = config.num_experts  # 128
        self.experts_per_token = config.experts_per_token  # 4
```


### How Routing Works

When a batch of tokens arrives at an MoE layer:

1. **Router computes scores**: A small gating network produces scores for each token-expert pair
2. **Top-k selection**: Each token selects its top-k experts based on scores
3. **Token grouping**: Tokens are implicitly grouped by their assigned experts
4. **Expert computation**: Each expert processes its assigned tokens

Here's the routing code from the PyTorch implementation:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    t = self.norm(x)
    g = self.gate(t)  # [num_tokens, num_experts] scores
    
    # Select top-4 experts per token
    experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices  # [num_tokens, 4]
```

For 100 tokens, `expert_indices` might look like:

```python
expert_indices = [
    [3, 17, 45, 92],   # Token 0 → Experts 3, 17, 45, 92
    [3, 8, 22, 101],   # Token 1 → Experts 3, 8, 22, 101
    [17, 45, 67, 88],  # Token 2 → Experts 17, 45, 67, 88
    # ...
]
```

### Interactive MoE Routing Visualization

The following visualization shows how tokens are routed through MLP Block 1 and MLP Block 2 in a Mixture of Experts architecture:

<div id="moe-routing-viz"></div>
<script src="/assets/js/moe-routing-visualizer.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        new MoERoutingVisualizer('moe-routing-viz');
    });
</script>

### The Computational Challenge

After routing, different experts receive different numbers of tokens:

- Expert 1: 35 tokens → needs (35 × 512) × (512 × 2048) GEMM
- Expert 2: 28 tokens → needs (28 × 512) × (512 × 2048) GEMM  
- Expert 3: 37 tokens → needs (37 × 512) × (512 × 2048) GEMM

Each multiplication has the same weight dimensions but different batch sizes. This is where Group GEMM becomes essential.

## What is Group GEMM?

### Definition

Group GEMM performs multiple independent matrix multiplications in a single batched operation, where each multiplication can have different dimensions.

**Comparison:**

- **Standard GEMM**: `C = A × B` for single matrix pair
- **Batched GEMM**: Multiple multiplications with same dimensions
- **Group GEMM**: Multiple multiplications with different dimensions

### Why Not Just Use Multiple Kernel Launches?

Launching separate GEMM kernels for each expert has significant overhead:

```python
# Naive approach: 128 separate kernel launches
for expert_id in range(128):
    tokens = get_tokens_for_expert(expert_id)
    if len(tokens) > 0:
        output = torch.matmul(tokens, expert_weights[expert_id])
```

**Costs:**
- Kernel launch overhead: ~5-10 μs per launch
- GPU idle time between launches
- No opportunity to share resources across experts
- Poor utilization when experts have few tokens

For 128 experts, this adds ~1 ms of pure overhead per layer, plus serialization inefficiencies.

## Triton Group GEMM Implementation

The [Triton Group GEMM kernel](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) provides an efficient implementation using static on-device scheduling.

### Key Design: Fixed Number of Thread Blocks

```python
grid = lambda META: (META['NUM_SM'], )
```

The kernel launches exactly NUM_SM thread blocks (CTAs), typically matching the number of streaming multiprocessors on the GPU (e.g., 108 for A100). This is fundamentally different from standard GEMM kernels that launch `(M/BLOCK_M) × (N/BLOCK_N)` blocks.

### Work Distribution Algorithm

```python
@triton.jit
def grouped_matmul_kernel(
    group_a_ptrs,      # Pointers to A matrices
    group_b_ptrs,      # Pointers to B matrices
    group_c_ptrs,      # Pointers to C matrices
    group_gemm_sizes,  # (M, N, K) for each GEMM
    g_lds,             # Leading dimensions
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        
        num_tiles = (gm // BLOCK_SIZE_M) * (gn // BLOCK_SIZE_N)
        
        while (tile_idx >= last_problem_end and 
               tile_idx < last_problem_end + num_tiles):
            # Compute this tile
            # ...
            tile_idx += NUM_SM  # Strided access pattern
        
        last_problem_end += num_tiles
```

### Work Distribution Example

Consider 4 GEMMs with tile counts:
- GEMM 0: 64 tiles (positions 0-63)
- GEMM 1: 16 tiles (positions 64-79)
- GEMM 2: 4 tiles (positions 80-83)
- GEMM 3: 1 tile (position 84)

With NUM_SM = 108:

- **CTA 0**: Processes tile 0 (from GEMM 0), then tile_idx becomes 108 (done)
- **CTA 10**: Processes tile 10 (from GEMM 0), then done
- **CTA 64**: Processes tile 64 (tile 0 of GEMM 1), then done
- **CTA 70**: Processes tile 70 (tile 6 of GEMM 1), then done

Each CTA picks up tiles in a strided pattern, ensuring work is distributed even when GEMMs have very different sizes.

### The Core GEMM Loop

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
    tl.multiple_of(a_ptrs, [16, 16])  # Alignment hints for vectorization
    tl.multiple_of(b_ptrs, [16, 16])
    
    a = tl.load(a_ptrs)  # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    b = tl.load(b_ptrs)  # [BLOCK_SIZE_K, BLOCK_SIZE_N]
    accumulator += tl.dot(a, b)  # Tensor Core operation
    
    a_ptrs += BLOCK_SIZE_K
    b_ptrs += BLOCK_SIZE_K * ldb
```

The `tl.dot()` operation compiles to Tensor Core instructions on NVIDIA GPUs, providing high throughput for the matrix multiplication.

## Hardware Optimization Details

### Tensor Core Utilization

Modern NVIDIA GPUs include specialized Tensor Cores for matrix operations:

- **A100**: 16×8×16 matrix operations (FP16 inputs, FP32 accumulator)
- **H100**: 16×16×16 or larger operations

The tile size `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32` is chosen to maximize Tensor Core utilization while fitting in shared memory.

### Memory Bandwidth Analysis

For a 128×128 tile computation:

**Memory traffic per tile:**
- Load A: 128×32×2 bytes = 8 KB
- Load B: 32×128×2 bytes = 8 KB  
- Store C: 128×128×2 bytes = 32 KB
- Total: 48 KB

**Compute:**
- Operations: 128×128×(2×32) = 1,048,576 FLOPs

**Arithmetic intensity:** 1,048,576 FLOPs / 48 KB = 21.8 FLOPs/byte

This is compute-bound on modern GPUs (A100 needs ~14 TB/s for peak performance, has ~2 TB/s HBM bandwidth), which is ideal.

### Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 
                       'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
                       'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
        # ...
    ],
    key=['group_size'],
)
```

The optimal configuration depends on the problem characteristics:

- **Small GEMMs**: Fewer CTAs (NUM_SM=64-84) avoids idle threads
- **Large GEMMs**: More CTAs (NUM_SM=128+) improves latency hiding through oversubscription

## MXFP4 Quantization

The [OpenAI GPT-OSS quantized implementation](https://github.com/openai/gpt-oss) uses MXFP4 format to reduce memory requirements by approximately 4×.

### What is MXFP4?

MXFP4 (Microscaling FP4) is a block-based quantization format from the [OCP Microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):

- Tensors divided into blocks of 32 elements
- Each block shares a single 8-bit exponent (E8M0 format)
- Each element stored as 4-bit E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit)

**Storage:** (32 elements × 4 bits + 8 bits scale) / 32 = 4.25 bits per element

This compares to 16 bits per element for BF16, providing approximately 3.76× compression.

### Memory Impact for MoE

For a model with 128 experts, 36 layers, and hidden size 2880:

**Regular (BF16) per layer:**
- mlp1_weight: 128 × 2880 × 5760 × 2 bytes = 4.24 GB
- mlp2_weight: 128 × 5760 × 2880 × 2 bytes = 4.24 GB  
- Total: 8.48 GB per layer

**Quantized (MXFP4) per layer:**
- mlp1_weight: ~1.06 GB
- mlp2_weight: ~1.06 GB
- Total: ~2.12 GB per layer

**Model-wide savings:** (8.48 - 2.12) GB × 36 layers = 229 GB

### Quantization Implementation

```python
self.mlp1_weight_tensor, self.mlp1_weight_mx = quantize_mx4(
    torch.empty(
        (config.num_experts, config.hidden_size, config.intermediate_size * 2),
        device=device,
        dtype=torch.bfloat16,
    ),
)
self.mlp1_weight = torch.nn.Parameter(
    self.mlp1_weight_tensor.storage.data, 
    requires_grad=False
)
```

The `quantize_mx4` function returns:
- `mlp1_weight_tensor`: Packed 4-bit values
- `mlp1_weight_mx`: 8-bit scale factors

### Fused Kernel Implementation

The quantized version uses a single fused Triton kernel:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    t = self.norm(x).view(batch_size * n_ctx, dim)
    
    t = moe(
        t,
        self.gate["weight"],
        self.mlp1_weight_tensor, self.mlp1_weight_mx,  # Quantized
        self.mlp2_weight_tensor, self.mlp2_weight_mx,  # Quantized
        self.gate["bias"].float(),
        self.mlp1_bias.float(),
        self.mlp2_bias.float(),
        experts_per_token=self.experts_per_token,
        num_experts=self.num_experts,
        swiglu_limit=self.swiglu_limit,
    )
    
    return x + t.view(batch_size, n_ctx, dim)
```

This single kernel performs:
1. Gating and top-k expert selection
2. On-the-fly MXFP4 dequantization
3. Group GEMM for first expert layer
4. SwiGLU activation
5. Group GEMM for second expert layer  
6. Weighted combination of expert outputs

### On-the-fly Dequantization

Inside the kernel, weights are dequantized as they're loaded:

```python
# Conceptual operation inside the kernel
for block in weight_blocks:
    scale = load_scale(weight_mx, block_idx)  # Load 8-bit scale
    packed_values = load_packed_4bit(weight_tensor, block_idx)
    
    for i in range(32):
        fp4_value = extract_4bits(packed_values, i)
        bf16_value = dequantize_e2m1(fp4_value, scale)
        # Use immediately in computation
```

This approach provides several benefits:
- Reduced memory bandwidth (4.25 bits vs 16 bits per weight)
- Better cache utilization (more weights fit in GPU cache)
- Minimal computational overhead (dequantization is fast)

### Accuracy Considerations

Research on microscaling formats has shown that MXFP4 maintains model quality through its fine-grained per-block quantization. Each block of 32 elements having its own scale factor helps capture local variations in weight magnitudes while keeping quantization error low.

## Performance Analysis

### Bandwidth Savings

With MXFP4, the memory bandwidth requirement drops by approximately 3.76×:

**BF16 baseline:**
- Per-token: 4 experts × 2 layers × (2880 × 5760 × 2 bytes) ≈ 264 MB
- Bandwidth limited: At 2 TB/s, ~7.6k tokens/second maximum

**MXFP4:**  
- Per-token: 4 experts × 2 layers × (2880 × 5760 × 0.53 bytes) ≈ 70 MB
- Bandwidth limited: At 2 TB/s, ~28.6k tokens/second maximum

This 3.76× improvement in theoretical throughput translates to real speedups when memory bandwidth is the bottleneck.

### Kernel Fusion Benefits

Comparing the PyTorch and Triton implementations:

**PyTorch (torch/model.py):**
- Separate operations: gating, indexing, einsum, activation, combining
- Multiple kernel launches
- Weights in BF16

**Triton (triton/model.py):**
- Single fused kernel
- All operations in one launch
- On-the-fly dequantization
- Optimized Group GEMM

The fused approach reduces kernel launch overhead and improves data locality, leading to better GPU utilization.

## Implementation Comparison

### PyTorch Version

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    t = self.norm(x)
    g = self.gate(t)
    experts = torch.topk(g, k=self.experts_per_token, dim=-1)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices

    # Index into weights
    mlp1_weight = self.mlp1_weight[expert_indices, ...]
    mlp1_bias = self.mlp1_bias[expert_indices, ...]
    
    # First layer
    t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    t = swiglu(t, limit=self.swiglu_limit)

    # Second layer
    mlp2_weight = self.mlp2_weight[expert_indices, ...]
    mlp2_bias = self.mlp2_bias[expert_indices, ...]
    t = torch.einsum("beck,bek->bec", mlp2_weight, t)
    t += mlp2_bias

    # Combine expert outputs
    t = torch.einsum("bec,be->bc", t, expert_weights)
    return x + t
```

This implementation uses einsums which PyTorch optimizes internally, but still involves multiple kernel launches and stores full BF16 weights.

### Triton Quantized Version

The Triton version replaces all of the above with a single `moe()` kernel call that handles routing, dequantization, computation, and combining. The kernel can be found in the triton_kernels package.

## Practical Considerations

### When to Use Group GEMM

Group GEMM provides the most benefit when:

1. You have multiple GEMMs with varying dimensions
2. Individual GEMMs are small enough that kernel launch overhead matters
3. Total work is large enough to utilize all SMs
4. You can fuse operations into the Group GEMM kernel

For MoE models, all these conditions typically hold.

### Trade-offs with Quantization

MXFP4 quantization provides:

**Benefits:**
- 3.76× memory reduction
- 3.76× bandwidth reduction  
- Faster inference on bandwidth-bound workloads
- More experts can fit in GPU memory

**Considerations:**
- Requires custom kernels for efficient dequantization
- Block size (32 elements) affects granularity
- Some accuracy loss (typically minimal with proper training)

## Conclusion

Group GEMM and MXFP4 quantization address complementary challenges in MoE inference. Group GEMM solves the scheduling problem of executing many variable-sized matrix multiplications efficiently, while MXFP4 reduces the memory footprint and bandwidth requirements of storing expert weights.

The Triton implementation demonstrates how these techniques can be combined into a single fused kernel, achieving better performance than launching separate operations. For the 128-expert configuration examined here, the quantized version reduces memory usage by approximately 229 GB across 36 layers while maintaining computational efficiency through on-the-fly dequantization.

These optimizations are particularly relevant as models continue to scale. The ability to fit more experts in memory and process tokens more quickly enables practical deployment of larger MoE models.

## References

- [OpenAI GPT-OSS Repository](https://github.com/openai/gpt-oss)
- [Triton Group GEMM Tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Microscaling Data Formats for Deep Learning (Paper)](https://arxiv.org/pdf/2310.10537)