---
title: "Explore GEMMs - optimization journey"
description: >-
  CUDA matrix multiplication optimization explorations
date: 2025-10-01
categories: [Blog]
tags: [Triton, CUDA, GPU, GEMM, Performance]
pin: true
math: true
author: ks
---

Matrix multiplication (GEMM - General Matrix Multiply) is one of the most fundamental operations in deep learning. Understanding how to optimize GEMM kernels on GPUs provides deep insights into GPU architecture, memory hierarchies, and parallel computing principles.

If the reader is remotely interested in CUDA or Triton programmin, they have likely come across [this gemm of a post](https://siboehm.com/articles/22/CUDA-MMM) -- pun intended -- by Simon Boehm. We'll build on this and walk through several optimization stages, each with some interactive visualizations to help understand the concepts. In addition, we will try to do most of the work in triton instead.

## GEMM Basics

GEMM (General Matrix Multiply) is a fundamental operation defined as:

$$C = \alpha AB + \beta C$$

where:
- $A$ is an $M \times K$ matrix
- $B$ is a $K \times N$ matrix
- $C$ is an $M \times N$ matrix (both input and output)
- $\alpha$ and $\beta$ are scalar coefficients

**Key Points:**
- The standard matrix product $C = AB$ is a special case with $\alpha = 1$ and $\beta = 0$
- When $\beta \neq 0$, GEMM accumulates into a pre-existing matrix $C$
- This formulation enables fused operations, avoiding separate kernel launches

### Computational Complexity

Each element $C[i,j]$ requires a dot product:

$$C[i,j] = \alpha \sum_{k=0}^{K-1} A[i,k] \times B[k,j] + \beta C[i,j]$$

For matrices of size $M \times K$, $K \times N$:
- Total dot products: $M \times N$
- Operations per dot product: $2K$ (K multiplies + K adds) + 3 scalar ops ($\alpha$, $\beta$, addition of $M \times N$ matrix)
- **Total FLOPs**: $2MNK + MK$ (dominated by dot products)

For a $4096 \times 4096$ matrix multiplication ($M = N = K = 4096$):
- Total operations: $2 \times 4096^3 \approx 137$ GFLOPs
- Memory required: $3 \times 4096^2 \times 4$ bytes $\approx$ 201 MB (float32)
- **Arithmetic Intensity**: $\frac{137 \text{ GFLOPs}}{201 \text{ MB}} \approx 682$ FLOPs/byte

## Code Repository

Full implementation of all kernels discussed in this post:

<div class="github-card" data-user="gpusgobrr" data-repo="explore-gemm" data-width="100%" data-height="" data-theme="default"></div>

## Hardware Specifications

All benchmarks in this post were run on an **NVIDIA GeForce RTX 4090** 🚀. Below are the key specifications:

<div style="overflow-x: auto; margin: 20px 0;">
  <table style="width: 100%; border-collapse: collapse; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 10px; overflow: hidden;">
    <thead>
      <tr style="background: linear-gradient(90deg, #76b900 0%, #53a401 100%);">
        <th style="padding: 15px; text-align: left; color: white; font-weight: 600; border-bottom: 2px solid #76b900;">⚡ Specification</th>
        <th style="padding: 15px; text-align: left; color: white; font-weight: 600; border-bottom: 2px solid #76b900;">🎯 RTX 4090 (Ada Lovelace)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background: rgba(118, 185, 0, 0.05);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🏗️ <strong>Architecture</strong></td>
        <td style="padding: 12px; color: #76b900; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">Ada Lovelace (TSMC 4nm)</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🔢 <strong>CUDA Cores</strong></td>
        <td style="padding: 12px; color: #4fc3f7; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">16,384</td>
      </tr>
      <tr style="background: rgba(118, 185, 0, 0.05);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🎛️ <strong>Streaming Multiprocessors (SMs)</strong></td>
        <td style="padding: 12px; color: #4fc3f7; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">128</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">⚡ <strong>GPU Boost Clock</strong></td>
        <td style="padding: 12px; color: #ffb74d; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">2,520 MHz</td>
      </tr>
      <tr style="background: rgba(255, 193, 7, 0.1);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">💪 <strong>FP32 Performance</strong></td>
        <td style="padding: 12px; color: #ffc107; font-weight: 700; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.1);">82.6 TFLOPS 🔥</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🧠 <strong>Tensor Cores</strong></td>
        <td style="padding: 12px; color: #ba68c8; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">512 (4th Gen)</td>
      </tr>
      <tr style="background: rgba(118, 185, 0, 0.05);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🚀 <strong>Tensor Performance (FP8)</strong></td>
        <td style="padding: 12px; color: #ba68c8; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">660.6 TFLOPS</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">✨ <strong>RT Cores</strong></td>
        <td style="padding: 12px; color: #81c784; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">128 (3rd Gen)</td>
      </tr>
      <tr style="background: rgba(33, 150, 243, 0.1);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">💾 <strong>Memory Size</strong></td>
        <td style="padding: 12px; color: #42a5f5; font-weight: 700; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.1);">24 GB GDDR6X</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">⏱️ <strong>Memory Clock</strong></td>
        <td style="padding: 12px; color: #42a5f5; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">21 Gbps</td>
      </tr>
      <tr style="background: rgba(33, 150, 243, 0.1);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🌊 <strong>Memory Bandwidth</strong></td>
        <td style="padding: 12px; color: #42a5f5; font-weight: 700; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.1);">1,008 GB/s 💨</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">📦 <strong>L1 Cache / Shared Memory (Total)</strong></td>
        <td style="padding: 12px; color: #ff7043; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">16,384 KB (16 MB)</td>
      </tr>
      <tr style="background: rgba(118, 185, 0, 0.05);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🗄️ <strong>L2 Cache</strong></td>
        <td style="padding: 12px; color: #ff7043; font-weight: 700; font-size: 16px; border-bottom: 1px solid rgba(255,255,255,0.1);">72 MB 📈</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🔧 <strong>Shared Memory per SM</strong></td>
        <td style="padding: 12px; color: #ff7043; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">128 KB</td>
      </tr>
      <tr style="background: rgba(118, 185, 0, 0.05);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">📝 <strong>Registers per SM</strong></td>
        <td style="padding: 12px; color: #ff7043; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">256 KB</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">⚡ <strong>TGP</strong></td>
        <td style="padding: 12px; color: #f57c00; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">450 W</td>
      </tr>
      <tr style="background: rgba(118, 185, 0, 0.05);">
        <td style="padding: 12px; color: #e0e0e0; border-bottom: 1px solid rgba(255,255,255,0.1);">🔬 <strong>Transistor Count</strong></td>
        <td style="padding: 12px; color: #9575cd; font-weight: 600; border-bottom: 1px solid rgba(255,255,255,0.1);">76.3 Billion</td>
      </tr>
      <tr>
        <td style="padding: 12px; color: #e0e0e0;">📏 <strong>Die Size</strong></td>
        <td style="padding: 12px; color: #9575cd; font-weight: 600;">608.5 mm²</td>
      </tr>
    </tbody>
  </table>
</div>

> **Key Takeaways for GEMM Performance:**
> - **128 SMs** with 128 KB shared memory each = massive parallelism opportunity
> - **82.6 TFLOPS FP32** theoretical peak (our target to approach)
> - **1,008 GB/s** memory bandwidth = need to minimize global memory access
> - **72 MB L2 cache** = excellent data reuse potential
> - **16,384 KB total L1/shared memory** across all SMs = critical for tiling strategies
{: .prompt-tip}

### SM Architecture

![SM architecture](/assets/4090_sm.png)

*Source: [NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)*

## Kernel 1: Naive Implementation

### Concept

The simplest approach to calculate GEMM assign each thread to compute one output element. Here is a naive implementation for matrix maltiply GEMM kernel

```c
__global__ void sgemm_naive(int M, int N, int K,
                            float alpha, const float *A,
                            const float *B, float beta,
                            float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
```

#### Caller

We operate on torch Tensors directly to call the above kernel:

```cpp

void sgemm_naive(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                 torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch: 16x16 threads per block
    constexpr int threads_per_block = 32;
    dim3 block_dim(threads_per_block, threads_per_block);
    dim3 grid_dim(CEIL_DIV(num_rows_a, threads_per_block),
                  CEIL_DIV(num_cols_b, threads_per_block));

    // Launch kernel
    sgemm_naive_kernel<<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
```


> **What does the memory access pattern look like in this naive case?** 
>
> At a high level, each thread independently:
> 1. Loads one row of $A$ (K elements)
> 2. Loads one column of $B$ (K elements)
> 3. Computes dot product
> 4. Writes one element to $C$
{: .prompt-tip}

Let's look an interactive visualization of it below

<div id="naive-viz"></div>

> **Problem**: Threads access memory in a scattered, non-coalesced pattern.
{: .prompt-warning}

I ran the naive kernel and pytorch gemm kernels on my RTX 4090 to experiment. Let's see the results below:

```
2025-10-07 07:11:33.403 | INFO     | __main__:run_benchmarks:745 - 📊 Comparison (baseline: PyTorch)
2025-10-07 07:11:33.404 | INFO     | __main__:run_benchmarks:746 - ────────────────────────────────────────────────────────────────────────────────
2025-10-07 07:11:33.404 | INFO     | __main__:run_benchmarks:770 - PyTorch             : 1.00× (baseline) 🎯
2025-10-07 07:11:33.404 | INFO     | __main__:run_benchmarks:773 - CUDA Naive          : 0.01× 🐢
2025-10-07 07:11:33.404 | INFO     | __main__:run_benchmarks:775 - 

2025-10-07 07:11:33.406 | INFO     | __main__:run_benchmarks:655 - ================================================================================
2025-10-07 07:11:33.406 | INFO     | __main__:run_benchmarks:656 - 📐 Matrix dimensions: (4096, 4096) @ (4096, 4096) = (4096, 4096)
2025-10-07 07:11:33.406 | INFO     | __main__:run_benchmarks:661 - 💾 Expected memory usage: 0.20 GB (0.07 GB per matrix)
2025-10-07 07:11:33.406 | INFO     | __main__:run_benchmarks:664 - ================================================================================
2025-10-07 07:11:33.406 | INFO     | __main__:run_benchmarks:700 - 
🔵 Benchmarking PyTorch...
2025-10-07 07:11:33.599 | INFO     | __main__:run_benchmarks:713 -    ⏱️  Time: 1.6782 ms (min: 1.6128, max: 2.0460)
2025-10-07 07:11:33.599 | SUCCESS  | __main__:run_benchmarks:716 -    💪 Performance: 81.90 TFLOPS
2025-10-07 07:11:33.599 | SUCCESS  | __main__:run_benchmarks:717 -    🌊 Bandwidth: 119.97 GB/s
2025-10-07 07:11:33.599 | INFO     | __main__:run_benchmarks:700 - 
🔴 Benchmarking CUDA Naive...
2025-10-07 07:11:58.810 | INFO     | __main__:run_benchmarks:713 -    ⏱️  Time: 222.9623 ms (min: 219.4291, max: 232.7135)
2025-10-07 07:11:58.810 | SUCCESS  | __main__:run_benchmarks:716 -    💪 Performance: 0.62 TFLOPS
2025-10-07 07:11:58.810 | SUCCESS  | __main__:run_benchmarks:717 -    🌊 Bandwidth: 0.90 GB/s
2025-10-07 07:11:58.810 | INFO     | __main__:run_benchmarks:744 - 
```
> For M = N = K = 4096:
> - The naive CUDA kernel is 133× slower than PyTorch (0.01× speedup)
> - Achieves only 0.76% of PyTorch's TFLOPS
> - Bandwidth utilization is 133× worse (0.90 GB/s vs 119.97 GB/s)
{: .prompt-info}

### Benchmark Results

Below are the full benchmark results comparing the naive CUDA kernel against PyTorch's optimized GEMM implementation across different shapes:

![Naive Only](/assets/explore_gemm_naive_only.png)

As we can see, the naive implementation is significantly slower than PyTorch's optimized kernel, achieving only ~1% of PyTorch's performance. 


## Kernel 2: Global Memory Coalescing

### Why Memory Coalescing Matters

#### The Hardware Perspective

To understand memory coalescing, we need to understand GPU execution hierarchy:

**Threads → Warps → Thread Blocks → Streaming Multiprocessors (SMs)**

1. **Threads**: Individual execution units in your CUDA kernel
2. **Warps**: Groups of 32 threads that execute the same instruction simultaneously (SIMT - Single Instruction, Multiple Thread)
3. **Thread Blocks**: Logical groupings of threads (up to 1024 threads) that share resources and can synchronize
4. **Streaming Multiprocessors (SMs)**: The physical processors on the GPU that execute thread blocks

> A modern GPU like the H100 has **144 SMs**, each capable of running up to **2,048 threads concurrently** (64 warps). Each SM has **4 warp schedulers** that issue instructions to warps every clock cycle.
{: .prompt-tip}

> On my RTX 4090, I have 128 SMs and it allows **1024 threads concurrently** (32 warps).
{: .prompt-info}

> The key insight: **warps are the fundamental unit of execution**. All 32 threads in a warp execute the same instruction at the same time. When these threads access consecutive memory addresses, the hardware can combine their memory requests into a single transaction. Modern GPU DRAM systems can fetch large contiguous blocks (32B, 64B, or 128B cache lines) in one transaction. Without coalescing, the same 32 accesses could require 32 separate transactions → **32x increase in memory traffic** ⚠️.
{: .prompt-tip}


**Why SMs matter for GEMM**:
- Each SM has limited resources (registers, shared memory)
- Multiple thread blocks compete for these resources
- SMs can switch between warps in a **single clock cycle** (1000x faster than CPU context switches)
- This enables **latency hiding**: while one warp waits for memory, another executes
- GEMM efficiency depends on keeping all warp schedulers busy with coalesced memory access patterns

**Coalescing Example**:
- 32 threads each load a 4-byte float
- Total data: 128 bytes
- **Coalesced**: 1 memory transaction (128-byte cache line)
- **Uncoalesced**: Up to 32 separate transactions

### Concept

Now that we have general hardware fundatmentals, we can see that the naive kernel's memory access pattern is inefficient. When threads in a warp access scattered memory locations, each access requires a separate memory transaction. Memory coalescing restructures the thread-to-output mapping so threads in the same warp access consecutive memory locations, enabling the hardware to combine multiple accesses into a single transaction.

#### Thread-to-Output Remapping

```cuda
template <const uint block_size>
__global__ void sgemm_global_mem_coalesce_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                                 float alpha, const float *matrix_a,
                                                 const float *matrix_b, float beta, float *matrix_c)
{
    // Map 1D thread ID to 2D output position for coalesced memory access
    // *** KEY CHANGE wrt NAIVE kernel
    const int output_row = blockIdx.x * block_size + (threadIdx.x / block_size);
    const int output_col = blockIdx.y * block_size + (threadIdx.x % block_size);
    // *** KEY CHANGE wrt NAIVE kernel

    // Boundary check for non-multiple of block size
    if (output_row < num_rows_a && output_col < num_cols_b)
    {
        float accumulator = 0.0f;
        for (int k_idx = 0; k_idx < num_cols_a; ++k_idx)
        {
            accumulator += matrix_a[output_row * num_cols_a + k_idx] *
                          matrix_b[k_idx * num_cols_b + output_col];
        }
        const int output_idx = output_row * num_cols_b + output_col;
        matrix_c[output_idx] = alpha * accumulator + beta * matrix_c[output_idx];
    }
}

```

### Caller

```cpp
void sgemm_global_mem_coalesce(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                               torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch: 1D blocks with block_size^2 threads (32x32 = 1024 threads per block)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size);
    dim3 grid_dim(CEIL_DIV(num_rows_a, block_size),
                  CEIL_DIV(num_cols_b, block_size));

    // Launch kernel
    sgemm_global_mem_coalesce_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
```

The key change from the naive kernel:

Threads with consecutive `threadIdx.x` now access consecutive elements in the same row of A—enabling coalescing.

### Memory Access Visualization

Let's visualize how the coalesced kernel accesses memory during matrix multiplication. Notice how threads in the same warp now access the **same row** of matrix A, enabling memory coalescing:

<div id="coalesced-matrix-viz"></div>

<div id="index-transform-viz"></div>

#### The Performance Impact

Just like the naive version, we ran a benchmark for N = M = K = 4096 to get the FLOPs and memory bandwidth numbers.

```
────────────────────────────────────────────────────────────────────────────────
2025-10-07 10:41:26.466 | INFO     | __main__:run_benchmarks:745 - 📊 Comparison (baseline: PyTorch)
2025-10-07 10:41:26.466 | INFO     | __main__:run_benchmarks:746 - ────────────────────────────────────────────────────────────────────────────────
2025-10-07 10:41:26.466 | INFO     | __main__:run_benchmarks:770 - PyTorch             : 1.00× (baseline) 🎯
2025-10-07 10:41:26.466 | INFO     | __main__:run_benchmarks:773 - CUDA Naive          : 0.01× 🐢
2025-10-07 10:41:26.466 | INFO     | __main__:run_benchmarks:773 - CUDA Coalesced      : 0.07× 🐢
2025-10-07 10:41:26.466 | INFO     | __main__:run_benchmarks:775 - 

2025-10-07 10:41:26.468 | INFO     | __main__:run_benchmarks:655 - ================================================================================
2025-10-07 10:41:26.468 | INFO     | __main__:run_benchmarks:656 - 📐 Matrix dimensions: (4096, 4096) @ (4096, 4096) = (4096, 4096)
2025-10-07 10:41:26.468 | INFO     | __main__:run_benchmarks:661 - 💾 Expected memory usage: 0.20 GB (0.07 GB per matrix)
2025-10-07 10:41:26.468 | INFO     | __main__:run_benchmarks:664 - ================================================================================
2025-10-07 10:41:26.469 | INFO     | __main__:run_benchmarks:700 - 
🔵 Benchmarking PyTorch...
2025-10-07 10:41:26.663 | INFO     | __main__:run_benchmarks:713 -    ⏱️  Time: 1.6991 ms (min: 1.6189, max: 1.9333)
2025-10-07 10:41:26.663 | SUCCESS  | __main__:run_benchmarks:716 -    💪 Performance: 80.89 TFLOPS
2025-10-07 10:41:26.663 | SUCCESS  | __main__:run_benchmarks:717 -    🌊 Bandwidth: 118.49 GB/s
2025-10-07 10:41:26.663 | INFO     | __main__:run_benchmarks:700 - 
🔴 Benchmarking CUDA Naive...
2025-10-07 10:41:51.896 | INFO     | __main__:run_benchmarks:713 -    ⏱️  Time: 222.4662 ms (min: 219.6237, max: 235.8671)
2025-10-07 10:41:51.896 | SUCCESS  | __main__:run_benchmarks:716 -    💪 Performance: 0.62 TFLOPS
2025-10-07 10:41:51.896 | SUCCESS  | __main__:run_benchmarks:717 -    🌊 Bandwidth: 0.90 GB/s
2025-10-07 10:41:51.896 | INFO     | __main__:run_benchmarks:700 - 
🟢 Benchmarking CUDA Coalesced...
2025-10-07 10:41:55.180 | INFO     | __main__:run_benchmarks:713 -    ⏱️  Time: 29.0288 ms (min: 28.4549, max: 29.5383)
2025-10-07 10:41:55.180 | SUCCESS  | __main__:run_benchmarks:716 -    💪 Performance: 4.73 TFLOPS
2025-10-07 10:41:55.180 | SUCCESS  | __main__:run_benchmarks:717 -    🌊 Bandwidth: 6.94 GB/s
2025-10-07 10:41:55.180 | INFO     | __main__:run_benchmarks:744 - 
────────────────────────────────────────────────────────────────────────────────

```

> **Throughput:**
> - 7.63× TFLOPS improvement (0.62 → 4.73 TFLOPS)
> - Still only 5.8% of PyTorch's performance, but a significant step forward
> 
> **Bandwidth:**
> - 7.71× bandwidth improvement (0.90 → 6.94 GB/s)
> - Better memory utilization through coalesced access patterns
{: .prompt-info}


Below are the full benchmark results comparing the naive CUDA kernel against PyTorch's optimized GEMM implementation across different shapes:

![Global coalesced](/assets/explore_gemm_global_coalesced.png)

We can see that performance improved but still way slower than the pytorch version.

## Kernel 3: Shared Memory Caching

### GPU Memory Hierarchy

Before diving into shared memory optimization, let's understand the RTX 4090's memory hierarchy and why shared memory is so critical for performance:

<div id="memory-hierarchy-viz"></div>


Shared memory provides **bandwidth advantage** compared to global memory:

| Memory Type | Bandwidth | Latency | Location |
|------------|-----------|---------|----------|
| **Global Memory (GDDR6X)** | ~1 TB/s | 400-800 cycles | Off-chip |
| **Shared Memory (L1)** | ~14 TB/s | 20-30 cycles | On-chip |


Even with coalesced access, both the naive and coalesced kernels repeatedly read the same data from global memory:
- Each element of matrix $A$ is read $N$ times (once per column of $B$)
- Each element of matrix $B$ is read $M$ times (once per row of $A$)
- For a 1024×1024 matrix multiplication: **each element is read ~1000 times from slow global memory**

#### The Shared Memory Solution

The RTX 4090 has **128 KB of shared memory per SM**, physically located on-chip. This memory is:

1. **Partitioned among thread blocks**: Each block gets its own chunk of shared memory
2. **Shared within a block**: All threads in a block can access the same shared memory
3. **Explicitly managed**: We control what data goes into shared memory
4. **Much faster**: 14 TB/s vs 1 TB/s global memory bandwidth
5. **Lower latency**: ~20-30 cycles vs 400-800 cycles

With 128 SMs on the RTX 4090, that's a total of **16.4 MB of shared memory** distributed across the chip.

### Concept

Now, instead of reading from global memory for every operation, we:

1. **Load tiles** (chunks) of $A$ and $B$ from global memory into shared memory
2. **Compute partial results** using the tiles in fast shared memory
3. **Reuse** the tile data across multiple threads without re-reading from global memory
4. **Slide the tiles** across matrices $A$ and $B$ to compute the final result


```cuda

constexpr uint BLOCKSIZE =  32;

__global__ void sgemm_shared_mem_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                        float alpha, const float *matrix_a,
                                        const float *matrix_b, float beta,
                                        float *matrix_c)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[BLOCKSIZE * BLOCKSIZE];
    __shared__ float tile_b[BLOCKSIZE * BLOCKSIZE];

    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    // Calculate global row and column indices for this thread
    const int global_row = block_row * BLOCKSIZE + thread_row;
    const int global_col = block_col * BLOCKSIZE + thread_col;

    // Move pointers to the starting position for this block
    matrix_a += block_row * BLOCKSIZE * num_cols_a;  // row=block_row, col=0
    matrix_b += block_col * BLOCKSIZE;               // row=0, col=block_col
    matrix_c += block_row * BLOCKSIZE * num_cols_b + block_col * BLOCKSIZE;

    float accumulator = 0.0f;

    // Loop over all tiles along K dimension
    for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BLOCKSIZE)
    {
        // Load tile from matrix A into shared memory with bounds checking
        // threadCol is consecutive for coalesced memory access
        if (global_row < num_rows_a && (tile_idx + thread_col) < num_cols_a) {
            tile_a[thread_row * BLOCKSIZE + thread_col] =
                matrix_a[thread_row * num_cols_a + thread_col];
        } else {
            tile_a[thread_row * BLOCKSIZE + thread_col] = 0.0f;
        }

        // Load tile from matrix B into shared memory with bounds checking
        // threadCol is consecutive for coalesced memory access
        if ((tile_idx + thread_row) < num_cols_a && global_col < num_cols_b) {
            tile_b[thread_row * BLOCKSIZE + thread_col] =
                matrix_b[thread_row * num_cols_b + thread_col];
        } else {
            tile_b[thread_row * BLOCKSIZE + thread_col] = 0.0f;
        }

        // Block threads until cache is fully populated
        __syncthreads();

        // Advance pointers to next tile
        matrix_a += BLOCKSIZE;
        matrix_b += BLOCKSIZE * num_cols_b;

        // Compute partial dot product using shared memory
        for (int dot_idx = 0; dot_idx < BLOCKSIZE; ++dot_idx)
        {
            accumulator += tile_a[thread_row * BLOCKSIZE + dot_idx] *
                          tile_b[dot_idx * BLOCKSIZE + thread_col];
        }

        // Sync again to avoid faster threads fetching next block before slower threads finish
        __syncthreads();
    }
}
```

#### Caller

```cpp
void sgemm_shared_mem(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch: 1D blocks with BLOCKSIZE^2 threads (32x32 = 1024 threads per block)
    dim3 block_dim(BLOCKSIZE * BLOCKSIZE);
    dim3 grid_dim(CEIL_DIV(num_rows_a, BLOCKSIZE),
                  CEIL_DIV(num_cols_b, BLOCKSIZE));

    // Launch kernel
    sgemm_shared_mem_kernel<<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
```

<div id="shared-memory-viz"></div>

### Performance Analysis

Running the shared memory kernel for M = N = K = 4096:

```
────────────────────────────────────────────────────────────────────────────────
2025-10-08 06:57:47.542 | INFO     | __main__:run_benchmarks:766 - 📊 Comparison (baseline: PyTorch)
2025-10-08 06:57:47.542 | INFO     | __main__:run_benchmarks:767 - ────────────────────────────────────────────────────────────────────────────────
2025-10-08 06:57:47.542 | INFO     | __main__:run_benchmarks:791 - PyTorch             : 1.00× (baseline) 🎯
2025-10-08 06:57:47.542 | INFO     | __main__:run_benchmarks:794 - CUDA Naive          : 0.01× 🐢
2025-10-08 06:57:47.542 | INFO     | __main__:run_benchmarks:794 - CUDA Coalesced      : 0.06× 🐢
2025-10-08 06:57:47.542 | INFO     | __main__:run_benchmarks:794 - CUDA Shared Mem     : 0.09× 🐢
2025-10-08 06:57:47.542 | INFO     | __main__:run_benchmarks:796 -

2025-10-08 06:57:47.544 | INFO     | __main__:run_benchmarks:675 - ================================================================================
2025-10-08 06:57:47.544 | INFO     | __main__:run_benchmarks:676 - 📐 Matrix dimensions: (4096, 4096) @ (4096, 4096) = (4096, 4096)
2025-10-08 06:57:47.544 | INFO     | __main__:run_benchmarks:681 - 💾 Expected memory usage: 0.20 GB (0.07 GB per matrix)
2025-10-08 06:57:47.544 | INFO     | __main__:run_benchmarks:684 - ================================================================================
2025-10-08 06:57:47.545 | INFO     | __main__:run_benchmarks:721 -
🔵 Benchmarking PyTorch...
2025-10-08 06:57:47.743 | INFO     | __main__:run_benchmarks:734 -    ⏱️  Time: 1.7483 ms (min: 1.6170, max: 2.0234)
2025-10-08 06:57:47.743 | SUCCESS  | __main__:run_benchmarks:737 -    💪 Performance: 78.61 TFLOPS
2025-10-08 06:57:47.743 | SUCCESS  | __main__:run_benchmarks:738 -    🌊 Bandwidth: 115.16 GB/s
2025-10-08 06:57:47.743 | INFO     | __main__:run_benchmarks:721 -
🔴 Benchmarking CUDA Naive...
2025-10-08 06:58:11.687 | INFO     | __main__:run_benchmarks:734 -    ⏱️  Time: 214.7259 ms (min: 209.7582, max: 226.1811)
2025-10-08 06:58:11.687 | SUCCESS  | __main__:run_benchmarks:737 -    💪 Performance: 0.64 TFLOPS
2025-10-08 06:58:11.687 | SUCCESS  | __main__:run_benchmarks:738 -    🌊 Bandwidth: 0.94 GB/s
2025-10-08 06:58:11.687 | INFO     | __main__:run_benchmarks:721 -
🟢 Benchmarking CUDA Coalesced...
2025-10-08 06:58:14.762 | INFO     | __main__:run_benchmarks:734 -    ⏱️  Time: 27.8368 ms (min: 26.9005, max: 29.5887)
2025-10-08 06:58:14.762 | SUCCESS  | __main__:run_benchmarks:737 -    💪 Performance: 4.94 TFLOPS
2025-10-08 06:58:14.762 | SUCCESS  | __main__:run_benchmarks:738 -    🌊 Bandwidth: 7.23 GB/s
2025-10-08 06:58:14.762 | INFO     | __main__:run_benchmarks:721 -
🟣 Benchmarking CUDA Shared Mem...
2025-10-08 06:58:17.257 | INFO     | __main__:run_benchmarks:734 -    ⏱️  Time: 22.5301 ms (min: 21.8061, max: 24.0466)
2025-10-08 06:58:17.257 | SUCCESS  | __main__:run_benchmarks:737 -    💪 Performance: 6.10 TFLOPS
2025-10-08 06:58:17.257 | SUCCESS  | __main__:run_benchmarks:738 -    🌊 Bandwidth: 8.94 GB/s
2025-10-08 06:58:17.257 | INFO     | __main__:run_benchmarks:765 -
────────────────────────────────────────────────────────────────────────────────
```

> **Performance Improvement:**
> - **1.24× TFLOPS improvement** over coalesced (4.94 → 6.10 TFLOPS)
> - **1.24× bandwidth improvement** (7.23 → 8.94 GB/s)
> - **9.5× faster than naive** (0.64 → 6.10 TFLOPS)
> - Still only **7.8% of PyTorch's performance**, indicating more optimization needed
>
> **Key Insight:**
> - Shared memory caching provides modest improvement (~24%) over just coalescing
> - The relatively small gain suggests we're not yet effectively hiding memory latency
> - Need additional optimizations like thread-level tiling to improve arithmetic intensity
{: .prompt-info}

Below are all the results from the benchmarking:

![Kernel with shared mem](/assets/explore_gemm_shared_mem.png)

## Understanding GPU Occupancy

Before diving into more advanced optimizations, we need to understand **occupancy**—a critical metric that determines how well we utilize the GPU's resources.

### What is Occupancy?

**Occupancy** is the ratio of active warps to the maximum number of possible warps per SM:

$$\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}$$

For the RTX 4090 (Compute Capability 8.9):
- **Maximum warps per SM**: 48 warps
- **Maximum threads per SM**: 1,536 threads (48 warps × 32 threads/warp)
- **Maximum thread blocks per SM**: 32 blocks

### Why Occupancy Matters

GPUs hide memory latency through **massive parallelism**. When one warp waits for memory, the SM immediately switches to execute another warp—with **zero overhead** (no register state swapping).

> **Key Insight:** Higher occupancy → More warps available → Better latency hiding → Higher performance
{: .prompt-tip}

However, **occupancy is not everything**. A kernel with 100% occupancy but poor memory access patterns will still perform poorly. 

### Factors Limiting Occupancy

Occupancy is limited by three hardware resources per SM:

#### 1. **Registers (256 KB per SM on RTX 4090)**

Each thread gets registers from the SM's register file. More registers per thread → fewer concurrent threads.

**Maximum threads calculation:**

$$\text{Max Threads} = \min\left(\frac{65536 \text{ registers/SM}}{\text{registers per thread}}, 1536\right)$$

| Registers/Thread | Max Threads | Active Warps | Occupancy | Status |
|------------------|-------------|--------------|-----------|--------|
| 32 | 1,536 (limited by max) | 48 | 100% | ✅ Optimal |
| 64 | 1,024 | 32 | 66.7% | ⚠️ Good |
| 128 | 512 | 16 | 33.3% | ❌ Poor |

#### 2. **Shared Memory (128 KB per SM on RTX 4090)**

Shared memory is partitioned among thread blocks on the same SM.

**Maximum blocks calculation:**

$$\text{Max Blocks} = \min\left(\frac{131072 \text{ bytes/SM}}{\text{shared memory per block}}, \underbrace{32}_{\text{hardware limit}}\right)$$

> Note: 32 is **maximum resident blocks per SM** (hardware limit for RTX 4090)
{: .prompt-info}

| Shared Memory/Block | Max Blocks | Notes | Status |
|---------------------|------------|-------|--------|
| 0 KB | 32 | No shared memory usage | ✅ Maximum blocks |
| 32 KB | 4 | Good for moderate tiling | ✅ Good |
| 64 KB | 2 | Large tiles, fewer blocks | ⚠️ Acceptable |
| 96 KB | 1 | Very large tiles, single block | ❌ Poor if block is small |

#### 3. **Thread Block Size**

The number of threads per block affects how many blocks can fit on an SM.

| Threads/Block | Warps/Block | Max Blocks/SM | Active Warps | Occupancy | Status |
|---------------|-------------|---------------|--------------|-----------|--------|
| 128 | 4 | 12 | 48 | 100% | ✅ Optimal |
| 256 | 8 | 6 | 48 | 100% | ✅ Optimal |
| 512 | 16 | 3 | 48 | 100% | ✅ Optimal |
| 1024 | 32 | 1 | 32 | 66.7% | ⚠️ Limited by block size |

### Calculating Occupancy for Our Shared Memory Kernel

Let's analyze our current shared memory kernel:

```cuda
constexpr uint BLOCKSIZE = 32;
dim3 block_dim(BLOCKSIZE * BLOCKSIZE);  // 1024 threads per block
```

**Shared memory usage:**
```cuda
__shared__ float tile_a[32 * 32];  // 4 KB
__shared__ float tile_b[32 * 32];  // 4 KB
// Total: 8 KB per block
```

**Occupancy calculation:**

1. **Threads per block**: 1,024 threads (32 warps)
2. **Blocks per SM (thread limit)**: $\lfloor 1536 / 1024 \rfloor = 1$ block
3. **Blocks per SM (shared memory limit)**: $\lfloor 131072 / 8192 \rfloor = 16$ blocks
4. **Blocks per SM (block limit)**: 32 blocks (max)
5. **Actual blocks per SM**: $\min(1, 16, 32) = 1$ block

**Active warps**: $1 \text{ block} \times 32 \text{ warps/block} = 32 \text{ warps}$

**Occupancy**: $\frac{32}{48} = 66.7\%$ ⚠️

> **Problem**: Our large block size (1,024 threads) limits us to only 1 block per SM, resulting in just 66.7% occupancy. 66.7% occupancy is not necessarily bad but let's see more details using nsight-compute
{: .prompt-warning}

### Nsight Compute Profiling

First let's confirm our occupancy calculation and it matches. 

![shared_mem_occupancy](/assets/explore_gemm_shared_mem_occupancy.png)

Now, looking at the summary - it provides some ideas on why the kernel is still slow:

![ncu summary shared mem](/assets/explore_gemm_summary_shared_mem_ncu.png)

Next, looking at the instruction mix, we can see that LDS dominates the instruction mix -- LDS = load within shared memory window -- **which is not good.**

![LDS Too much](/assets/explore_gemm_lds_too_much.png)

> NVIDIA provides [`cudaOccupancyMaxActiveBlocksPerMultiprocessor()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g5a5d67a3c907371559ba692195e8a38c) to calculate theoretical occupancy at runtime. Most profilers (Nsight Compute) also report achieved occupancy.
{: .prompt-tip}

## Kernel 4: 1D Block Tiling

Now that we understand that in Kernel 3, each thread was computing a single output element of matrix C, meaning each thread needed to load elements from shared memory repeatedly, with memory accesses dominating the execution time. Next, instead of each thread computing exactly one output element of the tile, each thread computes multiple output elements along one dimension. To support this, we fetch some data from SMEM into registers (for reuse) within each thread, reducing repeated SMEM loads. 


> In essence, we are trying to improve the arithmetic intensity of the kernel, which effectively means computing more results per thread with the same loaded data i.e. increase FLOPS/byte
{: .prompt-tip}


### Kernel

```cuda
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                            float alpha, const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    const uint thread_row = threadIdx.x / BN;
    const uint thread_col = threadIdx.x % BN;

    // Calculate global row and column indices for this thread
    const int global_row = block_row * BM + thread_row * TM;
    const int global_col = block_col * BN + thread_col;

    // Move pointers to the starting position for this block
    matrix_a += block_row * BM * num_cols_a;  // row=block_row, col=0
    matrix_b += block_col * BN;               // row=0, col=block_col
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    // Allocate thread-local cache for results in registerfile
    float thread_results[TM] = {0.0f};

    // Loop over all tiles along K dimension
    for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BK)
    {
        // Load tile from matrix A into shared memory with bounds checking
        // Each thread loads one element from A
        const uint a_row = threadIdx.x / BK;
        const uint a_col = threadIdx.x % BK;
        if ((block_row * BM + a_row) < num_rows_a && (tile_idx + a_col) < num_cols_a) {
            tile_a[a_row * BK + a_col] = matrix_a[a_row * num_cols_a + a_col];
        } else {
            tile_a[a_row * BK + a_col] = 0.0f;
        }

        // Load tile from matrix B into shared memory with bounds checking
        // Each thread loads one element from B
        const uint b_row = threadIdx.x / BN;
        const uint b_col = threadIdx.x % BN;
        if ((tile_idx + b_row) < num_cols_a && (block_col * BN + b_col) < num_cols_b) {
            tile_b[b_row * BN + b_col] = matrix_b[b_row * num_cols_b + b_col];
        } else {
            tile_b[b_row * BN + b_col] = 0.0f;
        }

        __syncthreads();

        // Advance pointers to next tile
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        // Calculate per-thread results
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // We make the dotproduct loop the outside loop, which facilitates
            // reuse of the tile_b entry, which we can cache in a tmp var.
            float b_tmp = tile_b[dot_idx * BN + thread_col];
            for (uint res_idx = 0; res_idx < TM; ++res_idx) {
                thread_results[res_idx] +=
                    tile_a[(thread_row * TM + res_idx) * BK + dot_idx] * b_tmp;
            }
        }

        __syncthreads();
    }

    // Write results to global memory: C = α*(A@B)+β*C
    for (uint res_idx = 0; res_idx < TM; ++res_idx) {
        int row = global_row + res_idx;
        if (row < num_rows_a && global_col < num_cols_b) {
            matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col] =
                alpha * thread_results[res_idx] +
                beta * matrix_c[(thread_row * TM + res_idx) * num_cols_b + thread_col];
        }
    }
}
```

### Caller

```cuda

void sgemm_blocktiling_1d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Template parameters for kernel
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    // Configure kernel launch
    // Number of threads = (BM / TM) * BN = (64 / 8) * 64 = 512 threads per block
    dim3 block_dim((BM / TM) * BN);
    dim3 grid_dim(CEIL_DIV(num_rows_a, BM),
                  CEIL_DIV(num_cols_b, BN));

    // Launch kernel
    sgemm_blocktiling_1d_kernel<BM, BN, BK, TM><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

```

<div id="1d-tiling-viz"></div>

<!-- ### Memory Hierarchy Pipeline

The visualization below shows the complete data flow through the GPU memory hierarchy for 1D block tiling:

<div id="1d-pipeline-viz"></div> -->

### Performance Results

```
🟠 Benchmarking CUDA 1D Block Tiling...
2025-10-11 07:30:59.509 | INFO     | __main__:run_benchmarks:755 -    ⏱️  Time: 7.4340 ms (min: 7.4025, max: 7.4721)
2025-10-11 07:30:59.509 | SUCCESS  | __main__:run_benchmarks:758 -    💪 Performance: 18.49 TFLOPS
2025-10-11 07:30:59.509 | SUCCESS  | __main__:run_benchmarks:759 -    🌊 Bandwidth: 27.08 GB/s
```

> **Performance Improvement:**
> - **3.03× TFLOPS improvement** over shared memory (6.10 → 18.49 TFLOPS)
> - **3.03× bandwidth improvement** (8.94 → 27.08 GB/s)
> - **28.9× faster than naive** (0.64 → 18.49 TFLOPS)
> - Achieved **21.8% of PyTorch's performance** (84.62 TFLOPS)
>
> **Key Insight:**
> - Register-level tiling provides **3× improvement** by increasing arithmetic intensity
> - By caching `b_tmp` in registers and reusing it TM times, we reduce shared memory traffic by ~37.5%
> - Each thread now computes TM=8 outputs instead of 1, amortizing memory access costs
{: .prompt-info}

**Comparison vs Previous Kernels:**

| Kernel | Time (ms) | TFLOPS | vs PyTorch | Speedup over Naive |
|--------|-----------|---------|------------|-------------------|
| Naive | 214.24 | 0.64 | 0.8% | 1.0× |
| Coalesced | 28.62 | 4.80 | 5.7% | 7.5× |
| Shared Memory | 22.52 | 6.10 | 7.2% | 9.5× |
| **1D Block Tiling** | **7.43** | **18.49** | **21.8%** | **28.9×** |
| PyTorch  | 1.62 | 84.62 | 100% | 132.1× |

Let's look at overall performance across several batch sizes

![1D block tiling performance](/assets/explore_gemm_id_blocktiling_performance.png)

### NCU profiling

Now, let's look at how things have changed and confirm our results using ncu!

#### NCU Summary Stats

![NCU summary stats](/assets/explore_gemm_1d_tiling_ncu_1.png)

> We can see the warning related to MIO Throttle Stalls and Shared Store Bank Conflicts are gone now
{: .prompt-tip}

#### NCU Instruction Mix

![NCU execution mix](/assets/explore_gemm_1d_tiling_ncu_2.png)

> LDS is no longer the majority of instruction mix. 
{: .prompt-tip}

#### Memory Charts Comparison

The shared memory bandwidth used is also showed in the memory charts. See the comparison below:

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
  <div>
    <img src="/assets/explore_gemm_1d_smem_ncu_memory.png" alt="Shared Memory Kernel - NCU Memory Profile" style="width: 100%; border-radius: 8px;">
    <p style="text-align: center; margin-top: 10px; font-size: 14px; color: #666;"><strong>Shared Memory Kernel</strong> - Higher SMEM traffic, lower throughput</p>
  </div>
  <div>
    <img src="/assets/explore_gemm_1d_block_tiling_ncu_memory.png" alt="1D Block Tiling Kernel - NCU Memory Profile" style="width: 100%; border-radius: 8px;">
    <p style="text-align: center; margin-top: 10px; font-size: 14px; color: #666;"><strong>1D Block Tiling Kernel</strong> - Register caching reduces SMEM traffic, 3× faster</p>
  </div>
</div>

> Key takeaway from 1D blocktiling approach is as we calculate more values per thread, we reduce the number of loads/stores per result i.e. increase arithmetic intensity
{: .prompt-tip}

## Kernel 5: 2D Block Tiling

### Concept

2D block tiling extends the 1D approach by having each thread compute a **TM × TN tile** of outputs instead of just TM outputs. This creates bidirectional register reuse:
- Each value from A (loaded into `register_m[TM]`) is reused across **TN computations**
- Each value from B (loaded into `register_n[TN]`) is reused across **TM computations**
- This forms an **outer product** pattern that dramatically increases arithmetic intensity

**Key Improvement over 1D Tiling:**
- 1D tiling: Each thread computes TM outputs (e.g., 8 outputs)
- 2D tiling: Each thread computes TM × TN outputs (e.g., 8 × 8 = 64 outputs)
- Result: **8× more computation per thread** with only marginally more register usage

We implement **two separate kernels** such that a **Main Kernel** (No Bounds Checking) handles all **interior blocks** where every memory access is guaranteed to be in-bounds -- which helps with zero thread divergence since all threads can execute the same logic. An **Edge Kernel** (With Bounds Checking) handles **boundary blocks** at the right edge, bottom edge, and corner. 


### Kernel

We show just the main kernel -- I had Claude put a bunch of comments on this kernel so steps are clear. 

```cuda
// ==================== MAIN KERNEL (NO BOUNDS CHECKING) ====================
// This kernel handles all interior blocks where we know all memory accesses are in-bounds
// No thread divergence, maximum performance
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                            float alpha, const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    // Shared memory tiles for A and B
    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    // Calculate thread position within the block
    // Each thread is responsible for computing a TM x TN output tile
    // Total threads per block = (BM / TM) * (BN / TN)
    const uint thread_row = threadIdx.x / (BN / TN); // Which row of thread tiles
    const uint thread_col = threadIdx.x % (BN / TN); // Which column of thread tiles

    // Thread count and loading strategy
    // We have (BM/TM) * (BN/TN) = 64 threads
    // tile_a needs BM * BK = 64 * 8 = 512 elements
    // tile_b needs BK * BN = 8 * 64 = 512 elements
    // Each thread must load 512/64 = 8 elements from each tile
    const uint num_threads = (BM / TM) * (BN / TN);

    // Position input/output matrix pointers at the start of this block's tile
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    // Allocate thread-local storage in registers for:
    // 1. Final results: TM x TN output values this thread computes
    // 2. register_m: TM values from matrix A (reused across TN computations)
    // 3. register_n: TN values from matrix B (reused across TM computations)
    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    // Outer loop over block tiles along K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
// ==================== LOAD TILES INTO SHARED MEMORY ====================

// Load tile from matrix A into shared memory
// Layout: tile_a is BM x BK (64 x 8 = 512 elements)
// With 64 threads, each thread loads 512/64 = 8 elements
// NO BOUNDS CHECKING - assumes all accesses are in-bounds
#pragma unroll
        for (uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads)
        {
            uint load_idx = threadIdx.x + load_offset;
            uint a_row = load_idx / BK;
            uint a_col = load_idx % BK;
            tile_a[load_idx] = matrix_a[a_row * num_cols_a + a_col];
        }

// Load tile from matrix B into shared memory
// Layout: tile_b is BK x BN (8 x 64 = 512 elements)
// With 64 threads, each thread loads 512/64 = 8 elements
// NO BOUNDS CHECKING - assumes all accesses are in-bounds
#pragma unroll
        for (uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads)
        {
            uint load_idx = threadIdx.x + load_offset;
            uint b_row = load_idx / BN;
            uint b_col = load_idx % BN;
            tile_b[load_idx] = matrix_b[b_row * num_cols_b + b_col];
        }

        __syncthreads();

        // Advance block tile pointers for next iteration
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        // ==================== COMPUTE USING REGISTER BLOCKING ====================

        // For each element along the K dimension of the current block tile
        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            // Load TM elements from tile_a into registers
            // These are the elements in column dot_idx, rows [thread_row*TM : thread_row*TM+TM)
            // We load these once and reuse them for all TN columns
            for (uint i = 0; i < TM; ++i)
            {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
            }

            // Load TN elements from tile_b into registers
            // These are the elements in row dot_idx, columns [thread_col*TN : thread_col*TN+TN)
            // We load these once and reuse them for all TM rows
            for (uint i = 0; i < TN; ++i)
            {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            // Compute outer product of register_m and register_n, accumulating into thread_results
            // This is the key 2D blocking: we compute TM x TN results using cached values
            // For each result position (res_m, res_n), compute: result += register_m[res_m] * register_n[res_n]
            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                {
                    // Store in row-major order: thread_results[res_idx_m * TN + res_idx_n]
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

// ==================== WRITE RESULTS TO GLOBAL MEMORY ====================

// Write the TM x TN tile of results computed by this thread back to global memory
// Apply scaling: C = alpha * (A @ B) + beta * C
// NO BOUNDS CHECKING - assumes all accesses are in-bounds
#pragma unroll
    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
    {
#pragma unroll
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
        {
            const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b +
                               (thread_col * TN + res_idx_n);
            matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                              beta * matrix_c[c_idx];
        }
    }
}

```

### Caller

```cuda

void sgemm_blocktiling_2d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Template parameters for kernel
    // BM, BN: Block tile dimensions (64x64 output block per thread block)
    // BK: Inner dimension block size (8 elements processed per iteration)
    // TM, TN: Thread tile dimensions (8x8 output tile per thread)
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Configure kernel launch
    // Number of threads = (BM / TM) * (BN / TN) = (64 / 8) * (64 / 8) = 8 * 8 = 64 threads per block
    dim3 block_dim((BM / TM) * (BN / TN));

    // Calculate number of complete blocks and edge blocks
    const int num_blocks_m = CEIL_DIV(num_rows_a, BM);
    const int num_blocks_n = CEIL_DIV(num_cols_b, BN);
    const int main_blocks_m = num_rows_a / BM; // Complete blocks in M dimension
    const int main_blocks_n = num_cols_b / BN; // Complete blocks in N dimension

    // Launch main kernel for interior blocks (no bounds checking needed)
    if (main_blocks_m > 0 && main_blocks_n > 0)
    {
        dim3 main_grid(main_blocks_m, main_blocks_n);
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN><<<main_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
    }

    // Launch edge kernel for right edge (last column of blocks)
    if (main_blocks_m > 0 && num_blocks_n > main_blocks_n)
    {
        dim3 edge_right_grid(main_blocks_m, 1);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_right_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            0, main_blocks_n);
    }

    // Launch edge kernel for bottom edge (last row of blocks)
    if (num_blocks_m > main_blocks_m && main_blocks_n > 0)
    {
        dim3 edge_bottom_grid(1, main_blocks_n);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_bottom_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            main_blocks_m, 0);
    }

    // Launch edge kernel for bottom-right corner (if both edges exist)
    if (num_blocks_m > main_blocks_m && num_blocks_n > main_blocks_n)
    {
        dim3 edge_corner_grid(1, 1);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_corner_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            main_blocks_m, main_blocks_n);
    }
}

```

<div id="2d-tiling-viz"></div>

### Performance Results

Looking at the benchmark results for 4096×4096 matrices, the 2D block tiling kernel shows furthre performance gains:



> **Performance Improvement:**
> - **1.66× TFLOPS improvement** over 1D block tiling (18.40 → 30.55 TFLOPS)
> - **1.66× bandwidth improvement** (26.96 → 44.75 GB/s)
> - **38.7% of PyTorch's performance** (79.00 TFLOPS)
> - **46.9× faster than naive** (0.652 → 30.55 TFLOPS)
{: .prompt-info}

**Comparison vs Previous Kernels (4096×4096):**

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs PyTorch | Speedup over Naive |
|--------|-----------|---------|------------------|------------|-------------------|
| Naive | 210.75 | 0.65 | 0.96 | 0.8% | 1.0× |
| Coalesced | 26.96 | 5.10 | 7.47 | 6.5% | 7.8× |
| Shared Memory | 22.67 | 6.06 | 8.88 | 7.7% | 9.3× |
| 1D Block Tiling | 7.47 | 18.40 | 26.96 | 23.3% | 28.2× |
| **2D Block Tiling** | **4.50** | **30.55** | **44.75** | **38.7%** | **46.9×** |
| PyTorch | 1.74 | 79.00 | 115.73 | 100% | 121.2× |

### Performance Across Matrix Sizes

![2D block tiling performance](/assets/explore_gemm_2d_blocktiling_performance.png)

As next steps, when we check our kernel in Nsight compute, we get more pointers to improve performance:

> - **L1TEX Global Store Access Pattern:** The memory access pattern for global stores to L1TEX might not be optimal. On average, only 4.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global stores.
> - **Shared Load Bank Conflicts:** The memory access pattern for shared loads might not be optimal and causes on average a 5.6 - way bank conflict across all 167772160 shared load requests.This results in 503316480 bank conflicts, which represent 53.57% of the overall 939534036 wavefronts for shared loads. Check the Source Counters section for uncoalesced shared loads.
> - **Uncoalesced Shared Accesses:** This kernel has uncoalesced shared accesses resulting in a total of 369098752 excessive wavefronts (37% of the total 1006632960 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide has an example on optimizing shared memory accesses.

![NCU After 2D Tiling](/assets/explore_gemm_ncu_after_2d_tiling.png)

## Kernel 6: Vectorized Memory Access

### Concept

Many CUDA kernels are **bandwidth-bound**, meaning their performance is limited by memory transfer speed rather than compute throughput. One powerful optimization is to use **vectorized memory operations** to increase effective bandwidth utilization.

CUDA provides built-in vector types (`float2`, `float4`, `int2`, `int4`) that enable loading or storing multiple values in a single instruction. Instead of issuing four separate 32-bit loads, a single `float4` load can fetch **128 bits (16 bytes)** at once.

> Good reference for Vectorized Memory Access for Performance: [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
{: .prompt-tip}

In our 2D block tiling kernel, each thread loads elements from shared memory to registers. Currently, these loads happen as individual 32-bit transactions:

```cuda
// Current approach: scalar loads
for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
    // Load TM elements from tile_a into registers
    for (uint i = 0; i < TM; ++i) {
        register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];  // 32-bit load
    }

    // Load TN elements from tile_b into registers
    for (uint i = 0; i < TN; ++i) {
        register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];  // 32-bit load
    }

    // ... compute outer product
}
```

If `TM=8`, we're issuing 8 separate load instructions. With vectorization, we can combine these into 2 loads of `float4`:

```cuda
// Vectorized approach: load 4 elements at once
for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
    // Load 8 elements from tile_a using 2× float4 instead of 8× float
    *reinterpret_cast<float4*>(&register_m[0]) =
        *reinterpret_cast<const float4*>(&tile_a[(thread_row * TM) * BK + dot_idx]);
    *reinterpret_cast<float4*>(&register_m[4]) =
        *reinterpret_cast<const float4*>(&tile_a[(thread_row * TM + 4) * BK + dot_idx]);

    // Load 8 elements from tile_b using 2× float4 instead of 8× float
    *reinterpret_cast<float4*>(&register_n[0]) =
        *reinterpret_cast<const float4*>(&tile_b[dot_idx * BN + thread_col * TN]);
    *reinterpret_cast<float4*>(&register_n[4]) =
        *reinterpret_cast<const float4*>(&tile_b[dot_idx * BN + thread_col * TN + 4]);
}
```

### How Vector Loads Work

When you load a `float4`, the compiler emits a **128-bit vectorized instruction** (`LDG.E.128`) instead of four 32-bit loads (`LDG.E`):

| Load Type | Size | Instruction | Elements | Instructions Needed |
|-----------|------|-------------|----------|---------------------|
| `float` | 32 bits | `LDG.E` | 1 | 4 (for 4 elements) |
| `float2` | 64 bits | `LDG.E.64` | 2 | 2 (for 4 elements) |
| `float4` | 128 bits | `LDG.E.128` | 4 | 1 (for 4 elements) |


> Vectorized loads only work efficiently when data is properly aligned to vector size e.g. 16 bytes for `float4` and access patterns respect alignment boundaries
{: .prompt-info}

```cuda
// ✅ Valid: base pointer from cudaMalloc is aligned
float4* vec_ptr = reinterpret_cast<float4*>(device_array);
float4 data = vec_ptr[i];  // OK if i maintains alignment

// ❌ Invalid: offset breaks alignment
float* offset_ptr = device_array + 1;  // Now only 4-byte aligned
float4* bad_vec = reinterpret_cast<float4*>(offset_ptr);  // Misaligned!
```

#### Vectorized load example:
```cuda
/*
- 1 vectorized load instruction
- Guaranteed single 128-bit memory transaction
- Hardware-enforced coalescing
*/
float4 vec = *reinterpret_cast<const float4*>(&A[i]);  // LDG.E.128 (128-bit)
float a = vec.x;
float b = vec.y;
float c = vec.z;
float d = vec.w;
```

> **Trade-Offs**: Vectorized `float4` loads reduce memory transactions but increase register pressure (potentially lowering occupancy), require tail handling for non-divisible sizes, and may not benefit shared memory accesses with strided layouts without restructuring.
>
> **NOTE: In our case, we will skip the tail handling and assume the inputs is aligned to the block sizes used in the kernel just to simplify the kernel**
{: .prompt-info }

### Kernel

```cuda
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorize_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                       float alpha, const float *matrix_a,
                                       const float *matrix_b, float beta,
                                       float *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // Thread indices for computing output tile
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint thread_row = threadIdx.x / (BN / TN);

    // Shared memory tiles - stored in column-major for A to enable coalescing
    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    // Position matrix pointers at the start of this block's tile
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    // Thread indices for vectorized loading
    // Load 4 floats at a time using float4
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4);
    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4);

    // Allocate register storage
    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    // Outer loop over K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        // ==================== VECTORIZED LOAD FROM GLOBAL MEMORY ====================

        // Load tile_a using float4 vectorized loads
        // Store in transposed layout: tile_a[col][row] for coalesced shared memory access
        float4 tmp_a = reinterpret_cast<const float4*>(
            &matrix_a[inner_row_a * num_cols_a + inner_col_a * 4])[0];
        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a] = tmp_a.w;

        // Load tile_b using float4 vectorized loads
        // Store in row-major layout: tile_b[row][col]
        float4 tmp_b = reinterpret_cast<const float4*>(
            &matrix_b[inner_row_b * num_cols_b + inner_col_b * 4])[0];
        tile_b[inner_row_b * BN + inner_col_b * 4 + 0] = tmp_b.x;
        tile_b[inner_row_b * BN + inner_col_b * 4 + 1] = tmp_b.y;
        tile_b[inner_row_b * BN + inner_col_b * 4 + 2] = tmp_b.z;
        tile_b[inner_row_b * BN + inner_col_b * 4 + 3] = tmp_b.w;

        __syncthreads();

        // Advance pointers for next tile
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        // ==================== COMPUTE USING REGISTER BLOCKING ====================

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            // Load TM elements from tile_a (transposed layout)
            #pragma unroll
            for (uint i = 0; i < TM; ++i) {
                register_m[i] = tile_a[dot_idx * BM + thread_row * TM + i];
            }

            // Load TN elements from tile_b
            #pragma unroll
            for (uint i = 0; i < TN; ++i) {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            // Outer product accumulation
            #pragma unroll
            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                #pragma unroll
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

    // ==================== WRITE RESULTS TO GLOBAL MEMORY ====================

    // Write results with alpha/beta scaling
    #pragma unroll
    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
        #pragma unroll
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
            const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b +
                               (thread_col * TN + res_idx_n);
            matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] +
                              beta * matrix_c[c_idx];
        }
    }
}
```

### Caller

```cuda
void sgemm_vectorize(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                     torch::Tensor &output_matrix, float alpha, float beta)
{
    // Validate inputs
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Template parameters for kernel
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Validate dimensions are multiples of tile sizes
    TORCH_CHECK(num_rows_a % BM == 0, "Matrix A rows must be multiple of ", BM);
    TORCH_CHECK(num_cols_a % BK == 0, "Matrix A cols must be multiple of ", BK);
    TORCH_CHECK(num_cols_b % BN == 0, "Matrix B cols must be multiple of ", BN);

    // Configure kernel launch
    dim3 block_dim((BM / TM) * (BN / TN));
    dim3 grid_dim(num_cols_b / BN, num_rows_a / BM);

    // Launch kernel
    sgemm_vectorize_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}
```

### Performance Analysis

The vectorized kernel shows **mixed results** — performance **degrades** compared to 2D block tiling for small to medium matrices but shows **significant improvements** at larger sizes.

> **Performance at 4096×4096:**
> - **1.27× TFLOPS improvement** over 2D block tiling (30.85 → 39.00 TFLOPS)
> - **1.26× bandwidth improvement** (45.19 → 57.14 GB/s)
> - **46.3% of PyTorch's performance** (84.23 TFLOPS)
> - **59.7× faster than naive** (0.653 → 39.00 TFLOPS)
{: .prompt-info}

**Comparison vs Previous Kernels (4096×4096):**

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs PyTorch | Speedup over Naive |
|--------|-----------|---------|------------------|------------|-------------------|
| Naive | 210.33 | 0.65 | 0.96 | 0.8% | 1.0× |
| Coalesced | 26.92 | 5.11 | 7.48 | 6.1% | 7.8× |
| Shared Memory | 22.42 | 6.13 | 8.98 | 7.3% | 9.4× |
| 1D Block Tiling | 7.44 | 18.48 | 27.07 | 21.9% | 28.3× |
| 2D Block Tiling | 4.46 | 30.85 | 45.19 | 36.6% | 47.3× |
| **Vectorized** | **3.52** | **39.00** | **57.14** | **46.3%** | **59.7×** |
| PyTorch | 1.63 | 84.23 | 123.38 | 100% | 129.0× |

### Performance Across Matrix Sizes

![Vectorized performance](/assets/explore_gemm_performance_vectorized.png)


## WarpTiling


After optimizing thread-level and block-level tiling, the next step is **warp-level tiling**—exploiting the natural 32-thread warp execution unit in NVIDIA GPUs to achieve better register reuse and computation efficiency.

#### What is Warp Tiling? Tiling Hierarchy in CUTLASS

A **warp** is the fundamental execution unit in NVIDIA GPUs consisting of 32 threads that execute in SIMT (Single Instruction, Multiple Thread) fashion. Warp tiling introduces an additional level in the memory hierarchy:

The NVIDIA CUTLASS library implements a sophisticated tiling strategy that mirrors the GPU's memory hierarchy at multiple levels:

![CUTLASS Memory Hierarchy](/assets/explore_gemm_cutlass_hierarchy.png)
*Source: [NVIDIA CUTLASS Blog](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)*

**Key Levels:**

1. **Device-level GEMM**: Operates on entire matrices in global memory
2. **Threadblock-level GEMM**: Each block loads tiles into shared memory (our current 2D block tiling)
3. **Warp-level GEMM**: Each warp loads fragments from shared memory into registers
4. **Thread-level GEMM**: Individual threads compute on register data
5. **Instruction-level**: Hardware instructions (e.g., Tensor Cores on modern GPUs)

#### Why Warp Tiling Matters

{: .prompt-info }
> **Performance Rationale**: Warp tiling reduces **shared memory bandwidth** as a bottleneck by maximizing **register-level data reuse**. Fragments are kept small in the K dimension to maximize compute intensity relative to data movement.

**Benefits:**

1. **Register Fragment Reuse**: Each warp loads small matrix fragments (typically 16×16 or 8×8) from shared memory into registers and reuses them across multiple computations
2. **Reduced Shared Memory Pressure**: By keeping fragments small in K dimension, warps maximize computation per byte loaded from shared memory
3. **Better Instruction-Level Parallelism (ILP)**: Warps can issue independent math instructions to CUDA cores while simultaneously loading next fragments
4. **Double Buffering**: Warps maintain two register fragment sets—one for current computation, one for prefetching next data

#### Tiling Hierarchy Example

For a typical CUTLASS GEMM configuration:

```
Threadblock Tile: 128×128×8  (BM=128, BN=128, BK=8)
    ├─ Warp Tile: 64×64×8     (4 warps process this threadblock)
    │   └─ Thread Tile: 8×8   (Each of 32 threads in warp computes 8×8)
```

**Compute Intensity Calculation:**

- **Without warp tiling**: Each element loaded from shared memory → used once
- **With warp tiling**: Each element loaded → reused across 8×8 thread tile = 64 FMA operations per load
- **Result**: ~64× improvement in compute intensity

#### Implementation Challenges

1. **Register Pressure**: Warp tiles require significant register storage (e.g., 8×8 float tile = 64 registers per thread)
2. **Warp Synchronization**: Requires explicit `__syncwarp()` or implicit warp-synchronous execution
3. **Fragment Loading**: Must carefully orchestrate loads from shared memory to avoid bank conflicts
4. **Complexity**: Adds another dimension to autotuning (warp tile sizes on top of block/thread tiles)

#### Modern Extensions: Warp Specialization

In NVIDIA's latest CUTLASS implementations (Hopper GPUs), **warp specialization** takes this further:

- **Producer warps**: Dedicated to loading data from global → shared memory (using TMA - Tensor Memory Accelerator)
- **Consumer warps**: Dedicated to computation using WGMMA (Warp Group Matrix Multiply-Accumulate)
- **Asynchronous pipelining**: Producer and consumer warps operate concurrently with minimal synchronization

This represents the state-of-the-art in GEMM optimization, achieving >90% of theoretical peak performance on modern hardware.

{: .prompt-tip }
> **Further Reading**: For implementation details, see [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html) and the [Colfax CUTLASS Tutorial](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/).

#### Implementation: From 2D Block Tiling to Warp Tiling

Let's extend our previous 2D block tiling kernel to incorporate warp-level tiling. The key difference is adding an intermediate warp-level processing layer between block tiles and thread tiles.

**Evolution of the Tiling Hierarchy:**

```
Previous (2D Block Tiling):
Threadblock Tile (BM×BN) → Thread Tile (TM×TN)
                ↓
         All threads cooperate

New (Warp Tiling):
Threadblock Tile (BM×BN) → Warp Tile (WM×WN) → Warp Subtile (WSUBM×WSUBN) → Thread Tile (TM×TN)
                ↓                    ↓                      ↓
           All warps          Single warp (32 threads)    Individual threads
```

**Template Parameters:**

```cpp
template <const int BM,      // Block tile M dimension (e.g., 128)
          const int BN,      // Block tile N dimension (e.g., 128)
          const int BK,      // Block tile K dimension (e.g., 16)
          const int WM,      // Warp tile M dimension (e.g., 64)
          const int WN,      // Warp tile N dimension (e.g., 64)
          const int WNITER,  // Warp subtile iterations in N (e.g., 4)
          const int TM,      // Thread tile M dimension (e.g., 8)
          const int TN,      // Thread tile N dimension (e.g., 4)
          const int NUM_THREADS> // Threads per block (e.g., 128)
```

**Computed Values:**

```cpp
// Number of warp subtile iterations in M dimension
WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER)

// Warp subtile dimensions
WSUBM = WM / WMITER  // Height of each warp subtile
WSUBN = WN / WNITER  // Width of each warp subtile
```

**Example Configuration:**

For `BM=128, BN=128, WM=64, WN=64` with `NUM_THREADS=128`:
- **Warps per block**: `(BM/WM) × (BN/WN) = 2 × 2 = 4 warps`
- **Threads needed**: `4 warps × 32 threads/warp = 128 threads` ✓

#### Key Implementation Details

**1. Warp Placement Within Block**

Each thread determines which warp it belongs to and where that warp sits in the block tile:

```cpp
const uint warp_idx = threadIdx.x / WARPSIZE;       // Which warp [0-3 for 128 threads]
const uint warp_col = warp_idx % (BN / WN);         // Warp's column in block
const uint warp_row = warp_idx / (BN / WN);         // Warp's row in block
```

**2. Thread Placement Within Warp Subtile**

Each thread determines its position within the warp's subtile:

```cpp
const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;           // [0-31]
const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN); // Column index
const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN); // Row index
```

For `WSUBN=16, TN=4`: each row has `16/4 = 4 threads`, so 32 threads fill `32/4 = 8 rows`.

**3. Warp-Level Register Caching**

The critical optimization—each warp loads **entire warp tile fragments** into registers:

```cpp
// Thread-local storage
float thread_results[WMITER * TM * WNITER * TN] = {0.0f};  // Final accumulation
float register_m[WMITER * TM] = {0.0f};                     // Cache for A fragments
float register_n[WNITER * TN] = {0.0f};                     // Cache for B fragments
```

For `WMITER=2, TM=8, WNITER=4, TN=4`:
- `thread_results`: `2×8×4×4 = 256 floats`
- `register_m`: `2×8 = 16 floats` (covers both M subtiles)
- `register_n`: `4×4 = 16 floats` (covers all 4 N subtiles)

**4. Warp Tile Processing**

The core computation processes the entire warp tile by iterating over warp subtiles:

```cpp
template <...>
__device__ void process_warp_tile(...)
{
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
        // Load all warp subtile fragments for this K slice
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint i = 0; i < TM; ++i) {
                register_m[wsub_row_idx * TM + i] =
                    tile_a[(dot_idx * BM) + warp_row * WM +
                           wsub_row_idx * WSUBM + thread_row_in_warp * TM + i];
            }
        }

        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (uint i = 0; i < TN; ++i) {
                register_n[wsub_col_idx * TN + i] =
                    tile_b[(dot_idx * BN) + warp_col * WN +
                           wsub_col_idx * WSUBN + thread_col_in_warp * TN + i];
            }
        }

        // Compute outer product across all warp subtiles
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                        thread_results[(wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                      (wsub_col_idx * TN) + res_idx_n] +=
                            register_m[wsub_row_idx * TM + res_idx_m] *
                            register_n[wsub_col_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}
```

**Why This Works:**

1. **Shared Memory to Register Ratio**: Each element loaded from shared memory is reused across `TM × TN` operations
2. **Warp-Level Cooperation**: All 32 threads in a warp load complementary fragments simultaneously
3. **Register Reuse**: Fragments stay in registers across the entire BK iteration, minimizing shared memory traffic
4. **ILP Opportunity**: The nested loops expose instruction-level parallelism for the compiler

**5. Visualization of Warp Subtiling**

For a concrete example with `WM=64, WN=64, WMITER=2, WNITER=4`:

```
Warp Tile (64×64)
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  Subtile 0,0    │  Subtile 0,1    │  Subtile 0,2    │  Subtile 0,3    │
│   (32×16)       │   (32×16)       │   (32×16)       │   (32×16)       │
│  WSUBM=32       │                 │                 │                 │
│  WSUBN=16       │                 │                 │                 │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│  Subtile 1,0    │  Subtile 1,1    │  Subtile 1,2    │  Subtile 1,3    │
│   (32×16)       │   (32×16)       │   (32×16)       │   (32×16)       │
│                 │                 │                 │                 │
│                 │                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
     ↑
     Each subtile processed by 32 threads with TM=8, TN=4 thread tiles
     Thread layout: 4 threads/row × 8 rows = 32 threads per subtile
```

#### Comparison: 2D Block Tiling vs Warp Tiling

| Aspect | 2D Block Tiling | Warp Tiling |
|--------|----------------|-------------|
| **Granularity** | Thread-level only | Warp-level + Thread-level |
| **Register Usage** | `TM × TN` per thread | `WMITER × TM × WNITER × TN` per thread |
| **Shared Memory Reuse** | Each load → `TM × TN` reuse | Each load → `TM × TN` reuse (better ILP) |
| **Code Complexity** | Moderate | High (3-level hierarchy) |
| **Performance** | Good | Better (especially on larger matrices) |
| **Flexibility** | 2 tuning parameters (TM, TN) | 4 tuning parameters (WM, WN, WMITER, WNITER) |

The warp tiling approach exposes more optimization opportunities but requires careful tuning of additional parameters to balance register pressure, shared memory bandwidth, and instruction throughput.

## CUDA vs Triton: A Comprehensive Comparison

### Programming Model

| Aspect | CUDA | Triton |
|--------|------|--------|
| **Abstraction Level** | Low-level, explicit control | High-level, compiler-managed |
| **Thread Management** | Manual `threadIdx`, `blockIdx` | Implicit via `program_id` |
| **Memory Management** | Explicit `__shared__`, `__syncthreads()` | Automatic shared memory promotion |
| **Vectorization** | Manual `float4`, alignment | Automatic compiler optimization |
| **Register Allocation** | Manual arrays, explicit indexing | Compiler-managed |
| **Code Verbosity** | ~200-300 lines for optimized kernel | ~50-80 lines for same performance |

### Example: Loading a Tile

**CUDA**:
```cuda
__shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
int tx = threadIdx.x;
int ty = threadIdx.y;

// Each thread loads one element
As[ty * BLOCK_SIZE + tx] = A[globalRow * K + globalCol];
__syncthreads();

// Use shared memory
for (int k = 0; k < BLOCK_SIZE; k++) {
    sum += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
}
```

**Triton**:
```python
# Load tile - Triton handles shared memory automatically
offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
offs_k = k + tl.arange(0, BLOCK_SIZE)
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
a = tl.load(a_ptrs, mask=...)

# Compute - shared memory and synchronization handled automatically
accumulator += tl.dot(a, b)
```



## Performance Summary

<div id="performance-comparison"></div>

### Why cuBLAS is Still Faster

1. **Tensor Cores**: Hardware-accelerated matrix operations (for FP16/INT8)
2. **Warp-level Primitives**: Direct warp shuffle and cooperative operations
3. **Advanced Scheduling**: Better handling of memory latency
4. **Assembly Optimization**: Hand-tuned PTX/SASS code
5. **Architecture-specific Tuning**: Optimized per GPU generation

## Further Optimizations

Beyond what we covered:

1. **Warp-level Matrix Operations**: Use `wmma` (Warp Matrix Multiply-Accumulate)
2. **Asynchronous Memory Copy**: Overlap computation with memory transfers
3. **Double Buffering**: Load next tile while computing current
4. **Register Blocking**: More sophisticated register reuse patterns
5. **Mixed Precision**: FP16/BF16 compute with FP32 accumulation
6. **Persistent Kernels**: Keep GPU occupied across multiple operations


## References

### CUDA Resources
- [Simon Boehm's CUDA Matrix Multiplication](https://siboehm.com/articles/22/CUDA-MMM) - Original blog post that inspired this guide
- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Official CUDA documentation
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass) - NVIDIA's high-performance GEMM library
- [Dissecting the NVIDIA Volta GPU Architecture](https://arxiv.org/abs/1804.06826) - Deep dive into GPU architecture

### Triton Resources
- [Triton Language Documentation](https://triton-lang.org/) - Official Triton documentation
- [Triton Tutorial: Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) - Official GEMM tutorial
- [OpenAI Triton: A GPU Programming Language](https://openai.com/research/triton) - Original announcement
- [Triton GitHub Repository](https://github.com/openai/triton) - Source code and examples

### GPU Optimization
- [NVIDIA Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Best practices for CUDA optimization
- [GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/contributors) - Advanced GPU programming techniques
- [Understanding GPU Memory](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/) - Memory coalescing explained
- [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)


<script src="/assets/js/gemm-optimization-visualizer.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all visualizations (only if div exists)
    if (document.getElementById('naive-viz')) new NaiveKernelViz('naive-viz');
    if (document.getElementById('coalesced-matrix-viz')) new CoalescedMatrixViz('coalesced-matrix-viz');
    if (document.getElementById('memory-hierarchy-viz')) new MemoryHierarchyViz('memory-hierarchy-viz');
    if (document.getElementById('index-transform-viz')) new IndexTransformViz('index-transform-viz');
    if (document.getElementById('shared-memory-viz')) new SharedMemoryViz('shared-memory-viz');
    if (document.getElementById('1d-tiling-viz')) new Tiling1DViz('1d-tiling-viz');
    if (document.getElementById('1d-pipeline-viz')) new Tiling1DPipelineViz('1d-pipeline-viz');
    if (document.getElementById('2d-tiling-viz')) new Tiling2DViz('2d-tiling-viz');
    if (document.getElementById('vectorized-viz')) new VectorizedViz('vectorized-viz');
    if (document.getElementById('performance-comparison')) new PerformanceComparison('performance-comparison');
});
</script>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
