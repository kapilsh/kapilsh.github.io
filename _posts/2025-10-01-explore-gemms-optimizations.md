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

Matrix multiplication (GEMM - General Matrix Multiply) is a fundamental operations in deep learning. Understanding how to optimize GEMM kernels on GPUs provides good insights into GPU architecture, compute and memory hierarchy.

If the reader is remotely interested in CUDA or Triton programming, they have likely come across [this gemm of a post](https://siboehm.com/articles/22/CUDA-MMM) -- pun intended -- by Simon Boehm. There was [another amazing post](https://www.aleksagordic.com/blog/matmul) on GPU architecture from Aleksa Gordić. This one is more recent and I encountered it while writing this one. 

In this post, we'll build on these and walk through several optimization stages, each with some interactive visualizations to help understand the concepts.

## Code Repository and Structure

[Full implementation of all kernels](https://github.com/gpusgobrr/explore-gemm)

In the following sections, we will go over several steps in optimizing GEMM kernel, each time discussing a new concept. In the later sections, we will look at fp16/bf16 matmul kernels and tensor cores. In the end, we'll look into cutlass to show how it maps APIs to the GPU menory hierarchy. 

## GEMM Basics

GEMM (General Matrix Multiply) is a fundamental operation defined as:

$$C = \alpha AB + \beta C$$

where:
- $A$ is an $M \times K$ matrix
- $B$ is a $K \times N$ matrix
- $C$ is an $M \times N$ matrix (both input and output)
- $\alpha$ and $\beta$ are scalar coefficients


{: .prompt-tip}
> - The standard matrix product $C = AB$ is a special case with $\alpha = 1$ and $\beta = 0$
> - When $\beta \neq 0$, GEMM accumulates into a pre-existing matrix $C$
> - This formulation also enables fused operations, avoiding separate kernel launches. We'll see this in cutlass section.

### Computational Complexity

Each element $C[i,j]$ requires a dot product:

$$C[i,j] = \alpha \sum_{k=0}^{K-1} A[i,k] \times B[k,j] + \beta C[i,j]$$

For matrices of size $M \times K$, $K \times N$:
- Total dot products: $M \times N$
- Operations per dot product: $2K$ (K multiplies + K adds) + 3 scalar ops ($\alpha$, $\beta$, addition of $M \times N$ matrix)
- **Total FLOPs**: $2MNK + MK$ (dominated by dot products)

{: .prompt-tip}
> For a $4096 \times 4096$ matrix multiplication ($M = N = K = 4096$):
> - Total operations: $2 \times 4096^3 \approx 137$ GFLOPs
> - Memory required: $3 \times 4096^2 \times 4$ bytes $\approx$ 201 MB (float32)
> - **Arithmetic Intensity**: $\frac{137 \text{ GFLOPs}}{201 \text{ MB}} \approx 682$ FLOPs/byte

## Hardware Specifications

All benchmarks in this post were run on an **NVIDIA GeForce RTX 4090** 🚀. Below are the key specifications:

### SM Architecture

![SM architecture](/assets/4090_sm.png)

*Source: [NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)*

| Specification | RTX 4090 (Ada Lovelace) |
|---------------|--------------------------|
| **Architecture** | Ada Lovelace (TSMC 4nm) |
| **CUDA Cores** | 16,384 |
| **Streaming Multiprocessors (SMs)** | 128 |
| **GPU Boost Clock** | 2,520 MHz |
| **FP32 Performance** | 82.6 TFLOPS |
| **Tensor Cores** | 512 (4th Gen) |
| **Tensor Performance (FP8)** | 660.6 TFLOPS |
| **RT Cores** | 128 (3rd Gen) |
| **Memory Size** | 24 GB GDDR6X |
| **Memory Clock** | 21 Gbps |
| **Memory Bandwidth** | 1,008 GB/s |
| **L1 Cache / Shared Memory (Total)** | 16,384 KB (16 MB) |
| **L2 Cache** | 72 MB |
| **Shared Memory per SM** | 128 KB |
| **Registers per SM** | 256 KB |
| **TGP** | 450 W |
| **Transistor Count** | 76.3 Billion |
| **Die Size** | 608.5 mm² |

> **Important hardware details for GEMM Performance:**
> - **128 SMs** with 128 KB shared memory each
> - **82.6 TFLOPS FP32** theoretical peak
> - **1,008 GB/s** max memory bandwidth
> - **72 MB L2 cache**
> - **16,384 KB total L1/shared memory**
{: .prompt-tip}

## Naive Implementation

### Concept

The simplest approach to calculate GEMM assign each thread to compute one output element.

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

### Kernel

Here is a naive implementation for matrix multiply GEMM kernel:

```c
template <const uint block_size>
__global__ void sgemm_naive_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                   float alpha, const float *matrix_a,
                                   const float *matrix_b, float beta, float *output_matrix)
{
    // Map 1D thread ID to 2D output position for coalesced memory access
    const int output_row = blockIdx.x * block_size + (threadIdx.x % block_size);
    const int output_col = blockIdx.y * block_size + (threadIdx.x / block_size);

    // Boundary check for non-multiple of block size
    if (output_row < num_rows_a && output_col < num_cols_b)
    {
        float accumulator = 0.0f;
        for (int k_idx = 0; k_idx < num_cols_a; ++k_idx)
        {
            accumulator += matrix_a[output_row * num_cols_a + k_idx] *
                           matrix_b[k_idx * num_cols_b + output_col];
        }
        // C = α*(A@B)+β*C
        const int output_idx = output_row * num_cols_b + output_col;
        output_matrix[output_idx] = alpha * accumulator + beta * output_matrix[output_idx];
    }
}
```

### Caller

We operate on torch Tensors directly to call the above kernel:

```cpp

namespace {
    constexpr int ceil_div(int m, int n) {
        return (m + n - 1) / n;
    }
}

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

    // Configure kernel launch: 1D blocks with block_size^2 threads (32x32 = 1024 threads per block)
    constexpr uint block_size = 32;
    dim3 block_dim(block_size * block_size);
    dim3 grid_dim(ceil_div(num_rows_a, block_size),
                  ceil_div(num_cols_b, block_size));

    // Launch kernel
    sgemm_naive_kernel<block_size><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

```

### Performance Analysis

I ran the naive kernel and PyTorch GEMM kernels on my RTX 4090 to experiment. Let's see the results below:

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

Below are the full benchmark results comparing the naive CUDA kernel against PyTorch's optimized GEMM implementation across different shapes:

![Naive Only](/assets/explore_gemm_naive_only.png)

As we can see, the naive implementation is significantly slower than PyTorch's optimized kernel, achieving only ~1% of PyTorch's performance. 

## Global Memory Coalescing

Let's take a brief diggresion before we look into why naive kernel is so slow.

### Concept

Slides from [an NVidia GTC talk](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41101/) is a really good reference to understand the memory hierarchy details and why memory access patterns are the primary consideration when we think about GPU/CUDA performance. 

#### Some Basics

To understand memory coalescing, we need to understand GPU execution hierarchy:

1. **Threads**: Individual execution units in your CUDA kernel
2. **Warps**: Groups of 32 threads that execute the same instruction simultaneously (SIMT - Single Instruction, Multiple Thread)
3. **Thread Blocks**: Logical groupings of threads (up to 1024 threads) that share resources and can synchronize
4. **Streaming Multiprocessors (SMs)**: The physical processors on the GPU that execute thread blocks

**Warps are the fundamental unit of execution** such that all 32 threads in a warp execute the same instruction at the same time. Also, when these threads access consecutive memory addresses, the hardware can combine their memory requests into a single transaction. Modern GPU DRAM systems can fetch large contiguous blocks (32B, 64B, or 128B cache lines) in one transaction. Without coalescing, the same 32 accesses could require 32 separate transactions.

**In addition, other things to note about SMs include**:
- Each SM has limited resources (registers, shared memory)
- Multiple thread blocks compete for these resources
- SMs can switch between warps in a **single clock cycle**, enabling **latency hiding** while one warp waits for memory, another executes
- GEMM efficiency depends on keeping all warp schedulers busy with coalesced memory access patterns

#### Why is Naive Kernel Slow?

Let's add some debug info to see what threads map to what values on A, B

```diff
+            if (threadIdx.x < 64)
+            {
+                printf("Thread %d ; Warp %d: Multiplying A[%d][%d] * B[%d][%d] = C[%d][%d]\n",
+                       threadIdx.x, threadIdx.x / 32, output_row, k_idx,
+                       k_idx, output_col,
+                       output_row, output_col);
+            }
```

```text
Thread 0 ; Warp 0: Multiplying A[0][0] * B[0][0] = C[0][0]
Thread 1 ; Warp 0: Multiplying A[1][0] * B[0][0] = C[1][0]
...
Thread 30 ; Warp 0: Multiplying A[30][0] * B[0][0] = C[30][0]
Thread 31 ; Warp 0: Multiplying A[31][0] * B[0][0] = C[31][0]
Thread 32 ; Warp 1: Multiplying A[0][0] * B[0][1] = C[0][1]
Thread 33 ; Warp 1: Multiplying A[1][0] * B[0][1] = C[1][1]
...
Thread 62 ; Warp 1: Multiplying A[30][0] * B[0][1] = C[30][1]
Thread 63 ; Warp 1: Multiplying A[31][0] * B[0][1] = C[31][1]
```

Now that we have general hardware fundamentals, we can see that the naive kernel's memory access pattern is inefficient. For each thread in a warp, it is accessing `A[k][0]` values where k is the thread id in a warp. When threads in a warp access scattered memory locations, each access requires a separate memory transaction. It is totally solvable by memory coalescing such that we restructure the thread-to-output mapping so threads in the same warp access consecutive memory locations, enabling the hardware to combine multiple accesses into a single transaction.

Turns out the only change we need is swap % and / across row and col calculations for C.

```c
    // Map 1D thread ID to 2D output position for coalesced memory access
    // *** KEY CHANGE wrt NAIVE kernel
    const int output_row = blockIdx.x * block_size + (threadIdx.x / block_size);
    const int output_col = blockIdx.y * block_size + (threadIdx.x % block_size);
    // *** KEY CHANGE wrt NAIVE kernel
```

#### Memory Access Visualization

Let's visualize how the coalesced kernel accesses memory during matrix multiplication. Notice how threads in the same warp now access the **same row** of matrix A, enabling memory coalescing. We will take a simple example with `block_size=4`:

<div id="index-transform-viz"></div>

<div id="coalesced-matrix-viz"></div>

### Kernel

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

{: .prompt-info}
> The key change from the naive kernel:
> 
> Threads with consecutive `threadIdx.x` now access consecutive elements in the same row of A, enabling coalescing. I have removed the calling wrapper since it is the same as the last kernel and for brevity.

### Performance Analysis

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
> - Still only 5.8% of PyTorch's performance, but a significant step up
> 
> **Bandwidth:**
> - 7.71× bandwidth improvement (0.90 → 6.94 GB/s)
> - Better memory utilization through coalesced access patterns
{: .prompt-info}


Below are the full benchmark results comparing the naive CUDA kernel against PyTorch's optimized GEMM implementation across different shapes:

![Global coalesced](/assets/explore_gemm_global_coalesced.png)

We can see that performance improved but still way slower than the pytorch version.

## Shared Memory Caching

### Concept

#### GPU Memory Hierarchy

Before diving into shared memory optimization, let's understand the RTX 4090's memory hierarchy and why shared memory is so critical for performance:

<div id="memory-hierarchy-viz"></div>


Shared memory provides **bandwidth advantage** compared to global memory:

| Memory Type | Bandwidth | Latency | Location |
|------------|-----------|---------|----------|
| **Global Memory (GDDR6X)** | ~1 TB/s | 400-800 cycles | Off-chip |
| **Shared Memory (L1)** | ~14 TB/s | 20-30 cycles | On-chip |


{: .prompt-warning}
> Even with coalesced access, both the naive and coalesced kernels repeatedly read the same data from global memory:
> - Each element of matrix $A$ is read $N$ times (once per column of $B$)
> - Each element of matrix $B$ is read $M$ times (once per row of $A$)
> - For a 1024×1024 matrix multiplication: **each element is read ~1000 times from slow global memory**

#### Using the Shared Memory

The RTX 4090 has **128 KB of shared memory per SM** that serves as a fast on-chip cache physically located much closer to the compute units than global memory. This shared memory is partitioned among thread blocks (each block gets its own chunk), accessible by all threads within a block, and delivers dramatically better performance with ~14 TB/s bandwidth and 20-30 cycle latency compared to global memory's 1 TB/s and 400-800 cycle latency. 

With 128 SMs on the RTX 4090, there's a total of **16.4 MB of shared memory** distributed across the chip. Instead of repeatedly reading the same data slow GMEM, our optimization strategy loads tiles (chunks) of matrices A and B into this fast shared memory, computes partial results using the cached tile data with high reuse across multiple threads, then slides these tiles across the matrices to compute the final result—effectively transforming a bandwidth-bound problem into a compute-bound one.

<div id="shared-memory-viz"></div>

### Kernel

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

### Caller

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
> **Key Insight:**d
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

For the RTX 4090 (Compute Capability 8.9), below are the device specs:

| Specification | RTX 4090 Value |
|---------------|----------------|
| **Total Global Memory** | 24,209 MB |
| **Shared Memory per Block** | 49,152 bytes (48 KB) |
| **Shared Memory per SM** | 102,400 bytes (100 KB) |
| **Registers per Block** | 65,536 |
| **Warp Size** | 32 threads |
| **Max Threads per Block** | 1,024 |
| **Max Threads per SM** | 1,536 |
| **Number of SMs** | 128 |
| **Max Blocks per SM** | 24 |
| **Max Grid Dimensions** | [2,147,483,647, 65,535, 65,535] |
| **Max Thread Block Dimensions** | [1,024, 1,024, 64] |
| **Base Clock Rate** | 2,520 MHz |
| **Memory Clock Rate** | 10,501 MHz |
| **Memory Bus Width** | 384 bits |
| **L2 Cache Size** | 75,497,472 bytes (72 MB) |


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

## 1D Block Tiling

### Concept

Now that we understand that in the previous kernel, each thread was computing a single output element of matrix C, meaning each thread needed to load elements from shared memory repeatedly, with memory accesses dominating the execution time. Next, instead of each thread computing exactly one output element of the tile, each thread computes multiple output elements along one dimension. To support this, we fetch some data from SMEM into registers (for reuse) within each thread, reducing repeated SMEM loads.

> In essence, we are trying to improve the arithmetic intensity of the kernel, which effectively means computing more results per thread with the same loaded data i.e. increase FLOPS/byte
{: .prompt-tip}

<div id="1d-tiling-viz"></div>

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

### Performance Analysis

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

## 2D Block Tiling

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

### Performance Analysis

Looking at the benchmark results for 4096×4096 matrices, the 2D block tiling kernel shows further performance gains:



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

#### Performance Across Matrix Sizes

![2D block tiling performance](/assets/explore_gemm_2d_blocktiling_performance.png)

As next steps, when we check our kernel in Nsight compute, we get more pointers to improve performance:

> - **L1TEX Global Store Access Pattern:** The memory access pattern for global stores to L1TEX might not be optimal. On average, only 4.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global stores.
> - **Shared Load Bank Conflicts:** The memory access pattern for shared loads might not be optimal and causes on average a 5.6 - way bank conflict across all 167772160 shared load requests.This results in 503316480 bank conflicts, which represent 53.57% of the overall 939534036 wavefronts for shared loads. Check the Source Counters section for uncoalesced shared loads.
> - **Uncoalesced Shared Accesses:** This kernel has uncoalesced shared accesses resulting in a total of 369098752 excessive wavefronts (37% of the total 1006632960 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide has an example on optimizing shared memory accesses.

![NCU After 2D Tiling](/assets/explore_gemm_ncu_after_2d_tiling.png)

## Vectorized Memory Access

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

#### Performance Across Matrix Sizes

![Vectorized performance](/assets/explore_gemm_performance_vectorized.png)


## Warp Tiling

### Concept

After optimizing thread-level and block-level tiling, the next step is **warp-level tiling**—exploiting the natural 32-thread warp execution unit in NVIDIA GPUs to achieve better register reuse and computation efficiency.

#### What is Warp Tiling?

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

#### Memory Access Pattern Comparison

```
Without Warp Tiling (2D Block Tiling):
  For each K iteration:
    Each thread: Load 1 element from As, 1 element from Bs → Compute TM×TN FMAs
    Shared memory accesses: O(NUM_THREADS) per K iteration

With Warp Tiling:
  For each K iteration:
    Each warp: Load WSUBM×WSUBN fragment → Store in registers
    Each thread in warp: Compute TM×TN FMAs using register data
    Repeat for all warp subtiles
    Shared memory accesses: O(NUM_THREADS) per K iteration, but higher compute per access
```

The critical difference: with warp tiling, each shared memory load is amortized over more computation because the data stays in registers for the entire warp tile processing loop.

<div id="warp-tiling-viz"></div>

{: .prompt-tip }
> [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html) has a lot more details and is pretty comprehensive in explaining different layers of tiling.

### Kernel

```cuda
constexpr int WARPSIZE = 32;

// ==================== HELPER FUNCTIONS ====================

// Load data from global memory to shared memory with vectorized access
template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const float *matrix_a, const float *matrix_b,
                               float *tile_a, float *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b)
{
    // Load tile_a with float4 vectorized loads and transpose
    for (uint offset = 0; offset + row_stride_a <= BM; offset += row_stride_a) {
        const float4 tmp_a = reinterpret_cast<const float4*>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 4])[0];
        // Transpose while storing to shared memory
        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
    }

    // Load tile_b with float4 vectorized loads
    for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b) {
        reinterpret_cast<float4*>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4*>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 4])[0];
    }
}

// Process warptile: compute using warp subtiling
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                   const float *tile_a, const float *tile_b,
                                   const uint warp_row, const uint warp_col,
                                   const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    // Loop over BK dimension
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
        // Populate registers for entire warptile
        // Load WMITER * TM elements from tile_a (covers all warp subtiles in M dimension)
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint i = 0; i < TM; ++i) {
                register_m[wsub_row_idx * TM + i] =
                    tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
            }
        }

        // Load WNITER * TN elements from tile_b (covers all warp subtiles in N dimension)
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (uint i = 0; i < TN; ++i) {
                register_n[wsub_col_idx * TN + i] =
                    tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i];
            }
        }

        // Execute warptile matmul across all warp subtiles
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
                // Calculate per-thread results for this warp subtile
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

// ==================== WARPTILING KERNEL ====================

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_warptiling_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                        float alpha, const float *matrix_a, const float *matrix_b,
                        float beta, float *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // Warp-level placement within threadblock
    const uint warp_idx = threadIdx.x / WARPSIZE;          // Which warp this thread belongs to
    const uint warp_col = warp_idx % (BN / WN);            // Warp's column in block tile
    const uint warp_row = warp_idx / (BN / WN);            // Warp's row in block tile

    // Warp subtile dimensions
    // WMITER: number of subtile iterations in M dimension per warp
    // Formula: total warp work (WM*WN) / work per thread per iteration (WARPSIZE*TM*TN*WNITER)
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;  // Warp subtile height
    constexpr uint WSUBN = WN / WNITER;  // Warp subtile width

    // Thread placement within warp subtile
    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;           // [0, 31]
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN); // Column within subtile
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN); // Row within subtile

    // Shared memory for block tiles
    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    // Position matrix pointers at start of this block's tile
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    // Position output pointer at this warp's tile
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b + block_col * BN + warp_col * WN;

    // Thread indices for loading data into shared memory
    // Load 4 floats at a time using float4
    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4);
    constexpr uint row_stride_a = (NUM_THREADS * 4) / BK;

    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4);
    constexpr uint row_stride_b = NUM_THREADS / (BN / 4);

    // Thread-local storage in registers
    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    // Cache for warptile computation
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    // Outer loop over block tiles along K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        // Load block tile from global memory to shared memory
        load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        // Process warptile from shared memory
        process_warp_tile<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        // Advance to next block tile
        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        __syncthreads();
    }

    // ==================== WRITE RESULTS TO GLOBAL MEMORY ====================

    // Write results for each warp subtile with vectorized stores
    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            // Move pointer to current warp subtile
            float *matrix_c_interim = matrix_c + (wsub_row_idx * WSUBM) * num_cols_b +
                                     wsub_col_idx * WSUBN;

            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1) {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {
                    // Load C vector into registers
                    float4 tmp_c = reinterpret_cast<float4*>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                         thread_col_in_warp * TN + res_idx_n])[0];

                    // Perform GEMM update in registers
                    const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                       wsub_col_idx * TN + res_idx_n;
                    tmp_c.x = alpha * thread_results[res_idx + 0] + beta * tmp_c.x;
                    tmp_c.y = alpha * thread_results[res_idx + 1] + beta * tmp_c.y;
                    tmp_c.z = alpha * thread_results[res_idx + 2] + beta * tmp_c.z;
                    tmp_c.w = alpha * thread_results[res_idx + 3] + beta * tmp_c.w;

                    // Write back with vectorized store
                    reinterpret_cast<float4*>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                         thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
                }
            }
        }
    }
}
```

### Caller

```
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
void sgemm_warptiling(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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

    TORCH_CHECK(matrix_b.size(0) == num_cols_a,
                "Matrix dimensions must match: A is MxK, B must be KxN");

    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");

    // Validate dimensions are multiples of tile sizes
    TORCH_CHECK(num_rows_a % BM == 0, "Matrix A rows must be multiple of ", BM);
    TORCH_CHECK(num_cols_a % BK == 0, "Matrix A cols must be multiple of ", BK);
    TORCH_CHECK(num_cols_b % BN == 0, "Matrix B cols must be multiple of ", BN);

    // Validate warptiling constraints
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    static_assert(WMITER * WSUBM == WM, "WMITER * WSUBM must equal WM");
    static_assert(WNITER * WSUBN == WN, "WNITER * WSUBN must equal WN");
    static_assert((BM % WM == 0) && (BN % WN == 0), "Block tile must be divisible by warp tile");
    static_assert((WSUBM % TM == 0) && (WSUBN % TN == 0), "Warp subtile must be divisible by thread tile");

    // Get raw device pointers
    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Configure kernel launch
    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(CEIL_DIV(num_cols_b, BN), CEIL_DIV(num_rows_a, BM));

    // Launch kernel
    sgemm_warptiling_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

// Default configuration wrapper
void sgemm_warptiling_default(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                               torch::Tensor &output_matrix, float alpha, float beta)
{
    // Default configuration: BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4, NUM_THREADS=128
    // This gives: WMITER=2, WSUBM=32, WSUBN=16
    // Warps per block: (128*128)/(64*64) = 4 warps
    // Threads needed: 4 warps * 32 threads/warp = 128 threads
    sgemm_warptiling<128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}

```

### Performance Analysis

The warp tiling kernel demonstrates **impressive performance gains** at large matrix sizes, achieving the **best performance of all our hand-written kernels**. However, it shows **interesting complexity tradeoffs** at smaller sizes.

> **Performance at 4096×4096:**
> - **1.17× TFLOPS improvement** over vectorized kernel (39.07 → 45.82 TFLOPS)
> - **1.17× bandwidth improvement** (57.24 → 67.12 GB/s)
> - **54.4% of PyTorch's performance** (84.19 TFLOPS)
> - **70.1× faster than naive** (0.654 → 45.82 TFLOPS)
{: .prompt-info}

**Comparison vs Previous Kernels (4096×4096):**

| Kernel | Time (ms) | TFLOPS | Bandwidth (GB/s) | vs PyTorch | Speedup over Naive |
|--------|-----------|---------|------------------|------------|-------------------|
| Naive | 210.29 | 0.65 | 0.96 | 0.8% | 1.0× |
| Coalesced | 26.95 | 5.10 | 7.47 | 6.1% | 7.8× |
| Shared Memory | 22.40 | 6.14 | 8.99 | 7.3% | 9.4× |
| 1D Block Tiling | 7.42 | 18.52 | 27.13 | 22.0% | 28.3× |
| 2D Block Tiling | 4.45 | 30.89 | 45.25 | 36.7% | 47.2× |
| Vectorized | 3.52 | 39.07 | 57.24 | 46.4% | 59.8× |
| **Warp Tiling** | **3.00** | **45.82** | **67.12** | **54.4%** | **70.1×** |
| PyTorch | 1.63 | 84.19 | 123.33 | 100% | 128.8× |

#### Performance Across Matrix Sizes

![Across matrices warptiling](/assets/explore_gemm_warptiling_performance.png)


| Matrix Size | Time (ms) | TFLOPS | vs PyTorch | Speedup vs 2D Tiling |
|-------------|-----------|---------|------------|---------------------|
| 128×128 | 0.021 | 0.20 | 40.2% | 0.81× |
| 512×512 | 0.058 | 4.60 | 18.9% | 0.79× |
| 1024×1024 | 0.103 | 20.77 | 29.8% | 0.90× |
| 2048×2048 | 0.396 | 43.38 | 52.6% | 1.14× |
| 4096×4096 | 3.00 | 45.82 | 54.4% | 1.17× |
| 8192×8192 | 25.69 | 42.80 | 50.9% | 1.12× |


## Supporting 16-bit Precision Types

### Concept

So far, we have worked only on fp32 kernels. Most modern workloads increasingly use 16-bit floating-point formats (FP16 and BF16) to reduce memory bandwidth requirements and increase throughput. While our warp tiling kernel works for FP32, we can extend it to support lower-precision computations in 16-bit.

| Format | Sign | Exponent | Mantissa | Range | Precision |
|--------|------|----------|----------|-------|-----------|
| **FP32** | 1 bit | 8 bits | 23 bits | ±3.4e38 | ~7 decimal digits |
| **FP16** | 1 bit | 5 bits | 10 bits | ±65,504 | ~3 decimal digits |
| **BF16** | 1 bit | 8 bits | 7 bits | ±3.4e38 | ~2 decimal digits |

**Our Approach:**
Load inputs as FP16/BF16 → Convert to FP32 in registers → Accumulate in FP32 → Write FP32 output

This gives us the bandwidth benefits of 16-bit with the numerical stability of 32-bit accumulation.

#### Implementation Strategy

The multi-dtype kernel extends warp tiling with three key changes:

1. **Type-parameterized loads**: Template parameter `InputType` can be `float`, `half`, or `nv_bfloat16`
2. **Dtype-specific vectorization**: `half2`/`nv_bfloat162` for 16-bit types vs `float4` for FP32
3. **Conversion on load**: Convert 16-bit values to FP32 immediately after loading into registers

```
Memory Hierarchy with Mixed Precision:
┌─────────────────────────────────────────────────────────────┐
│ Global Memory (DRAM)                                         │
│ InputType (FP16/BF16/FP32) - Lower bandwidth for 16-bit     │
└───────────────────────┬─────────────────────────────────────┘
                        ↓ Block loads tile
┌─────────────────────────────────────────────────────────────┐
│ Shared Memory (On-chip)                                      │
│ InputType (same as global) - 2× capacity with 16-bit        │
└───────────────────────┬─────────────────────────────────────┘
                        ↓ Warp loads fragments + converts
┌─────────────────────────────────────────────────────────────┐
│ Register File (Warp-level cache)                             │
│ float (always FP32) - Computation in full precision         │
└───────────────────────┬─────────────────────────────────────┘
                        ↓ Thread computes TM×TN tile
┌─────────────────────────────────────────────────────────────┐
│ Accumulator Registers                                        │
│ float (FP32 accumulation) - Numerical stability             │
└───────────────────────┬─────────────────────────────────────┘
                        ↓ Write back
┌─────────────────────────────────────────────────────────────┐
│ Global Memory Output                                         │
│ float (FP32 output) - Full precision results                │
└─────────────────────────────────────────────────────────────┘
```

### Kernel

The multi-dtype kernel uses C++ templates and compile-time specialization to support different input types while maintaining optimal performance.

#### Type Traits for Vectorization

First, we define type traits to map each input type to its vectorized form:

```cuda
// Helper to get the vectorized type for each input type
template <typename T>
struct VecType {};

template <>
struct VecType<float> {
    using type = float4;  // Load 4 floats at once
};

template <>
struct VecType<half> {
    using type = half2;  // Load 2 halfs at once
};

template <>
struct VecType<nv_bfloat16> {
    using type = nv_bfloat162;  // Load 2 bf16s at once
};

// Vector size in elements
template <typename T>
constexpr int vec_size() { return 4; }

template <>
constexpr int vec_size<half>() { return 2; }

template <>
constexpr int vec_size<nv_bfloat16>() { return 2; }
```

#### Type Conversion Helpers

CUDA provides intrinsics to convert 16-bit types to FP32:

```cuda
// Type conversion helper for 16-bit types to float
__device__ __forceinline__ float to_float(half x) {
    return __half2float(x);
}

__device__ __forceinline__ float to_float(nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float to_float(float x) {
    return x;  // No-op for FP32
}
```

#### Dtype-Aware Loading

The `load_from_gmem` function now uses `constexpr` to specialize vector operations based on input type:

```cuda
template <typename InputType, const int BM, const int BN, const int BK,
          const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const InputType *matrix_a, const InputType *matrix_b,
                               InputType *tile_a, InputType *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b)
{
    constexpr int VEC_SIZE = vec_size<InputType>();
    using VecT = typename VecType<InputType>::type;

    // Load tile_a with vectorized loads and transpose
    for (uint offset = 0; offset + row_stride_a <= BM; offset += row_stride_a) {
        const VecT tmp_a = reinterpret_cast<const VecT*>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * VEC_SIZE])[0];

        // Transpose while storing to shared memory
        if constexpr (VEC_SIZE == 4) {
            // FP32 case: load 4 elements
            tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
            tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
            tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
            tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
        } else {
            // FP16/BF16 case: load 2 elements
            tile_a[(inner_col_a * 2 + 0) * BM + inner_row_a + offset] = tmp_a.x;
            tile_a[(inner_col_a * 2 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        }
    }

    // Load tile_b with vectorized loads (no transpose)
    for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b) {
        reinterpret_cast<VecT*>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * VEC_SIZE])[0] =
            reinterpret_cast<const VecT*>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * VEC_SIZE])[0];
    }
}
```

**Key features:**
- Uses `constexpr if` to branch at compile-time based on vector size
- Maintains the same transpose-on-load optimization for matrix A
- No runtime overhead—compiler generates separate code paths for each type

#### Warp Tile Processing with Conversion

The `process_warp_tile` function converts inputs to FP32 as they're loaded into registers:

```cuda
template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const InputType *tile_a, const InputType *tile_b,
                                  const uint warp_row, const uint warp_col,
                                  const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    // Loop over BK dimension
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
        // Load WMITER * TM elements from tile_a and convert to float
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx) {
            for (uint i = 0; i < TM; ++i) {
                // Convert to float for computation (works for float, half, nv_bfloat16)
                register_m[wsub_row_idx * TM + i] = to_float(
                    tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i]);
            }
        }

        // Load WNITER * TN elements from tile_b and convert to float
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx) {
            for (uint i = 0; i < TN; ++i) {
                register_n[wsub_col_idx * TN + i] = to_float(
                    tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i]);
            }
        }

        // Execute warptile matmul (same as before - all FP32 now)
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

**Conversion happens once per K iteration**, not per FMA operation, so the overhead is minimal. After conversion, the computation is identical to the FP32 kernel.

#### Main Kernel

The main kernel template is nearly identical to the FP32 warp tiling kernel, just templated on `InputType`:

```cuda
template <typename InputType, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_warptiling_multidtype_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                   float alpha, const InputType *matrix_a, const InputType *matrix_b,
                                   float beta, float *matrix_c)
{
    // Same warp and thread placement as FP32 kernel
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;
    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    // Warp subtile dimensions
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    // Thread placement within warp subtile
    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    // Shared memory for block tiles (InputType - 16-bit or 32-bit)
    __shared__ InputType tile_a[BM * BK];
    __shared__ InputType tile_b[BK * BN];

    // Position matrix pointers
    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b + block_col * BN + warp_col * WN;

    // Thread-local storage (always FP32 for accumulation)
    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    // Outer loop over block tiles along K dimension
    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        // Load block tile from global memory to shared memory
        load_from_gmem<InputType, BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        // Process warptile with conversion to FP32
        process_warp_tile<InputType, BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        matrix_a += BK;
        matrix_b += BK * num_cols_b;
        __syncthreads();
    }

    // Write results (always FP32 output)
    // ... vectorized stores as before ...
}
```

### Caller

We provide three public API functions—one for each supported dtype:

```cpp
// FP32 version - delegate to original FP32 warptiling
void sgemm_warptiling_fp32(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");

    // Delegate to the original FP32 warptiling kernel (no conversion overhead)
    sgemm_warptiling_default(matrix_a, matrix_b, output_matrix, alpha, beta);
}

// FP16 version - use the multi-dtype kernel
void sgemm_warptiling_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat16, "Matrix A must be float16");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat16, "Matrix B must be float16");

    sgemm_warptiling_multidtype<half, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}

// BF16 version - use the multi-dtype kernel
void sgemm_warptiling_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta)
{
    TORCH_CHECK(matrix_a.dtype() == torch::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b.dtype() == torch::kBFloat16, "Matrix B must be bfloat16");

    sgemm_warptiling_multidtype<nv_bfloat16, 128, 128, 16, 64, 64, 4, 8, 4, 128>(
        matrix_a, matrix_b, output_matrix, alpha, beta);
}
```

**Design rationale:**
- **Separate functions**: Cleaner API—user chooses dtype explicitly
- **Type safety**: Compile-time checks ensure A and B have matching dtypes
- **FP32 optimization**: FP32 path uses the original kernel (no conversion overhead)
- **Output always FP32**: Consistent with cuBLAS and PyTorch mixed-precision training

### Performance Analysis

Performance results will be added here once benchmarking is complete.

## Further Optimizations

Beyond what we covered:

1. **Warp-level Matrix Operations**: Use `wmma` (Warp Matrix Multiply-Accumulate)
2. **Asynchronous Memory Copy**: Overlap computation with memory transfers
3. **Double Buffering**: Load next tile while computing current
4. **Register Blocking**: More sophisticated register reuse patterns
5. **Mixed Precision**: FP16/BF16 compute with FP32 accumulation
6. **Persistent Kernels**: Keep GPU occupied across multiple operations


## References

- [Simon Boehm's CUDA Matrix Multiplication](https://siboehm.com/articles/22/CUDA-MMM) - Original blog post that inspired this article
- [Simon's GEMM Repo](https://github.com/siboehm/SGEMM_CUDA/tree/master)
- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [Triton Tutorial: Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [NVIDIA Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Understanding GPU Memory](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [Reddit: What's the point of warp-level gemm](https://www.reddit.com/r/CUDA/comments/1hk4410/whats_the_point_of_warplevel_gemm/)
- [Implementing Strassen’s Algorithm with CUTLASS on NVIDIA Volta GPUs](https://arxiv.org/pdf/1808.07984)
- [Lei Mao's Blog](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)
- [Advanced Matrix Multiplication Optimization on NVIDIA GPUs](https://salykova.github.io/gemm-gpu)
- [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
- [NVIDIA ADA GPU ARCHITECTURE](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Modal GPU Glossary](https://modal.com/gpu-glossary/readme)



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
    if (document.getElementById('warp-tiling-viz')) new WarpTilingViz('warp-tiling-viz');
    if (document.getElementById('performance-comparison')) new PerformanceComparison('performance-comparison');
});
</script>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
