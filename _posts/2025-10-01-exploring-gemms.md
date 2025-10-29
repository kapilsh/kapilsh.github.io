---
title: "Exploring GEMMs - Naive to Cutlass"
description: >-
  Interactive walkthrough of optimization techniques for GEMMs
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

Let's take a brief digression before we look into why naive kernel is so slow.

### Concept

Apart from [PMPP book](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311), slides from [an NVidia GTC talk](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41101/) is a really good reference to understand the memory hierarchy details and why memory access patterns are the primary consideration when we think about GPU/CUDA performance. For recent architectures, cutlass/cute documentation is really good reference as well.

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

> **Problem**: Threads access memory in a scattered, non-coalesced pattern.
{: .prompt-warning}

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

Before diving into shared memory optimization, let's understand the RTX 4090's memory hierarchy and why shared memory is so critical for performance. Shared memory provides significant **bandwidth advantage** compared to global memory, since it on chip. As a result, latency of loads/stores are at least one order of magnitude better compared to global memory. I could not find any public numbers on the differences though.

{: .prompt-warning}
> Even with coalesced access, both the naive and coalesced kernels repeatedly read the same data from global memory:
> - Each element of matrix $A$ is read $N$ times (once per column of $B$)
> - Each element of matrix $B$ is read $M$ times (once per row of $A$)

#### Using the Shared Memory

RTX 4090 has **128 KB of shared memory per SM** that serves as a fast on-chip cache/shared memory. This shared memory is partitioned among thread blocks (each block gets its own chunk), accessible by all threads within a block. With 128 SMs on the RTX 4090, there's a total of **16.4 MB of shared memory** distributed across the chip. Instead of repeatedly reading the same data from slow global memory, as an optimization strategy, we can load tiles (chunks) of matrices A and B into this fast shared memory, compute partial results using the cached tile data with high reuse across multiple threads, then slide these tiles across the matrices to compute the final result—effectively transforming a bandwidth-bound problem into a compute-bound one. Let's look at how this works below:

<div id="shared-memory-viz"></div>

{: .prompt-info}
> NOTE: We need to synchronize threads after both loading the data into shared memory and also after finishing the tiled matmuls to ensure: 
> 1) All data is loaded before we do the matmul calculations
> 2) All calculated data is stored back into matrix C before going to next tiles

### Kernel

```c

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
    const uint global_row = block_row * BLOCKSIZE + thread_row;
    const uint global_col = block_col * BLOCKSIZE + thread_col;

    // Move pointers to the starting position for this block
    matrix_a += block_row * BLOCKSIZE * num_cols_a; // row=block_row, col=0
    matrix_b += block_col * BLOCKSIZE;              // row=0, col=block_col
    matrix_c += block_row * BLOCKSIZE * num_cols_b + block_col * BLOCKSIZE;

    float accumulator = 0.0f;

    // Loop over all tiles along K dimension
    for (int tile_idx = 0; tile_idx < num_cols_a; tile_idx += BLOCKSIZE)
    {
        // Load tile from matrix A into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if (global_row < num_rows_a && (tile_idx + thread_col) < num_cols_a)
        {
            tile_a[thread_row * BLOCKSIZE + thread_col] =
                matrix_a[thread_row * num_cols_a + thread_col];
        }
        else
        {
            tile_a[thread_row * BLOCKSIZE + thread_col] = 0.0f;
        }

        // Load tile from matrix B into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if ((tile_idx + thread_row) < num_cols_a && global_col < num_cols_b)
        {
            tile_b[thread_row * BLOCKSIZE + thread_col] =
                matrix_b[thread_row * num_cols_b + thread_col];
        }
        else
        {
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

    // Write result to global memory with bounds checking: C = α*(A@B)+β*C
    if (global_row < num_rows_a && global_col < num_cols_b)
    {
        matrix_c[thread_row * num_cols_b + thread_col] =
            alpha * accumulator + beta * matrix_c[thread_row * num_cols_b + thread_col];
    }
}
```

{: .prompt-info}
> Again, we have removed the caller code since it looks similar to previous kernels

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
{: .prompt-info}

Below are all the results from the benchmarking:

![Kernel with shared mem](/assets/explore_gemm_shared_mem.png)

## Understanding GPU Occupancy

Before diving into more advanced optimizations, we need to understand **occupancy**—a critical metric that determines how well we utilize the GPU's resources.

### What is Occupancy?

**Occupancy** is the ratio of active warps to the maximum number of possible warps per SM:

$$\text{Occupancy} = \frac{\text{Active Warps per SM}}{\text{Maximum Warps per SM}}$$

Here is a really nice diagram from [Modal GPU Glossary](https://modal.com/gpu-glossary/perf/occupancy)

![GPU Glossary occupancy](/assets/explore_gemms_occupancy_modal.png)

For the RTX 4090 (Compute Capability 8.9), below are the relevant device specs:

| Specification | RTX 4090 Value |
|---------------|----------------|
| **Registers per Block** | 65,536 |
| **Max Threads per Block** | 1,024 |
| **Max Threads per SM** | 1,536 |
| **Number of SMs** | 128 |
| **Max Blocks per SM** | 24 |
| **Shared Memory per SM** | 102,400 bytes (100 KB) |
| **Shared Memory per Block** | 49,152 bytes (48 KB) |

You can load these details directly using `cudaDeviceProp`:

```cpp
std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);

    std::cout << "Device " << i << ": " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
    std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
    std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "  Number of SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n";
    std::cout << "  Max grid dimensions: ["
                << prop.maxGridSize[0] << ", "
                << prop.maxGridSize[1] << ", "
                << prop.maxGridSize[2] << "]\n";
    std::cout << "  Max threads dim (block): ["
                << prop.maxThreadsDim[0] << ", "
                << prop.maxThreadsDim[1] << ", "
                << prop.maxThreadsDim[2] << "]\n";
    std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz\n";
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
    std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor << " registers\n";
    std::cout << "  Registers per Block: " << prop.regsPerBlock;
    std::cout << "  Registers per Block: " << prop.warpSize;

    std::cout << std::endl;
}
```

### Why Occupancy Matters

GPUs hide memory latency through **massive parallelism**. When one warp waits for memory, the SM immediately switches to execute another warp. Occupancy does not enable performance directly but primariy through latency hiding. Higher occupancy effectively means more warps available for compute and we can hide latency more effectively. If occupancy is low, hardware sits idle waiting for data. However, **occupancy is not everything**. A kernel with 100% occupancy but poor memory access patterns can still perform poorly since that could lead to lower register/shared memory per thread. So, it is imporatant to keep into account other factors apart from just optimizing for occupancy. 

#### Key Considerations

Now, each thread gets registers from the SM's register file. More registers per thread implies fewer concurrent threads.

$$\text{Max Threads} = \min\left(\frac{65536 \text{ registers/SM}}{\text{registers per thread}}, 1536\right)$$

| Registers/Thread | Max Threads | Active Warps | Occupancy |
|------------------|-------------|--------------|-----------|
| 32 | 1,536 (limited by max) | 48 | 100% |
| 64 | 1,024 | 32 | 66.7% |
| 128 | 512 | 16 | 33.3% |

In addition, shared memory is partitioned among thread blocks on the same SM.

$$\text{Max Blocks} = \min\left(\frac{102400 \text{ bytes/SM}}{\text{shared memory per block}}, \underbrace{32}_{\text{hardware limit}}\right)$$

> Note: 32 is **maximum resident blocks per SM** (hardware limit for RTX 4090)
{: .prompt-info}

| Shared Memory/Block | Max Blocks | Notes |
|---------------------|------------|-------|
| 0 KB | 32 | No shared memory usage |
| 24 KB | 4 | Good for moderate tiling |
| 48 KB (max for RTX 4090) | 2 | Large tiles, fewer blocks |

Lastly, number of threads per block affects how many blocks can fit on an SM and hence affecting occupancy.

| Threads/Block | Warps/Block | Max Blocks/SM | Active Warps | Occupancy |
|---------------|-------------|---------------|--------------|-----------|
| 128 | 4 | 12 | 48 | 100% |
| 256 | 8 | 6 | 48 | 100% |
| 512 | 16 | 3 | 48 | 100% |
| 1024 | 32 | 1 | 32 | 66.7% |

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
3. **Blocks per SM (shared memory limit)**: $\lfloor 102400 / 8192 \rfloor = 12$ blocks
4. **Blocks per SM (block limit)**: 32 blocks (max)
5. **Actual blocks per SM**: $\min(1, 12, 32) = 1$ block

**Active warps**: $1 \text{ block} \times 32 \text{ warps/block} = 32 \text{ warps}$

**Occupancy**: $\frac{32}{48} = 66.7\%$

> **Problem**: Our large block size (1,024 threads) limits us to only 1 block per SM, resulting in just 66.7% occupancy. 66.7% occupancy is not necessarily bad but let's see more details using nsight-compute
{: .prompt-warning}

### Nsight Compute Profiling

First let's confirm our occupancy calculation and it matches.

![shared_mem_occupancy](/assets/explore_gemm_shared_mem_occupancy.png)

Now, looking at the summary - it provides some ideas on why the kernel is still slow:

![ncu summary shared mem](/assets/explore_gemm_summary_shared_mem_ncu.png)

Next, looking at the instruction mix, we can see that LDS dominates the instruction mix -- LDS = load within shared memory window -- **which is not good.**

![LDS Too much](/assets/explore_gemm_lds_too_much.png)

> Use Tip I Found: NVIDIA provides [`cudaOccupancyMaxActiveBlocksPerMultiprocessor()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g5a5d67a3c907371559ba692195e8a38c) to calculate theoretical occupancy at runtime. Most profilers (Nsight Compute) also report achieved occupancy.
{: .prompt-tip}

## 1D Block Tiling

### Concept

Now that we understand that in the previous kernel, each thread was computing a single output element of matrix C, meaning each thread needed to load elements from shared memory repeatedly, with memory accesses dominating the execution time. 

Next, instead of each thread computing exactly one output element of the tile, each thread computes multiple output elements along one dimension. To support this, we fetch some data from SMEM into registers (for reuse) within each thread, reducing repeated SMEM loads.

> In essence, we are trying to improve the arithmetic intensity of the kernel, which effectively means computing more results per thread with the same loaded data i.e. increase FLOPS/byte
{: .prompt-tip}

To accomplish this, we effectively just add TM accumulators i.e. 

```c
float thread_results[TM] = {0.0f};
```
and later we calculate TM outputs per thread

```c
for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
    float b_tmp = tile_b[dot_idx * BN + thread_col];
    for (uint res_idx = 0; res_idx < TM; ++res_idx) {
        thread_results[res_idx] +=
            tile_a[(thread_row * TM + res_idx) * BK + dot_idx] * b_tmp;
    }
}

```

<div id="1d-tiling-viz"></div>

### Kernel

Full listing below:

{: .prompt-info}
> NOTE: [Simon Boehm's Kernels](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/4_kernel_1D_blocktiling.cuh) did not handle bounds check i.e. non block multiple kernels will be incorrect. I tried to add the bounds check here but we can see that it starts to make the code fairly complex. We will the bounds check as long as we can but will drop it later sections to focus on core concepts.

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

```c

void sgemm_blocktiling_1d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta)
{
    // ... rest of the code is similar 
    // ...
    // Template parameters for kernel
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    // Configure kernel launch
    // Tiling strategy : 
    // - BM x BK from A, BK x BN from B
    // - TM values per thread
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
- Each value from A (loaded into `register_m[TM]`) is reused across TN computations
- Each value from B (loaded into `register_n[TN]`) is reused across TM computations
- This forms an "outer product" pattern and increases arithmetic intensity

2D tiling enables each thread to compute TM × TN outputs (e.g., 8 × 8 = 64 outputs) instead of just TM outputs (e.g., 8 outputs) in 1D tiling, resulting in TN× more computation per thread with more register usage. Of course, we would need to keep within the register memory bounds but this is how we can increase arithmetic intensity further by reusing shared memory. 

In the code, I implemented two separate kernels such that a **Main Kernel** (No Bounds Checking) handles all **interior blocks** where every memory access is guaranteed to be in-bounds -- which helps with zero thread divergence since all threads can execute the same logic. An **Edge Kernel** (With Bounds Checking) handles **boundary blocks** at the right edge, bottom edge, and corner. This structures the code well but the code is starting to look fairly complex at this point.

### Kernel

We show just the main kernel -- I had Claude put a bunch of comments on this kernel so steps are clear. 

```cuda
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                            float alpha, const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c)
{
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    const uint thread_row = threadIdx.x / (BN / TN);
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint num_threads = (BM / TM) * (BN / TN);

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
#pragma unroll
        for (uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads)
        {
            uint load_idx = threadIdx.x + load_offset;
            uint a_row = load_idx / BK;
            uint a_col = load_idx % BK;
            tile_a[load_idx] = matrix_a[a_row * num_cols_a + a_col];
        }

#pragma unroll
        for (uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads)
        {
            uint load_idx = threadIdx.x + load_offset;
            uint b_row = load_idx / BN;
            uint b_col = load_idx % BN;
            tile_b[load_idx] = matrix_b[b_row * num_cols_b + b_col];
        }

        __syncthreads();

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
            }

            for (uint i = 0; i < TN; ++i)
            {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            // Each thread calculates TM x TN outputs
            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                {
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

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

You will notice we are using `#pragma unroll` directives in the code. **What does `#pragma unroll` do?**

It is a compiler optimization that can, for example, replace a piece of code like

```cpp
for (int i = 0; i < 5; i++ )
    b[i] = i;
```
with

```cpp
b[0] = 0;
b[1] = 1;
b[2] = 2;
b[3] = 3;
b[4] = 4;
```

by putting #pragma unroll directive right before the loop. The good thing about the unrolled version is that it involves less processing load for the processor. 


### Caller

```cuda

void sgemm_blocktiling_2d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta)
{
    // ... same as most prevoius kernelz
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    // Number of threads launches as part of the block
    // (64 / 8) * (64 / 8) = 64
    dim3 block_dim((BM / TM) * (BN / TN));

    const int num_blocks_m = ceil_div(num_rows_a, BM);
    const int num_blocks_n = ceil_div(num_cols_b, BN);
    const int main_blocks_m = num_rows_a / BM;
    const int main_blocks_n = num_cols_b / BN;

    if (main_blocks_m > 0 && main_blocks_n > 0)
    {
        dim3 main_grid(main_blocks_m, main_blocks_n);
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN><<<main_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
    }

    // ... handle edge kernels
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

{: .prompt-info}
> - **L1TEX Global Store Access Pattern:** The memory access pattern for global stores to L1TEX might not be optimal. On average, only 4.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global stores.
> - **Shared Load Bank Conflicts:** The memory access pattern for shared loads might not be optimal and causes on average a 5.6 - way bank conflict across all 167772160 shared load requests.This results in 503316480 bank conflicts, which represent 53.57% of the overall 939534036 wavefronts for shared loads. Check the Source Counters section for uncoalesced shared loads.
> - **Uncoalesced Shared Accesses:** This kernel has uncoalesced shared accesses resulting in a total of 369098752 excessive wavefronts (37% of the total 1006632960 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source locations. The CUDA Best Practices Guide has an example on optimizing shared memory accesses.

![NCU After 2D Tiling](/assets/explore_gemm_ncu_after_2d_tiling.png)

## Vectorized Memory Access

### Concept

As we have realized already, GEMMs are **bandwidth-bound**, meaning their performance is limited by memory transfer speed rather than compute throughput on GPUs. Another optimization that can increase effective bandwidth utilization is to use **vectorized memory operations**.

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

If `TM=8`, we're issuing 8 separate load instructions. With vectorization, we can combine these into 2 loads of `float4`. Let's see how this works:

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
{: .prompt-warning}


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

{: .prompt-info}
> We have removed the caller code since that is unchanged

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

After optimizing thread-level and block-level tiling, the next step is **warp-level tiling** to exploit the natural 32-thread warp execution unit. This can be leveraged to achieve better register reuse and computation efficiency.

#### What is Warp Tiling?

As we have discussed previously, a **warp** is the fundamental execution unit in NVIDIA GPUs consisting of 32 threads that execute in SIMT (Single Instruction, Multiple Thread) fashion. Warp tiling introduces an additional level in the memory hierarchy. FWIW - we are slowly approaching cutlass territory. Cutlass library implements a sophisticated tiling strategy that mirrors the GPU's memory hierarchy at multiple levels:

![CUTLASS Memory Hierarchy](/assets/explore_gemm_cutlass_hierarchy.png)
*Source: [NVIDIA CUTLASS Blog](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)*

To take advantage of the this hierarchy, warp provide another layer of tiling such that while loading data from shared memory into registers, we do it at warp tile level. In addition, each thread in the warp then calculates its own small fraction of computation. 

![Nvidia blog warp tiling](/assets/explore_gemms_warp_tiling_nvidia_blog_2.png)

Thread tile (next level tiling) is effectively 2D Tiling that we discussed in the previous section. 

![Nvidia blog thread tiling](/assets/explore_gemms_thread_tiling_nvidia_blog.png)

Pseudo code for this process looks like:

```
for (k = 0; k < K; k += BK) {
    // 1. Load A_tile[BM x BK] and B_tile[BK x BN] into shared memory
    __syncthreads();

    // 2. Each warp loads its own subtile of A and B into registers
    //    (from shared memory)
    for (int kk = 0; kk < BK; ++kk) {
        // 3. Each thread computes on its own 4x4 tile in registers
        C_reg[m][n] += A_frag[m][kk] * B_frag[kk][n];
    }

    __syncthreads();
}

// 4. Write C_reg results from each thread to global C

```

In the later sections, we will look into tensorcores and this is key requirement for us to get there. But let's first look how this works through below visualization:

<div id="warp-tiling-viz"></div>

{: .prompt-tip }
> [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html) has a lot more details and is pretty comprehensive in explaining different layers of tiling.

### Kernel

{: .prompt-info}
> NOTE: I have removed the non-block size mulitple kernel handling below since the code is getting fairly complicated with multiple layers of nesting for different levels of tiles. I will continue to ignore tail handling in subsequent sections. 

```cuda
constexpr int WARPSIZE = 32;

/*
Warp-level tiling GEMM kernel.
Hierarchy: Block (BM x BN) → Warps (WM x WN) → Warp Subtiles (WSUBM x WSUBN) → Thread Tiles (TM x TN)
*/
template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int num_cols_b, int num_cols_a,
                               const float *matrix_a, const float *matrix_b,
                               float *tile_a, float *tile_b,
                               int inner_row_a, int inner_col_a,
                               int inner_row_b, int inner_col_b)
{
    for (uint offset = 0; offset + row_stride_a <= BM; offset += row_stride_a)
    {
        const float4 tmp_a = reinterpret_cast<const float4 *>(
            &matrix_a[(inner_row_a + offset) * num_cols_a + inner_col_a * 4])[0];
        tile_a[(inner_col_a * 4 + 0) * BM + inner_row_a + offset] = tmp_a.x;
        tile_a[(inner_col_a * 4 + 1) * BM + inner_row_a + offset] = tmp_a.y;
        tile_a[(inner_col_a * 4 + 2) * BM + inner_row_a + offset] = tmp_a.z;
        tile_a[(inner_col_a * 4 + 3) * BM + inner_row_a + offset] = tmp_a.w;
    }

    for (uint offset = 0; offset + row_stride_b <= BK; offset += row_stride_b)
    {
        reinterpret_cast<float4 *>(
            &tile_b[(inner_row_b + offset) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4 *>(
                &matrix_b[(inner_row_b + offset) * num_cols_b + inner_col_b * 4])[0];
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_warp_tile(float *register_m, float *register_n, float *thread_results,
                                  const float *tile_a, const float *tile_b,
                                  const uint warp_row, const uint warp_col,
                                  const uint thread_row_in_warp, const uint thread_col_in_warp)
{
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                register_m[wsub_row_idx * TM + i] =
                    tile_a[(dot_idx * BM) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
            }
        }

        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            for (uint i = 0; i < TN; ++i)
            {
                register_n[wsub_col_idx * TN + i] =
                    tile_b[(dot_idx * BN) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i];
            }
        }

        for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
        {
            for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
            {
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
                {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                    {
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

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemm_warptiling_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                            float alpha, const float *matrix_a, const float *matrix_b,
                            float beta, float *matrix_c)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_col = warp_idx % (BN / WN);
    const uint warp_row = warp_idx / (BN / WN);

    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += (block_row * BM + warp_row * WM) * num_cols_b + block_col * BN + warp_col * WN;

    const uint inner_row_a = threadIdx.x / (BK / 4);
    const uint inner_col_a = threadIdx.x % (BK / 4);
    constexpr uint row_stride_a = (NUM_THREADS * 4) / BK;

    const uint inner_row_b = threadIdx.x / (BN / 4);
    const uint inner_col_b = threadIdx.x % (BN / 4);
    constexpr uint row_stride_b = NUM_THREADS / (BN / 4);

    float thread_results[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK)
    {
        load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
            num_cols_b, num_cols_a, matrix_a, matrix_b, tile_a, tile_b,
            inner_row_a, inner_col_a, inner_row_b, inner_col_b);

        __syncthreads();

        process_warp_tile<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            register_m, register_n, thread_results, tile_a, tile_b,
            warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        __syncthreads();
    }

    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
    {
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            float *matrix_c_interim = matrix_c + (wsub_row_idx * WSUBM) * num_cols_b +
                                      wsub_col_idx * WSUBN;

            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4)
                {
                    float4 tmp_c = reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                          thread_col_in_warp * TN + res_idx_n])[0];

                    const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                        wsub_col_idx * TN + res_idx_n;
                    tmp_c.x = alpha * thread_results[res_idx + 0] + beta * tmp_c.x;
                    tmp_c.y = alpha * thread_results[res_idx + 1] + beta * tmp_c.y;
                    tmp_c.z = alpha * thread_results[res_idx + 2] + beta * tmp_c.z;
                    tmp_c.w = alpha * thread_results[res_idx + 3] + beta * tmp_c.w;

                    reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * num_cols_b +
                                          thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
                }
            }
        }
    }
}
```

### Caller

```c
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
void sgemm_warptiling(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                      torch::Tensor &output_matrix, float alpha, float beta)
{
    // .. similar to previous kernels
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    static_assert(WMITER * WSUBM == WM, "WMITER * WSUBM must equal WM");
    static_assert(WNITER * WSUBN == WN, "WNITER * WSUBN must equal WN");
    static_assert((BM % WM == 0) && (BN % WN == 0), "Block tile must be divisible by warp tile");
    static_assert((WSUBM % TM == 0) && (WSUBN % TN == 0), "Warp subtile must be divisible by thread tile");

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

The warp tiling kernel demonstrates further performance gains at large matrix sizes, achieving the best performance so far of all our hand-written kernels. It also shows interesting complexity tradeoffs at smaller sizes.

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

| Matrix Size | Time (ms) | TFLOPS | vs PyTorch | Speedup vs 2D Tiling |
|-------------|-----------|---------|------------|---------------------|
| 128×128 | 0.021 | 0.20 | 40.2% | 0.81× |
| 512×512 | 0.058 | 4.60 | 18.9% | 0.79× |
| 1024×1024 | 0.103 | 20.77 | 29.8% | 0.90× |
| 2048×2048 | 0.396 | 43.38 | 52.6% | 1.14× |
| 4096×4096 | 3.00 | 45.82 | 54.4% | 1.17× |
| 8192×8192 | 25.69 | 42.80 | 50.9% | 1.12× |

![Across matrices warptiling](/assets/explore_gemm_warptiling_performance.png)

## Supporting 16-bit GEMM

So far, we have worked only on fp32 kernels. Most workloads today increasingly use 16-bit floating-point formats (FP16 and BF16) and even lower precisions such as fp8, fp4, etc to reduce memory bandwidth requirements and increase throughput. While our warp tiling kernel works for FP32, we can extend it to support lower-precision computations in 16-bit. This mostly requires us adapt the kernel to handle multiple dtypes, which is just templating. Although, one thing to note is to get good numeric behavior on the lower prevision kernels, we need to ensure we do accumulation in higher precision such as fp32. 

I am leaving out the kernel implementation for this part so we can focus on WMMA and Tensor Cores next. However, here is the baseline performance after porting warptiling technique for fp32 above to fp16/bf16. Spoiler alert: the performance is pretty bad and we are down to about 1/4th of pytorch performance.

> NOTE: I haven't done any tuning of the tile sizes here - just took fp32 kernel as is and update types and handled vectorized loads using float2!
{: .prompt-info}

![FP16 Baseline](/assets/explore_gemm_fp16_baseline.png)

![BFP16 Baseline](/assets/explore_gemm_bf16_baseline.png)

> Quick Takeaway: without tensorcores we can't match the performance of pytorch. So, let's look at that next:

## WMMA and Tensorcores

The warp-tiling structure that we discussed earlier can also be implemented using nvidia WMMA api (Warp Matrix Multiply-Accumulate). WMMA provides extensions to the tiling structure and exposes Tensorcore MMA operations. 

> NOTE: CUTLASS provide even more high-level abstractions for GEMMs. We will discuss that later
{: .prompt-info}

Before we dig further into WMMA, let's look at what are Tensorcores. At a high level, Tensorcores provide warp-level collective operation for MMA such that 32 threads within a warp collectively hold MMA operands. Below we represent 4 x 4 x 4 matrix processing array performing `D = A * B + C`.

![WMMA](/assets/explore_gemms_wmma_nvidia_blog.png)

Digging a little into specific, here is a good common use case for fp16 matmul with fp32 accumulation:

![Tensorcore example](/assets/explore_gemms_tensorcore_nvidia_gtc_presentation_1.png)

can be codified as

```c
float D[4];
uint32_t const A[2];
uint32_t const B;
float const C[4];
// Example targets 16-by-8-by-8 Tensor Core operation
asm(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    " { %0, %1, %2, %3 }, "
    " { %4, %5}, "
    " %6, "
    " { %7, %8, %9, %10 };"
    :
    "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    :
    "r"(A[0]), "r"(A[1]),
    "r"(B),
    "f"(C[0]), "f"(C[1])
);
```

> Here, `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` can be understood as `mma.sync.aligned` instruction over matrix dimentsion of `M=16, N=8, K=8 (computes C[16×8] = A[16×8] × B[8×8] + C[16×8])`. `row.col` represents row-major/column-major layouts for A and B, respectivelly and finally `f32.f16.f16.f32` represents data types such that output is 32-bit float (fp32) with A and B as fp16 and accumulation happening in fp32.
{: .prompt-info}

More details on Tensorcore instructions in [this GTC talk from 2019](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/).


### WMMA

So, as we see above, the general idea is that NVidia provides instruction sets for Tensor Cores to do warp level matmuls that allow us to calculate thread tile matrix multiplications in single instructions. These instructions allow warp-wide MMA operations. Quoted from original [Cutlass post](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/):

> Figure shows the warp tile structure that targets the CUDA WMMA API. Calls to `wmma::load_matrix_sync` load fragments of A and B into instances of the `nvcuda::wmma::fragment<>` template, and the accumulator elements for the warp tile are structured as an array of `nvcuda::wmma::fragment<accumulator>` objects. These fragments store a 2D matrix distributed among the threads of the warp. Finally, calls to `nvcuda::wmma::mma_sync()` for each accumulator fragment (and corresponding fragments from A and B) compute the warp-wide matrix multiply-accumulate operation using Tensor Cores.
>
> **Source:** [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
{: .prompt-tip}

![WMMA 2](/assets/explore_gemms_wmma_nvidia_blog_2.png)

Next, let's transform our original warp-tiled kernel for 16-bit precision to wmma api. We should be able to map tile sizes directly to wmma api.

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
- [Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma-and-friends)
- [Programming Tensor Cores](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)
- [CUDA Techniques to Maximize Memory Bandwidth and Hide Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)
- [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [NVIDIA Tensor Core Evolution: From Volta To Blackwell
](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)


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
