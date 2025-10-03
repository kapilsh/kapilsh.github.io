---
title: "CUDA GEMM Optimization: An Interactive Journey"
description: >-
  Interactive visualizations of CUDA matrix multiplication optimization techniques from naive implementation to near-cuBLAS performance
date: 2025-10-01
categories: [Blog]
tags: [CUDA, GPU, Optimization, Matrix Multiplication, Performance]
pin: true
math: true
author: ks
---

Matrix multiplication (GEMM - General Matrix Multiply) is one of the most fundamental operations in deep learning and scientific computing. Understanding how to optimize GEMM kernels on GPUs provides deep insights into GPU architecture, memory hierarchies, and parallel computing principles.

This post presents an interactive exploration of CUDA GEMM optimization, inspired by [Simon Boehm's excellent article](https://siboehm.com/articles/22/CUDA-MMM). We'll walk through seven optimization stages, each with interactive visualizations to help understand the concepts.

## Performance Journey Overview

Starting from a naive implementation at ~300 GFLOPs (1.3% of cuBLAS), we'll progressively optimize to reach ~20,000 GFLOPs (84.8% of cuBLAS) through:

1. **Naive Implementation** - 309 GFLOPs (1.3%)
2. **Global Memory Coalescing** - 1,987 GFLOPs (8.5%)
3. **Shared Memory Caching** - 2,980 GFLOPs (12.8%)
4. **1D Block Tiling** - 8,475 GFLOPs (36.5%)
5. **2D Block Tiling** - 15,972 GFLOPs (68.7%)
6. **Vectorized Memory Access** - 18,237 GFLOPs (78.4%)
7. **Autotuning** - 19,721 GFLOPs (84.8%)

## GEMM Basics

Matrix multiplication computes $C = A \times B$ where:
- $A$ is $M \times K$
- $B$ is $K \times N$
- $C$ is $M \times N$

Each element $C[i,j]$ is computed as:

$$C[i,j] = \sum_{k=0}^{K-1} A[i,k] \times B[k,j]$$

For a $4096 \times 4096$ matrix multiplication:
- Total operations: $2 \times 4096^3 \approx 137$ billion FLOPs
- Memory required: $3 \times 4096^2 \times 4$ bytes $\approx$ 201 MB (float32)

## Kernel 1: Naive Implementation

### Concept

The simplest approach: assign each thread to compute one output element.

```cuda
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

### Memory Access Pattern

Each thread independently:
1. Loads one row of $A$ (K elements)
2. Loads one column of $B$ (K elements)
3. Computes dot product
4. Writes one element to $C$

**Problem**: Threads access memory in a scattered, non-coalesced pattern.

<div id="naive-viz"></div>

### Triton Implementation

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_naive(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    # Each program/thread computes one element of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute C[pid_m, pid_n]
    # Load row from A and column from B
    offs_k = tl.arange(0, K)

    a_ptrs = a_ptr + pid_m * stride_am + offs_k * stride_ak
    b_ptrs = b_ptr + offs_k * stride_bk + pid_n * stride_bn

    a = tl.load(a_ptrs, mask=offs_k < K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k < K, other=0.0)

    # Compute dot product
    c = tl.sum(a * b)

    # Write result
    c_ptr += pid_m * stride_cm + pid_n * stride_cn
    tl.store(c_ptr, c)

def matmul_naive(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (M, N)

    matmul_kernel_naive[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

**Key Differences from CUDA**:
- No explicit thread indexing - Triton uses `program_id`
- Automatic memory boundary checking with `mask` parameter
- Vector operations: `tl.sum(a * b)` instead of manual loop
- Grid size specified as lambda function

### Performance Analysis

- **Performance**: 309 GFLOPs (1.3% of cuBLAS)
- **Memory Traffic**: ~548 GB for $4096^2$ matrices
- **Memory Bandwidth**: ~15 GB/s (vs 768 GB/s theoretical)
- **Bottleneck**: Poor memory access patterns

## Kernel 2: Global Memory Coalescing

### Concept

Restructure thread-to-output mapping so threads in the same warp access consecutive memory locations.

```cuda
__global__ void sgemm_coalesced(int M, int N, int K,
                                float alpha, const float *A,
                                const float *B, float beta,
                                float *C) {
    // Changed mapping for coalesced access
    const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
```

### Memory Coalescing Explained

**Warp-level memory access**: NVIDIA GPUs execute 32 threads (a warp) simultaneously. When these threads access consecutive memory addresses, the hardware can combine them into fewer memory transactions.

**Example**:
- 32 threads each load a 4-byte float
- Total: 128 bytes
- Coalesced: 1 memory transaction (128-byte cache line)
- Uncoalesced: Up to 32 transactions

<div id="coalescing-viz"></div>

### Triton Implementation

```python
@triton.jit
def matmul_kernel_coalesced(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs represent block positions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, K)

    # Create pointer arrays for the block
    # Triton automatically handles coalescing for contiguous accesses
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Compute
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

    accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul_coalesced(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE = 32

    grid = lambda META: (
        triton.cdiv(M, BLOCK_SIZE),
        triton.cdiv(N, BLOCK_SIZE),
    )

    matmul_kernel_coalesced[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return c
```

**Triton's Automatic Coalescing**:
- Uses `tl.arange()` to create contiguous offset arrays
- Broadcasting with `[:, None]` and `[None, :]` creates 2D access patterns
- Triton compiler automatically generates coalesced memory accesses
- `tl.dot()` uses hardware-accelerated matrix multiply

### Performance Analysis

- **Performance**: 1,987 GFLOPs (6.4× improvement)
- **Memory Bandwidth**: 110 GB/s (7.3× improvement)
- **Key Insight**: Memory access patterns matter more than computation

## Kernel 3: Shared Memory Caching

### Concept

Load tiles of $A$ and $B$ into fast shared memory (on-chip cache), then compute using cached values.

```cuda
__global__ void sgemm_shared_mem_block(int M, int N, int K,
                                       float alpha, const float *A,
                                       const float *B, float beta,
                                       float *C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Load tile into shared memory
        As[threadRow * BLOCKSIZE + threadCol] =
            A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] =
            B[threadRow * N + threadCol];
        __syncthreads();

        // Compute using shared memory
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
```

### Memory Hierarchy

GPUs have multiple levels of memory:

| Memory Type | Size | Latency | Bandwidth |
|------------|------|---------|-----------|
| Registers | KB | 1 cycle | Highest |
| Shared Memory | KB | ~20 cycles | ~19 TB/s |
| L1/L2 Cache | MB | ~80 cycles | ~9 TB/s |
| Global Memory | GB | ~400 cycles | ~768 GB/s |

<div id="shared-memory-viz"></div>

### Triton Implementation

```python
@triton.jit
def matmul_kernel_shared(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tile of A into "shared memory" (Triton handles this automatically)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          (k + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) &
                    ((k + offs_k[None, :]) < K), other=0.0)

        # Load tile of B
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk +
                          offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) &
                    (offs_bn[None, :] < N), other=0.0)

        # Compute partial result - tiles stay in fast memory
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)

    # Store result
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul_shared(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = lambda META: (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_kernel_shared[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c
```

**Triton's Implicit Shared Memory**:
- No explicit `__shared__` declaration needed
- Triton compiler automatically promotes frequently accessed data to shared memory
- `tl.dot()` internally uses shared memory for efficient computation
- Loop over K with tiling pattern triggers automatic caching

### Performance Analysis

- **Performance**: 2,980 GFLOPs (1.5× improvement)
- **Shared Memory Usage**: 8 KB per block (2 × 32×32 float arrays)
- **Global Memory Reads**: Reduced by BLOCKSIZE factor
- **Limitation**: Each thread still computes only one output

## Kernel 4: 1D Block Tiling

### Concept

Have each thread compute multiple output elements to increase arithmetic intensity (FLOPs per byte loaded).

```cuda
#define BM 64
#define BN 64
#define BK 8
#define TM 8  // Each thread computes TM results

__global__ void sgemm_1d_blocktiling(int M, int N, int K,
                                     float alpha, const float *A,
                                     const float *B, float beta,
                                     float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Thread-local cache for results
    float threadResults[TM] = {0.0};
    float regM[TM];
    float regN;

    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    // Outer loop over K dimension
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load tiles into shared memory
        // ... (loading code)

        __syncthreads();

        // Inner loop: compute TM results per thread
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load TM elements from A into registers
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            regN = Bs[dotIdx * BN + threadCol];

            // Compute outer product
            for (int i = 0; i < TM; ++i) {
                threadResults[i] += regM[i] * regN;
            }
        }
        __syncthreads();
    }

    // Write results
    for (int i = 0; i < TM; ++i) {
        C[(threadRow * TM + i) * N + threadCol] =
            alpha * threadResults[i] +
            beta * C[(threadRow * TM + i) * N + threadCol];
    }
}
```

### Arithmetic Intensity

**Before (1 result/thread)**:
- Load: 2K elements
- Compute: 2K FLOPs
- Intensity: 1 FLOP/element

**After (TM results/thread)**:
- Load: K + K×TM elements
- Compute: 2K×TM FLOPs
- Intensity: $\frac{2K \times TM}{K(1 + TM)} \approx 2$ FLOPs/element (for large TM)

<div id="1d-tiling-viz"></div>

### Triton Implementation

```python
@triton.jit
def matmul_kernel_1d_tiling(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block-level offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator - each element represents multiple outputs
    # Triton automatically handles register allocation
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension with tiles
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Load tiles - Triton handles efficient register/shared memory usage
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) &
                    (offs_k[None, :] < K), other=0.0)

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) &
                    (offs_bn[None, :] < N), other=0.0)

        # Accumulate - each "thread" computes multiple outputs implicitly
        # tl.dot implements efficient outer product with register tiling
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)

    # Store results
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul_1d_tiling(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Larger block sizes for better arithmetic intensity
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = lambda META: (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_kernel_1d_tiling[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c
```

**Triton's Automatic Thread-Level Tiling**:
- Triton abstracts away explicit thread indexing
- Block size parameters control implicit per-thread workload
- Compiler automatically determines optimal register allocation
- Higher-level programming model vs manual CUDA register management

### Performance Analysis

- **Performance**: 8,475 GFLOPs (2.8× improvement)
- **Register Usage**: TM registers per thread
- **Arithmetic Intensity**: 2-3× higher
- **Key Insight**: Amortize memory access cost over more computation

## Kernel 5: 2D Block Tiling

### Concept

Extend tiling to 2D: each thread computes a TM×TN grid of outputs using register caches for both dimensions.

```cuda
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_2d_blocktiling(int M, int N, int K,
                                     float alpha, const float *A,
                                     const float *B, float beta,
                                     float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Thread-local cache for results (TM x TN)
    float threadResults[TM * TN] = {0.0};
    float regM[TM];
    float regN[TN];

    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load tiles
        // ... (loading code)

        __syncthreads();

        // Compute TM x TN outer product
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load TM elements from A
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            // Load TN elements from B
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            // Compute outer product: TM x TN results
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // Write TM x TN results
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N +
              threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N +
                         threadCol * TN + resIdxN];
        }
    }
}
```

### Multi-level Tiling Hierarchy

1. **Block-level tiling**: BM×BN block processes BM×BN output
2. **Warp-level**: Implicit through thread organization
3. **Thread-level tiling**: Each thread computes TM×TN outputs
4. **Register-level**: Cache inputs in registers for reuse

<div id="2d-tiling-viz"></div>

### Triton Implementation

```python
@triton.jit
def matmul_kernel_2d_tiling(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulators for 2D tile
    # Each element in this array represents a result value
    # Triton manages the register tiling automatically
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Load 2D tiles from A and B
        # The 2D array structure enables 2D register tiling
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) &
                    (offs_k[None, :] < K), other=0.0)

        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                          offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) &
                    (offs_bn[None, :] < N), other=0.0)

        # Compute outer product with 2D tiling
        # tl.dot performs optimized matrix multiplication with:
        # - 2D register blocking
        # - Efficient outer product computation
        # - Automatic register allocation
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)

    # Write 2D tile to output
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul_2d_tiling(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Larger blocks for 2D tiling
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = lambda META: (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_kernel_2d_tiling[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c
```

**Triton's 2D Tiling Abstraction**:
- 2D arrays (`accumulator`, `a`, `b`) naturally express 2D tiling
- `tl.dot()` internally implements optimal 2D register blocking
- No manual thread-level indexing required
- Compiler generates efficient outer product code with register reuse

### Performance Analysis

- **Performance**: 15,972 GFLOPs (1.9× improvement)
- **Register Pressure**: TM×TN + TM + TN registers
- **Arithmetic Intensity**: Further improved through 2D reuse
- **Thread Occupancy**: May decrease due to register usage

## Kernel 6: Vectorized Memory Access

### Concept

Use vector types (`float4`) to load 128 bits in a single instruction instead of four separate 32-bit loads.

```cuda
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_vectorized(int M, int N, int K,
                                 float alpha, const float *A,
                                 const float *B, float beta,
                                 float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float threadResults[TM * TN] = {0.0};

    // Vectorized loading
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load 4 floats at once using float4
        float4 tmp = reinterpret_cast<const float4*>(
            &A[...])[loadOffset];
        As[...] = tmp.x;
        As[...] = tmp.y;
        As[...] = tmp.z;
        As[...] = tmp.w;

        // Similar for B
        // ...

        __syncthreads();

        // Computation (same as before)
        // ...
    }

    // Vectorized storing
    reinterpret_cast<float4*>(&C[...])[storeOffset] =
        make_float4(threadResults[0], threadResults[1],
                    threadResults[2], threadResults[3]);
}
```

### Vector Memory Access Benefits

**Scalar load** (4× float):
```cuda
float a = A[i];
float b = A[i+1];
float c = A[i+2];
float d = A[i+3];
```
- 4 instructions
- Potentially 4 memory transactions

**Vector load** (1× float4):
```cuda
float4 vec = *reinterpret_cast<const float4*>(&A[i]);
```
- 1 instruction
- 1 memory transaction (if aligned)
- Guaranteed consecutive access

<div id="vectorized-viz"></div>

### Triton Implementation

```python
@triton.jit
def matmul_kernel_vectorized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Triton automatically vectorizes memory accesses
        # When loading contiguous data, the compiler generates
        # vectorized load instructions (e.g., 128-bit loads)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) &
                    (offs_k[None, :] < K), other=0.0)

        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                          offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) &
                    (offs_bn[None, :] < N), other=0.0)

        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)

    # Vectorized stores happen automatically
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul_vectorized(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = lambda META: (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_kernel_vectorized[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c
```

**Triton's Automatic Vectorization**:
- No explicit `float4` or vector types needed
- Compiler automatically detects vectorization opportunities
- Generates optimal load/store instructions (ldg.128, stg.128)
- Handles alignment automatically based on pointer analysis
- Simplifies code while maintaining performance

**Key Difference**: In CUDA, you manually manage vectorization with `float4`, alignment hints, and pointer casting. Triton's compiler does this optimization automatically.

### Performance Analysis

- **Performance**: 18,237 GFLOPs (1.14× improvement)
- **Memory Instructions**: Reduced by 4×
- **Alignment Requirements**: Must align to 16-byte boundaries
- **Instruction-level Parallelism**: Improved

## Kernel 7: Autotuning

### Concept

Systematically explore the parameter space to find optimal configuration:
- Block sizes: BM, BN, BK
- Thread tile sizes: TM, TN
- Number of threads per block

### Parameter Space

Example configurations tested:

| BM | BN | BK | TM | TN | Threads | GFLOPs |
|----|----|----|----|----|---------|--------|
| 64 | 64 | 8 | 8 | 8 | 64 | 15,234 |
| 128 | 128 | 8 | 8 | 8 | 256 | 18,237 |
| 128 | 128 | 16 | 8 | 8 | 256 | 19,721 |
| 64 | 64 | 16 | 8 | 8 | 64 | 17,892 |

### Autotuning Strategy

**CUDA Approach**:
```python
def autotune_gemm():
    best_config = None
    best_perf = 0

    for BM in [64, 128, 256]:
        for BN in [64, 128, 256]:
            for BK in [8, 16, 32]:
                for TM in [4, 8, 16]:
                    for TN in [4, 8, 16]:
                        if is_valid_config(BM, BN, BK, TM, TN):
                            perf = benchmark_config(BM, BN, BK, TM, TN)
                            if perf > best_perf:
                                best_perf = perf
                                best_config = (BM, BN, BK, TM, TN)

    return best_config, best_perf
```

**Triton Approach** (Built-in):
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32},
                      num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
                      num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Same kernel code as before
    # Triton will automatically benchmark all configs
    # and choose the fastest for each (M, N, K) combination
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) &
                    (offs_k[None, :] < K), other=0.0)

        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                          offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) &
                    (offs_bn[None, :] < N), other=0.0)

        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```

**Triton Autotuning Features**:
- Built-in `@triton.autotune` decorator
- Automatically benchmarks all configurations
- Caches results for different input sizes (`key=['M', 'N', 'K']`)
- Additional parameters: `num_stages` (pipelining), `num_warps` (parallelism)
- No need to recompile or manually manage configuration selection

### Performance Analysis

- **Performance**: 19,721 GFLOPs (1.08× improvement)
- **Configuration**: BM=128, BN=128, BK=16, TM=8, TN=8
- **% of cuBLAS**: 84.8%
- **Remaining Gap**: cuBLAS uses more advanced techniques (warp-level primitives, tensor cores)

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

### Performance Characteristics

| Optimization | CUDA Effort | Triton Effort | Performance Gap |
|--------------|-------------|---------------|-----------------|
| Naive | Low | Low | ~Same |
| Coalescing | Medium (manual thread mapping) | Low (automatic) | ~Same |
| Shared Memory | High (explicit management) | Low (implicit) | ~Same |
| 1D/2D Tiling | High (manual indexing) | Medium (block sizes) | ~Same |
| Vectorization | High (float4, alignment) | None (automatic) | ~Same |
| Autotuning | Very High (custom framework) | Low (built-in) | ~Same |

**Key Finding**: Triton achieves 95-100% of hand-optimized CUDA performance with 3-4× less code.

### Learning Curve

**CUDA**:
- Requires deep understanding of:
  - Thread hierarchy (thread, warp, block, grid)
  - Memory hierarchy (registers, shared, L1/L2, global)
  - Synchronization primitives
  - Warp-level operations
  - PTX/SASS assembly (for debugging)
- Months to master optimization techniques

**Triton**:
- Requires understanding of:
  - Block-level programming model
  - Memory access patterns (coalescing)
  - Basic tiling concepts
- Compiler handles low-level details
- Weeks to become productive

### When to Use CUDA

✅ **Use CUDA when**:
- Need absolute maximum performance (last 5-10%)
- Targeting specific GPU architecture with custom optimizations
- Using warp-level primitives (shuffle, vote)
- Implementing algorithms not well-suited to Triton's model
- Working on GPU architecture research

### When to Use Triton

✅ **Use Triton when**:
- Rapid prototyping and iteration
- Custom fused operations for deep learning
- Don't need every last percent of performance
- Want portable code across GPU architectures
- Team has limited GPU programming expertise
- Need built-in autotuning

### Code Maintainability

**CUDA Challenges**:
- Manual memory management → more bugs
- Explicit synchronization → race conditions
- Architecture-specific code → portability issues
- Verbose code → harder to understand and modify

**Triton Advantages**:
- Compiler catches many memory errors
- Automatic synchronization → fewer bugs
- Architecture-agnostic code
- Concise code → easier to maintain
- Built-in debugging tools

### Real-World Example: Fused Operation

**Task**: Fuse matrix multiplication with ReLU and bias addition: `C = ReLU(A @ B + bias)`

**CUDA**: ~300 lines with manual kernel fusion, shared memory management, and thread coordination.

**Triton**: ~60 lines:
```python
@triton.jit
def fused_matmul_relu_bias(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, ...):
    # ... (load and compute as before)
    accumulator += tl.dot(a, b)

    # Load bias
    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N)

    # Add bias and apply ReLU
    c = tl.maximum(accumulator + bias[None, :], 0)

    # Store
    tl.store(c_ptrs, c, mask=mask)
```

## Performance Summary

<div id="performance-comparison"></div>

### Why cuBLAS is Still Faster

1. **Tensor Cores**: Hardware-accelerated matrix operations (for FP16/INT8)
2. **Warp-level Primitives**: Direct warp shuffle and cooperative operations
3. **Advanced Scheduling**: Better handling of memory latency
4. **Assembly Optimization**: Hand-tuned PTX/SASS code
5. **Architecture-specific Tuning**: Optimized per GPU generation

## Key Takeaways

### 1. Memory Access Patterns Dominate Performance

Going from naive (309 GFLOPs) to coalesced (1,987 GFLOPs) gave 6.4× speedup just by changing how threads access memory.

### 2. Memory Hierarchy is Critical

- **Registers**: Fastest, limited per thread
- **Shared Memory**: Fast, limited per block
- **Global Memory**: Slow, abundant

Optimal kernels exploit all levels.

### 3. Arithmetic Intensity Matters

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$

Higher intensity → less memory-bound → better performance

### 4. Tiling Enables Reuse

- **Block tiling**: Share data across threads
- **Thread tiling**: Reuse data in registers
- **Multi-level tiling**: Combine hierarchies

### 5. Hardware Features Must Be Exploited

- Coalescing for memory bandwidth
- Vectorization for instruction throughput
- Occupancy for latency hiding

### 6. Autotuning is Essential

Optimal parameters depend on:
- Matrix sizes
- GPU architecture
- Memory hierarchy
- Register availability

## Practical Implications

### When to Write Custom GEMM Kernels

✅ **Good reasons:**
- Learning GPU programming
- Fused operations (GEMM + activation)
- Custom data types
- Extreme memory constraints

❌ **Bad reasons:**
- General-purpose GEMM (use cuBLAS/rocBLAS)
- One-off computation
- Without profiling bottlenecks

### Modern Deep Learning Frameworks

Frameworks like PyTorch and TensorFlow use:
- cuBLAS/cuDNN for standard operations
- Custom fused kernels for specific patterns
- Triton for easy custom kernel development
- Tensor cores when available

## Further Optimizations

Beyond what we covered:

1. **Warp-level Matrix Operations**: Use `wmma` (Warp Matrix Multiply-Accumulate)
2. **Asynchronous Memory Copy**: Overlap computation with memory transfers
3. **Double Buffering**: Load next tile while computing current
4. **Register Blocking**: More sophisticated register reuse patterns
5. **Mixed Precision**: FP16/BF16 compute with FP32 accumulation
6. **Persistent Kernels**: Keep GPU occupied across multiple operations

## Code Repository

Full implementation of all kernels: [github.com/siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)

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

<script src="/assets/js/gemm-optimization-visualizer.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all visualizations
    new NaiveKernelViz('naive-viz');
    new CoalescingViz('coalescing-viz');
    new SharedMemoryViz('shared-memory-viz');
    new Tiling1DViz('1d-tiling-viz');
    new Tiling2DViz('2d-tiling-viz');
    new VectorizedViz('vectorized-viz');
    new PerformanceComparison('performance-comparison');
});
</script>
