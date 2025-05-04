---
title: Cuda Basics
description: >-
  Some notes on cuda compute and memory basics 
date: 2025-04-20
categories: [Blog, Tutorial]
tags: [AI, Machine Learning, CUDA]
pin: false
math: false
author: ks
---

In this post, we will go over some of the basics of compute and memory for a consumer grade GPU such as RTX 4090. This post is inspired by PMPP book and an excellent talk on GPU Mode.

I have had these notes sitting around on my computer for over a year. I ended up deciding to just share them publically.

## RTX 4090 Streaming MultiProcessor (SM)

![4090 SM Architecture](/assets/4090_sm.png)

[Source: NVIDIA ADA GPU ARCHITECTURE](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)


| Property                  | Value                     |
|---------------------------|---------------------------|
| Device                   | NVIDIA GeForce RTX 4090   |
| Compute capability       | 8.9                       |
| Total global memory      | 24209 MB                  |
| Shared memory per block  | 49152 bytes               |
| Shared memory per SM     | 102400 bytes              |
| Registers per block      | 65536                    |
| Warp size                | 32                        |
| Max threads per block    | 1024                      |
| Max threads per SM       | 1536                      |
| Number of SMs            | 128                       |
| Max blocks per SM        | 24                        |
| Max grid dimensions      | [2147483647, 65535, 65535]|
| Max threads dim (block)  | [1024, 1024, 64]          |
| Clock rate               | 2520 MHz                  |
| Memory Clock Rate        | 10501 MHz                 |
| Memory Bus Width         | 384 bits                  |
| L2 Cache Size            | 75497472 bytes            |



# NVIDIA RTX 4090 Architecture Overview

## Key Components of an SM (Streaming Multiprocessor)

In an SM (Streaming Multiprocessor) — that's where compute happens:

- **FP32 Cores**: 128 per SM (for regular floating point operations).
- **INT32 Cores**: 64 per SM (for integer operations).
- **Tensor Cores**: Special cores optimized for matrix multiplications (ideal for AI/ML workloads).
- **Load/Store Units**: Handle memory access (global/shared/local).
- **Special Function Units (SFUs)**: Perform transcendental math operations (e.g., sin, cos, exp).

---

## 🔹 Key Numbers for RTX 4090

- **128 SMs in total** (varies slightly based on chip binning).
- **16,384 FP32 cores** in total.
- **512 Tensor cores**.
- **128 RT Cores** (Ray Tracing units).
- **L2 Cache**: 72 MB (a massive upgrade compared to Ampere).

Additionally, clock speeds are significantly higher than earlier generations, often boosting beyond **2.5GHz** in real-world usage.

---

## 🔹 Scheduling on RTX 4090

### Warp-Based Scheduling

- A **warp** consists of 32 threads.
- The **Warp Scheduler** selects one or more warps to execute instructions.

### In Ada Lovelace (RTX 4090):

- Each SM has **four warp schedulers** and **eight dispatch units**.
- Each warp scheduler can issue instructions to **two execution units per cycle**.

---

### Instruction Issue

- At each clock cycle, a warp scheduler picks a ready warp and issues an instruction to available execution pipelines.
- **Dual-issue capability**: An SM can schedule an FP32 and an INT32 instruction simultaneously.

---

### Thread Grouping

- Warps are derived from **Cooperative Thread Arrays (CTAs)**, also known as thread blocks in CUDA.

---

### Memory Access

- **Coalesced memory access** is crucial for performance.
- Tensor cores have dedicated pathways for memory, leveraging shared memory and new optimizations.
- **FP8 precision** is introduced in Ada Lovelace for AI tensor operations.

---


## Example: Fused Softmax

```cpp
__global__ void fused_softmax_kernel(float* output, const float* input, int seq_len, int stride)
{

    // Get the SM ID using inline PTX
    unsigned int streaming_multiprocessor_id;
    asm("mov.u32 %0, %smid;" : "=r"(streaming_multiprocessor_id));

    // Print block and thread indices along with SM ID
    printf("BlockIdx: (%d, %d, %d), ThreadIdx: (%d, %d, %d), SM ID: %d\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           streaming_multiprocessor_id);

    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;

    // Shared memory for block-wide max and sum
    __shared__ float s_max;
    __shared__ float s_sum;

    // Base pointers for this row
    const float* row_input = input + row_idx * stride;
    float* row_output = output + row_idx * stride;

    // Step 1: Find max value (using warp-level reduction)
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
    {
        thread_max = max(thread_max, row_input[i]);
    }

    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset /= 2)
    {
        thread_max = max(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
    }

    // Store max in shared memory (first thread in warp)
    if (threadIdx.x == 0)
    {
        s_max = thread_max;
    }
    __syncthreads();

    // Step 2: Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
    {
        float val = exp(row_input[i] - s_max);
        thread_sum += val;
        row_output[i] = val; // Store intermediate result
    }

    // Warp-level reduction for sum
    for (int offset = 16; offset > 0; offset /= 2)
    {
        thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
    }

    // Store sum in shared memory (first thread in warp)
    if (threadIdx.x == 0)
    {
        s_sum = thread_sum;
    }
    __syncthreads();

    // Step 3: Normalize
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x)
    {
        row_output[i] /= s_sum;
    }
}
```


![Fused Softmax Execution](/assets/fused_softmax_digraph.png)


```shell
BlockIdx: (0, 123, 0), ThreadIdx: (79, 0, 0), SM ID: 119
... 31 more threads
BlockIdx: (0, 88, 0), ThreadIdx: (192, 0, 0), SM ID: 49
... 31 more threads
BlockIdx: (0, 99, 0), ThreadIdx: (832, 0, 0), SM ID: 71
... 31 more threads
```


## More Notes


- 32 fb32 units and half of them can to int32 operations
- Gen 4 tensor core units
- Thread block is assigned to one SM (max 1536 threads assignable to SM)
- No control over which block in a grid goes to which SM
- Each of these can execute one warp (32 threads) => 4 warps at a time
- 16K 32 bit registers shared between things scheduled ont the same block


### Threads, Warps, and Blocks
Launching a CUDA kernel with
- block layout: how many threads in a block, threads in a block are executed in parallel on the same SM, blocks are independent, cuda is free to assign arbitrary SMs to blocks, thread blocks are divided into warps of 32 threads
- grid layout: how many blocks should be launched
- warps should run in parallel, but we could have control flow. This can cause warp divergence


Getting good occupancy = balance resources
- Keep the SMs busy => use many blocks
- 1536 threads per SM => power of 2 block size, it's better to have 512, 1024, 2048 depending on the gpu
- Avoid fp64, int64
- Using too much shared memory or register files leads to limit on number of scheduled threads on SM
- Nsight compute tells you occupancy calculation



