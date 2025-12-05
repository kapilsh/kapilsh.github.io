---
title: "Learn CUTLASS the hard way - 2"
description: >-
  Exploring Hopper
date: 2025-11-29
categories: [Blog]
tags: [Triton, CUDA, GPU, GEMM, Performance]
pin: true
math: true
author: ks
---

This a continuation to learning CUTLASS journey I shared in my [previous post](../learn-cutlass-the-hard-way). In this post, I will explore Hopper architecture. There were several new architectural changes made to H100 and subsequent GPUs that have huge implications on the performance of GEMM kernels. However, before we jump into those - let's start with baselining the kernel we wrote for Ada (RTX 4090) on H100. 

## My Previous Baseline

#### Baseline Performance of `kernel_cutlass`

![Baseline Perf](/assets/explore_gemms_2_hopper_baseline_perf.png)

The baseline benchmark compares our original ~first~ CUTLASS kernel against PyTorch and other handwritten CUDA implementations on H100. Key observations:

- **Small matrices (64-1024)**: CUTLASS achieves 82-90% of PyTorch performance, delivering 95-320 TFLOPS. The gap is minimal at smaller sizes where both are memory-bound.
- **Medium matrices (1536-3072)**: Performance diverges significantly. PyTorch reaches 686 TFLOPS at 3K while CUTLASS delivers only 317 TFLOPS (46% speedup). This indicates our kernel isn't leveraging H100's architectural improvements.
- **Large matrices (4096-8192)**: The gap widens further. PyTorch achieves 754 TFLOPS at 4K and 685 TFLOPS at 8K, while CUTLASS plateaus at 396 TFLOPS and 375 TFLOPS respectively (52-55% speedup).

The performance cliff at medium-to-large matrix sizes reveals that our Ada-optimized kernel leaves significant H100 compute on the table. The hand-written implementations perform even worse.

#### Autotuned Results

![Autotuned Results Baseline](/assets/explore_gemms_2_hopper_baseline_autotuned.png)

Here is summary of the results after autotuning:

- **Small matrices (64-512)**: Autotuned configs achieve 2x speedup over PyTorch, reaching 30 TFLOPS at size 512. Best configs favor smaller tiles (64×32×32) with more stages (S4-S5), optimizing for low occupancy scenarios.
- **Sweet spot (1024)**: Peak relative performance at 170 TFLOPS (1.49x vs PyTorch). The optimal config (64×128×64 with S4) balances tile size and register pressure.
- **Performance inversion (2048+)**: PyTorch overtakes autotuned kernels. At 2K, PyTorch delivers 481 TFLOPS vs 402 TFLOPS (0.84x). At 8K, the gap grows to 654 vs 446 TFLOPS (0.68x).
- **Config evolution**: Larger matrices demand bigger tiles (128×256×64) but fewer stages (S3), shifting from register-bound to compute-bound regimes.

> It seems surprizing to me but autotuning successfully extracts ~2x performance for small GEMMs. However, the fundamental kernel architecture can't exploit H100's full potential at larger sizes, as was also shown in [Pranjal's Post on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) post. Looking at his results ther perf for larger sizes falls somewhere in the region of "Larger Tiles"
{: .prompt-info}

![Pranjal's Results](/assets/explore_gemms_2_pranjals_result.png)


## Key Changes in Hopper

Let's first cover some of Hopper's architectural features that matter most for GEMMs. 

### Thread Block Clusters

Hopper adds a new hierarchy level above thread blocks called Thread Block Clusters (TBC), groups of up to 8 thread blocks that can cooperate. This includes a distributed shared memory such that thread blocks in a cluster can directly access each other's shared memory. This allows for better data reuse across blocks without going through global memory. As a result, new programming pattern has emerged where we need to partition work across TBCs rather than just within block. 

Overall, TBCs boost performance by enabling cooperation across multiple SMs, giving kernels access to more threads and a larger effective shared-memory pool than a single thread block can provide.

![TBC](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/Thread-Block-Clusters-and-Grids-with-Clusters.jpg)

![TBC Shared Memory](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/Thread-Block-to-Thread-Block-data-exchange-A100-vs-H100-with-Clusters-1536x331.jpg)


### Tensor Memory Accelerator

Tensor Memory Accelerator (TMA) efficiently move tensors between global memory and shared memory. Instead of per-element address generation, TMA uses a copy descriptor that describes the transfer in terms of tensor dimensions and block coordinates. It leverages cute layouts extensively and handles address generation and data movement entirely in hardware. 

TMA operations are asynchronous and use shared-memory–based barriers. Only one thread in a warp issues the `cuda::memcpy_async`, while others simply wait on the barrier. It frees up threads to perform useful work while data transfers proceed in parallel. 

![TMA](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/Asynchronous-Memory-Copy-with-TMA-on-H100-vs-LDGSTS-Instruction-on-A100.jpg)

### Async Execution

Asynchronous barriers, introduced in Ampere, split synchronization into two non-blocking steps: Arrive, where threads signal they’ve finished producing data and can continue doing other work, and Wait, where threads block only when they actually need the results. This overlap boosts performance, and in Hopper, waiting threads can now sleep instead of spinning, reducing wasted cycles.

Hopper adds a more advanced primitive: the asynchronous transaction barrier. Like an async barrier, it uses Arrive and Wait, but it also tracks the amount of data produced. Threads perform Arrive operations that include a transaction (byte) count, and the Wait step blocks until both conditions are met: all threads arrived and the total data produced reaches a specified threshold.

These transaction barriers are crucial for asynchronous memory operations and efficient data exchange—such as the fast thread block–to–thread block communication enabled by clusters—because they synchronize based on both thread participation and actual data movement.

### Warp Group Instructions

### Native FP8 Support

Let's explore each of these in detail.


## Resources

- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)
- [CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)