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

This is a continuation to my nprevious post [Learn CUTLASS the Hard Way](../learn-cutlass-the-hard-way), where I will explore Hopper architecture. There were several new architectural changes that made it to H100 and had huge implications on the performance of GEMM kernels, specially for 16-bit and lower precision GEMMS. There is already a great post from Pranjal on his H100 Journey at [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog). Expect this post to be a bit more higher level than that as we will primarily use CUTLASS to leverage the same optimizations. However, before we jump into it - let's start with baselining the kernel we wrote for Ada (RTX 4090) on H100. 

## My Previous Baseline

We already had a CUTLASS kernel that we wrote in the previous post. It leveraged CUTLASS 2.x API and was primarily meant for Ada and older generation of GPUS. But, we can still run it on H100 to get a baseline performance. 

#### Baseline Performance of `kernel_cutlass`

![Baseline Perf](/assets/explore_gemms_2_hopper_baseline_perf.png)

Below we compare original *~first~ CUTLASS kernel* (and other handwritten CUDA implementations) against PyTorch on H100. Key observations:

| Matrix Size | CUTLASS Performance | PyTorch Performance |
|-------------|---------------------|---------------------|
| **Small (64-1024)** | 95-320 TFLOPS (82-90% of PyTorch) | N/A |
| **Medium (1536-3072)** | 317 TFLOPS at 3K | 686 TFLOPS at 3K |
| **Large (4096-8192)** | 396 TFLOPS at 4K<br>375 TFLOPS at 8K | 754 TFLOPS at 4K<br>685 TFLOPS at 8K |

> **Key Insights:**
> - **Small matrices**: Both implementations are memory-bound at smaller sizes.
> - **Medium matrices**: Performance diverges significantly and we can see our kernel isn't leveraging H100's architectural improvements.
> - **Large matrices**: The gap widens further as CUTLASS plateaus while PyTorch continues scaling until we hit compute boundness
> 
> The performance cliff at medium-to-large matrix sizes reveals that our Ada-optimized kernel leaves significant H100 compute on the table. The hand-written implementations perform even worse.
{: .prompt-info}


#### Autotuned Results

Let's run our original autotuned kernel now, which hopefully should perform slightly better:

![Autotuned Results Baseline](/assets/explore_gemms_2_hopper_baseline_autotuned.png)

Here is summary of the results after autotuning:

Autotuning successfully extracts ~2x speedup for small matrices (64-512), reaching 30 TFLOPS with smaller tiles (64×32×32) while using more pipeline stages (S4-S5). 

Performance peaks at 1024 (170 TFLOPS, 1.49x vs PyTorch) with optimal config 64×128×64 S4, but starts showing worse performance in compute bound regime for 2048+. 

As matrix size increases, optimal configs shift from smaller tiles with more stages to larger tiles (128×256×64) with fewer stages (S3), reflecting the transition from register-bound to compute-bound regimes.

> It seems surprizing to me but autotuning successfully extracts ~2x performance for small GEMMs but I am assuming it is just better register use for smaller sized kernels compared compared to PyTorch version, which I expect to be more generic. However, it is pretty visible that the fundamental kernel implementation can't exploit H100's full potential at larger sizes, as was also shown in [Pranjal's Post on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) post. 
{: .prompt-info}

Looking at his results the perf for larger sizes falls somewhere in the region of "Larger Tiles"

![Pranjal's Results](/assets/explore_gemms_2_pranjals_result.png)

## Key Changes in Hopper

Without making it sound like an NVIDIA marketing presentation, I will try to cover some of Hopper's architectural features that matter most for GEMMs. 

### Thread Block Clusters

Hopper added a new hierarchy level above thread blocks called **Thread Block Clusters (TBC)**, groups of up to 8 thread blocks that can cooperate. This includes a distributed shared memory such that thread blocks in a cluster can directly access each other's shared memory. This allows for better data reuse across blocks without going through global memory. As a result, new programming pattern has emerged where we need to partition work across TBCs rather than just within block. Overall, TBCs boost performance by enabling cooperation across multiple SMs, giving kernels access to more threads and a larger effective shared-memory pool than a single thread block can provide.

![TBC](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/Thread-Block-Clusters-and-Grids-with-Clusters.jpg)

![TBC Shared Memory](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/Thread-Block-to-Thread-Block-data-exchange-A100-vs-H100-with-Clusters-1536x331.jpg)

Source: [NVIDIA Hopper Architecture in Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

### Tensor Memory Accelerator

Tensor Memory Accelerator (TMA) is arguably one of the biggest architectural innovation in Hopper. TMA is efficiently a new way to move tensors from/to global memory to / from shared memory, while avoiding extra roundtrips of going through register file and instead writes / reads shared memory directly. In addition, data can be multi-casted to threadblocks in a cluster (see above for TBC). There are other features such as automatic bounds-checking, zero-out, etc that also help reduce overhead. 

We discussed this briefly in my previous post, but we are more and more towards async producer-consumer paradigms when it comes to GEMM kernels. TMA operations are also asynchronous and use shared-memory–based barriers, where only one thread in a warp issues the `cuda::memcpy_async`, while others simply wait on the barrier. It frees up threads to perform useful work while data transfers proceed in parallel.

#### Comparison to A100

![TMA](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/Asynchronous-Memory-Copy-with-TMA-on-H100-vs-LDGSTS-Instruction-on-A100.jpg)

Source: [NVIDIA Hopper Architecture in Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

#### New way to move data

![New way TMA](/assets/explore_gemms_2_tma_nvidia_presentation.png)

Source: [Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)

### Asynchronous Transaction Barriers

Asynchronous barriers were introduced in Ampere and split synchronization into two non-blocking steps: Arrive, where threads signal they’ve finished producing data and can continue doing other work, and Wait, where threads block only when they actually need the results. This overlap boosts performance. We already used this in our previous kernels in our pipelined kernels in previous post. In Hopper, waiting threads can now sleep instead of spinning, reducing wasted cycles. 

Hopper also has a more advanced primitive called the asynchronous transaction barrier. In addition to Arrive and Wait, it also tracks the amount of data produced. Threads perform Arrive operations that include a transaction (byte) count, and the Wait step blocks until both conditions are met: all threads arrived and the total data produced reaches a specified threshold. These transaction barriers are crucial for asynchronous memory operations and efficient data exchange that we discussed in previous sections. 

![Async Execution](/assets/explore_gemms_2_async_execution.jpg)

![Async Barrier](/assets/explore_gemms_2_async_barrier_a100vsh100.jpg)

Source: [Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)


### Warp Group Instructions

Hopper introduces the asynchronous warpgroup-level matrix multiply and accumulate operation (WGMMA). A warpgroup consists of four contiguous warps, i.e., 128 contiguous threads, where the warp-rank of the first warp is a multiple of four. The `wgmma.mma_async` instruction is executed collectively by all 128 threads in a warpgroup. 

The following matrix shapes are supported for bf16 dense computations for the wgmma.mma_async operation:

`m64n8k16`, `m64n16k16`, `m64n24k16`, `m64n32k16`, `m64n40k16`, `m64n48k16`, `m64n56k16`, `m64n64k16`, `m64n72k16`, `m64n80k16`, `m64n88k16`, `m64n96k16`, `m64n104k16`, `m64n112k16`, `m64n120k16`, `m64n128k16`, `m64n136k16`, `m64n144k16`, `m64n152k16`, `m64n160k16`, `m64n168k16`, `m64n176k16`, `m64n184k16`, `m64n192k16`, `m64n200k16`, `m64n208k16`, `m64n216k16`, `m64n224k16`, `m64n232k16`, `m64n240k16`, `m64n248k16`, `m64n256k16`

Source: [PTX Handbook](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape)

### Other Features
- **Native FP8 Support**: H100 also introduced native FP8 (8-bit floating-point) with two formats (E4M3 and E5M2)
- **Larger L2 Cache**: 25% L2 Cache increase to 50 MB compared to 40 MB 

We will ignore the nvlink, nvswitch features since we will just work on a single GPU.

## CUTLASS 3.x and Support for Hopper

Before we dive into implementing a Hopper-optimized kernel, let's quickly look at changes NVIDIA made in CUTLASS 3.x+ to introduce a a new five-layer hierarchy for GEMM Kernels to make them composable, and portable for future architectures. 

We have previously looked at the GEMM hierarchy as shown below:

![Cutlass 2.x](/assets/explore_gemms_2_cutlass_2x_gemm.png)

For Hopper and beyond, the API was changed to be centered around conceptual GEMM hierarchy instead of Hardware Hierarchy:

![Cutlass 3.x](/assets/explore_gemms_2_cutlass_3x_gemm.png)

The way this translates into GEMM API is shown below. For more detailed intro on GEMM API, [CUTLASS documentation on CUTLASS 3.x GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#) is pretty useful, though I think the documentation was a bit better.

The Collective layer is particularly important for Hopper+ kernels. It's where temporal micro-kernels orchestrate the producer-consumer pattern we discussed earlier—producer warps issuing TMA loads while consumer warps execute WGMMA operations, all coordinated through asynchronous transaction barriers.

<div id="cutlass-3x-hierarchy-viz"></div>


### Producer-Consumer Pipeline

The combination of TMA, async barriers, and warp specialization enables a powerful producer-consumer pattern in Hopper GEMMs. Producer warps handle data movement (TMA loads), while consumer warps focus purely on computation (WGMMA). This separation, combined with software pipelining, allows memory transfers and computation to overlap nearly perfectly.

<div id="hopper-gemm-pipeline-viz"></div>

## Resources

- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)
- [Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)
- [CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)


<script src="/assets/js/hopper-gemm-pipeline.js"></script>