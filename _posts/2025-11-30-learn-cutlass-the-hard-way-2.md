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

This is a continuation to my nprevious post [Learn CUTLASS the Hard Way](../learn-cutlass-the-hard-way), where I will explore Hopper architecture. There were several new architectural changes that made it to H100 and had huge implications on the performance of GEMM kernels, specially for 16-bit and lower precision GEMMS. There is already a great post from Pranjal on his H100 Journey at [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog). Expect this post to be a bit more higher level than that as we will primarily use CUTLASS to leverage the same optimizations. 

## Setup

Verda etc

## My Previous Baseline

However, before we jump into Hopper kernels, let's start with baselining the kernel we wrote for Ada (RTX 4090) on H100. We already had a CUTLASS kernel that we wrote in the previous post. It leveraged CUTLASS 2.x API and was primarily meant for Ada and older generation of GPUS. But, we can still run it on H100 to get a baseline performance. 

#### Baseline Performance of `kernel_cutlass`

![Baseline Perf](/assets/explore_gemms_2_hopper_baseline_perf.png)

Below we compare original *~first~ CUTLASS kernel* (and other handwritten CUDA implementations) against PyTorch on H100. Key observations:

| Matrix Size | CUTLASS Performance | PyTorch Performance |
|-------------|---------------------|---------------------|
| **Small (64-1024)** | 95-320 TFLOPS (82-90% of PyTorch) | N/A |
| **Medium (1536-3072)** | 317 TFLOPS at 3K | 686 TFLOPS at 3K |
| **Large (4096-8192)** | 396 TFLOPS at 4K<br>375 TFLOPS at 8K | 754 TFLOPS at 4K<br>685 TFLOPS at 8K |

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

> The Collective layer is particularly important for Hopper+ kernels. It's where temporal micro-kernels orchestrate the producer-consumer pattern we discussed earlier, where producer warps issue TMA loads and consumer warps execute WGMMA operations, all coordinated through asynchronous transaction barriers. 
{: .prompt-info}

The way this translates into GEMM API is visualized below. For more detailed intro on GEMM API, [CUTLASS documentation on CUTLASS 3.x GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html#) is pretty useful, though I think the documentation was a bit better.

Example kernels are useful though. Most useful example in the CUTLASS repo I found was [Hopper GEMM with Collective Builder](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder) example. 

<div id="cutlass-3x-hierarchy-viz"></div>

### Warp Specialization

[Warp specialization (also called spatial partitioning)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#spatial-partitioning-also-known-as-warp-specialization) is a technique where different warps within a thread block are assigned distinct roles rather than executing identical work. Instead of all warps performing the same operations on different data, specialized warps focus on specific tasks i.e. some handle data movement while others perform computation. As GEMMs move to async producer/consumer paradigms, this becomes essential so that a single warp is not holding all the required registers and shared memory resources. It also allows for handling unpredictable cycle counts for memory loads / TMA more efficiently. We will see more paradigms in this later but overall it reduces "empty bubbles" such that some warps can continue executing while others wait on blocking operations. 

#### Simple Warp Specialized Producer-Consumer Pattern

The visualization below shows a simple warp-specialized kernel with one producer and one consumer such that:

- **Producer warps** initiate asynchronous data transfers (e.g., using `cuda::memcpy_async` or TMA operations)
- **Consumer warps** perform computations on previously loaded data
- Both groups coordinate using asynchronous barriers

In this arrangement, we overlap memory operations with computation to reduce idle time and improving throughput. Actual timelines can look different depending on memory/compute boundness but this just aims to demostrate the general idea.

<div id="warp-specialization-viz"></div>

## TMA Warp Specialized Kernel

### Basic/Naive

Let's start with baseline TMA Warp Specialized version of the Kernel. We will skip basic kernels such as TMA without Warp Specialization since we already know those would end up being slower and is already discussed in some of the other posts linked. Instead, we can start with Hopper specific kernels that provide some reasonable baseline performance. 

Let's look at the most important pieces of the kernel. For the whole kernel, see [Full Hopper CUTLASS kernel](https://github.com/gpusgobrr/explore-gemm/blob/main/cuda/15_kernel_cutlass_hopper.cu).


First, we define the Element types and Layouts. We pass RowMajor tensors as is to the kernel.

```c
using ElementA = cutlass::bfloat16_t;
using ElementB = ElementType;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Alignment (16-byte for TMA)
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
```

Next, we define the TileShape and Thread Block Cluster shapes. We use `Shape<_1, _1, _1>` meaning not using TBC.

```c
// Tile and cluster configuration for H100
using TileShape = Shape<_128, _128, _64>; // CTA tile (M, N, K)
using ClusterShape = Shape<_1, _1, _1>;   // Not using Thread block cluster
```
Next, we define the GEMM Op, which is mostly boilerplate. You can see that we use the `KernelSchedule` and `EpilogueSchedule` for TMA warp specialized kernel. For software pipelinining, we start with a 2 Stage Kernel. 

```c
// Warp specialization schedules
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

// Build mainloop collective with automatic stage count calculation
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCount<2>, // 2 stages hard coded
    KernelSchedule>::CollectiveOp;

// Build epilogue collective
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule>::CollectiveOp;

// Assemble the kernel (using non-batched shape)
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

To define the strides, we use the cute utililities provided by cutlass to set the corresponding shapes. Note that `{M, K, 1}` essentially means `M x K` matrix since we use number of batches as 1. 

```c
// Problem size (non-batched GEMM)
auto problem_shape = make_shape(M, N, K);

// Stride types for row-major layouts
using StrideA = typename Config::GemmKernel::StrideA;
using StrideB = typename Config::GemmKernel::StrideB;
using StrideC = typename Config::GemmKernel::StrideC;
using StrideD = typename Config::GemmKernel::StrideD;

auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
```

Let's see the performance results compared to PyTorch.

![2 Stage Kernel TMA WS](/assets/explore_gemm_2_basic_2_stage_tma_warp_specialized.png)

> Overall, the performance for large matrices is only about 20-25% - which is actually worse than what we saw with the original CUTLASS 2.x based baseline we tested. 
{: .prompt-warning}

We can see from the Speed of Light analysis in NCU that SM and Memory throughput is pretty bad, as expected.

![SOL Analysis 2 stage](/assets/explore_gemms_2_stage_2_ncu_sol.png)

### Increasing/Auto Stage Count

**We haven't tuned anything here, so let's start with just updating the number of stages from hard-coded value of 2 to Auto.** Only thing we need to change is stage argument:

```diff
     // Warp specialization schedules
     using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
@@ -57,6 +58,7 @@ struct CutlassHopperGemmConfig
         ElementAccumulator,
         TileShape,
         ClusterShape,
-        cutlass::gemm::collective::StageCount<2>, // 2 stages hard coded
+        cutlass::gemm::collective::StageCountAuto,
         KernelSchedule>::CollectiveOp;
```

![Stage Auto Benchmark](/assets/explore_gemms_basic_stage_auto_benchmark.png)

> We pretty much doubled our performance for medium to larger matrices and saw marginal improvements in smaller matrices.
{: .prompt-info}

Looking at the NCU Speed of Light profile for size 4096, we can see that increase in TFLOPS is directly proporsional to the SM and Memory throughput. 

![Stage Auto NCU SOL](/assets/explore_gemms_2_ncu_sol_stage_auto.png)

Now, looking at the Memory Chart, I saw the L2 cache hit rate improve to 73% from 63% but nothing much changed. If you noticed previosly, we set the cluster shape to `Shape<_1, _1, _1>` i.e. we do not share any date across the thread block cluster. This is validated by the 0 input/output from `DSMEM`.

![Memory Chart Stage Auto](/assets/explore_gemms_2_stage_auto_tma_ws_memory_chart.png)

**Hence, a natural place to tune next is use Thread Block Clusters.**

### Thread Block Cluster

Next, we will test a `1 x 2 x 1` Thread Block Cluster Tile to cooperate across 2 SMs along the reduction dimension. This should allows threads from 2 SMs at a time to be able to share memory across 2 thread blocks.

```diff
--- a/cuda/15_kernel_cutlass_hopper.cu
+++ b/cuda/15_kernel_cutlass_hopper.cu
@@ -42,8 +42,7 @@ struct CutlassHopperGemmConfig
 
     // Tile and cluster configuration for H100
     using TileShape = Shape<_128, _128, _64>; // CTA tile (M, N, K)
-    using ClusterShape = Shape<_1, _1, _1>;   // Not TBC
+    using ClusterShape = Shape<_1, _2, _1>;   // Thread block cluster

```

![TBC 2 Benchmark](/assets/explore_gemms_2_tbc_2_benchmark.png)

> We see a modest 5% improvement in performance by enabling Thread Block Clusters, which is decent but not much. We are still hovering around 45-55% of Pytorch performance for any batch size > 1024. 
{: .prompt-info}

## Warp-specialized Persistent Cooperative Kernel

The Persistent Cooperative kernel extends basic warp specialization with the following:

- Persistent Thread Blocks, launching one thread block per output tile and occupy a fixed number of thread blocks specified in [`KernelHardwareInfo`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/kernel_hardware_info.hpp). For example, 132 for H100 and each processing multiple tiles. This amortizes kernel launch overhead and improves SM utilization.

- Two consumer warp groups acting as Cooperative Consumers, which split each output tile in half across the M dimension. This reduces register pressure per consumer, enabling larger tiles that improve arithmetic intensity and cache reuse.

- [`TileScheduler`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp) will dynamically assigns tiles to persistent thread blocks, considering cluster geometry and SM availability. Thread blocks atomically grab next tiles until the work queue empties.

Key changes in the code will look like:

> I initially tried to keep the number of stages as Auto but that led to a runtime error that had no useful. After debugging for a bit, I landed on constant stages = 5
{: .prompt-warning}


```diff
diff --git a/cuda/15_kernel_cutlass_hopper.cu b/cuda/15_kernel_cutlass_hopper.cu
index f02922c..07b1997 100644
--- a/cuda/15_kernel_cutlass_hopper.cu
+++ b/cuda/15_kernel_cutlass_hopper.cu
@@ -45,8 +45,12 @@ struct CutlassHopperGemmConfig
     using ClusterShape = Shape<_1, _2, _1>;   // Thread block cluster
 
     // Warp specialization schedules
-    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
-    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
+    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
+    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
+    using TileSchedulerType = cutlass::gemm::PersistentScheduler;
 
     // Build mainloop collective with automatic stage count calculation
     using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
@@ -57,7 +61,6 @@ struct CutlassHopperGemmConfig
         ElementAccumulator,
         TileShape,
         ClusterShape,
-         cutlass::gemm::collective::StageCountAuto,
+        // cutlass::gemm::collective::StageCount<5>,
         KernelSchedule>::CollectiveOp;
 
@@ -78,7 +81,8 @@ struct CutlassHopperGemmConfig
     using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
         Shape<int, int, int>,
         CollectiveMainloop,
-        CollectiveEpilogue>;
+        CollectiveEpilogue,
+        TileSchedulerType>;
 
     using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
 };

```

Note that we already previously had `cutlass::KernelHardwareInfo hw_info`, which be useful now.

#### Hardware info

```cpp
cutlass::KernelHardwareInfo hw_info;
hw_info.device_id = 0;
hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
```

<div id="persistent-cooperative-viz"></div>

![Persistent Cooperative](/assets/explore_gemms_2_persistent_cooperative_stage_5.png)

> We got a nice boost in performance for larger matrices and now our kernel is ranging from 60-70% of PyTorch performance. We are able to achieve upto 480-490 TFLOPS for 4096-8192 batch sizes compared to Pytorch which is in the range of 700-750 TFLOPS.
{: .prompt-info}

> We observe a regression for smaller matrices that we'll address later during autotuning when we explore different tile sizes and stage counts optimized for memory-bound workloads.
{: .prompt-warning}

## Ping Pong Schedule

The Ping-Pong schedule extends the Cooperative pattern to overlap epilogue with mainloop computation. In Cooperative Schedule, both consumer groups work on the same output tile, sharing A/B buffers. When both finish their MMA operations, tensor cores sit idle during the epilogue (storing results to global memory). This sequential execution leaves performance on the table.

> In vanilla persistent cooperative scheduling both consumer Warp Groups have the dependency to the same resources i.e. the smem tile for matrix A/B/C/D since they deal with the same tiles
{: .prompt-warning}

In Ping-Pong Schedule, the tile scheduler assigns each consumer a different output tile. The producer uses ordered sequence barriers to fill buffers alternately. While consumer 1 executes MMA operations, consumer 2 performs its epilogue. They then swap roles—maximizing tensor core utilization by overlapping computation with memory writes.

![PyTorch Ping Pong Schedule](/assets/explore_gemms_2_pytorch_ping_pong_fp8_blog.png)


# TODO: Visualization

## CTA Rasterization and Swizzle

We discussed swizzling in the previous blog post but we will cover it a bit more here. The way I understand it, [CTA rasterization](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#threadblock-rasterization) defines the order in which thread blocks map to GEMM tiles and are executed on the GPU. A naive row-major launch often leads to poor L2 reuse, redundant global loads, and memory partition camping. As we previously covered, CTA swizzling remaps `(blockIdx.x, blockIdx.y)` to improve spatial and temporal locality across tiles. Proper swizzling increases reuse of A and B tiles in L2, reduces DRAM traffic, and improves SM load balance. We did not any swizzling techniques on the RTX 4090 we used in previous post but on modern GPUs (Hopper/Blackwell), swizzling is critical for cluster-level execution, enabling shared memory reuse and efficient async/TMA pipelines. [Bert Maher has a nice writeup](https://github.com/bertmaher/simplegemm/tree/main) where he describes his process of understanding swizzling while reproducing pingpong swizzleed kernel from scratch. 

## Stream-K Scheduling

### Wave Quantization Problem

Standard GEMM kernels partition output tiles across SMs in discrete waves when the number of work units exceeds number of available SMs. Hence, when work tiles don't divide evenly by SM count, the final partial wave leaves SMs idle i.e. **wave quantization**.

> On H100 SXM5 with 132 SMs, computing 133 tiles requires 2 full waves i.e. identical cost to computing 264 tiles. The 133rd tile effectively halves device utilization.
{: .prompt-info}

<div id="wave-quantization-viz"></div>

For more details, see [prior work from Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) on this:

![WQ CR](/assets/wave_quantization_colfax.png)

Source: [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)

#### Data-Parallel Approach: Simple but Costly

The most direct solution is to reduce tile size and creating more work units to fill partial waves. In our 4 SM example, this would create 18 tiles instead of 9, improving utilization from 75% to 90%. However, smaller tiles degrade efficiency due to loss of arithmetic intensity drops. (refer to tiling sections from prevous post). In essense, this will reduce latency-hiding opportunities for the warp scheduler. Hence, While data-parallel tiling improves wave balance, the per-tile performance loss often negates any gains—making it an incomplete solution.

### Split-K: Partitioning Along Reduction Dimension

Next natural extenstion here is Split-K, where we could divide tiles along the K-dimension into a constant number of pieces (e.g., splitting a 128×128×128 tile into two 128×128×64 pieces). Unlike splitting M or N, this increases work units without shrinking output tile dimensions. This can preserve arithmetic intensity better than data-parallel approach. Since each CTA accumulates only partial results for its output tile, CTAs collaborating on the same tile perform turnstile reduction in a global memory workspace such that each waits at a barrier for CTAs processing earlier K-slices, reduces its partial results into the workspace, then signals completion. The final CTA reduces from the workspace to accumulators and executes the epilogue.

> More splits improve wave balance but degrade K-tile efficiency (lower arithmetic intensity, fewer latency-hiding opportunities) and increase synchronization overhead (barriers + GMEM traffic). Hence sweet spot depends on problem and hardware specific tuning.
{: .prompt-tip}

<div id="split-k-viz"></div>


### Stream-K: Fractional Tile Assignment

That brings us to Stream-K, which aims to eliminate wave quantization completely by assigning each persistent CTA a fractional number of work tiles. In our 9-tile, 4-SM example, each SM computes exactly 2.25 tiles instead of discrete waves. SM0 processes tiles 0, 1, and ¼ of tile 2; SM1 completes tile 2, processes tile 3, and starts half of tile 4. Split tiles partition along K-dimension using turnstile reduction (like Split-K), but with temporal scheduling—early K-pieces compute well before final pieces, minimizing barrier wait times.

This eliminates quantization entirely. Total time approaches 2.25 work units (vs 3 waves in the naive approach) with only minimal synchronization overhead. Most original 128×128×128 tiles remain intact, maintaining high arithmetic intensity and full WGMMA instruction availability. Temporal scheduling ensures epilogue-computing CTAs rarely wait i.e. earlier collaborators finish K-slices far in advance. The trade-off is additional GMEM workspace for partial tile sharing between CTAs, similar to Split-K but with the better load balancing.

<div id="stream-k-viz"></div>

### Hybrid Stream-K

Hybrid Stream-K combines both approaches: it uses Stream-K scheduling for one full wave plus the partial wave, then switches to conventional data-parallel scheduling for remaining complete tiles. This design recovers cache locality benefits while eliminating quantization effects. Since all CTAs process the same total amount of Stream-K work, they finish this phase simultaneously before proceeding to standard tiling—balancing load and cache efficiency.

<div id="hopper-gemm-pipeline-viz"></div>

## Resources

- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)
- [Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)
- [CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
- [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html)
- [CUDA C++ Programming Guide: Spatial Partitioning (Warp Specialization)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#spatial-partitioning-also-known-as-warp-specialization)
- [Warp Specialization Blog Post](https://rohany.github.io/blog/warp-specialization/)
- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
- [Why we have both Ping-pong and Cooperative schedule?](https://github.com/NVIDIA/cutlass/issues/2181)
- [Stream-K Paper](https://arxiv.org/pdf/2301.03598)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant)
- [bertmaher/simplegemm](https://github.com/bertmaher/simplegemm)
- [ColfaxResearch/cfx-article-src](https://github.com/ColfaxResearch/cfx-article-src)


<script src="/assets/js/hopper-gemm-pipeline.js"></script>