---
title: Triton Kernels - Fused Softmax - 2
description: >-
  Worklog: Performance debugging Triton Kernel
date: 2025-09-20
categories: [Blog]
tags: [AI, Machine Learning, Triton, GPU, Kernels, Profiling]
pin: true
math: false
author: ks
---

As a recap, in the [prevous post](../triton-kernels-softmax) I implemented a Triton softmax kernel. The kernel achieves competitive performance with PyTorch at small-to-medium vocabulary sizes but shows degradation at very large vocabularies (32K+) and large batch_sizes. In this post, we will use some profiling tools to debug why that might be the case. 

## NCU Profiler

I provided a snapshot of ncu-ui in the previous post, but let's look deeper into the results. You can run a full ncu profiler from the cli like below and then inspect the results in the ui.

```shell
# Pytorch version
ncu --set full  -o output python softmax_benchmark.py --M 8192 --N 32000  --iters 1000 --backend torch
# Triton version
ncu --set full -o output python softmax_benchmark.py --M 8192 --N 32000  --iters 1000 --backend triton
```

The profiling results provide you a lot of details, but on careful inspection -- couple of things jumped out.

### Scheduler and Warp State Statistics

#### Triton

![Triton warp state](/assets/softmax_warp_state_stats_triton.png)

#### Torch
![Torch warp state](/assets/softmax_warp_state_stats_torch.png)

> - The metric **Warp Cycles per Issued Instruction** 📊 reflects the average latency between consecutive instructions. The slower kernel shows a higher value **(14.43 vs. 11.99)** ⚠️, meaning warps spend more cycles stalled before issuing the next instruction. 
> - This indicates reduced instruction throughput 📉 and a greater need for warp parallelism to hide latency. The stall breakdown should be examined 🔍 to identify whether memory, dependencies, or occupancy limits are the main cause.
{: .prompt-info}

### Warp stalls 

#### Triton

![Triton warp stalls](/assets/softmax_ncu_warp_stalls_triton.png)

#### Torch
![Torch warp stalls](/assets/softmax_ncu_warp_stalls_torch.png)

We can see from the above distribution that warps are stalled a lot more for the top 4-5 locations for Triton vs Torch. NCU allows you click into where the stalls are coming from. Let's see that next:

![Stall compare](/assets/softmax_ncu_stall_sass_compare.png)

The Specific Instruction: @P0 CALL.REL.NOINC 0x70efb7c4c750

```
@P0   CALL.REL.NOINC 0x70efb7c4c750    138    2052096    20537
```

This instruction is:

- `@P0`: Predicated execution - only executes if predicate register P0 is true
- `CALL.REL.NOINC`: A relative function call with no automatic increment
- `0x70efb7c4c750`: The target address to call
- `138`: Cycle number when this instruction executes
- `2052096`: Number of threads executing this instruction
- `20537`: Stall cycles - **this is why it's concerning!**

Looking at the instructions where `0x70efb7c4c750` is used:

```assemby
466    @!P0  MOV R35, R18     132    1    0                                
467    @!P0  BRA 0x70efb7c4cd30     132    1                
468          FSETP.GEU.FTZ.AND P0, PT, R18, RZ, PT     132    0
469    @!P0  IMAD.MOV.U32 R35, RZ, RZ, 0x7fffffff     132    0           
470    @!P0  BRA 0x70efb7c4cd30     132    0
471          FSETP.GTU.FTZ.AND P0, PT, |R18|, +INF , PT     132    0
472    @P0   FADD.FTZ R35, R18, 1     132    0
473    @P0   BRA 0x70efb7c4cd30     132    0    0
```

This appears to be implementing a **floating-point validation and sanitization routine** - likely part of some validation for NaN, inf.

```
466  @!P0  MOV R35, R18        # If P0 is false, copy R18 to R35
467  @!P0  BRA 0x70efb7c4cd30  # If P0 is false, branch to exit
```
This handles some initial condition where if P0 is false, it just passes through the input value in R18.

```
468  FSETP.GEU.FTZ.AND P0, PT, R18, RZ, PT  # Set P0 = (R18 >= 0.0)
469  @!P0  IMAD.MOV.U32 R35, RZ, RZ, 0x7fffffff  # If negative, set R35 = NaN
470  @!P0  BRA 0x70efb7c4cd30              # If negative, branch to exit
```
**Purpose**: Handle negative numbers (probably related to `sqrt`)
- Tests if R18 (input) is greater than or equal to zero
- If the number is **negative** (!P0), it sets R35 to `0x7fffffff` (which is NaN in IEEE 754) and exits
- This suggests the function only accepts non-negative inputs

```
471  FSETP.GTU.FTZ.AND P0, PT, |R18|, +INF, PT  # Set P0 = (|R18| > +INF)
472  @P0   FADD.FTZ R35, R18, 1               # If NaN, add 1 (propagates NaN)
473  @P0   BRA 0x70efb7c4cd30                 # If NaN, branch to exit
```
**Purpose**: Handle NaN inputs
- Tests if the absolute value of R18 is greater than infinity (only true for NaN)
- If input is **NaN**, it adds 1 to propagate the NaN and exits
- The `FADD.FTZ R35, R18, 1` ensures NaN propagation (NaN + anything = NaN)

This suggests the kernel is spending significant time validating inputs resulting in warp divergence.

> I tried changing the max value to something like -100.0 but that did not seem to help. 
{: .prompt-info}

Comparing this to torch version:

```
388	@P0   CALL.REL.NOINC 0x70175fc4c750 	138	2052096	16792	
```

> `0x70175fc4c750` is not used anywhere else which seems to indicate there is no such branching in the torch version. 
{: .prompt-info}

I have uploaded the assembly to gists for detailed comparison:

### SASS Assembly Code Comparison

<div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 1px solid #0284c7; border-radius: 12px; padding: 24px; margin: 24px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
  <div style="margin-bottom: 16px;">
    <h4 style="color: #0369a1; margin: 0 0 8px 0; font-size: 1.1em;">🔍 Assembly Code Analysis</h4>
    <p style="color: #374151; margin: 0; line-height: 1.5;">
      The following gist contains the SASS assembly output for both Triton and PyTorch kernels. You can see the differences in instruction patterns and optimization strategies between the two implementations.
    </p>
  </div>

  <div class="gist-wrapper" style="background-color: white; padding: 20px; border-radius: 8px; max-height: 600px; overflow-y: auto; border: 1px solid #cbd5e1;">
    <script src="https://gist.github.com/kapilsh/e05eb60eda61c3c43d4a131b34ad29dc.js"></script>
  </div>

  <div style="margin-top: 16px; padding: 12px; background-color: rgba(255, 255, 255, 0.7); border-radius: 6px;">
    <strong style="color: #0369a1;">📁 Files in the gist:</strong>
    <ul style="margin: 8px 0 0 0; color: #374151;">
      <li><code>triton_sass</code> - Assembly output from the Triton softmax kernel</li>
      <li><code>torch_sass.txt</code> - Assembly output from PyTorch's optimized softmax implementation</li>
    </ul>
  </div>
</div>

<style>
.gist-wrapper .gist-file {
  background-color: white !important;
}
.gist-wrapper .gist-data {
  background-color: white !important;
}
.gist-wrapper .highlight {
  background-color: white !important;
}
.gist-wrapper .blob-code-inner {
  background-color: white !important;
  color: #333 !important;
}
.gist-wrapper .blob-num {
  background-color: #f8f8f8 !important;
  color: #666 !important;
}
.gist-wrapper .gist-meta {
  background-color: #f8f8f8 !important;
}
</style>

## Summary

Performance profiling using NCU revealed that the Triton softmax kernel shows degraded performance due to increased warp stalls. 

Key findings:

- **Warp latency**: Triton kernel shows 14.43 vs 11.99 cycles per instruction compared to PyTorch
- **Stall cause**: Excessive time spent in floating-point validation routines (`CALL.REL.NOINC 0x70efb7c4c750`)
- **Warp divergence**: The validation code handles NaN/infinity checks causing significant branching overhead
- **Assembly difference**: PyTorch implementation avoids this validation bottleneck entirely

The performance gap stems from compiler-generated safety checks rather than algorithmic differences but that is s debugging exercise for another day! ✌️ 


### Code

**[🔗 Triton Kernels Implementation](https://github.com/kapilsh/gpt-oss-scratch/tree/main/kernels)**