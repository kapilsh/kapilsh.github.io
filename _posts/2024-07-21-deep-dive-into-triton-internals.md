---
title: Deep Dive into Triton Internals
description: >-
  Missing tutorial on how triton program gets converted to cuda kernels under the hood
date: 2024-08-01
categories: [Blog, Tutorial]
tags: [AI, Triton, CUDA, Machine Learning, Compiler]
pin: true
math: true
author: ks
---

## Cuda Compilation

Before we dive into triton, it is useful to understand the cuda compilation process using nvcc. Beloe diagram from [NVIDIA docs](https://docs.nvidia.com/cuda/archive/11.6.1/cuda-compiler-driver-nvcc/index.html#cuda-compilation-trajectory) depicts the whole process of how cuda is compiled to executable code.

![Cuda compilation](/assets/cuda_compile.jpg)

- The CUDA compiler (nvcc) preprocesses the input program for device compilation. This involves handling directives like #include and macros relevant to device code.
- The preprocessed device code is compiled into CUDA binary (cubin) and/or PTX (Parallel Thread Execution) intermediate code.
  - PTX is a low-level virtual machine and assembly language for CUDA. It allows for optimizations and is architecture-independent.
  - Cubin is a GPU-specific binary code optimized for the target architecture.
- Both cubin and PTX are placed into a fatbinary, which contains multiple versions of the code optimized for different GPU architectures.

Let us look at a simple example of cuda -> ptx. You can do that pretty easily using [Compiler Explorer](https://godbolt.org/z/GcYKT7cfo). It is very a silly example for demonstration purposes.

```cuda
#include <stdio.h>

// Size of array
#define N 1024

// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}
```

```
.visible .entry add_vectors(double*, double*, double*)(
        .param .u64 add_vectors(double*, double*, double*)_param_0,
        .param .u64 add_vectors(double*, double*, double*)_param_1,
        .param .u64 add_vectors(double*, double*, double*)_param_2
)
{

        ld.param.u64    %rd1, [add_vectors(double*, double*, double*)_param_0];
        ld.param.u64    %rd2, [add_vectors(double*, double*, double*)_param_1];
        ld.param.u64    %rd3, [add_vectors(double*, double*, double*)_param_2];
        mov.u32         %r2, %ntid.x;
        mov.u32         %r3, %ctaid.x;
        mov.u32         %r4, %tid.x;
        mad.lo.s32      %r1, %r2, %r3, %r4;
        setp.gt.s32     %p1, %r1, 1023;
        @%p1 bra        $L__BB0_2;

        cvta.to.global.u64      %rd4, %rd1;
        mul.wide.s32    %rd5, %r1, 8;
        add.s64         %rd6, %rd4, %rd5;
        cvta.to.global.u64      %rd7, %rd2;
        add.s64         %rd8, %rd7, %rd5;
        ld.global.f64   %fd1, [%rd8];
        ld.global.f64   %fd2, [%rd6];
        add.f64         %fd3, %fd2, %fd1;
        cvta.to.global.u64      %rd9, %rd3;
        add.s64         %rd10, %rd9, %rd5;
        st.global.f64   [%rd10], %fd3;

$L__BB0_2:
        ret;

}
```

## Triton Compiler

Let's start with the [original paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) that showed a high level overview of triton jit process. 

![Triton compiler from paper](/assets/triton_compiler_paper.png)

Though at high level, it still holds but the backend infrastructure was completely swapped to use [MLIR at the end of 2022](https://github.com/triton-lang/triton/pull/1004).

![Triton MLIR PR Merge](/assets/triton-mlir-pr.png)

Documentation on the updated triton compilation process is sparse to non-existent. The best resources that I found that went into some details on the compilation are below:

### Triton Compiler Talk (Pytorch Conference 2023)

{% include embed/youtube.html id='AtbnRIzpwho' %}

### Triton conference 2023 by Ian Bearman (Microsoft)

Another Talk that is helpful and goes into MLIR details:

{% include embed/youtube.html id='y2V3ucS1pfQ' %}

Couple of relevant slides that covers high-level compilation process

![Triton Compiler Middle Layer](/assets/triton-compiler-middle-layer.png)

![Triton Compiler Pytorch Conf](/assets/triton-compiler-pytorch-conf.png)


After reading through code, unit tests and watching these talks, here is my general summary on how the process works end to end.

- Triton takes the Python code written in its DSL and parses it. This code typically involves specifying tensor operations, kernels, and other high-level abstractions.
- The parsed Python code is translated into an intermediate representation. This IR is designed to capture the high-level structure of the computation while being amenable to various optimizations. 
- Various optimization passes are applied to the IR. These optimizations can include loop unrolling, memory access optimizations, constant folding, CSE, etc to improve the efficiency of the generated code.
- Progressive IR generation is done via MLIR dialects - Triton, TritonGPU, TritonNVIDIAGPU.  
- The optimized Triton IR is then lowered to LLVM IR. Triton leverages LLVM to perform further optimizations and to facilitate the generation of machine code.
- The LLVM IR is then used to generate CUDA code. This involves translating the LLVM IR to PTX code, which is the intermediate representation used by NVIDIA's CUDA compiler (nvcc).
- The generated PTX code is then compiled Just-In-Time (JIT) into a CUDA binary (CUBIN). This compilation is done using NVIDIA's JIT compiler, which converts the PTX code into machine code that can be executed on the GPU.

### Zoomed in view: Triton IR -> LLVM IR

![Triton to LLVM](/assets/triton-to-llvm-ir.png)

### Optimization Passes

#### MLIR general optimizations
• CSE, DCE, Inlining, …

#### TritonGPU specific optimizations
- Pipeline
- Prefetch
- Matmul accelerate
- Coalesce
- Remove layout

#### TritonNVIDIAGPU specific optimizations
• TMA Materialization
• TMA Multicast
• Async Dot
• Warp Specialization

## Python AST -> Triton IR, PTX, CUBIN

Next, we look into how to get the Triton IR, PTX, and CUBIN from Python code.


## Sources

- [Cuda Compilations](https://leimao.github.io/blog/CUDA-Compilation/)
- [Cuda compilation](https://docs.nvidia.com/cuda/archive/11.6.1/cuda-compiler-driver-nvcc/index.html#gpu-compilation)
- [Triton Source Code](https://github.com/triton-lang/triton/blob/aa3ac0a146def686877685b4fb8897db64789c7a/python/test/unit/tools/test_aot.py#L427)
- [MLIR](https://mlir.llvm.org/)
- [Semi Analysis Blog](https://www.semianalysis.com/p/nvidiaopenaitritonpytorch)
- [Technical Overview of Triton and Pytorch 2.0](https://www.jokeren.tech/slides/Triton_bsc.pdf)
- [Lei Mao BLog](https://leimao.github.io/)
- [Triton MLIR Dialects](https://github.com/triton-lang/triton/tree/b0f8332c7dedb6ce3a2cf365e53391775d4e4a2e/include/triton/Dialect)
