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

![Cuda compilation](/assets/cuda_compile.png)

## Sources


- [Cuda Compilations](https://leimao.github.io/blog/CUDA-Compilation/)
- [Cuda compilation](https://docs.nvidia.com/cuda/archive/11.6.1/cuda-compiler-driver-nvcc/index.html#gpu-compilation)
- [Triton Source Code](https://github.com/triton-lang/triton/blob/aa3ac0a146def686877685b4fb8897db64789c7a/python/test/unit/tools/test_aot.py#L427)
- [MLIR](https://mlir.llvm.org/)
- [Semi Analysis Blog](https://www.semianalysis.com/p/nvidiaopenaitritonpytorch)

