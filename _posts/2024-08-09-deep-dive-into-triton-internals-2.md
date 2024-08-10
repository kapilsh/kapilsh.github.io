---
title: Deep Dive into Triton Internals (Part 2)
description: >-
  Let's explore triton compiler with debugger
date: 2024-08-09
categories: [Blog, Tutorial]
tags: [AI, Triton, CUDA, Machine Learning, Compiler]
pin: true
math: true
author: ks
---

In the [previous post](../deep-dive-into-triton-internals/), we explored the Triton compiler internals and code generation pipeline under the hood. In this post, we will explore further with debugger to understand what happens when `triton.compile` is called in the frontend. We will use a debugger to peak into the compiler and understand the code generation process. 

We covered in the previous post that you can generate the code for the kernel by using triton compiler directly. Follow the instructions in [triton repo](https://github.com/triton-lang/triton) to fork and install triton from source. 

> Triton's internals might change in the future. The git hash for the version that I am working on is:
> 
> ```shell
> git rev-parse origin/master
> # b35bd67059a3e2bf7f6407ca0f098eb6f796d56f
> ```
{: .prompt-tip}

Here's the command again to compile one of the tutorial examples:

```shell
$ python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  python/tutorials/01-vector-add.py
```
This will generate 2 files: `add_kernel.9969bdda_0123.c`, `add_kernel.9969bdda_0123.h`! Let's look into how the files are generated:



