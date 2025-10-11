---
title: AI GPU Programming - Cuda Basics
description: >-
  Basic building blocks for GPU programming
date: 2025-08-27
categories: [Blog, Tutorial]
tags: [AI, Machine Learning, CUDA]
pin: false
math: false
author: ks
---

## Cuda Programming Basics

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform and programming model. It allows you to write C/C++-like code that runs on the GPU, which is highly parallel and suitable for tasks like matrix operations, image processing, and machine learning.
Here’s a quick explanation of how CUDA programming works, especially the concepts of blockDim, threadIdx, and blockIdx.

## 🧠 The Execution Model

When you launch a CUDA kernel (a function that runs on the GPU), you define:
Grid: A collection of blocks
Block: A collection of threads.
Each thread executes the same kernel code, but works on different data depending on its thread ID.

## 💡 Key Built-In Variables

These three built-in variables help identify which thread is executing:
1. `threadIdx`
* Identifies the thread within a block
* It’s a 3D index: threadIdx.x, threadIdx.y, threadIdx.z

2. `blockIdx`
* Identifies the block within the grid
* Also 3D: blockIdx.x, blockIdx.y, blockIdx.z.

3. `blockDim`
* Tells how many threads are in a block (in each dimension)
* 3D: blockDim.x, blockDim.y, blockDim.z



## 🧮 Calculating Global Thread Index
Often, you want a 1D global thread index to map each thread to an element in an array. You compute it like this:
int idx = blockIdx.x * blockDim.x + threadIdx.x;

This gives each thread a unique index across the entire grid.

### ✅ Example: Add Two Arrays
Here's a minimal CUDA example to add two arrays:

```c
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

Launching the Kernel

```c
int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
```


##  🪟 Visualization - vector_add
Let’s say we vector_add A and B with both having 8 elements each. We use block size of 4 and 2 thread blocks. 

It means: 

```c
blockDim.x = 4
blockIdx.x = 0, 1
threadIdx.x = 0, 1, 2, 3
blockIdx.x * blockDim.x + threadIdx.x: 0, 1, 2, 3, 4, 5, 6, 7
```

![Vector Add](/assets/vector_add_gpu.png)

## 🪟 Visualization - image_grayscale

Let's take below example

* Image size: 32 x 32 pixels
* Block size: 16 x 16 threads
* Grid size: 2 x 2 blocks (since 32 / 16 = 2)


```c
int width = 32, height = 32;
float *d_input, *d_output;
size_t size = width * height * sizeof(float);

cudaMalloc(&d_input, size);
cudaMalloc(&d_output, size);

// Assume input is copied from host

dim3 blockDim(16, 16);
dim3 gridDim((width + 15) / 16, (height + 15) / 16);

image_grayscale<<<gridDim, blockDim>>>(d_input, d_output, width, height);
cudaDeviceSynchronize();
```

Mapping out how blocks will run computations:

```c
blockIdx.x = 0, 1
blockIdx.y = 0, 1

threadIdx.x = 0 to 15
threadIdx.y = 0 to 15

For block (0,0):
    thread (0,0) => pixel (0,0)
    thread (15,15) => pixel (15,15)

For block (1,0):
    thread (0,0) => pixel (16,0)
    thread (15,15) => pixel (31,15)

For block (1,1):
    thread (0,0) => pixel (16,16)
    thread (15,15) => pixel (31,31)
..
```

![Gary scaling](/assets/gray_scale_gpu.png)

## 🔁 Summary
* `threadIdx`: Thread ID within a block
* `blockIdx`: Block ID within the grid
* `blockDim`: Number of threads per block
* `gridDim`: (Optional) Number of blocks in the grid

## 🎮 Interactive CUDA Execution Visualizer

Below is an interactive visualization showing how CUDA grids, blocks, warps, and threads execute in parallel on a GPU. This demonstrates a simple vector addition kernel with 4 blocks (2×2 grid) and 128 threads per block (16×8).

<div id="cuda-visualizer" style="margin: 2rem 0;"></div>
<script src="/assets/js/cuda-visualizer.js"></script>

