---
title: "Learn CUTLASS the hard way - Cute Layouts"
description: >-
  Exploring CUTE and Layouts
date: 2026-02-19
categories: [Blog]
tags: [CUDA, Cute, GEMM]
pin: true
math: false
author: ks
---

I have been exploring CUTE more recently and the hardest concept to wrap your head around when starting with CUTE/CuteDSL is **layouts**. Everything else builds on top of them, so getting a solid mental model early saves a lot of pain. This post is my attempt to demystify them. In the era of AI slop, hoping this can serve as a good reference for folks who are trying to get started with CUTE. 

## Shapes and Strides

No matter how many dimensions a tensor has, it lives in a **flat 1D buffer** in memory — a contiguous array of elements. Shape and strides are the bookkeeping layer that lets you treat that flat buffer as a multi-dimensional object.

A tensor's **shape** is a tuple of dimension sizes. A 2D tensor with shape `(M, N)` has `M × N` elements accessible via row index `i ∈ [0, M)` and column index `j ∈ [0, N)`.

**Strides** are the jumps. For each dimension, the stride is how many elements you skip in the flat buffer when you advance one step along that axis. Given coordinates `(i, j)`, the element's position in the buffer is:

```
offset(i, j) = i × stride_0 + j × stride_1
```

This is just an inner product of the coordinate vector with the stride vector. It generalizes to any number of dimensions:

```
offset(i₀, i₁, ..., iₙ) = i₀×s₀ + i₁×s₁ + ... + iₙ×sₙ
```

### PyTorch uses the same model

Every `torch.Tensor` carries a shape, a stride tuple, and a pointer into a flat storage buffer. You can inspect this directly:

```python
import torch

t = torch.zeros(4, 8)
print(t.shape)    # torch.Size([4, 8])
print(t.stride()) # (8, 1)   ← row-major by default
```

`t[i, j]` lives at position `i*8 + j*1` in the underlying storage — same formula. The shape tells you valid index ranges; the strides tell you where to find each element.

**Transpose is free — no copy.** Swapping two dimensions just swaps their strides:

```python
t_T = t.T
print(t_T.shape)    # torch.Size([8, 4])
print(t_T.stride()) # (1, 8)
```

`t` and `t_T` point to the same 32 floats. Nothing moved.

**Slicing produces non-unit strides.** Taking every other row doubles the row stride:

```python
t_skip = t[::2, :]
print(t_skip.shape)    # torch.Size([2, 8])
print(t_skip.stride()) # (16, 1)
```

**Contiguity.** A tensor is *contiguous* (row-major) when each stride equals the product of all dimension sizes to its right: `stride[i] = shape[i+1] × shape[i+2] × ...`. PyTorch exposes this:

```python
t.is_contiguous()      # True
t.T.is_contiguous()    # False — strides are (1, 8), not (4, 1)
t.T.contiguous()       # forces a copy into fresh row-major storage
```

This is why `view()` requires a contiguous input — it reinterprets the stride pattern without copying. `reshape()` calls `contiguous()` silently when needed.

> PyTorch strides count in **elements**, not bytes — unlike NumPy which uses bytes. CUTE follows the same element-based convention.
{: .prompt-info}

**Why GPU kernels care.** On a GPU, threads in a warp execute in lockstep. If 32 consecutive threads each load one element, coalesced access requires those 32 elements to be adjacent in memory (consecutive offsets). A row-major access pattern along columns achieves this; a column-major access along rows does not — each thread's offset jumps by the row stride, scattering loads across memory.

This is the direct motivation for caring about layouts in CUDA code. The choice of strides determines whether your memory access is coalesced or scattered, and the difference can be an order of magnitude in bandwidth.

In CUTE, a `Layout` is the `(Shape, Stride)` pair made first-class — written as `(M,N):(s0,s1)`. Everything in CUTLASS is built from composing and transforming these pairs.

### Row-Major: `(4,8):(8,1)`

In row-major (C-order) layout, elements within the same row are contiguous in memory. Moving one step along columns (`j`) costs a stride of 1 — adjacent slots. Moving one step along rows (`i`) costs a stride of 8, skipping over a full row.

<div id="cute-row-major-viz"></div>

The memory bar at the bottom shows how the 32 elements land in the flat buffer. Row 0 occupies offsets `0–7`, row 1 occupies `8–15`, and so on — rows land as contiguous blocks.

### Column-Major: `(4,8):(1,4)`

Flip the strides: stride of `1` along rows (`i`), stride of `4` along columns (`j`). Now columns are contiguous — elements `(0,0)`, `(1,0)`, `(2,0)`, `(3,0)` map to offsets `0, 1, 2, 3`.

<div id="cute-col-major-viz"></div>

Column 0 occupies offsets `0–3`, column 1 occupies `4–7`, etc. This is the default layout for FORTRAN and what cuBLAS expects. The same 32 values, the same buffer — just a different stride pair changes the traversal order completely.

### 3D Tensor: `(4,4,4):(1,4,16)`

Extending to three dimensions is mechanical — add a third stride:

```
offset(i, j, k) = i×1 + j×4 + k×16
```

Each 2D slice (fixed `k`) is a 4×4 column-major matrix. The stride along `k` is `16 = 4×4`, exactly the size of one slice. The four grids below show slices `k=0` through `k=3`; each slice's base offset is `k×16`.

<div id="cute-3d-viz"></div>

With the flat-buffer model in mind, the 3D case is the same arithmetic — just one more term. CUTE extends this further with hierarchical (nested) shapes and strides, letting you express tiled layouts in a single `Layout` object. That's what makes it powerful for expressing the register and shared memory layouts that tensor core operations require.

<script src="{{ '/assets/js/cute-layout-visualizer.js' | relative_url }}"></script>

