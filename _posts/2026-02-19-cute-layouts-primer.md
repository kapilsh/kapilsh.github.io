---
title: "Learn CUTLASS the hard way - Cute Layouts"
description: >-
  Exploring CUTE and Layouts
date: 2026-02-19
categories: [Blog]
tags: [CUDA, Cute, GEMM]
pin: true
math: true
author: ks
---

I have been exploring CUTE more recently and the hardest concept to wrap your head around when starting with CUTE/CuteDSL is **layouts**. Everything else builds on top of them, so getting a solid mental model early saves a lot of pain. This post is my attempt to demystify them. In the era of AI slop, hoping this can serve as a good reference for folks who are trying to get started with CUTE. 

## Shapes and Strides

No matter how many dimensions a tensor has, it lives in a **flat 1D buffer** in memory — a contiguous array of elements. Shape and strides are the bookkeeping layer that lets you treat that flat buffer as a multi-dimensional object.

A tensor's **shape** is a tuple of dimension sizes. A 2D tensor with shape `(M, N)` has `M × N` elements accessible via row index `i ∈ [0, M)` and column index `j ∈ [0, N)`.

**Strides** are the jumps. For each dimension, the stride is how many elements you skip in the flat buffer when you advance one step along that axis. Given coordinates `(i, j)`, the element's position in the buffer is:

$$\text{offset}(i,\, j) = i \times s_0 + j \times s_1$$

This is just an inner product of the coordinate vector with the stride vector. It generalizes to any number of dimensions:

$$\text{offset}(i_0, i_1, \ldots, i_n) = \sum_{k=0}^{n} i_k \times s_k$$

### PyTorch uses the same model

Every `torch.Tensor` carries a shape, a stride tuple, and a pointer into a flat storage buffer. You can inspect this directly:

```python
import torch

t = torch.zeros(4, 8)
print(t.shape)    # torch.Size([4, 8])
print(t.stride()) # (8, 1)   ← row-major by default
```

`t[i, j]` lives at position `i*8 + j*1` in the underlying storage — same formula. The shape tells you valid index ranges; the strides tell you where to find each element.

For a **contiguous** tensor, strides are fully determined by the shape — you don't get to choose them independently:

- **Row-major (C-order):** strides are **suffix products** of the shape. Each stride is the product of all dimension sizes that come *after* it.

```python
# shape (4, 8, 2) — row-major strides
# stride[0] = 8 * 2 = 16
# stride[1] = 2
# stride[2] = 1
t = torch.zeros(4, 8, 2)
print(t.stride())  # (16, 2, 1)
```

- **Column-major (Fortran-order):** strides are **prefix products**. Each stride is the product of all dimension sizes that come *before* it.

```python
# shape (4, 8, 2) — column-major strides
# stride[0] = 1
# stride[1] = 4
# stride[2] = 4 * 8 = 32
t_col = torch.zeros(4, 8, 2).permute(2, 1, 0).contiguous().permute(2, 1, 0)
print(t_col.stride())  # (1, 4, 32)
```

For the 2D case this simplifies cleanly — shape `(M, N)` gives strides `(N, 1)` for row-major and `(1, M)` for column-major:

```python
t_rm = torch.zeros(4, 8)
t_cm = torch.zeros(8, 4).T.contiguous().T   # allocate transposed, force contiguous, transpose back
print(t_rm.stride())  # (8, 1)  — suffix products: [8, 1]
print(t_cm.stride())  # (1, 4)  — prefix products: [1, 4]
```

The suffix/prefix product rule is exactly the formula for contiguity: `stride[i] = prod(shape[i+1:])` for row-major, `stride[i] = prod(shape[:i])` for column-major.

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

> **Mode** is CUTE's term for a dimension. A 2D layout has Mode 0 and Mode 1; a 3D layout adds Mode 2. For the flat layouts in this post, Mode and dimension are interchangeable — Mode 0 is the first dimension (rows), Mode 1 the second (columns), Mode 2 the third (depth). The term generalizes when layouts are nested: a single mode can itself carry a `(shape, stride)` pair to express tiled memory patterns, which is where "mode" becomes more expressive than "dimension".
{: .prompt-info}

### Row-Major: `(4,8):(8,1)`

In row-major (C-order) layout, elements within the same row are contiguous in memory. Moving one step along columns (`j`) costs a stride of 1 — adjacent slots. Moving one step along rows (`i`) costs a stride of 8, skipping over a full row.

<div id="cute-row-major-viz"></div>

The memory bar at the bottom shows how the 32 elements land in the flat buffer. Row 0 occupies offsets `0–7`, row 1 occupies `8–15`, and so on — rows land as contiguous blocks.

### Column-Major: `(4,8):(1,4)`

Flip the strides: stride of `1` along rows (`i`), stride of `4` along columns (`j`). Now columns are contiguous — elements `(0,0)`, `(1,0)`, `(2,0)`, `(3,0)` map to offsets `0, 1, 2, 3`.

<div id="cute-col-major-viz"></div>

Column 0 occupies offsets `0–3`, column 1 occupies `4–7`, etc. This is the default layout for FORTRAN and what cuBLAS expects. The same 32 values, the same buffer — just a different stride pair changes the traversal order completely.

### 3D Tensor Row-Major: `(4,4,4):(16,4,1)`

Flip to row-major (C-order): strides are suffix products of the shape, so `k` (the last dimension) gets stride `1`, `j` gets `4`, and `i` gets `4×4 = 16`:

$$\text{offset}(i,\, j,\, k) = i \times 16 + j \times 4 + k \times 1$$

Now `k` is the fastest-varying axis — the four elements `(i,j,0..3)` are adjacent in the buffer.

<div id="cute-3d-row-major-viz"></div>

### 3D Tensor Column-Major: `(4,4,4):(1,4,16)`

Extending to three dimensions is mechanical — add a third stride:

$$\text{offset}(i,\, j,\, k) = i \times 1 + j \times 4 + k \times 16$$

Each 2D slice (fixed `k`) is a 4×4 column-major matrix. The stride along `k` is `16 = 4×4`, exactly the size of one slice. The four grids below show slices `k=0` through `k=3`; each slice's base offset is `k×16`. Compare the Linear Memory bar against the row-major version above: here the k-slices are completely separated (k=0 owns `0–15`, k=1 owns `16–31`), while in row-major they were interleaved.

<div id="cute-3d-viz"></div>

### Hierarchical Column-Major: `((4,4),4):((1,4),16)`

The same 64-element buffer, now addressed as a **2-mode hierarchical layout**. Mode 0 groups `i` and `j` into a nested `(4,4):(1,4)` column-major submatrix; Mode 1 is `k` with stride 16:

$$\text{offset}\big((i,\,j),\,k\big) = i \times 1 + j \times 4 + k \times 16$$

The formula is identical to the flat version — nothing in memory changed. Only the **coordinate structure** changed: you address this tensor with a 2-tuple `((i,j), k)` instead of `(i, j, k)`. The grid groups rows by `j` (outer sub-mode of Mode 0), with `i` varying within each group and `k` across columns.

<div id="cute-3d-col-major-hier-viz"></div>

### Hierarchical Row-Major: `(4,(4,4)):(16,(4,1))`

Symmetrically, the row-major tensor groups `j` and `k` into Mode 1 as a nested `(4,4):(4,1)` row-major submatrix. Mode 0 remains `i` with stride 16:

$$\text{offset}\big(i,\,(j,\,k)\big) = i \times 16 + j \times 4 + k \times 1$$

The coordinate is now `(i, (j,k))`. The grid groups columns by `j` (outer sub-mode of Mode 1) with `k` varying within each group. This is exactly the structure CUTLASS uses to encode tiled GEMM layouts — thread-block tile, warp tile, and register tile each become nested sub-modes in a single `Layout` expression.

<div id="cute-3d-row-major-hier-viz"></div>

With the flat-buffer model in mind, the 3D case is the same arithmetic — just one more term. CUTE extends this further with hierarchical (nested) shapes and strides, letting you express tiled layouts in a single `Layout` object. That's what makes it powerful for expressing the register and shared memory layouts that tensor core operations require.

## BLAS Formats

BLAS GEMM computes `C = α·op(A)·op(B) + β·C`. In CUTE, this is expressed over `A(M,K)` and `B(N,K)` — both share the K (contraction) dimension. The NT/TN/NN/TT names describe the storage layout of each matrix:

| Name | Stride-1 dimension | Memory layout | BLAS op |
|------|-------------------|---------------|---------|
| M-major | M | column-major | op(A) = A (Not transposed) |
| K-major | K | row-major | op(A) = A^T (Transposed) |
| N-major | N | column-major | op(B) = B^T (Transposed) |

> In CUTE's convention, B has shape (N,K) — K is the second dimension. So a column-major B (N-major) is what classical BLAS calls a "transposed B."
{: .prompt-info}

The four combinations with M=4, K=6, N=2:

| Format | A layout | B layout |
|--------|----------|----------|
| NT | M-major `(4,6):(1,4)` | N-major `(2,6):(1,2)` |
| TN | K-major `(4,6):(6,1)` | K-major `(2,6):(6,1)` |
| NN | M-major `(4,6):(1,4)` | K-major `(2,6):(6,1)` |
| TT | K-major `(4,6):(6,1)` | N-major `(2,6):(1,2)` |

> Cells are colored by **K-column index** — the same color appears at the same `k` position in both A and B. The linear memory bar shows how the coloring pattern changes between column-major (grouped blocks) and row-major (interleaved) storage.
{: .prompt-info}

### NT: A M-major, B N-major

Both A and B are stored column-major. Each column (length M or N) is contiguous in memory. The linear memory bar shows blocks of 4 same-color cells — each K-column lands as a contiguous group.

<div id="cute-blas-nt-viz"></div>

### TN: A K-major, B K-major

Both A and B are stored row-major. Each row (length K=6) is contiguous. The memory bar shows an interleaved color pattern — k=0,1,2,3,4,5 repeating as each M (or N) row lands sequentially.

<div id="cute-blas-tn-viz"></div>

### NN: A M-major, B K-major

A is column-major (M-major), B is row-major (K-major). A mixed layout — A groups by K-column in memory, B groups by N-row.

<div id="cute-blas-nn-viz"></div>

### TT: A K-major, B N-major

A is row-major (K-major), B is column-major (N-major). The other mixed layout.

<div id="cute-blas-tt-viz"></div>

## nvfp4 Layout

nvfp4 (e2m1) is NVIDIA's 4-bit float for Blackwell (B200/GB200) tensor cores — **1 sign + 2 exponent + 1 mantissa bit**. It achieves 2× memory density over fp8 via two mechanisms stacked on top of a standard K-major layout.

### Byte packing

Two 4-bit elements share one byte — the even-indexed element occupies the **low nibble** (bits 3:0), the odd-indexed element the **high nibble** (bits 7:4):

$$\text{byte}[b] = e_{2b+1} \;\|\; e_{2b} \qquad \text{(hi nibble \;|\; lo nibble)}$$

For a K-major tensor `(M,K):(K,1)`, elements `(m, k)` and `(m, k+1)` (k even) share byte `(m·K + k)/2`.

### Block scaling

The 4-bit dynamic range (values: `0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6`) is too narrow for most activations. Every **16 consecutive K-elements** therefore share one `fp8(e4m3fnuz)` scale factor:

$$\hat{a}(m,\,k) = a_{\text{fp4}}(m,\,k) \;\times\; \text{sfa}\!\left(m,\,\lfloor k/16 \rfloor\right)$$

This creates a two-level K-axis structure in CUTE hierarchical notation:

| Tensor | CUTE layout | Role |
|--------|-------------|------|
| `a` data | `(M, (K/16, 16)) : (K, (16, 1))` | outer = scale group, inner = 16 packed elements |
| `sfa` scales | `(M, K/16) : (K/16, 1)` | one fp8 per scale group per row |

Both are K-major (stride-1 in K). The K constraint of divisibility by 64 ensures 16-byte TMA alignment and 32-byte SMEM load alignment (`K/2` bytes per row divisible by 32).

The visualization shows M=4, K=16 (one complete scale block). Cells with the same color share a byte (B0–B7 mark the 8 byte-pairs). The `sfa[m]` cell to the right is the single fp8 scale covering all 16 K-elements of that row.

<div id="cute-nvfp4-viz"></div>

### Batched layout: `(M, K, L):(K, 1, M×K)`

The competition tensor `a` adds a batch dimension L. K-major strides generalize directly to three dimensions:

$$\text{offset}(m,\, k,\, l) = l \cdot M \cdot K + m \cdot K + k$$

Batch `l` occupies bytes `[l·M·K/2, (l+1)·M·K/2)` — **batches are completely sequential in memory**, each an independent K-major `(M, K)` nvfp4 matrix. The scale tensor `sfa(M, K/16, L):(K/16, 1, M·K/16)` follows the same sequential batch structure.

The visualization uses M=2, K=8, L=2, BLOCK=8 for clarity (actual format: BLOCK=16, K≥64). Color encodes both batch (color family) and row (shade) — the memory bar makes the batch boundary at byte `l×M×K/2` immediately visible.

<div id="cute-nvfp4-batched-viz"></div>

<script src="{{ '/assets/js/cute-layout-visualizer.js' | relative_url }}"></script>

## CUTE DSL

Let's look at few layout examples from CUTE DSL. We just convert the C++ examples given in docs to python.

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def print_layouts():
    # s8: Int<8>{} -- static 8
    s8 = cute.make_layout(8)

    # d8: dynamic 8
    d8 = cute.make_layout(cutlass.Int32(8))

    # s2xs4: static 2 x static 4
    s2xs4 = cute.make_layout((2, 4))
    # s2xd4: static 2 x dynamic 4
    s2xd4 = cute.make_layout((2, cutlass.Int32(4)))

    # s2xd4_a: custom stride (12, 1)
    s2xd4_a = cute.make_layout((2, cutlass.Int32(4)), stride=(12, 1))
    # s2xd4_col: LayoutLeft (column-major, same as default)
    s2xd4_col = cute.make_layout((2, cutlass.Int32(4)))
    # s2xd4_row: LayoutRight (row-major)
    s2xd4_row = cute.make_ordered_layout((2, cutlass.Int32(4)), order=(1, 0))

    # s2xh4: hierarchical shape (2,(2,2)) with explicit stride
    s2xh4 = cute.make_layout(
        (cutlass.Int32(2), (cutlass.Int32(2), cutlass.Int32(2))),
        stride=(cutlass.Int32(4), (cutlass.Int32(2), cutlass.Int32(1))),
    )
    # s2xh4_col: same shape as s2xh4 but LayoutLeft (column-major)
    s2xh4_col = cute.make_layout(cute.shape(s2xh4))

    cute.printf("[{}] s8        :  {}\n", cute.is_static(s8), s8)
    cute.printf("[{}] d8        :  {}\n", cute.is_static(d8), d8)
    cute.printf("[{}] s2xs4     :  {}\n", cute.is_static(s2xs4), s2xs4)
    cute.printf("[{}] s2xd4     :  {}\n", cute.is_static(s2xd4), s2xd4)
    cute.printf("[{}] s2xd4_a   :  {}\n", cute.is_static(s2xd4_a), s2xd4_a)
    cute.printf("[{}] s2xd4_col :  {}\n", cute.is_static(s2xd4_col), s2xd4_col)
    cute.printf("[{}] s2xd4_row :  {}\n", cute.is_static(s2xd4_row), s2xd4_row)
    cute.printf("[{}] s2xh4     :  {}\n", cute.is_static(s2xh4), s2xh4)
    cute.printf("[{}] s2xh4_col :  {}\n", cute.is_static(s2xh4_col), s2xh4_col)
  
print_layouts()
```

```
[1] s8        :  8:1
[0] d8        :  8:1
[1] s2xs4     :  (2,4):(1,2)
[0] s2xd4     :  (2,4):(1,2)
[0] s2xd4_a   :  (2,4):(12,1)
[0] s2xd4_col :  (2,4):(1,2)
[0] s2xd4_row :  (2,4):(4,1)
[0] s2xh4     :  (2,(2,2)):(4,(2,1))
[0] s2xh4_col :  (2,(2,2)):(1,(2,4))
```

