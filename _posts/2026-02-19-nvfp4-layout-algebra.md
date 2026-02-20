---
title: "Understanding NVFP4 Layout Algebra"
description: >-
  A step-by-step visual walkthrough of the to_blocked transform that converts a raw (M, K/16) FP8 scale-factor tensor into the blocked memory layout required by Blackwell SM100, SM120, and SM121 (DGX Spark) MMA instructions.
date: 2026-02-19
categories: [Blog]
tags: [CUDA, GPU, CUTLASS, CuTe, FP4, Quantization, Blackwell]
pin: true
math: false
author: ks
---

Javascript visualizations below were created with the help of Claude Code.

---

When you run FP4 matrix-multiply on a Blackwell GPU, you need to hand the hardware not just your FP4 weight
matrix but also a carefully rearranged tensor of FP8 _scale factors_.  The rearrangement is a five-step
transform called `to_blocked`.  This post walks through every step in detail, with interactive visualizations
of the data movement for a concrete `M=256, K=128` example.

## What is the scale-factor tensor, and why does it need a special layout?

Every 16 consecutive FP4 elements in the K dimension share one FP8 scale factor.
For a matrix A of shape `(M=256, K=128)` that gives `128/16 = 8` scale factors per row, so the raw
scale-factor tensor is `sf = (256, 8)`.

The Blackwell MMA instruction (`mma.sync.aligned.block_scale` on SM120/SM121, `tcgen05.mma.blockscaled`
on SM100) processes A in **128-row × 4-k-slot atoms** — one warp-group height × one 32-bit register lane.
The job of `to_blocked` is to take the row-major `(256, 8)` tensor and reorder it into the exact memory
layout the hardware's scale-factor load path expects.

### Color legend

Each cell in the visualizations below represents one FP8 scale factor.  Color encodes which 32-row
**warp-band** that element belongs to inside its 128-row block.

<div id="nvfp4-legend"></div>

---

## Setup — Input: `(256, 8)` scale-factor matrix

For `M=256, K=128` with `sf_vec_size=16`:

- `n_row_blocks = ceil(256 / 128) = 2`
- `n_col_blocks = ceil(8 / 4) = 2`

There are **four outer tile regions** in the input (a 2×2 grid), each covering 128 rows × 4 k-columns.
The grid below shows a sample of rows from each warp-band.  Notice that elements from the same tile are
_not_ contiguous in memory — they are interleaved across the full 256-row extent of the tensor.  That is
what `to_blocked` fixes.

<div id="nvfp4-setup-viz"></div>

---

## Step 1 — `view(2, 128, 2, 4)`: expose all four block boundaries

```python
blocks = padded.view(2, 128, 2, 4)
# dims: [row_block, row_within, col_block, col_within]
#         {0,1}      0..127      {0,1}       0..3
```

This is a **zero-copy reshape**.  Element `(r, c)` maps to
`[r//128, r%128, c//4, c%4]` = `[row_block, row_within, col_block, col_within]`.
The four logical tile quadrants are now accessible via explicit indices.

> **Why bother?**  The tile boundaries (every 128 rows, every 4 k-columns) are implied by arithmetic
> in the flat `(256, 8)` tensor.  This step makes those boundaries _explicit dimensions_ so that the
> permute in Step 2 can reorder them cleanly.

<div id="nvfp4-step1-viz"></div>

---

## Step 2 — `permute(0, 2, 1, 3)`: bring `col_block` next to `row_block`

```python
# BEFORE: (row_block=2, row_within=128, col_block=2, col_within=4)
blocks = blocks.permute(0, 2, 1, 3)
# AFTER:  (row_block=2, col_block=2, row_within=128, col_within=4)
```

After Step 1 the two "which tile am I in?" dimensions — `row_block` and `col_block` — are separated by
the 128-element `row_within` dimension.  When Step 3 calls `reshape(-1, ...)`, PyTorch reads dimensions
left-to-right (C-order / row-major).  If `row_block` and `col_block` are not adjacent, the collapsed
tile ordering comes out wrong.

The comparison below shows the difference:

<div id="nvfp4-step2-viz"></div>

> **No-op when `n_col_blocks=1`.**  With only one k-block there is no ordering to get wrong — swapping
> a size-1 dimension is algebraically meaningless.  This is why the single-block case hid the purpose
> of this step.

---

## Step 3 — `reshape(-1, 4, 32, 4)`: merge outer-block dims; split 128 rows into warp-bands

```python
rearranged = blocks.reshape(-1, 4, 32, 4)
# shape: (4,         4,          32,            4)
#         outer_tile  warp_band   thread_in_warp  k_col
```

Two things happen simultaneously:

1. The `-1` collapses `(row_block=2, col_block=2)` into **4 outer tiles** in the order guaranteed by
   the permute in Step 2: `T0=(rb0,cb0)` → `T1=(rb0,cb1)` → `T2=(rb1,cb0)` → `T3=(rb1,cb1)`.
2. The 128-row dimension inside each tile is split into `4 × 32` — the **warp decomposition**.
   A standard 4-warp threadblock has 128 threads (one per row), divided into four warp-bands of 32.

<div id="nvfp4-step3-viz"></div>

---

## Step 4 — `transpose(1, 2)`: thread becomes row, `warp_band` becomes inner column

```python
rearranged = rearranged.transpose(1, 2)
# shape: (4,         32,             4,           4)
#         outer_tile  thread_in_warp  warp_band    k_col
```

After Step 3 the shape is `(tile, warp_band, thread, k_col)`.  The hardware wants
**each of the 32 threads in a warp to load one complete row** of the final `(32×16)` output tile.
For that, the thread index must be the leading tile dimension (i.e. the row index).

This transpose simply moves `thread` before `warp_band`.  It is a no-data-move stride reinterpretation.
After it, `warp_band` and `k_col` sit next to each other as the last two dimensions — ready to merge
into a 16-wide column in Step 5.

---

## Step 5 — `reshape(-1, 32, 16)`: merge `warp_band × k_col` into 16-wide columns

```python
rearranged = rearranged.reshape(-1, 32, 16)
# shape: (4,    32,             16)
#               thread_in_warp  col = warp_band*4 + k_col
```

`(warp_band=4, k_col=4)` → `16`.  The output is **4 tiles, each `(32 rows × 16 columns)`**.

**Reading the final tile:**

| Columns | Content |
|---------|---------|
| 0–3     | Scale factors for row `t + 0×32` (warp-band 0), k-positions 0,1,2,3 |
| 4–7     | Scale factors for row `t + 1×32` (warp-band 1), k-positions 0,1,2,3 |
| 8–11    | Scale factors for row `t + 2×32` (warp-band 2), k-positions 0,1,2,3 |
| 12–15   | Scale factors for row `t + 3×32` (warp-band 3), k-positions 0,1,2,3 |

Thread `t` reads row `t`.  All 32 threads issue one load each, mapping to 32 consecutive FP8 bytes =
one 128-byte cache line per warp-band slice.  The 4 columns per warp-band slot = 32 bits = one
32-bit register lane — exactly what the `mma.sync.aligned.block_scale` instruction expects.

After `flatten()` all 4 tiles are laid end-to-end in memory for `torch._scaled_mm`.

**Complete element mapping:**

| Output index | Formula |
|---|---|
| `tile_idx` | `(r // 128) × 2  +  (c // 4)` |
| `tile_row` | `r % 32` (thread slot 0–31) |
| `tile_col` | `((r % 128) // 32) × 4  +  (c % 4)` |

Tile memory offsets: T0 @ 0, T1 @ 512, T2 @ 1024, T3 @ 1536.

<div id="nvfp4-final-tiles"></div>

---

## Why the permute is a no-op at `n_col_blocks=1` but critical here

`reshape(-1, ...)` reads dimensions in **left-to-right (C-order) order**.

- **Without the permute**, shape after `view` is `(row_block=2, row_within=128, col_block=2, col_within=4)`.
  Collapsing the first two dims makes `row_within` the inner loop, so the tile sequence comes out
  rb-interleaved with cb: `(rb0,cb0)` → `(rb1,cb0)` → `(rb0,cb1)` → `(rb1,cb1)`.
  The MMA tile iterator sees mismatched row-blocks back to back.

- **With `permute(0,2,1,3)`**, shape becomes `(row_block=2, col_block=2, row_within=128, col_within=4)`.
  Both block-index dimensions sit at the front, so the collapse produces the correct order:
  all col-blocks for rb0 first, then all col-blocks for rb1.

When `n_col_blocks=1` there is only one k-block; swapping a size-1 dimension changes nothing and the
permute is algebraically a no-op.  That is why simpler examples (with a single k-block) never surface
the bug.

---

## CuTe layout — grounded in CUTLASS examples 79b and 72b

### The two fundamental constants

The CUTLASS class `Sm1xxBlockScaledConfig` (from `sm100_blockscaled_layout.hpp`, shared across SM100,
SM120, and SM121) defines:

```cpp
using Blk_MN    = Int<128>;  // rows per tile = MMA atom M-height
using Blk_SF    = Int<4>;    // k-slots per tile = 4 FP8 = 32 bits = one register lane
using Blk_Elems = Int<512>;  // total FP8 elements per tile = 32 rows × 16 cols
```

The 32-bit constraint comes from the hardware: the MMA unit reads scale factors as 32-bit register
lanes, so 4 consecutive FP8 values (one warp-band slot) fit exactly in one lane.

### The GmemLayoutSFA that CUTLASS 79b passes to the kernel

```cpp
// After all 5 steps, to_blocked() produces a buffer whose CuTe layout is:
using GmemLayoutSFA = Layout<
  Shape <Shape<_32, _4>,    Shape<_4,   _2>,    _2 >,
  Stride<Stride<_1,  _32>,  Stride<_128, _512>, _512>
>;

// dim 0 — M atom: Shape<_32,_4>, Stride<_1,_32>
//   inner _32: thread index within warp  (stride=1 → coalesced load ✓)
//   outer  _4: warp-band index (0–3)     (stride=32 → 32-byte offset per band)
//
// dim 1 — K atom: Shape<_4,_2>, Stride<_128,_512>
//   inner _4: k-position within block    (stride=128 = one 32-row column)
//   outer _2: k-block index              (stride=512 = one full tile)
//
// dim 2 — M tiles: _2, stride _512
//   2 row-blocks; each starts one full tile (512 bytes) later
```

This layout is returned by `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1))` —
the same call appears identically in both example 72b (SM100 / B200) and example 79b (SM120 / RTX 50-series,
SM121 / DGX Spark GB10).

### SM120/SM121 (DGX Spark) vs SM100 (B200) — what changes, what doesn't

| | SM100 (B200) | SM120/SM121 (RTX 50 / DGX Spark) |
|---|---|---|
| MMA instruction | `tcgen05.mma.blockscaled` | `mma.sync.aligned.block_scale` |
| Scale factor path | UTCCP: gmem → TMEM → MMA | TMA: gmem → SMEM → MMA |
| TMEM | ✓ available | ✗ not present |
| Cluster multicast | ✓ optional | ✗ always `<1,1,1>` |
| `GmemLayoutSFA` | `Shape<(32,4),(4,2),2>` | **identical** |
| `to_blocked()` output valid? | ✓ | ✓ |

The PyTorch `to_blocked()` output you generate on CPU is correct input for example 79b on DGX Spark
(SM121) without any changes.  The `Sm1xxBlkScaledConfig` class name's `1xx` suffix explicitly covers
the whole family.

---

## References

- [CUTLASS example 79b — SM120 GeForce FP4 GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu)
- [CUTLASS example 72b — SM100 B200 FP4 GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/)
- [sm100_blockscaled_layout.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/detail/sm100_blockscaled_layout.hpp)
- [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
- [NVIDIA Blackwell Architecture Technical Brief](https://www.nvidia.com/en-us/data-center/tensor-cores/)

<script src="/assets/js/nvfp4-layout-viz.js"></script>
