---
title: Deep Dive into Triton Internals (Part 3)
description: >-
  Enough MLIR and LLVM to be dangerous 
date: 2024-09-04
categories: [ Blog, Tutorial ]
tags: [ AI, Triton, CUDA, Machine Learning, Compiler ]
pin: true
math: true
author: ks
---

This post is a continuation of the series on Triton internals. In this post, we will dive into the MLIR and LLVM
internals of Triton.

Previous posts covered the following topics:

- [Triton Compilation Process](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)
- [Triton Compiler Frontend](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/)

## Commit Hash

Triton is an active project. To ensure that the information in this post is relevant, I have used the following commit
hash:

```shell
$ git rev-parse main
7480ef5028b724cb434b7841b016c6d6debf3b84
```

## Background: Brief Overview of MLIR

- What is MLIR?
- What is table-gen?
- How does MLIR relate to LLVM?
- How does Triton use MLIR?
- I wnat to know more about MLIR. Where do I start?

## Review: Triton Python Bindings

In the previous post, we started looking into the NVidia backend. We encountered `add_stages` function that runs through
different stages of the compilation and produces progressive IR in the process.

```python
def add_stages(self, stages, options):
    stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
    stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
    stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.capability)
    stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)
```

We can find these implementations in the `triton/third_party/<vendor/provider>/backend/compiler.py` file. I only have
NVidia GPU so let's look at NVidia backend.

Code Pointers:

- [`make_ttir`](https://github.com/triton-lang/triton/blob/7480ef5028b724cb434b7841b016c6d6debf3b84/third_party/nvidia/backend/compiler.py#L170-L182)
- [`make_ttgir`](https://github.com/triton-lang/triton/blob/7480ef5028b724cb434b7841b016c6d6debf3b84/third_party/nvidia/backend/compiler.py#L185-L228)
- others are in the same file

## Triton IR (TTIR)

If we look at first of these functions - `make_ttir`. As the first step in the compilation process, `make_ttir`
effectively generates a tensor program.

```python
def make_ttir(mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(mod)
    return mod
```

We notice that it ends up using ir and passes modules from libtriton.

### from triton._C.libtriton import ir, passes, llvm, nvidia

These modules are defined in
pybind11. [`triton/python/src/main.cc`](https://github.com/triton-lang/triton/blob/7480ef5028b724cb434b7841b016c6d6debf3b84/python/src/main.cc#L46-L55).

```cpp
PYBIND11_MODULE(libtriton, m) {
  m.doc() = "Python bindings to the C++ Triton API";
  init_triton_stacktrace_hook(m);
  init_triton_env_vars(m);
  ### IR and PASSESS MODULES ###
  init_triton_ir(m.def_submodule("ir"));
  init_triton_passes(m.def_submodule("passes"));
  ### --------------------- ###
  init_triton_interpreter(m.def_submodule("interpreter"));
  init_triton_llvm(m.def_submodule("llvm"));
  FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)
}
```

So, from here we will start digging into `ir` and `passes` modules.

### IR Module

We covered `ir` module briefly in the previous post and discovered that it provides
the [MLIR PassManager](https://mlir.llvm.org/doxygen/classmlir_1_1PassManager.html#details) reference.

At a high level, PassManager is a container for a sequence of passes. It provides a way to run a sequence of passes on a
module. Below is some sample code:

```cpp
mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get());
mlir::PassManager pm(module);

// Add some passes
pm.addPass(mlir::createCSEPass());
pm.addPass(mlir::createDeadCodeEliminationPass());

pm.run(module);
```

### Passes Module

The `passes` module provides a set of passes that can be added to the PassManager. There are submodules for different
types of passes such as common, ttir, ttgir.

For
example, [`passes/common`](https://github.com/triton-lang/triton/blob/7480ef5028b724cb434b7841b016c6d6debf3b84/python/src/passes.cc#L26-L34)
provides common passes like CSE, LICM, etc. Let's look some of these:

### Passes.Common Passes

All the common passes are generic MLIR passes that are available in the MLIR library. Triton wraps these passes in its
own API.

Triton does constant propagation, dead code elimination, inlining, canonicalization, common subexpression elimination,
and loop invariant code motion (hoisting) as part of the common passes.

```cpp
void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
}
```

Reference to documentation for these passes: [MLIR Passes](https://mlir.llvm.org/doxygen/namespacemlir.html)

#### Passes.TTIR Passes

Triton IR passes are defined in the `passes/ttir` module. These passes are specific to Triton and defined
in [triton/lib/Dialect/Triton/Transforms](https://github.com/triton-lang/triton/tree/7480ef5028b724cb434b7841b016c6d6debf3b84/lib/Dialect/Triton/Transforms).

Triton runs passes such as combining ops together, reorder operations to move broadcasting, splats, and tensor pointer
load/stores.

These passes were covered in
the [previous post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/#inliner-pass).

Reference table-gen
file: [TTIR Passes](https://github.com/triton-lang/triton/blob/7480ef5028b724cb434b7841b016c6d6debf3b84/include/triton/Dialect/Triton/Transforms/Passes.td#L1)

## Triton GPU IR (TTGIR)

The next stage in the compilation process is to generate the GPU IR. This is done by the `make_ttgir` function. Below is
the reference NVIDIA version:

```python
def make_ttgir(mod, metadata, opt, capability):
    ...truncated...
    # TTIR -> TTGIR
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    ...truncated...
    nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    passes.ttgpuir.add_accelerate_matmul(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    passes.common.add_cse(pm)
    if capability // 10 >= 8:
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages)
    ...truncated...
    pm.run(mod)
    return mod
```

We can see that Triton GPU IR generation relies on some `common`, `nvidia`, and `ttgpuir` passes. We have already
covered the common passes.

At this stage, Triton compiler applies several gpu specific optimization passes to the IR previously generated.

Some of the well-known optimization passes to the pass manager, including:

- Coalescing
- F32 dot product optimization
- CTA planning
- Thread locality
- Matrix multiplication acceleration
- Optimization of dot operands
- Data de-duplication -
- Instructions reordering
- TMA lowering
  and several others.

I feel that this is good level to cover MLIR passes for GPU lowering. Beyond this, we will get too much into
implementatin details.

## LLIR and PTX Generation

We covered LLIR and PTX Generation in [part 1](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/), so I
won't cover it again here. However, the process remains similar here where passes are applied to the IR from previous
pass and IR is lowered to LLVM IR followed by PTX.

## LLM can explain MLIR code

Let's dump Triton GPU IR and see if we can understand it. We will of course leverage an LLM to help us. 🔥 🚀🚀🚀 🔥

```shell
$ python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  python/tutorials/01-vector-add.py
```

This will put the compiled kernel and IR artifacts in `~/.triton/cache/` directory. For example, on my machine:

```shell
-> % ll -R ~/.triton/cache
/home/ksharma/.triton/cache:
total 16K
drwxrwxr-x 2 ksharma ksharma 4.0K Sep  5 22:03 ht7F5vagGWCbXzZnYSv8P9EWINzpLWcHYcdPh9m8Dvg
drwxrwxr-x 2 ksharma ksharma 4.0K Sep  5 22:04 KhrPjVaTuMNfu7c00XjCY5ZXpWaGyn97op4fqj6nD_Q
drwxrwxr-x 2 ksharma ksharma 4.0K Sep  5 22:03 P4-eJXPkRvvD0Z7CcR_QHVX4oqH1l6K0oPt8Posthe0
drwxrwxr-x 2 ksharma ksharma 4.0K Sep  5 22:03 RK5K7n7w7g3VToYM9EYn47bO2r6HoiisdZiDAbimv2A

/home/ksharma/.triton/cache/ht7F5vagGWCbXzZnYSv8P9EWINzpLWcHYcdPh9m8Dvg:
total 36K
-rw-rw-r-- 1 ksharma ksharma 6.7K Sep  5 22:03 add_kernel.cubin
-rw-rw-r-- 1 ksharma ksharma  685 Sep  5 22:03 add_kernel.json
-rw-rw-r-- 1 ksharma ksharma 6.6K Sep  5 22:03 add_kernel.llir
-rw-rw-r-- 1 ksharma ksharma 3.9K Sep  5 22:03 add_kernel.ptx
-rw-rw-r-- 1 ksharma ksharma 3.4K Sep  5 22:03 add_kernel.ttgir
-rw-rw-r-- 1 ksharma ksharma 3.0K Sep  5 22:03 add_kernel.ttir
-rw-rw-r-- 1 ksharma ksharma  679 Sep  5 22:03 __grp__add_kernel.json

/home/ksharma/.triton/cache/KhrPjVaTuMNfu7c00XjCY5ZXpWaGyn97op4fqj6nD_Q:
total 36K
-rw-rw-r-- 1 ksharma ksharma 5.5K Sep  5 22:04 add_kernel.cubin
-rw-rw-r-- 1 ksharma ksharma  686 Sep  5 22:04 add_kernel.json
-rw-rw-r-- 1 ksharma ksharma 4.1K Sep  5 22:04 add_kernel.llir
-rw-rw-r-- 1 ksharma ksharma 2.9K Sep  5 22:04 add_kernel.ptx
-rw-rw-r-- 1 ksharma ksharma 3.3K Sep  5 22:04 add_kernel.ttgir
-rw-rw-r-- 1 ksharma ksharma 2.9K Sep  5 22:04 add_kernel.ttir
-rw-rw-r-- 1 ksharma ksharma  679 Sep  5 22:04 __grp__add_kernel.json

/home/ksharma/.triton/cache/P4-eJXPkRvvD0Z7CcR_QHVX4oqH1l6K0oPt8Posthe0:
total 28K
-rw-rw-r-- 1 ksharma ksharma 26K Sep  5 22:03 cuda_utils.so

/home/ksharma/.triton/cache/RK5K7n7w7g3VToYM9EYn47bO2r6HoiisdZiDAbimv2A:
total 20K
-rw-rw-r-- 1 ksharma ksharma 17K Sep  5 22:03 __triton_launcher.so
```

I used [`~/.triton/cache/ht7F5vagGWCbXzZnYSv8P9EWINzpLWcHYcdPh9m8Dvg/add_kernel.ttgir`](https://gist.github.com/kapilsh/cee55b221d7f91dc64fbac46f018163c)
to [prompt Meta.AI to explain the MLIR code](https://www.meta.ai/s/XtXQfTfkswTkiVVn/). It does pretty good job! 🚀🚀🚀 

Here's a summary:

### 1. Module attributes

The first section defines the module attributes, which provide gpu metadata about the generated code:

```
module attributes {
  "triton_gpu.num-ctas" = 1 : i32,
  "triton_gpu.num-warps" = 4 : i32,
  triton_gpu.target = "cuda:89",
  "triton_gpu.threads-per-warp" = 32 : i32
}
```

### 2. Program ID and Range Creation

The code initializes kernel execution by creating a constant, retrieving the program ID, and generating a range of indices from 0 to 1024.

```
%c1024_i32 = arith.constant 1024 : i32
%0 = tt.get_program_id x : i32
%1 = arith.muli %0, %c1024_i32 : i32
%2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
```

### 3. Splat and add operations

Creates a tensor of i32 values by splatting the value %1 to a tensor of shape (1024,) and then adds the tensor %3 to the range %2 element-wise using arith.addi.

> arith is another MLIR dialect that provides a set of arithmetic operations.
{: .prompt-tip}

```
%3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
%4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
```

### 4. Load and store operations

The code generates a tensor of pointers to f32 values, offsets them using tt.addptr, and loads the resulting values from memory using tt.load.

```
%7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
...
tt.return
```

Below is the full split of the MLIR code by sections. We can map each of the instructions back to the python (triton DSL) code. I am currently working on a tool for that. 🔥🔥🔥

```
Section 1: unknown
  %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
Section 2: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:37:24
  %0 = tt.get_program_id x : i32 loc(#loc2)
Section 3: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:42:24
  %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc3)
Section 4: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:43:41
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked> loc(#loc4)
Section 5: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:43:28
  %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked> loc(#loc5)
  %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked> loc(#loc5)
Section 6: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:45:21
  %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked> loc(#loc6)
  %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked> loc(#loc6)
Section 7: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:48:24
  %7 = tt.splat %arg0 : !tt.ptr -> tensor<1024x!tt.ptr, #blocked> loc(#loc7)
  %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr, #blocked>, tensor<1024xi32, #blocked> loc(#loc7)
Section 8: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:48:16
  %9 = tt.load %8, %6 : tensor<1024x!tt.ptr, #blocked> loc(#loc8)
Section 9: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:49:24
  %10 = tt.splat %arg1 : !tt.ptr -> tensor<1024x!tt.ptr, #blocked> loc(#loc9)
  %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr, #blocked>, tensor<1024xi32, #blocked> loc(#loc9)
Section 10: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:49:16
  %12 = tt.load %11, %6 : tensor<1024x!tt.ptr, #blocked> loc(#loc10)
Section 11: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:50:17
  %13 = arith.addf %9, %12 : tensor<1024xf32, #blocked> loc(#loc11)
Section 12: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:52:26
  %14 = tt.splat %arg2 : !tt.ptr -> tensor<1024x!tt.ptr, #blocked> loc(#loc12)
  %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr, #blocked>, tensor<1024xi32, #blocked> loc(#loc12)
Section 13: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:52:35
  tt.store %15, %13, %6 : tensor<1024x!tt.ptr, #blocked> loc(#loc13)
Section 14: /home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py:52:4
  tt.return loc(#loc14)
```

```shell

## Triton MLIR Dumps

We can dump the MLIR IR using the `MLIR_ENABLE_DUMP` environment variable. This will dump the IR at different stages of the compilation process.

```shell
# Triton cache interferes with the dump when a dump already exists
# so, clear the cache. I could not find a way to disable the cache for now
$ rm -rf ~/.triton/cache/*
$ conda activate learning-triton
$ MLIR_ENABLE_TIMING=1 MLIR_ENABLE_DUMP=1 python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  python/tutorials/01-vector-add.py &> /tmp/my_triton_info_all.txt
```

[Example MLIR Dump](https://gist.github.com/kapilsh/924e81e0e9fe4c1b9b58f1825419baea)


