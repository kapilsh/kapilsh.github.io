---
title: Deep Dive into Triton Internals (Part 3)
description: >-
  Enough MLIR to be dangerous - how Triton uses MLIR passes to progressively lower IR
date: 2024-09-07
categories: [ Blog, Tutorial ]
tags: [ AI, Triton, CUDA, Machine Learning, Compiler ]
pin: true
author: ks
---

This post is a continuation of the series on Triton internals. In this post, we will dive into the MLIR
internals of Triton. We go deeper into the backend compilation process and look at the MLIR passes that Triton uses to progressively lower the IR to the target hardware.

Previous posts covered the following topics:

- **Part 1:** [Triton Compilation Process](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)
- **Part 2:** [Triton Compiler Frontend](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/)

## Commit Hash

Triton is an active project. To ensure that the information in this post is relevant, I have used the following commit
hash:

```shell
$ git rev-parse main
9baa051fa9dd00cd7255e750c71224153aecd3f0
```

## Review: Triton Python Bindings

This post expects familiarity with key concepts covered in previous posts. For example, we expect familiarity (but not necessarily deep understanding) with Triton Compiler, MLIR (and table-gen), and the Triton Python bindings.

When I started working on this series of posts, I found one of the TensorFlow videos pretty useful in understanding MLIR applications to ML compilers and accelerators. Although it is not specific to Triton, it provides a good overview of the concepts. 

{% include embed/youtube.html id='R5LLIj8EMxw' %}

## Triton Backend Compilation

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

We can find these implementations in the `triton/third_party/<vendor/provider>/backend/compiler.py` files. I only have
NVidia GPU so let's look at NVidia backend.

Code Pointers:

- [`make_ttir`](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/third_party/nvidia/backend/compiler.py#L178-L190)
- [`make_ttgir`](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/third_party/nvidia/backend/compiler.py#L193-L236)
- others are in the same file

## Triton IR (TTIR)

If we look at first of these functions - `make_ttir`. it effectively generates a tensor program as the first step in the compilation process. This is done by running a series of passes on the input module. Below is the implementation:

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

We notice that it ends up using `ir` and `passes` modules from libtriton.

### from triton._C.libtriton import ir, passes, llvm, nvidia

These modules are defined using pybind11 in [`triton/python/src/main.cc`](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/python/src/main.cc#L46-L55) and provide python bindings for compiler passes defined in the C++ layer.

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

### PassManager

At a high level, `PassManager` is a container for a sequence of passes. It provides a way to run a sequence of passes on a
module and progressively lower IR down the target hardware. 

#### Hello world example

Here is s simple example of how to use the PassManager:

```cpp
mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get());
mlir::PassManager pm(module);

// Add some passes
pm.addPass(mlir::createCSEPass());
pm.addPass(mlir::createDeadCodeEliminationPass());

pm.run(module);
```
This "Hello world" example shows how to run two passes on a module. The first pass is Common Subexpression Elimination (CSE), and the second is Dead Code Elimination (DCE).

Let's say we had a toy language with the following function:

```
def foo(b, c):
  a = b + c
  d = b + c
  e = 2 * a
  return d
```

After running the CSE pass, the code would be optimized to:

```
def foo(b, c):
  a = b + c
  d = a
  e = 2 * a
  return d
```

The CSE pass identified that `b + c` was computed twice and replaced the second computation with the value of `a`.

In the next step, the DCE pass would remove the `e` variable since it is not used in the function. The final optimized code would be:

```
def foo(b, c):
  a = b + c
  d = a
  return d
```

### Passes Module

The `passes` module provides a set of passes that can be added to the PassManager. There are submodules for different
types of passes such as common, ttir, ttgir.

For
example, [`passes/common`](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/python/src/passes.cc#L26-L34)
provides CSE, LICM, etc. Let's look some of these:

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
in [triton/lib/Dialect/Triton/Transforms](https://github.com/triton-lang/triton/tree/9baa051fa9dd00cd7255e750c71224153aecd3f0/lib/Dialect/Triton/Transforms).

Triton runs passes such as combining ops together, reorder operations to move broadcasting, splats, and tensor pointer
load/stores.

- These passes were covered in
the [previous post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/#inliner-pass).
- Reference table-gen
file: [TTIR Passes](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/include/triton/Dialect/Triton/Transforms/Passes.td#L1)

## Triton GPU IR (TTGIR)

Reference table-gen files

- [TTGIR Passes](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/include/triton/Dialect/TritonGPU/Transforms/Passes.td)
- [TTNVGPUIR Passes (NVidia specific)](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td)

The next stage in the compilation process is to generate the Triton GPU IR. This is done by the `make_ttgir` function. Below is
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
covered the `common` passes. As part of `nvidia` and `ttgpuir` passes, triton compiler applies several gpu specific optimization passes to the IR previously generated. Some of the well-known optimization passes include:

- Coalescing
- F32 dot product optimization
- CTA planning
- Thread locality
- Matrix multiplication acceleration
- Optimization of dot operands
- Data de-duplication
- Instructions reordering
- TMA lowering, etc.

I feel that this is good level to cover MLIR passes for GPU lowering. Beyond this, we will get too much into
implementation details. Reader is encouraged to look at the triton codebase and read code for specific passes to learn more.

> Example: [Accelerate Matmul](https://github.com/triton-lang/triton/blob/9baa051fa9dd00cd7255e750c71224153aecd3f0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L386). Please trace through the directory for more transforms.
{: .prompt-tip}

### LLIR and PTX Generation

We covered LLIR and PTX Generation in [part 1](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/), so I
won't cover it again here. However, the process remains similar, where passes are applied to the GPU IR to lower it to LLVM IR followed by PTX.

## IR Walkthrough

Let's do a walkthrough of IR generated after one of the many passes. Triton GPU IR will be an interesting pick since IR should have more specificities of GPU ops.

First, we will compile one of the tutorial example from triton repo. (`vector-add`)

```shell
$ python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  python/tutorials/01-vector-add.py
```

> This will put the compiled kernel and IR artifacts in `~/.triton/cache/` directory. 
{: .prompt-tip}

After compilation, I see the following files in the cache directory:

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

Let's look into `ht7F5vagGWCbXzZnYSv8P9EWINzpLWcHYcdPh9m8Dvg`. 
I uploaded the content of `~/.triton/cache/ht7F5vagGWCbXzZnYSv8P9EWINzpLWcHYcdPh9m8Dvg/add_kernel.ttgir` to a [gist](https://gist.github.com/kapilsh/cee55b221d7f91dc64fbac46f018163c). Let's go through it step by step:

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

### 2. Kernel function

```
tt.func public @add_kernel(
  %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
  %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
  %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
  %arg3: i32 {tt.divisibility = 16 : i32}
) attributes {noinline = false} {
  ...
}
```

This should look familiar since it maps directly to the python code:

```python
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
```

- `%arg0`, `%arg1`, and `%arg2`: Pointers to float32 arrays with divisibility 16 (meaning the arrays are aligned to 16-byte boundaries).
- `%arg3`: An integer with divisibility 16
- `BLOCK_SIZE` is missing. We will see later that it is defined as a constant in the IR.

### 3. Program ID and Range Creation

The code initializes kernel execution by creating a constant (`BLOCK_SIZE`), retrieving the program ID, and generating a range of indices from 0 to 1024.

```
%c1024_i32 = arith.constant 1024 : i32
%0 = tt.get_program_id x : i32
%1 = arith.muli %0, %c1024_i32 : i32
%2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
```

### 4. Splat and add operations

Creates a tensor of `i32` values by splatting (broadcasting) the value `%1` to a tensor of shape `(1024,)` and then adds the tensor `%3` to the range `%2` element-wise using `arith.addi`.

> `arith` is another MLIR dialect that provides a set of arithmetic operations. [Arith Dialet](https://mlir.llvm.org/docs/Dialects/ArithOps/)
{: .prompt-info}

```
%3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
%4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
```

### 5. Load and store operations

The code generates a tensor of pointers to `f32` values, offsets them using `tt.addptr`, and loads the resulting values from memory using `tt.load`.

Finally, stores the output of the `arith.addf` operation to the memory location pointed to by `%15` based on condition `%6`. `%15` is the `output_ptr`.

```
%7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
%8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
%9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
...
tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>, #blocked> loc(#loc13)
tt.return
```

We can map each of the instructions back to the python (triton DSL) code as well. See the splits below after parsing the IR based location info. 

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

## Triton MLIR Dumps

Triton has a bunch of environment variable that provide ways to debug and get backend logging/info. Among these, `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all kernels. 
- If you are interested in diving deeper into the IR generated by Triton, you can use this MLIR dump feature
- This could also be useful for debugging tricky bugs. 

```shell
# Triton cache interferes with the dump when a dump already exists
# so, clear the cache. I could not find a way to disable the cache for now
$ MLIR_ENABLE_TIMING=1 MLIR_ENABLE_DUMP=1 python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  python/tutorials/01-vector-add.py
```

### Sample output

[Example MLIR Dump](https://gist.github.com/kapilsh/924e81e0e9fe4c1b9b58f1825419baea)

I am hacking on a tool to parse these dumps and provide a more readable output. I will update this post once it is ready. However, for now here is a sample output before/after a pass:

![Python to MLIR Mapping](/assets/python-to-mlir.png)

## Conclusion

In this post, we looked at the MLIR passes that Triton uses to progressively lower the IR to the target hardware. We walked through the Triton GPU IR generated after one of the many passes and saw how it maps back to the original python code.

