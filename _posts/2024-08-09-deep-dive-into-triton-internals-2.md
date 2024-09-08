---
title: Deep Dive into Triton Internals (Part 2)
description: >-
  What happens when triton.compile is called in the frontend? 
date: 2024-08-09
categories: [Blog, Tutorial]
tags: [AI, Triton, CUDA, Machine Learning, Compiler]
pin: true
math: true
author: ks
---

In the [previous post](../deep-dive-into-triton-internals/), we explored the Triton compiler internals and code generation pipeline under the hood. In this post, we will explore further to understand what happens when `triton.compile` is called in the frontend. We will initially use python debugger to peak into the python compiler layer and then look at the native layer in C/C++.

This is part 2 of the deep dive. Other posts in the series:
- [Deep Dive into Triton Internals (Part 1)](../deep-dive-into-triton-internals).
- [Deep Dive into Triton Internals (Part 3)](../deep-dive-into-triton-internals-3).

## Git Hash

> Triton's internals might change in the future. The git hash for the version that I am working on is:
> 
> ```shell
> git rev-parse origin/master # at the time of writing
> # 14025786d108596cfd99700caa4f438938c2ceba
> ```
{: .prompt-tip}

I forked the official triton repo at the above hash to [my forked repo](https://github.com/kapilsh/triton). You should see the hash in the links for all the code pointers. 

Follow the instructions in [triton repo](https://github.com/triton-lang/triton) to fork and install triton from source.

## Getting Started

We covered in the previous post that you can generate the code for the kernel by using triton compiler directly. 

Here's the command again to compile one of the tutorial examples:

```shell
$ python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  python/tutorials/01-vector-add.py
```
This will generate 2 files: `add_kernel.9969bdda_0123.c`, `add_kernel.9969bdda_0123.h`! Let's look into how the files are generated:

## AST Source Code Generation

- [AST source code generation](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/triton/tools/compile.py#L110) is the first step in the Triton compiler pipeline.
- Command-line arguments allow specifying:
  - The path to the Python file containing the kernel.
  - The name of the kernel to compile.
  - The number of warps and stages for the kernel.
  - The output name and path for the compiled kernel.
  - The signature of the kernel, including types and constexpr values.
  - The launch grid for the kernel.
- Most of the command line arguments feed into the `ASTSource` constructor:

```python
src = triton.compiler.ASTSource(fn=kernel, constants=constants, signature=signature, attrs=attrs)
```

We can inspect what is inside ASTSource by setting a breakpoint:

```diff
$ git --no-pager diff
diff --git a/python/triton/tools/compile.py b/python/triton/tools/compile.py
index 872332b0..b2fe5abf 100644
--- a/python/triton/tools/compile.py
+++ b/python/triton/tools/compile.py
@@ -108,6 +108,7 @@ if __name__ == "__main__":
     for i in equal_to_1:
         constants.update({i: 1})
     src = triton.compiler.ASTSource(fn=kernel, constants=constants, signature=signature, attrs=attrs)
+    import ipdb; ipdb.set_trace()
     opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
     ccinfo = triton.compile(src, options=opts)
     arg_names = []
```

```shell
-> % python3 python/triton/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1024,1024 \
  python/tutorials/01-vector-add.py
...
> /home/ksharma/dev/git/triton/python/triton/tools/compile.py(112)<module>()
    111     import ipdb; ipdb.set_trace()
--> 112     opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    113     ccinfo = triton.compile(src, options=opts)

ipdb> src
<triton.compiler.compiler.ASTSource object at 0x7ecefad51af0>
```

For reference, [ASTSource](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/triton/compiler/compiler.py#L88-L116) is defined in `triton/compiler/compiler.py`: 

```shell
ipdb> dir(src)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'attrs', 'constants', 'ext', 'fn', 'hash', 'make_ir', 'name', 'parse_options', 'signature']
ipdb> src.signature
{0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}
ipdb> src .attrs
AttrsDescriptor(divisible_by_16=[], equal_to_1=[])
ipdb> src.fn
JITFunction(01-vector-add:add_kernel)
ipdb> src.fn.arg_names
['x_ptr', 'y_ptr', 'output_ptr', 'n_elements', 'BLOCK_SIZE']
ipdb> src.fn.params[0]
<triton.runtime.jit.KernelParam object at 0x7ecff9464e90>
ipdb> src.fn.params[0].name
'x_ptr'
ipdb> print(src.fn.src)
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    ...truncated...
    output = x + y                                                                                                                      
    # Write x + y back to DRAM.                                                                                                         
    tl.store(output_ptr + offsets, output, mask=mask)                                 
```

We can see that the AST Source has all the kernel information, code, including the signature, attributes, and the source code.

Let's take a couple of steps forward:

```shell
ipdb> n
> /home/ksharma/dev/git/triton/python/triton/tools/compile.py(113)<module>()
    112     opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
--> 113     ccinfo = triton.compile(src, options=opts)
    114     arg_names = []

ipdb> n
> /home/ksharma/dev/git/triton/python/triton/tools/compile.py(114)<module>()
    113     ccinfo = triton.compile(src, options=opts)
--> 114     arg_names = []
    115     arg_types = []

ipdb> ccinfo
<triton.compiler.compiler.CompiledKernel object at 0x7ecef81fdc40>
```

> `ccinfo` is essentially what we looked at in the [previous post](../deep-dive-into-triton-internals/#ttir-triton-ir). It contains the compiled kernel information, including the LLVM IR, PTX, CUBIN, etc.
{: .prompt-info}

```shell
ipdb> ccinfo.asm.keys()
dict_keys(['ttir', 'ttgir', 'llir', 'ptx', 'cubin'])
ipdb> print(ccinfo.asm["ttir"])
#loc = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0)
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg1: !tt.ptr<f32> loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg2: !tt.ptr<f32> loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg3: i32 loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0)) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c64_i32 : i32 loc(#loc3)
    ...truncated...
```

Most of the work for compilation is done at this point. In the [next few lines](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/triton/tools/compile.py#L124-L145), kernel is dumped into .c/.h files as we saw in the previous post. All the info is passed to kernel source/header templates and inserted using [python formatting shenanigans](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/triton/tools/compile.py#L142-L145).

- [C source template](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/triton/tools/compile.c)
- [Header template](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/triton/tools/compile.h)

## Step into `triton.compile`

Let's now step into `triton.compile` to see what happens inside the compiler. That's where the magic happens!

```shell
ipdb> n
> /home/ksharma/dev/git/triton/python/triton/tools/compile.py(113)<module>()
    112     opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
--> 113     ccinfo = triton.compile(src, options=opts)
    114     arg_names = []

ipdb> s
--Call--
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(226)compile()
    225 
--> 226 def compile(src, target=None, options=None):
    227     if target is None:

ipdb> options
{'num_warps': 1, 'num_stages': 3}
ipdb> src
<triton.compiler.compiler.ASTSource object at 0x7fe3cf7c1dc0>
```

Now we are in [`triton/compiler/compiler.py`](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/triton/compiler/compiler.py#L227). 

After stepping a few times:

```shell

ipdb> n
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(231)compile()
    230     backend = make_backend(target)
--> 231     ir_source = not isinstance(src, ASTSource)
    232     # create backend

ipdb> backend
<nvidia.CUDABackend object at 0x7fe4cb7dcef0>

...step forward...

ipdb> options
CUDAOptions(num_warps=1, num_ctas=1, num_stages=3, maxnreg=None, cluster_dims=(1, 1, 1), ptx_version=None, enable_fp_fusion=True, allow_fp8e4nv=True, allow_fp8e4b15=True, default_dot_input_precision='tf32', allowed_dot_input_precisions=('tf32', 'tf32x3', 'ieee'), max_num_imprecise_acc_default=0, extern_libs=(('libdevice', '/home/ksharma/dev/git/triton/python/triton/backends/nvidia/lib/libdevice.10.bc'),), debug=False, backend_name='cuda')

```

This is where the backend options are created/parsed. I would expect this code to be generic and should handle AMD backend as well. Let's keep stepping through:

```shell
...step forward...
ipdb> s
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(277)compile()
    276     import ipdb; ipdb.set_trace()
--> 277     context = ir.context()
    278     ir.load_dialects(context)

ipdb> import pprint
ipdb> pprint.pprint(metadata)
{'allow_fp8e4b15': True,
 'allow_fp8e4nv': True,
 'allowed_dot_input_precisions': ('tf32', 'tf32x3', 'ieee'),
 'backend_name': 'cuda',
 'cluster_dims': (1, 1, 1),
 'debug': False,
 'default_dot_input_precision': 'tf32',
 'enable_fp_fusion': True,
 'extern_libs': (('libdevice',
                  '/home/ksharma/dev/git/triton/python/triton/backends/nvidia/lib/libdevice.10.bc'),),
 'hash': 'c8abb49242c7120a41c83f2e04bf352aac3f33813783a2ccf837a9f62e0f66d7',
 'max_num_imprecise_acc_default': 0,
 'maxnreg': None,
 'num_ctas': 1,
 'num_stages': 3,
 'num_warps': 1,
 'ptx_version': None,
 'target': GPUTarget(backend='cuda', arch=89, warp_size=32)}
```

 We now see that the target backend is set - nvidia CUDABackend! 

```shell

ipdb> n
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(279)compile()
    278     ir.load_dialects(context)
--> 279     backend.load_dialects(context)
    280     codegen_fns = backend.get_codegen_implementation()

ipdb> ir
<module 'triton._C.libtriton.ir'>

ipdb> n
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(280)compile()
    279     backend.load_dialects(context)
--> 280     codegen_fns = backend.get_codegen_implementation()
    281     try:

ipdb> backend
<nvidia.CUDABackend object at 0x73a501b3e0f0>
ipdb> n
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(281)compile()
    280     codegen_fns = backend.get_codegen_implementation()
--> 281     try:
    282         module = src.make_ir(options, codegen_fns, context)

ipdb> codegen_fns
{'convert_custom_types': <function convert_custom_float8_sm80 at 0x73a6c4c73100>, 'min_dot_size': <function min_dot_size.<locals>.<lambda> at 0x73a5b7f45080>}
```

Now we are starting to get into the IR generation part and most of this work happens in the C/C++ layer.

- `context`: [Code](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/src/ir.cc#L206-L215)
- `load_dialects`: [Code](https://github.com/triton-lang/triton/blob/6a9a0a6474afa20498f3b8ae9a8bbb872cad458b/python/src/ir.cc#L221-L231)
- `module`: [Code](https://github.com/triton-lang/triton/blob/6a9a0a6474afa20498f3b8ae9a8bbb872cad458b/python/src/ir.cc#L464)

i.e. all the C++ code that powers python frontend can be found in [python/src](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/python/src). 

Let's test this out. We can put a `stdout` in the C++ code to see if it gets printed. 

```diff
$ git --no-pager diff
diff --git a/python/src/ir.cc b/python/src/ir.cc
index 46095dcc..ec78a7fd 100644
--- a/python/src/ir.cc
+++ b/python/src/ir.cc
@@ -1,6 +1,7 @@
 ﻿#include <pybind11/functional.h>
 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>
+#include <iostream>
 
 #include "mlir/Bytecode/BytecodeWriter.h"
 #include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
@@ -219,6 +220,9 @@ void init_triton_ir(py::module &&m) {
       .def(py::init<llvm::SourceMgr &, MLIRContext *>());
 
   m.def("load_dialects", [](MLIRContext &context) {
+    std::cout << "==========================================" << std::endl;
+    std::cout << "Loading dialects" << std::endl;
+    std::cout << "==========================================" << std::endl;
     DialectRegistry registry;
     registry.insert<TritonDialect, ::mlir::triton::gpu::TritonGPUDialect,
                     math::MathDialect, arith::ArithDialect, index::IndexDialect,

```

> NOTE: Any changes to C++ code will require recompiling the triton python package: `pip install -e python # inside the triton repo`. [More Info](https://github.com/triton-lang/triton?tab=readme-ov-file#install-from-source)
{: .prompt-info}

And, low and behold, we see the print statement in the terminal:

```shell

ipdb> n
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(278)compile()
    277     context = ir.context()
--> 278     ir.load_dialects(context)
    279     backend.load_dialects(context)

ipdb> n
==========================================
Loading dialects
==========================================
> /home/ksharma/dev/git/triton/python/triton/compiler/compiler.py(279)compile()
    278     ir.load_dialects(context)
--> 279     backend.load_dialects(context)
    280     codegen_fns = backend.get_codegen_implementation()

```

## Compiler Backends

At this point, we reach the backend. The backend is responsible for generating the code for the target hardware. In this case, we are using the CUDA backend.

In the source code, the backend code is available in [third_party/nvidia](https://github.com/kapilsh/triton/tree/14025786d108596cfd99700caa4f438938c2ceba/third_party/nvidia). This gets sym-linked from `python/triton/backends/nvidia` during the build process. You can see below that AMD backend is also available in the same directory.


Here's the directory structure after build on my machine:

```shell

$ ll python/triton/backends 
total 16K
lrwxrwxrwx 1 ksharma ksharma   52 Aug 12 22:55 amd -> /home/ksharma/dev/git/triton/third_party/amd/backend
-rw-rw-r-- 1 ksharma ksharma 2.7K Aug  9 16:27 compiler.py
-rw-rw-r-- 1 ksharma ksharma  977 Aug  9 16:27 driver.py
-rw-rw-r-- 1 ksharma ksharma 1.6K Aug  9 16:27 __init__.py
lrwxrwxrwx 1 ksharma ksharma   55 Aug 12 22:55 nvidia -> /home/ksharma/dev/git/triton/third_party/nvidia/backend
drwxrwxr-x 2 ksharma ksharma 4.0K Aug  9 17:42 __pycache__

```

Let's look at the next two lines:

```python
backend.load_dialects(context)
codegen_fns = backend.get_codegen_implementation()
```

- `load_dialects` can be traced to the nvidia backend [here](https://github.com/triton-lang/triton/blob/6a9a0a6474afa20498f3b8ae9a8bbb872cad458b/third_party/nvidia/backend/compiler.py#L158-L159)
- `get_codegen_implementation` can be traced to the nvidia backend [here](https://github.com/triton-lang/triton/blob/6a9a0a6474afa20498f3b8ae9a8bbb872cad458b/third_party/nvidia/backend/compiler.py#L149-L156)

Looking deeper into the nvidia backend compiler code, we can find the actual code generation pointers for Cuda backend.

> `backend.add_stages` for cuda backend adds the different compiler stages. [Code](https://github.com/triton-lang/triton/blob/6a9a0a6474afa20498f3b8ae9a8bbb872cad458b/third_party/nvidia/backend/compiler.py#L346-L351)
{: .prompt-info}

```python
# From nvidia backend compiler.py
def add_stages(self, stages, options):
    stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
    stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
    stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.capability)
    stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)
```

Let's look at the `make_ttir` function to peal the onion further:

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

Looking into [`ir.pass_manager`](https://github.com/triton-lang/triton/blob/6a9a0a6474afa20498f3b8ae9a8bbb872cad458b/python/src/ir.cc#L1579), we see that it returns the [MLIR `PassManager`](https://mlir.llvm.org/doxygen/classmlir_1_1PassManager.html#details). This is where we start entering the MLIR layer. Let's edit this code to always print MLIR IR dump (which is controlled by `MLIR_ENABLE_DUMP` env variable). In addition, let's also print the diagnostics always.

```diff
$ git --no-pager diff
diff --git a/python/src/ir.cc b/python/src/ir.cc
index 46095dcc..b4f1aa22 100644
--- a/python/src/ir.cc
+++ b/python/src/ir.cc
@@ -1584,6 +1584,8 @@ void init_triton_ir(py::module &&m) {
              bool haveDiagnostics =
                  ::triton::tools::getBoolEnv("MLIR_ENABLE_DIAGNOSTICS");
              bool haveDump = ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
+             haveDiagnostics = true;
+             haveDump = true;
              std::string funcToDump;
              if (!haveDump) {
                funcToDump = triton::tools::getStrEnv("MLIR_ENABLE_DUMP");
```

Again, we need to rebuild `pip install -e python` to see the changes. It prints a lot of output but here is small snippet where MLIR is printed at `SymbolDCE` amd `ConvertTritonToTritonGPU` stage:

```shell 
...
// -----// IR Dump Before SymbolDCE (symbol-dce) ('builtin.module' operation) //----- //
#loc = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0)
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg1: !tt.ptr<f32> loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0), %arg2: !tt.ptr<f32> loc("/home/ksharma/dev/git/trit
on/python/tutorials/01-vector-add.py":28:0), %arg3: i32 loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0)) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)

...truncated...

// -----// IR Dump Before ConvertTritonToTritonGPU (convert-triton-to-tritongpu) ('builtin.module' operation) //----- //
#loc = loc("/home/ksharma/dev/git/triton/python/tutorials/01-vector-add.py":28:0)
...
```

> NOTE: Use the official triton guidance to print/dump debug info, IR, etc. [More Info](https://github.com/kapilsh/triton/tree/14025786d108596cfd99700caa4f438938c2ceba?tab=readme-ov-file#tips-for-hacking)
{: .prompt-tip}

## Triton C++ -> Python Bindings

Let's drill down further into the `make_ttir` function. This is where the actual MLIR passes are added:

```python
passes.common.add_inliner(pm)
```

`passes.common` maps to [`init_triton_passes_common`](https://github.com/triton-lang/triton/blob/9e955f1454095725dbb7bed96c8112092c02929e/python/src/passes.cc#L26-L34). 

Other passes are added in a similar way.

```python
passes.ttir.add_rewrite_tensor_pointer(pm)
passes.ttir.add_combine(pm)
passes.common.add_canonicalizer(pm)
passes.ttir.add_reorder_broadcast(pm)
passes.common.add_cse(pm)
passes.common.add_licm(pm)
passes.common.add_symbol_dce(pm)
```

In fact, all the passes are defined in the [`pythpn/src/passes.cc`](https://github.com/triton-lang/triton/blob/9e955f1454095725dbb7bed96c8112092c02929e/python/src/passes.cc) file. A few examples are:

- `init_triton_passes_ttir`
- `init_triton_passes_ttgpuir`

> For rest of the triton compiler passes, refer [`pythpn/src/passes.cc`](https://github.com/triton-lang/triton/blob/9e955f1454095725dbb7bed96c8112092c02929e/python/src/passes.cc#L83-L90)
{: .prompt-tip}

Finally, the C++ compiler backend is exposed to python through pybind11 bindings. The bindings are defined in [`python/src/main.cc`](https://github.com/triton-lang/triton/blob/9e955f1454095725dbb7bed96c8112092c02929e/python/src/main.cc#L46-L55)

## Compiler Passes

Let's look at a couple of passes individually to see what compiler passes are being applied to compile the kernel.

### Inliner Pass

```python
passes.common.add_inliner(pm)
```
and

```cpp
void init_triton_passes_common(py::module &&m) {
  ...
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ...
}
```

ultimately maps to `mlir::createInlinerPass`, which creates a pass which inlines calls and callable operations as defined by the CallGraph. [MLIR code pointer](https://mlir.llvm.org/doxygen/InlinerPass_8cpp_source.html)

### Rewrite Tensor Pointer Pass

```python
passes.ttir.add_rewrite_tensor_pointer(pm)
```
maps to 

```cpp

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ...
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createRewriteTensorPointerPass);
                     
  ...
}
```

where [`createRewriteTensorPointerPass`](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp#L570) is part of the Triton MLIR Dialect and returns [`RewriteTensorPointerPass`](https://github.com/kapilsh/triton/blob/14025786d108596cfd99700caa4f438938c2ceba/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp#L199)

```cpp
std::unique_ptr<Pass> triton::createRewriteTensorPointerPass() {
  return std::make_unique<RewriteTensorPointerPass>();
}
```

It is unclear to me what this pass does. However, I was able to trace down mlir-tblgen file for this pass and it is defined in [`include/triton/Dialect/Triton/Transforms/Passes.td`](https://github.com/triton-lang/triton/blob/9e955f1454095725dbb7bed96c8112092c02929e/include/triton/Dialect/Triton/Transforms/Passes.td#L31-L42) file. 

> Based on the summary and description, it seems like this pass rewrites tensor pointers to "legacy" pointers. But, I don't know what legacy pointers are!
> `let summary = "Rewrite load/stores with tensor pointers into legacy load/stores";`
{: .prompt-info}

### Other passes

There are a number of other passes that are applied and reader is encouraged to explore them further on their own.

## Final Thoughts

We are getting more and more into the MLIR layer and that might be another can of worms to open. This post is already getting pretty big so I will leave that exploration for another day. Hope this provides a good starting point for anyone interested in Triton compiler frontend-backend integration.


