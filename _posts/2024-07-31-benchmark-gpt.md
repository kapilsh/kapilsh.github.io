---
title: Exploring GPT2 (Part 2)
description: >-
  Benchmarking our own GPT2 model against Huggingface GPT2 model 
date: 2024-07-31
categories: [Blog, DevLog]
tags: [AI, Transformers, Pytorch, Machine Learning, LLM]
pin: true
author: ks
---

In the [previous post](../exploring-gpt2/), we explored the GPT2 model and how to generate text using it. In this post, we will benchmark our own GPT2 model against the Huggingface GPT2 model.

## Hardware specs

![RTX 4090](/assets/rtx-4090.png)

I have a RTX 4090 GPU with 24GB of VRAM on a linux desktop running Ubuntu 22.04. Here is the output of `nvidia-smi`:

```shell
$ nvidia-smi
Wed Jul 31 07:38:40 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0 Off |                  Off |
|  0%   37C    P8              21W / 450W |    355MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1949      G   /usr/lib/xorg/Xorg                          205MiB |
|    0   N/A  N/A      2170      G   /usr/bin/gnome-shell                         16MiB |
|    0   N/A  N/A      3823      G   ...seed-version=20240725-094838.558000       42MiB |
|    0   N/A  N/A      7258      G   ...erProcess --variations-seed-version       57MiB |
+---------------------------------------------------------------------------------------+
```

## Code

All code can be found in my [Github repository](https://github.com/kapilsh/ml-projects/tree/master/transformers). To simplify the benchmarking, I created a small script - `benchmark.py`:

- We will use tokens/s as the metric to measure the performance of different versions of the model. The metric should be agnostic to the batch_size we use.
- We will use the text we used to play around with GPT2 model in the previous post i.e. 1984 text. 
- I did a small test on what is the maximum batch_size I can fit into the GPU and found `batch_size = 8` being the ideal candidate that fits both models.

## Pytorch Eager

Let's start with the baseline model - Huggingface GPT2 model. 

### Huggingface GPT2

```shell
CUDA_LAUNCH_BLOCKING=1 python benchmark.py  --use-hf
# Mean tokens processed per second: 67347
```

![GPT2 HF](/assets/gpt2_benchmarks/baseline.png)

### Our GPT2

Now, let's check the GPT2 model that we created from scratch. 

```shell
CUDA_LAUNCH_BLOCKING=1 python benchmark.py
# Mean tokens processed per second: 117402
```

![Our GPT2](/assets/gpt2_benchmarks/our_gpt2.png)

| Model              | Tokens/s      | Perf |
|:-------------------|:--------------|-----:|
| Baseline (HF GPT2) | 67347         |    - |
| Our GPT2           | 117402        | +74% |

**NICE! Our Eager GPT2 model is 74% faster than the Huggingface GPT2 model.**

## Torch compile

So far we just tested the eager models. `torch.compile` promises us a better performance post compilation so let's try that.

### Huggingface GPT2

```shell
CUDA_LAUNCH_BLOCKING=1 python benchmark.py --use-hf --torch-compile
```

```
...
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 
```

**I LIED!**. `batch_size=8` is too large for the Huggingface GPT2 model. I had to reduce it to `batch_size=4` to fit the model into the GPU. 

```shell
CUDA_LAUNCH_BLOCKING=1 python benchmark.py --use-hf --torch-compile --batch-size=4
# Mean tokens processed per second: 78125
```

![GPT2 HF Compiled](/assets/gpt2_benchmarks/hf_gpt2_compiled.png)

We got a performance boost of 16% by using torch.compile. Let's see how our GPT2 model performs.

```shell
CUDA_LAUNCH_BLOCKING=1 python benchmark.py  --torch-compile
# Mean tokens processed per second: 98444
```

![Our GPT2 Compiled](/assets/gpt2_benchmarks/our_gpt2_compiled.png)

> That's interesting! We got worse performance with the compiled model than we did with the eager model. Although, it is still faster than eager or compiled Huggingface GPT2 model.
{: .prompt-warning}

Let's summarize the results so far:

| Model              | Tokens/s      | Perf |
|:-------------------|:--------------|-----:|
| Baseline (HF GPT2) | 67347         |    - |
| Our GPT2           | 117402        | +74% |
| HF GPT2 Compiled   | 78125         | +16% |
| Our GPT2 Compiled  | 98444         | +46% |

## Tensor Cores

Although, I was going to do this anyway but during the compilation process, I got a giant warning that I am not using Tensor Cores. So, let's enable that and see if we get any performance boost.

> /home/ksharma/anaconda3/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:124: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
{: .prompt-warning}

```shell
CUDA_LAUNCH_BLOCKING=1 python benchmark.py --use-hf  --torch-compile --use-tensorcores --batch-size=4
# Mean tokens processed per second: 89272
```

![GPT2 HF Tensor Cores](/assets/gpt2_benchmarks/hf_gpt2_compiled_tc.png)

Tensor cores gave a performance boost of another ~17% to the Huggingface GPT2 model. Let's see how our GPT2 model performs. 

```shell
CUDA_LAUNCH_BLOCKING=1 python benchmark.py --torch-compile --use-tensorcores
# Mean tokens processed per second: 149383
```

![Our GPT2 Tensor Cores](/assets/gpt2_benchmarks/our_gpt2_compiled_tc.png)

> Enabling Tensor Cores gave a performance boost of 51% to our GPT2 model w.r.t to the compiled model and 27% w.r.t to the eager model.
{: .prompt-info}

## Summary

Here is the summary of the results:

| Model                                 | Tokens/s |  Perf |
|:--------------------------------------|:---------|------:|
| Baseline (HF GPT2)                    | 67347    |     - |
| Our GPT2                              | 117402   |  +74% |
| HF GPT2 Compiled                      | 78125    |  +16% |
| Our GPT2 Compiled                     | 98444    |  +46% |
| HF GPT2 Compiled (with Tensor Cores)  | 89272    |  +33% |
| Our GPT2 Compiled (with Tensor Cores) | 149383   | +122% |

### Why is our GPT2 model faster?

I looked further into [huggingface documentation](https://huggingface.co/docs/transformers/en/model_doc/gpt2) and found that there is an option to pass `attn_implementation="flash_attention_2"` to pretrained model to use flash attention. The option is not enabled by default. 

I tried to enable it and got a nice error message:

> ValueError: GPT2LMHeadModel does not support Flash Attention 2.0 yet. Please request to add support where the model is hosted, on its model hub page: https://huggingface.co/gpt2/discussions/new or in the Transformers GitHub repo: https://github.com/huggingface
{: .prompt-danger}

So, this is where the story ends! It was fun exploring and benchmarking our GPT2 model against the Huggingface GPT2 model.
