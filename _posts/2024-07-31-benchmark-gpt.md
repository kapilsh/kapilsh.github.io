---
title: Exploring GPT2 (Part 2)
description: >-
  Benchmarking ourw own GPT2 model against Huggingface GPT2 model 
date: 2024-07-31
categories: [Blog, DevLog]
tags: [AI, Transformers, Pytorch, Machine Learning, LLM]
pin: true
author: ks
---

In the [previous post](../exploring-gpt2/), we explored the GPT2 model and how to generate text using it. In this post, we will benchmark our own GPT2 model against the Huggingface GPT2 model.

## Hardware specs

![RTX 4090](/assets/rtx-4090.png)

I have a RTX 4090 GPU with 24GB of memory. We will use this to perform the benchmarking.

Here is the output of `nvidia-smi`:

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

## Pytorch Eager

## Torch compile

## Tensor Cores

## Summary
