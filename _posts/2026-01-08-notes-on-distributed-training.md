---
title: Notes on distributed training
description: >-
  Scaling playbook
date: 2026-01-07
categories: [Blog]
tags: [AI, Machine Learning, GPU, TorchDistributed, Training]
pin: true
math: true
author: ks
---

I have been reading the excellent resource on large scale model training recently i.e. [Ultra Scale Playbook](https://huggingface.co/spaces/weege007/ultrascale-playbook). I learn best by writing down things as I was writing some notes on this, I decided to make these into a blog post. 

## Transformers models

Transformers on a basic level are learning sequences - typically that we feed sequences of tokens in a batch (batch size in tokens) split by sequence length i.e.

```python
batch_size_in_tokens = batch_size * sequence_length
```
Modern LLMs typically train with 4-60M tokens per batch, with recent models like DeepSeek using ~60M token batches over 14 trillion tokens

### Fundamental challenges to solve

All distributed training techniques tackle one or more of these challenges:

1. **Memory Usage** - The hard limitation. If a training step doesn't fit in GPU memory, training cannot proceed. Components include:
   - Model parameters
   - Gradients
   - Optimizer states
   - Activations (forward pass intermediate results)

2. **Compute Efficiency** - Hardware should spend maximum time computing, minimizing time on data transfers or waiting for other GPUs.

3. **Communication Overhead** - Minimize GPU idle time by:
   - Maximizing use of fast intra-node bandwidth (NVLink)
   - Minimizing use of slower inter-node bandwidth (InfiniBand/Ethernet)
   - Overlapping communication with computation whenever possible


### Memory Components and Requirements

For a transformer with N parameters in mixed precision training (BF16 + FP32):

**Per-Parameter Memory:**
- Parameters (BF16): 2 bytes
- Gradients (BF16): 2 bytes  
- FP32 parameters (master weights): 4 bytes
- Optimizer states (Adam - momentum + variance): 8 bytes
- **Total: 16 bytes per parameter**

With optional FP32 gradient accumulation: +4 bytes = 20 bytes per parameter.

**Example Memory Requirements:**

| Model Size | Memory (BF16 mixed precision) |
|-----------|-------------------------------|
| 1B        | 16-20 GB                      |
| 7B        | 112-140 GB                    |
| 70B       | 1,120-1,400 GB                |
| 405B      | 6,480-8,100 GB                |

#### Activation Memory

Activations scale with batch size and sequence length according to:

```
memory_activations = L × seq × bs × h × (34 + (5 × n_heads × seq) / h)
```

Where:
- L = number of layers
- seq = sequence length
- bs = batch size (samples)
- h = hidden dimension
- n_heads = number of attention heads

> **Key Observations:**
> - Linear scaling with batch size
> - Quadratic scaling with sequence length
> - For long sequences or large batches, activations dominate memory usage
> - Short sequences: parameters/gradients/optimizer states dominate
> - Long sequences (8k+ tokens): activations become the primary concern
{: .prompt-tip}


## Improvement 1: Gradient Checkpointing / Activation Recomputation


**Core Idea**: Discard activations during forward pass, recompute them on-the-fly during backward pass.

**Strategies:**

1. **Full Recomputation**
   - Checkpoint only at layer boundaries
   - Requires full forward pass during backward
   - Memory savings: ~70% for activations
   - Compute overhead: ~30-40% additional time

2. **Selective Recomputation** (Recommended)
   - Checkpoint expensive operations (feedforward)
   - Recompute cheap, memory-heavy operations (attention)
   - Memory savings: ~70% for GPT-3 175B
   - Compute overhead: ~2.7%
   - **Best practice**: FlashAttention natively integrates selective recomputation

**Memory Impact Example (8B model, seq=4k):**
- No recomputation: 45 GB activations
- Selective: 15 GB activations  
- Full: 8 GB activations


## References

- [Nanotron Ultrascaling Playbook](https://huggingface.co/spaces/weege007/ultrascale-playbook)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198)