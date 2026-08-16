---
title: "NCCL wire protocols: Simple, LL, and LL128"
description: >-
  How NCCL's three wire protocols trade payload bytes against synchronization cost
date: 2026-08-16
categories: [Blog]
tags: [NCCL, GPU, Distributed, Performance]
pin: true
math: false
author: ks
---

<style>
.nccl-fig {
  --n-text: #2C2C2A; --n-text2: #5F5E5A; --n-line: #888780;
  --n-box-stroke: #B4B2A9;
  --n-teal-fill: #E1F5EE;  --n-teal-stroke: #0F6E56;  --n-teal-text: #085041;
  --n-amber-fill: #FAEEDA; --n-amber-stroke: #854F0B; --n-amber-text: #633806;
  --n-red-fill: #FCEBEB;   --n-red-stroke: #A32D2D;   --n-red-text: #791F1F;
  --n-coral: #D85A30;
  width: 100%; height: auto; display: block; margin: 2rem 0; font-family: inherit;
}
@media (prefers-color-scheme: dark) {
  .nccl-fig {
    --n-text: #D3D1C7; --n-text2: #B4B2A9; --n-line: #888780;
    --n-box-stroke: #5F5E5A;
    --n-teal-fill: #085041;  --n-teal-stroke: #5DCAA5;  --n-teal-text: #9FE1CB;
    --n-amber-fill: #633806; --n-amber-stroke: #EF9F27; --n-amber-text: #FAC775;
    --n-red-fill: #791F1F;   --n-red-stroke: #F09595;   --n-red-text: #F7C1C1;
    --n-coral: #F0997B;
  }
}
.nccl-fig .t  { font-size: 14px; fill: var(--n-text); }
.nccl-fig .ts { font-size: 12px; fill: var(--n-text2); }
.nccl-fig .th { font-size: 14px; font-weight: 500; fill: var(--n-text); }
.nccl-fig .box { fill: none; stroke: var(--n-box-stroke); }
.nccl-fig .arr { stroke: var(--n-line); stroke-width: 1.5; fill: none; }
.nccl-fig .c-teal  rect { fill: var(--n-teal-fill);  stroke: var(--n-teal-stroke); }
.nccl-fig .c-teal  line { stroke: var(--n-teal-stroke); }
.nccl-fig .c-amber rect { fill: var(--n-amber-fill); stroke: var(--n-amber-stroke); }
.nccl-fig .c-red   rect { fill: var(--n-red-fill);   stroke: var(--n-red-stroke); }
.nccl-fig .c-teal  .th, .nccl-fig .c-teal  .ts { fill: var(--n-teal-text); }
.nccl-fig .c-amber .th, .nccl-fig .c-amber .ts { fill: var(--n-amber-text); }
.nccl-fig .c-red   .th, .nccl-fig .c-red   .ts { fill: var(--n-red-text); }
</style>

> NOTE: Mostly written by human in author's voice with some Claude assistence on research and diagrams
{: .prompt-info}

I was recently reading a [recent paper from NVidia](https://arxiv.org/pdf/2607.16100) on speed of light latency on GPU collectives. The main premise of the paper is focussed on using different wire protocols to establish low-latency NCCL collective performance baselines. I went down a rabbithole of looking at NCCL kernel selection for collectives. NCCL collective picks two things independently: an *algorithm* (Ring, Tree, CollNet, NVLS) and a *protocol* (Simple, LL, LL128). At a high level, 
- algorithm decides who talks to whom
- the protocol decides what the bytes on the wire i.e. wire protocol

In the post, we will focus on the latter since choosing right low latency wire protocol is specifically critical for inference, post-training workloads of today.


## Protocols

Below we cover the 3 main protocols that are currently available in NCCL.

### Simple

Data is written in large chunks such that 100% of the wire bytes are payload. Ordering between the payload stores and the synchronization signal is enforced by an explicit barrier using `__threadfence_system()` and a separate tail/flag store. This fence is expensive. It costs on the order of us on some paths, which is irrelevant for medium to larger payloads but dominate smaller messages let's say 4 KB. 

<svg class="nccl-fig" viewBox="0 0 680 330" role="img" xmlns="http://www.w3.org/2000/svg"><title>NCCL Simple protocol</title><desc>A full send-buffer slot of pure payload, followed by a thread fence, then a separate tail flag store, which the receiver spins on before reading the payload.</desc>
<defs><marker id="nccl-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="var(--n-line)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text class="ts" x="60" y="36">Send buffer slot (NCCL_BUFFSIZE / steps)</text>
<g class="c-teal">
<rect x="60" y="46" width="520" height="44" rx="4" stroke-width="0.5"/>
<line x1="125" y1="46" x2="125" y2="90" stroke-width="0.5"/>
<line x1="190" y1="46" x2="190" y2="90" stroke-width="0.5"/>
<line x1="255" y1="46" x2="255" y2="90" stroke-width="0.5"/>
<line x1="385" y1="46" x2="385" y2="90" stroke-width="0.5"/>
<line x1="450" y1="46" x2="450" y2="90" stroke-width="0.5"/>
<line x1="515" y1="46" x2="515" y2="90" stroke-width="0.5"/>
<text class="th" x="320" y="68" text-anchor="middle" dominant-baseline="central">Payload only — no inline metadata</text>
</g>
<text class="ts" x="320" y="110" text-anchor="middle">100% of wire bytes are data</text>
<line x1="320" y1="120" x2="320" y2="138" class="arr" marker-end="url(#nccl-arrow)"/>
<g class="c-red">
<rect x="170" y="140" width="300" height="34" rx="4" stroke-width="0.5"/>
<text class="th" x="320" y="157" text-anchor="middle" dominant-baseline="central">__threadfence_system()</text>
</g>
<line x1="320" y1="176" x2="320" y2="194" class="arr" marker-end="url(#nccl-arrow)"/>
<g class="c-amber">
<rect x="250" y="196" width="180" height="40" rx="4" stroke-width="0.5"/>
<text class="th" x="340" y="216" text-anchor="middle" dominant-baseline="central">tail / flag store</text>
</g>
<line x1="340" y1="238" x2="340" y2="256" class="arr" marker-end="url(#nccl-arrow)"/>
<rect class="box" x="60" y="258" width="520" height="52" rx="8" stroke-width="0.5"/>
<text class="th" x="320" y="277" text-anchor="middle" dominant-baseline="central">Receiver</text>
<text class="ts" x="320" y="295" text-anchor="middle" dominant-baseline="central">spins on tail, then reads the payload</text>
</svg>

### LL (Low Latency)

LL drops the fence entirely by exploiting naturally-aligned 8-byte atomic stores and replacing barrier fence with inlined synchronization flags. In essence, bandwidth is immaterial at lower latency so you can trade of more data on the wire for better latency. 

In LL, every 8-byte line is 4 B of data plus a 4 B flag, so the receiver polls the flag *inline*. Hence, there are no explicit barriers. Synchronization signal is embedded in the wire protocol. NCCL uses it below roughly 8 KB.


<svg class="nccl-fig" viewBox="0 0 680 256" role="img" xmlns="http://www.w3.org/2000/svg"><title>NCCL LL protocol</title><desc>Repeated 8-byte lines, each holding 4 bytes of data and a 4-byte flag, so the receiver polls a flag inline with the data it guards. Half of all bytes are flags.</desc>
<defs><marker id="nccl-arrow-2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="var(--n-line)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text class="ts" x="70" y="44">8-byte line, repeated</text>
<g class="c-teal"><rect x="70" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="100" y="74" text-anchor="middle" dominant-baseline="central">data</text><text class="ts" x="100" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<g class="c-amber"><rect x="130" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="160" y="74" text-anchor="middle" dominant-baseline="central">flag</text><text class="ts" x="160" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<g class="c-teal"><rect x="210" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="240" y="74" text-anchor="middle" dominant-baseline="central">data</text><text class="ts" x="240" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<g class="c-amber"><rect x="270" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="300" y="74" text-anchor="middle" dominant-baseline="central">flag</text><text class="ts" x="300" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<g class="c-teal"><rect x="350" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="380" y="74" text-anchor="middle" dominant-baseline="central">data</text><text class="ts" x="380" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<g class="c-amber"><rect x="410" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="440" y="74" text-anchor="middle" dominant-baseline="central">flag</text><text class="ts" x="440" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<g class="c-teal"><rect x="490" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="520" y="74" text-anchor="middle" dominant-baseline="central">data</text><text class="ts" x="520" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<g class="c-amber"><rect x="550" y="56" width="60" height="52" rx="4" stroke-width="0.5"/><text class="th" x="580" y="74" text-anchor="middle" dominant-baseline="central">flag</text><text class="ts" x="580" y="92" text-anchor="middle" dominant-baseline="central">4 B</text></g>
<text class="ts" x="340" y="130" text-anchor="middle">50% payload · 50% flags · no fence, no separate signal</text>
<line x1="340" y1="140" x2="340" y2="160" class="arr" marker-end="url(#nccl-arrow-2)"/>
<rect class="box" x="70" y="162" width="540" height="52" rx="8" stroke-width="0.5"/>
<text class="th" x="340" y="181" text-anchor="middle" dominant-baseline="central">Receiver polls the flag word</text>
<text class="ts" x="340" y="199" text-anchor="middle" dominant-baseline="central">data in that same 8 B line is valid — 8 B stores never tear</text>
</svg>

### LL128

LL128 uses the same idea as LL but expands it to a coarser granularity. It uses a 128-byte line that carries 15 × 8B payload (120B) plus one 8B signal flag word. Main tradeoff w.r.t. LL is that for medium sized messages, it gives better bandwidth utilization for transfering payload (6.25%) vs LL (50%).

One caveat to note is the dependency on 128B stores may be atomic for all transports. For example, NVLink guarantees atomicity for 128B stores but PCIe does not. 

> However, LL128 comes with stricter hardware requirements. It depends on atomic 128-byte writes, which must not be split or reordered by the memory system or interconnect. In systems where such operations are not guaranteed, due to PCIe limitations or other architectural constraints, NCCL disables LL128 to avoid data corruption. Protocol selection is thus influenced not only by message size, but also by systemlevel capabilities.
>
> **Source: [Demystifying NCCL](https://arxiv.org/pdf/2507.04786)**
{: .prompt-info}


<svg class="nccl-fig" viewBox="0 0 680 348" role="img" xmlns="http://www.w3.org/2000/svg"><title>NCCL LL128 protocol</title><desc>A 128-byte line of sixteen 8-byte words: fifteen data words followed by one flag word, giving 6.25 percent overhead, valid only where 128-byte stores are atomic.</desc>
<defs><marker id="nccl-arrow-3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="var(--n-line)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
<text class="ts" x="60" y="40">128-byte line — LL128_LINEELEMS = 16, DATAELEMS = 15</text>
<g class="c-teal">
<rect x="60" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="95" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="130" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="165" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="200" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="235" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="270" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="305" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="340" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="375" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="410" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="445" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="480" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="515" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
<rect x="550" y="52" width="34" height="44" rx="4" stroke-width="0.5"/>
</g>
<g class="c-amber"><rect x="585" y="52" width="34" height="44" rx="4" stroke-width="0.5"/></g>
<path d="M60 104V112H584V104" fill="none" stroke="var(--n-line)" stroke-width="0.5"/>
<path d="M585 104V112H619V104" fill="none" stroke="var(--n-line)" stroke-width="0.5"/>
<text class="ts" x="322" y="128" text-anchor="middle">15 × 8 B data words (120 B)</text>
<text class="ts" x="602" y="128" text-anchor="middle">flag 8 B</text>
<text class="ts" x="340" y="156" text-anchor="middle">1/16 overhead = 6.25% · ~93.75% of peak bandwidth</text>
<line x1="340" y1="166" x2="340" y2="186" class="arr" marker-end="url(#nccl-arrow-3)"/>
<rect class="box" x="60" y="188" width="560" height="52" rx="8" stroke-width="0.5"/>
<text class="th" x="340" y="207" text-anchor="middle" dominant-baseline="central">Receiver polls the 16th word</text>
<text class="ts" x="340" y="225" text-anchor="middle" dominant-baseline="central">the preceding 120 B in that line are valid — still no fence</text>
<g class="c-red">
<rect x="60" y="260" width="560" height="52" rx="8" stroke-width="0.5"/>
<text class="th" x="340" y="279" text-anchor="middle" dominant-baseline="central">Requires atomic 128 B stores</text>
<text class="ts" x="340" y="297" text-anchor="middle" dominant-baseline="central">guaranteed on NVLink, not on PCIe — hence the topology gate</text>
</g>
</svg>

### Summary

| | Payload efficiency | Sync cost | Typical range |
|---|---|---|---|
| Simple | 100% | `__threadfence_system()`, µs-scale | large messages |
| LL | ≤ 50% | none — inline 8 B flag | below ~8 KB |
| LL128 | ~93.75% | none — inline 8 B flag per 128 B | mid-size, NVLink only |

> - Latency: **LL < LL128 < Simple**
> - Bandwidth: **Simple > LL128 >> LL**
{: .prompt-info}

NCCL's tuning model picks per (algorithm, protocol, message size, topology) from its internal latency/bandwidth tables. 


## Fenceless synchronization

LL and LL128 are the same idea at two granularities, so it's worth describing the mechanism once rather than twice.

The problem both are solving: a receiver needs to know that a producer's data stores have landed before it reads them. The classic answer is Simple's — order the stores with a fence, then publish a separate flag, and have the consumer spin on that flag. The fence is what makes it correct and also what makes it slow.

The LL family's answer is to **make the flag and the data the same store**. If the hardware guarantees that a store of size *N* becomes visible atomically — all of it or none of it — then a flag placed inside that same *N* bytes cannot be observed before the data it accompanies. There is nothing left to order, so there is nothing to fence.

That reduces the whole protocol to one question: what's the largest store width the interconnect makes atomic?

- **8 bytes** is architecturally guaranteed for naturally-aligned accesses. That's LL, and it costs you 4 of every 8 bytes.
- **128 bytes** holds on NVLink. That's LL128, and it costs you 8 of every 128.

Everything else about the two protocols follows from that number.

### What the poll actually touches

The store is remote — it crosses the fabric into the peer's receive buffer. The poll is always local: each rank spins on its own buffer, never on a remote address.

<svg class="nccl-fig" viewBox="0 0 680 470" role="img" xmlns="http://www.w3.org/2000/svg"><title>LL128 flag polling across four NVLink-connected GPUs</title><desc>Four GPUs at the corners of an NVLink fabric, joined by matching right-angled connectors. Each holds a receive buffer of four data words plus a flag word. One GPU stores a 128-byte line directly into a peer's receive buffer over NVLink; the peer spins on its own local flag word and treats the preceding data as valid once the flag matches.</desc>
<defs><marker id="nccl-arrow-4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="var(--n-coral)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>

<path d="M180 160V200H258" fill="none" stroke="var(--n-line)" stroke-width="1" stroke-linejoin="round" stroke-linecap="round"/>
<path d="M500 160V200H422" fill="none" stroke="var(--n-line)" stroke-width="1" stroke-linejoin="round" stroke-linecap="round"/>
<path d="M180 300V260H258" fill="none" stroke="var(--n-line)" stroke-width="1" stroke-linejoin="round" stroke-linecap="round"/>
<path d="M500 300V260H422" fill="none" stroke="var(--n-line)" stroke-width="1" stroke-linejoin="round" stroke-linecap="round"/>

<rect x="260" y="180" width="160" height="100" rx="12" fill="none" stroke="var(--n-line)" stroke-width="0.5" stroke-dasharray="4 4"/>
<text class="th" x="340" y="218" text-anchor="middle" dominant-baseline="central">NVLink</text>
<text class="ts" x="340" y="238" text-anchor="middle" dominant-baseline="central">peer-mapped stores</text>

<rect class="box" x="40" y="50" width="190" height="110" rx="12" stroke-width="0.5"/>
<text class="th" x="135" y="72" text-anchor="middle" dominant-baseline="central">GPU 0</text>
<g class="c-teal"><rect x="71" y="100" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="97" y="100" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="123" y="100" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="149" y="100" width="24" height="24" rx="4" stroke-width="0.5"/></g>
<g class="c-amber"><rect x="175" y="100" width="24" height="24" rx="4" stroke-width="0.5"/></g>
<text class="ts" x="135" y="142" text-anchor="middle" dominant-baseline="central">stores into peer buffer</text>

<rect class="box" x="450" y="50" width="190" height="110" rx="12" stroke-width="0.5"/>
<text class="th" x="545" y="72" text-anchor="middle" dominant-baseline="central">GPU 1</text>
<g class="c-teal"><rect x="481" y="100" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="507" y="100" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="533" y="100" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="559" y="100" width="24" height="24" rx="4" stroke-width="0.5"/></g>
<g class="c-amber"><rect x="585" y="100" width="24" height="24" rx="4" stroke-width="1.5"/></g>
<text class="ts" x="545" y="142" text-anchor="middle" dominant-baseline="central">↻ flag hit → data valid</text>

<rect class="box" x="40" y="300" width="190" height="110" rx="12" stroke-width="0.5"/>
<text class="th" x="135" y="322" text-anchor="middle" dominant-baseline="central">GPU 2</text>
<g class="c-teal"><rect x="71" y="350" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="97" y="350" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="123" y="350" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="149" y="350" width="24" height="24" rx="4" stroke-width="0.5"/></g>
<g class="c-amber"><rect x="175" y="350" width="24" height="24" rx="4" stroke-width="0.5"/></g>
<text class="ts" x="135" y="392" text-anchor="middle" dominant-baseline="central">↻ spins on flag word</text>

<rect class="box" x="450" y="300" width="190" height="110" rx="12" stroke-width="0.5"/>
<text class="th" x="545" y="322" text-anchor="middle" dominant-baseline="central">GPU 3</text>
<g class="c-teal"><rect x="481" y="350" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="507" y="350" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="533" y="350" width="24" height="24" rx="4" stroke-width="0.5"/><rect x="559" y="350" width="24" height="24" rx="4" stroke-width="0.5"/></g>
<g class="c-amber"><rect x="585" y="350" width="24" height="24" rx="4" stroke-width="0.5"/></g>
<text class="ts" x="545" y="392" text-anchor="middle" dominant-baseline="central">↻ spins on flag word</text>

<text class="ts" x="340" y="90" text-anchor="middle">remote store, one 128 B line</text>
<line x1="232" y1="112" x2="446" y2="112" stroke="var(--n-coral)" stroke-width="1.5" marker-end="url(#nccl-arrow-4)"/>

<text class="ts" x="340" y="442" text-anchor="middle">the flag rides in the same line as the data it guards — the poll never leaves local memory</text>
</svg>

### Three details the picture compresses

**The flag value is a step counter, not a boolean.** NCCL writes `flag = step`, derived from the slot index, so the receiver compares against an *expected* value rather than testing for non-zero. That's what makes buffer reuse safe without a reset pass between steps: a stale line left over from step *N−1* simply doesn't match the value the poller wants for step *N*. A boolean flag would require zeroing the buffer between steps, which would reintroduce exactly the ordering problem the protocol just eliminated.

**The poll is a volatile load in a tight loop.** The LL128 receive path uses `ld.volatile.global.v4.u32`-class loads so the compiler can't hoist the read out of the loop and the value isn't served from a stale register. Only after the flag word matches does the warp consume the other 15 words of the line.

**Every rank is doing both jobs at once.** In a ring or tree allreduce each rank is simultaneously storing into its successor's buffer and spinning on lines arriving from its predecessor. What the diagram shows as a single arrow is really four concurrent producer/consumer pairs sharing the fabric.

### Reading the source

The two implementations live side by side and the structural difference between them is almost entirely line geometry:

- `nccl/src/device/prims_ll.h` — the 8-byte variant
- `nccl/src/device/prims_ll128.h` — `recvReduceSendCopy`, and the `LL128_LINEELEMS` / `LL128_DATAELEMS` split

Worth reading in that order. Once you've seen how the 4+4 split works in `prims_ll.h`, LL128 reads as the same state machine with a different constant, plus the extra bookkeeping needed because 15 doesn't divide evenly into anything convenient.