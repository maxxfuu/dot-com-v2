---
title: "Global Memory Coalesing From DRAM Up"
date: "2026-07-31"
summary: "DRAM row mechanics, 32-byte sectors, and the 13x you lose to uncoalesced access."
---

## Introduction 

The fundamental premise of CUDA is that its hardware architecture and programming model are dictated by the laws of physics. CUDA is an intentional co-design between hardware and software, built to extract maximum performance while working around physical constraints.

Getting the most out of a GPU means using all of its resources, so it is natural to look at peak FLOPS and ask how to saturate the compute units. But for most kernels, the limiter is memory bandwidth, not arithmetic.

With the current state of high performance computing hardware, compute capability has outpaced memory speed. Looking at the Ampere A100, it has 108 SMs at 1410 MHz, each capable of requesting 64 bytes per clock.

> Peak request rate = 64 B × 108 SMs × 1410 MHz ≈ 9,750 GB/s
> HBM2e bandwidth = 1,555 GB/s
> cThe ratio is 9,750 / 1,555 ≈ 6.3. 

The execution resources can ask for data roughly six times faster than the memory system can supply it. That same SM retires 64 FP64 FLOPs per clock, 32 FP64 units, one FMA each,  so the hardware is built to consume one byte per floating point operation, and HBM delivers a sixth of what that requires.

## Data Access Pattern

At the physical level, DRAM is a grid of capacitors organized in rows and columns. The grid, together with the circuitry that reads it, is a bank. Along the bank sits an array of sense amplifiers which forms the row buffer. Reading a row, starts with the hardware activating that row and pulling its contents into the buffer. Because those amplifiers can hold only one row's contents at a time, only one row per bank can be open.

Assume a row is already open and held in the row buffer. Reading from adjacent memory addresses is cheap because those addresses fall within the same row already sitting in the buffer, so the read costs only the column access (CL).

![Sequential access — one row, adjacent elements.](/videos/dram/dram-physics-1-sequential.mp4)

But if a GPU tries to read from a different row within the same bank, the bank must first close the open one. Since the sense amplifiers are still holding the previous row's data at full rail, a new row's tiny voltage swings cannot be detected against the bitlines, so the bitlines must be equalized back to the neutral reference voltage, precharge (tRP), before a new row can be activated and pulled into the sense amplifiers (tRCD). Only then can the read proceed (CL). The key insight is that a read hitting the currently open row pays CL alone, while a read to a different row in the same bank pays all three: tRP + tRCD + CL.

![Strided access — a new row for every element.](/videos/dram/dram-physics-2-strided.mp4)

NVIDIA's A100 has 1.55 TB/s of memory bandwidth, and an FP64 value is 8 bytes wide. For a kernel that performs one arithmetic operation per element it loads, achievable throughput is bandwidth × arithmetic intensity:

1.55 TB/s × (1 FLOP / 8 bytes) = 194 GFLOPS

That is the ceiling when every byte fetched is a byte used. If we widen the stride until access is basically random,  using the same kernel measures 14 GFLOPS. A decrease from 194 GLOPs to 14 GFLOPs is a 13.8x degradation; which is roughly 92% of bandwidth lost. 
Drawing from this, kernels must have the threads within a warp read adjacent addresses. Threads do not issue memory requests individually: a warp's 32 lanes issue together, and the hardware services them in 32-byte sectors. When those lanes read contiguous data, the warp's entire request falls inside 128 contiguous bytes, 4 sectors, and every byte fetched is a byte used. Scatter the same lanes and the hardware fetches up to 1024 bytes to deliver 128, while spraying those requests across rows and banks so that the tRP + tRCD penalty is paid over and over. The 13.8x above is those two effects compounding.


![Global Memory Coalesing](/videos/dram/coalesced-vs-strided.mp4)

Coalescing changes neither the FLOPs a kernel performs nor the bytes its algorithm must move. It changes the bytes actually moved to accomplish the same work, which is why two kernels running identical arithmetic can sit 13x apart.
