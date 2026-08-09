## From GPU Hardware Architecture To CUDA Kernels

## What is CUDA?
CUDA is an acronym for "compute unified device architecture", which essentially describes a wide array of computing units that can execute in parallel. However, depending on the context CUDA, it can either be describing a high-level device architecture or a parallel programming model for architectures with the CUDA design. 

The key insight is to understand that CUDA is a co-design between the hardware and the software.[^1] Writing good CUDA requires a solid fundamental understanding of the CUDA programming model and how it relates to the GPU hardware architecture.

## GPU Hardware

In a moden NVIDIA GPU, the most important hardware components relevant to optimizng a GEMM is the GPU die itself, the Streaming Multiprocessors (SM), everything inside the SM, L2 Cache, and the high bandwidth memory (HBM).

(Insert image of GPU Architecture) 

The exact organization and varies between different NVIDIA architectures, but these hardware components exists throughout all modern NVIDIA GPUs. For this article, we will be referencing the Blackwell Architecture, specifically the consumer grade GPU RTX 5080. 

## Streaming Multiprocessor

The Streaming Multiprocessor (SM) is the most important piece of hardware to understand. It is comparable to the CPU since the SM is what contains the hardware components that does the execution. The GPU contains multiple SMs, within the RTX 5080 the underlying GB203 die has 96 SMs, but only 84 of the SMs are enabled in the 5080 configuration.

Each SM contains 4 execution partition units or what I'd like to refer as sub-partitions, and each sub-partition containts the CUDA cores, Tensor Core, Load/Store Units, Warp Schedulers, Register File, Shared Memory, L1 Cache. 
In knowing this, you should immediately be aware that a CUDA kernel is not executing on "the GPU" operating as a monolithic processor like the CPU. Instead, work is distributed across all of the SMs within the GPU. 

## Execution Units
CUDA Cores are the general-purpose arithmetic execution units used for many ordinary CUDA operations. 

Tensor Cores, on the other hand, are specialized hardware designed to accelerate matrix multiplication and tensor operations, making them particularly important for deep learning workloads. 

Essentially, both are execution units that receive instructions and perform computations on data. The results of these computations are typically written back to registers, while Load/Store Units are responsible for moving data between registers and the memory hierarchy. CUDA Cores and Tensor Cores are both execution units within the SM, but they are specifized for different types of operations. 

## Load/Store Units 

The Load/Store Units (LSUs) are the hardware components responsible for issuing memory operations that move data between registers and the memory hierarchy. 

Practically speaking, when CUDA code is compiled into SASS instructions, a load or store instruction specifies the memory address involved and the registers used as the source or destination. For a load, the LSU initiates a request to the memory hierarchy. 

The memory system first checks the appropriate cache levels, such as L1 and L2. If the data is found in L1, it is returned to the destination register. If there is an L1 cache miss but an L2 cache hit, the data is retrieved from L2 and returned to the register. If both caches miss, the data must ultimately be fetched from device memory (VRAM). 

[Insert Image Here]

Since we are working with a consumer grade GPU, the device memory (VRAM) refers to GDDR7. ON data-center GPUs like the H100, the device memory (VRAM) is HBM.

## Warp Schedulers

The next big question following the introduction of Streaming Multiprocessor should be "how is work distributed across all of the SMs?". Each sub-partition owns a warp scheduler, and that scheduler is what decides, every cycle, which instruction the sub-partition issues next.

## Memory Hierarchy

The LSU walks a ladder of memories, and that ladder is the single most important thing to hold in your head for the rest of this article. Each level trades capacity for latency, and each one is visible to a different scope of the program:

```
  scope          level                 rough cost
  ─────          ─────                 ──────────
  thread    →    Registers             ~1 cycle
  block     →    Shared Memory / L1    tens of cycles
  device    →    L2 Cache              hundreds of cycles
  device    →    Device Memory (VRAM)  many hundreds of cycles
```

Registers are private to a single thread and are the only storage the execution units read operands from directly. They are also finite: the register file is a fixed budget per SM, split across every thread resident on it, so a kernel that asks for more registers per thread gets fewer threads resident at once.

Shared memory sits one step down. It is carved out of the same physical storage as L1 and it is private to a thread block, which makes it the one level the programmer manages by hand — you decide what goes in it and when. This is the level the entire optimization ladder is built around: if a value is going to be read many times by many threads, you want it in shared memory, read from device memory exactly once.

L2 is shared by every SM on the die and is managed entirely by the hardware. You do not place data in it, but you can still influence your hit rate by controlling which blocks touch which addresses at roughly the same time.

Device memory is the bottom of the ladder and the top of the capacity chart. It is where your matrices live when the kernel launches, it is an order of magnitude slower than anything above it, and every optimization in this article is ultimately an argument about how few times you can afford to touch it.

## CUDA Programming Model

The CUDA programming model is the abstraction you write to harness the powerful GPU you have on hand. CUDA exposes a hierarchical model enabling CUDA practioners to express parallel execution through the division of grid, blocks, threads. 

A grid is the complete collection of blocks. A block is a group of threads that execute and cooperate with one another. And a thread  is the smallest unit of execution exposed by the CUDA programming model. Each block of threads executes the same instruction, and each thread executes the kernel independently. 

```
[insert image]
Grid
└── Thread Blocks
    └── Threads
```

Every thread runs the same kernel body. The only thing separating one thread from another is its position, which CUDA exposes through four built-ins: `threadIdx` and `blockIdx` give position, `blockDim` and `gridDim` give shape. Combining them is what makes one program body do different work in every thread:

```cuda
const int row = blockIdx.y * blockDim.y + threadIdx.y;
```

## How the Program Maps to the Hardware

We now have both halves — the hardware on one side, the grid/block/thread abstraction on the other — so the useful question is which piece of the abstraction lands on which piece of silicon:

```
  programming model        hardware
  ─────────────────        ────────
  Grid                →    the whole GPU, all SMs
  Thread Block        →    exactly one SM
  Thread              →    a lane in one of that SM's sub-partitions
```

The middle row is the one that carries all the weight. A block is assigned to exactly one SM and stays resident there until every one of its threads has finished — it never migrates, and it is never split across two SMs. That is what makes the cooperation guarantee possible: the shared memory a block writes to is physical storage on the SM it landed on, and the barrier its threads synchronize at is hardware inside that SM.

An SM can host several blocks at the same time, and how many is not a number you set. It falls out of what each block asks for. Registers per thread come out of that SM's register file; shared memory per block comes out of that SM's shared memory; and no block may exceed 1024 threads. Ask for more of any of them and fewer blocks fit. This ratio — how much of an SM's capacity you actually keep occupied — is what we will later measure as occupancy, and it is why an optimization that looks free on paper can lose performance by quietly inflating register usage.

Nothing here says anything about *order*. The runtime hands blocks to whichever SM has room, in whatever order it likes, which is exactly why the programming model forbids you from assuming otherwise. It is also why the same binary fills an 84-SM RTX 5080 and a 16-SM laptop GPU without being recompiled.

What the mapping does not yet explain is what an SM does with a block once it has one. It does not run 1024 threads independently. It chops the block into warps.

## CUDA Execution Model

On the software level, a thread is a single unit of execution executed by the kernel. But on the hardware level, a single unit of execution is a warp. A warp is a collection of 32 threads which makes up one warp. Inside a block can contain mulitple warps.

Every block that is passed on to the SM gets split into warps and schedules those. A warp issues one instruction at a time on behalf of all 32 of its threads, so they advance in lockstep. When a branch sends them different ways, the warp walks each path in turn with the threads that took the other path switched off, so the two paths are serialized rather than run in parallel. This is why divergence inside a warp costs real time, and why it costs nothing when an entire warp takes the same branch.

The other half of the model is latency hiding. Many warps sit resident on an SM at once, and the warp scheduler issues each cycle from whichever ones are ready. When one warp stalls waiting on a global memory load, the scheduler simply issues from another. A CPU spends enormous transistor budget on caches and out-of-order execution to avoid stalling; a GPU accepts the stall and keeps enough warps in flight that something is always runnable. How well you keep the schedulers fed is what occupancy measures.

Nearly every optimization in this article follows from the warp. Coalescing is about what one warp's 32 addresses look like to the memory system, bank conflicts are about what they look like to shared memory, and tiling is about giving each warp enough arithmetic to chew on while other warps wait.

This section of the artcile covers the most bare-bone aespects of CUDA programming and CUDA hardware architecture. It should be just enough to get us started with writing the actual naive kernel. However, if you want to dive deeper into what happens when you run a CUDA kernel, check out Fergus Finn's blog. [^2]

:::check
During the forward pass, why do we store every `z` and `a` instead of discarding them?
---
The backward pass reuses them: computing each layer's gradients requires the activations produced during the forward pass, so recomputing them would double the work.
:::

[^1]: [How CUDA Programming Works - Stephen Jones, CUDA Architect](https://www.youtube.com/watch?v=QQceTDjA4f4&t=75s)
[^2]: [What happens when you run a CUDA kernel](https://fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/#an-interposition-hook)



