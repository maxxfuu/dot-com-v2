## From CUDA Kernels To NVIDIAs GPU Hardware Architecture

## What is CUDA?
CUDA is an acronym for "compute unified device architecture", which essentially describes a wide array of computing units that can execute in parallel. However, depending on the context CUDA, it can either be describing a high-level device architecture or a parallel programming model for architectures with the CUDA design. 

The key insight is to understand that CUDA is a co-design between the hardware and the software.[^1] Writing good CUDA requires a solid fundamental understanding of the CUDA programming model and how it relates to the GPU hardware architecture.

In a moden NVIDIA GPU, the most important hardware components relevant to optimizng a GEMM is the GPU die itself, the Streaming Multiprocessors (SM), everything inside the SM, L2 Cache, and the high bandwidth memory (HBM).

(Insert image of GPU Architecture) 

The exact organization and varies between different NVIDIA architectures, but these hardware components exists throughout all modern NVIDIA GPUs. For this article, we will be referencing the Blackwell Architecture, specifically the consumer grade GPU RTX 5080. 

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

## CUDA Execution Model

The programming model lets you pretend threads are independent. The execution model is what the hardware actually does with them, and its unit is not the thread — it is the **warp**, a group of 32 consecutive threads from one block.

An SM splits every block it is handed into warps and schedules those. A warp issues one instruction at a time on behalf of all 32 of its threads, so they advance in lockstep. When a branch sends them different ways, the warp walks each path in turn with the threads that took the other path switched off, so the two paths are serialized rather than run in parallel. This is why divergence inside a warp costs real time, and why it costs nothing when an entire warp takes the same branch.

The other half of the model is latency hiding. Many warps sit resident on an SM at once, and the warp scheduler issues each cycle from whichever ones are ready. When one warp stalls waiting on a global memory load, the scheduler simply issues from another. A CPU spends enormous transistor budget on caches and out-of-order execution to avoid stalling; a GPU accepts the stall and keeps enough warps in flight that something is always runnable. How well you keep the schedulers fed is what occupancy measures.

Nearly every optimization in this article follows from the warp. Coalescing is about what one warp's 32 addresses look like to the memory system, bank conflicts are about what they look like to shared memory, and tiling is about giving each warp enough arithmetic to chew on while other warps wait.

## Inside the GPU

### Streaming Multiprocessor 
The Streaming Multiprocessor (SM) is the most important piece of hardware to understand. It is comparable to the CPU since the SM is what contains the hardware components that does the execution. The GPU contains multiple SMs, within the RTX 5080 the underlying GB203 die has 96 SMs, but only 84 of the SMs are enabled in the 5080 configuration.

Each SM contains 4 execution partition units or what I'd like to refer as sub-partitions, and each sub-partition containts the CUDA cores, Tensor Core, Load/Store Units, Warp Schedulers, Register File, Shared Memory, L1 Cache. 

In knowing this, you should immediately be aware that a CUDA kernel is not executing on "the GPU" operating as a monolithic processor like the CPU. Instead, work is distributed across all of the SMs within the GPU. 

### Warp Schedulers

The next big question following the introduction of Streaming Multiprocessor should be "how is work distributed across all of the SMs?". 

### Execution Units
CUDA Cores are the general-purpose arithmetic execution units used for many ordinary CUDA operations. 

Tensor Cores, on the other hand, are specialized hardware designed to accelerate matrix multiplication and tensor operations, making them particularly important for deep learning workloads. 

Essentially, both are execution units that receive instructions and perform computations on data. The results of these computations are typically written back to registers, while Load/Store Units are responsible for moving data between registers and the memory hierarchy. CUDA Cores and Tensor Cores are both execution units within the SM, but they are specifized for different types of operations. 

### Load/Store Units 

The Load/Store Units (LSUs) are the hardware components responsible for issuing memory operations that move data between registers and the memory hierarchy. Practically speaking, when CUDA code is compiled into SASS instructions, a load or store instruction specifies the memory address involved and the registers used as the source or destination. For a load, the LSU initiates a request to the memory hierarchy. The memory system first checks the appropriate cache levels, such as L1 and L2. If the data is found in L1, it is returned to the destination register. If there is an L1 cache miss but an L2 cache hit, the data is retrieved from L2 and returned to the register. If both caches miss, the data must ultimately be fetched from device memory (VRAM). Since we are working with a consumer grade GPU, the device memory (VRAM) refers to GDDR7. ON data-center GPUs like the H100, the device memory (VRAM) is HBM.

The next big question to ask is, "how does the SM decide how and what to execute?". The answer to this question is the Warp Schedulers 


:::check
During the forward pass, why do we store every `z` and `a` instead of discarding them?
---
The backward pass reuses them: computing each layer's gradients requires the activations produced during the forward pass, so recomputing them would double the work.
:::

[^1]: [How CUDA Programming Works - Stephen Jones, CUDA Architect](https://www.youtube.com/watch?v=QQceTDjA4f4&t=75s)

