## From GPU Hardware to High-Performance Kernels

### What is CUDA?
CUDA is an acronym for "compute unified device architecture", which essentially describes a wide array of computing units that can execute in parallel. However, depending on the context CUDA, it can either be describing a high-level device architecture or a parallel programming model for architectures with the CUDA design. 

The key insight is to understand that CUDA is a co-design between the hardware and the software.[^1] Writing good CUDA requires a solid fundamental understanding of the CUDA memory hierarchy and how it relates to the hardware. 

In a moden NVIDIA GPU, the most important hardware components relevant to optimizng a GEMM is the GPU die itself, the Streaming Multiprocessors (SM), everything inside the SM, L2 Cache, and the high bandwidth memory (HBM).

The exact organization and varies between different NVIDIA architectures, but these hardware components exists throughout all modern NVIDIA GPUs. For this article, we will be referencing the Blackwell Architecture, specifically the consumer grade GPU RTX 5080. 

(Insert image of GPU Architecture) 

The Streaming Multiprocessor (SM) is the most important piece of hardware to understand. It is comparable to the CPU since the SM is what contains the hardware components that does the execution. The GPU contains multiple SMs, within the 5080 there are 96 SMs. Each SM contains 4 execution partition units or what I'd like to refer as sub-partitions, and each sub-partition containts the CUDA cores, Tensor Core, Load/Store Units, Warp Schedulers, Register File, Shared Memory, L1 Cache. 

In knowing this, you should immediately be aware that a CUDA kernel is not executing on "the GPU" operating as a monolithic processor like the CPU. Instead, work is distributed across all of the SMs within the GPU. 

## Execution Units
CUDA Cores are the general-purpose arithmetic execution units used for many ordinary CUDA operations. 

Tensor Cores, on the other hand, are specialized hardware designed to accelerate matrix multiplication and tensor operations, making them particularly important for deep learning workloads. 

Essentially, both are execution units that receive instructions and perform computations on data. The results of these computations are typically written back to registers, while Load/Store Units are responsible for moving data between registers and the memory hierarchy. CUDA Cores and Tensor Cores are both execution units within the SM, but they are specifized for different types of operations. 

## Load/Store Units 

The Load/Store Units (LSUs) are the hardware components responsible for issuing memory operations that move data between registers and the memory hierarchy. Practically speaking, when CUDA code is compiled into SASS instructions, a load or store instruction specifies the memory address involved and the registers used as the source or destination. For a load, the LSU initiates a request to the memory hierarchy. The memory system first checks the appropriate cache levels, such as L1 and L2. If the data is found in L1, it is returned to the destination register. If there is an L1 cache miss but an L2 cache hit, the data is retrieved from L2 and returned to the register. If both caches miss, the data must ultimately be fetched from device memory (VRAM). Since we are working with a consumer grade GPU, the device memory (VRAM) refers to GDDR7. ON data-center GPUs like the H100, the device memory (VRAM) is HBM.

## Warp Schedulers



The next big question to ask is, "how does the SM decide how and what to execute?". The answer to this question is the Warp Schedulers 






:::check
During the forward pass, why do we store every `z` and `a` instead of discarding them?
---
The backward pass reuses them: computing each layer's gradients requires the activations produced during the forward pass, so recomputing them would double the work.
:::

[^1]: [How CUDA Programming Works - Stephen Jones, CUDA Architect](https://www.youtube.com/watch?v=QQceTDjA4f4&t=75s)

