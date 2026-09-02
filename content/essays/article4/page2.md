## From CUDA Hardware Architecture To CUDA Kernels

## What is CUDA?
CUDA stands for Compute Unified Device Architecture. Depending on the context, CUDA can refer to NVIDIA's parallel computing platform and programming model, or more broadly to the ecosystem of hardware and software built around it. The CUDA programming model provides an abstraction for CUDA practioners to express massively parallel workloads, while NVIDIA's GPU architectures provide the underlying hardware that executes those workloads.

The key insight is to understand that CUDA is a co-design between the hardware and the software.[^1] Writing good CUDA requires a solid fundamental understanding of the CUDA programming model and how it relates to the GPU hardware architecture.

## GPU Hardware

In a modern NVIDIA GPU, the most important hardware components to understand for optimizing GEMM are the **GPU die, Streaming Multiprocessors (SMs), the hardware within each SM, the L2 cache, and the GPU’s high-bandwidth memory subsystem**.

![The GB203 die behind the RTX 5080: 7 GPCs, 42 TPCs, 84 SMs, a 64 MB unified L2 shared by every GPC, and eight 32-bit GDDR7 controllers.](/images/gemm/rtx-5080-die.png)

The exact organization of these components varies across NVIDIA GPU architectures, but the fundamental concepts remain consistent across modern NVIDIA GPUs. Throughout this article, we will use the **Blackwell architecture**, specifically the consumer-grade **GeForce RTX 5080**, as our reference point.[^2]

> Everything under the hardware section is throughly covered in *Programming Massively Parallel Processors*, basically chapters 1 through 4 of that book.[^3]

## Streaming Multiprocessor
The **Streaming Multiprocessor (SM)** is the most important piece of hardware to understand. You can think of an SM as a small, self-contained compute unit within the GPU: it contains the hardware responsible for scheduling and executing threads.

A GPU contains multiple SMs. In the RTX 5080, the underlying GB203 die contains **96 SMs**, but only **84 SMs are enabled** in the RTX 5080 configuration.

Each SM is divided into four execution partitions, which we'll refer to as SM sub-partitions. Each sub-partition contains its own warp scheduler, instruction dispatch units, register file, CUDA cores, Tensor Cores, and load/store units. The SM also contains other important resources, including shared memory and L1 cache, which are shared across the SM's sub-partitions.

<!--
![One SM of the RTX 5080. Each of the four sub-partitions owns its own warp scheduler and register file. The 128 KB of L1 and shared memory along the bottom is the one resource all four have to share.](/images/gemm/sm-blackwell.png)
-->

There are additional components within an SM, but the ones listed above are the most important for understanding the performance optimizations we'll make throughout this article.

In knowing this, you should immediately be aware that a CUDA kernel is not executing on "the GPU" operating as a monolithic processor like the CPU. Instead, work is distributed across all of the SMs within the GPU. 

## Execution Units
CUDA Cores are the general-purpose arithmetic execution units used for many ordinary CUDA operations. 

Tensor Cores on the other hand, are specialized hardware also for arithmetic execution, but specifically designed to accelerate matrix multiplication and tensor operations. This makes tensor cores remarkably performant, which is important for optimizing deep learning workloads. 

Both are execution units that receive instructions and perform computations on data. The results of these computations are typically written back to registers. And while Load/Store Units are responsible for moving data between registers and the memory hierarchy, CUDA Cores and Tensor Cores are both execution units within the SM, but they are specifized for different types of operations. 

## Load/Store Units 

The Load/Store Units (LSUs) are the hardware components responsible for issuing memory operations that move data between registers and the memory hierarchy.
Practically speaking, when CUDA code is compiled into SASS instructions through the nvcc compiler, there exists either a load or store instruction specifying the memory address involved and the registers used as the source or destination.

For a **load instruction**, the LSU initiates a request to the memory hierarchy. The memory system first checks the L1 cache. If the data is found within the L1 cache (L1 cache hit), the data is returned to the requesting thread. If the data is not found in (L1 cache miss), the request proceeds to L2 cache. If the data is found in L2, it is retrieved and returned to the thread. If not, this implies that there are two cache misses, where the data must ultimately be fetched from the device memory.

Since we are working with a consumer grade GPU, the device memory (VRAM) refers to GDDR7. In a data-center the GPUs such as the H100s’ (VRAM) is HBM.

Now for a **store instruction**, the process is very similar to the load instruction. The LSU first initiates a request to write data from the register to the memory hierarchy. How the data is stored is dependent entirely on the cache policy thats associated with the store instruction. 

For example, a write-back store instruction can cache data at multiple coherent cache level, but a cache-at-global-level store instruction will bypass the L1 and cache the data straight into the L2. Again, it all depends on the cache policy and the type of memory access performed by the store instruction. 

We won't cover the details of cache operators here but if you want to understand how store instructions interact with the cache hierarchy, I recommend reading NVIDIA's Cache Operators for Memory Store Instructions section in the PTX ISA documentation[^6]. Its just beyond the scope of this blog. 

## Warp Schedulers

Now a larger question you should have in the back of your head should be: "How does work actually get distributed across all of the SMs?". 

This responsibility is on the warp scheduler, but it starts with the **GigaThread Engine**. The **GigaThread Engine** sits on the GPU die and governs all of the SMs. Its whole job is to take the work you launched with your kernel and hand it to an avaliable SMs. 

Once a block is assigned to a SM by the GigaThread Engine, the threads within the whole block is divided into groups of 32 threads called a warp. At this point, know that the GigaThread engine determines which SM execuates a block, and the warp scheduler determines which warp executes an instruction next. 

As stated earlier in this chapter, each SM contains 4 sub-partitions, and each sub-partition contains one **warp scheduler**. So during each clock cycle, the warp scheduler selects an eligble warp and issues its next instructions to the execution units available to a sub-partition.   

We will cover more detail when we introduce the CUDA execution model later.

## Memory Hierarchy

The GPU memory hierarchy is also a very important concept to understand when it comes to optimizations. As you move down the hierarchy, the capacity of memory increases but that also increases and latency to move data around. 

At the highest level of the memory hierarchy, the registers are private to each thread which provides the fasts look-up/storage. The next level down is Shared Memory whichis a block of memory that the CUDA practioner has explicit over usually to store frequently used data. Going a level deeper is the L2 cache, is shared across all SMs and is managed automatically by the hardware. At the bottom most level is the VRAM, device memory is the largest memory but has the highest latency. 

![Every level down costs an order of magnitude more, the key is to stay higher memory region for as long as possible.](/images/gemm/memory-hierarchy.png)

The goal of optimizations is to keep data as high in the memory hierarchy as possible and aim for data resue before accessing the slower memory for the same data. 

## CUDA Programming Model

Everything discussed above is around the CUDA hardware, the first half of the CUDA co-design. The second half of the CUDA co-design is what you actually write, the CUDA software.

The CUDA programming model is the software abstraction you write to harness the powerful GPU you have on hand. The CUDA Programming Model exposes a hierarchical model enabling CUDA practioners to express parallel execution through the division of grid, blocks, threads. 

A grid is the complete collection of blocks. A block is a group of threads that executes instructions and have the capacity to cooperate with one another. And a thread is the smallest unit of execution exposed by the CUDA programming model. Every thread within a block executes the same instruction but just with different data. When writing a CUDA kernel, we are writing the instructions for a single thread. 

![Diagram of the CUDA Programming Model](/images/gemm/programming-model.png)

Now, if every thread is executes the same instruction, what makes a thread within a warp different from every other threads? The answer is the threads position. 

The CUDA programming model has 4 built-in APIs that helps determine the position of a thread. Each API also has 3 possible variables `x`, `y`, or  `z` which determines the dimension direction. Writing CUDA kernels within the context of deep learning generally takes place around the `x` and `y` plane, within the 2-dimensional therefore the `z` variable is rarely used. 

The `threadIdx` and `blockIdx` determines where a thread sits within its block and the block's position within the grid. `blockDim` and `gridDim` determines the shape of the block or grid, respectively.

![Zooming in one more level. Every thread in the block issues the same instructions; the index it computes from those built-ins is the only thing that sends it to a different element of the data.](/images/gemm/inside-thread-block.png)

```cuda
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;
```

The diagram above has eight threads shown because that's all I could reasonably fit on a page, but in a real kernel a block can hold upwards of **1024 threads**. Understand that every thread executes the same instructions such that there is no separate program for each thread. There is one kernel body that, and each thread's unqiue index determines which data it accesses. This is what people mean when they call CUDA an **SIMT** model: single instructions, multiple threads.

Threads in the same block can hand data to each other through shared memory and they can line up at a `__syncthreads()` barrier, where no thread moves past it until every thread in the block has reached it. Note that this only applies to threads that share the same block. Threads in *different* blocks do not share the same shared memory and cannot `__syncthreads()`. A block is assigned to exactly one SM and stays resident there until every one of the warps finishes executing, so the shared memory it writes to and the barrier it synchronizes on are both hardware sitting on that specific SM. 

## CUDA Execution Model

In the previous section we briefly dived into the CUDA programming model, that was introduction to the software abstraction to structure parallel code. Now we will dive into the CUDA exeuction model, the governing principles that determine how GPU hardware maps, schedules, and runs the software abstractions. 

On the software level, the thread is the smallest unit of execution. But on the hardware level, the smallest unit of execution is the **warp**: a group of 32 threads that the SM issues instructions for as one. A block is made up of multiple warps - a 1024-thread block is 32 of them.

![The SM never schedules a thread on its own. It schedules one of these.](/images/gemm/block-splits-into-warps.png)

You never declare a warp anywhere in your code. The hardware forms them for you by walking the block's threads in linear order - `threadIdx.x` first, then `y`, then `z` - and cutting every 32. Threads 0 through 31 become warp 0, threads 32 through 63 become warp 1, and so on down the block.

Which also means a block whose size is not a multiple of 32 still pays for whole warps. Launch 1000 threads and you do not get 31.25 warps, you get 32, and the last one runs with 8 lanes doing work and 24 doing nothing at all. This is the first place the number 32 shows up in this article, and it is why the block dimensions in real kernels are almost always a multiple of it.

When a block arrives at an SM, the SM splits it into warps and schedules those. A warp issues one instruction at a time on behalf of all 32 of its threads, so they advance in lockstep. When a branch sends them different ways, the warp walks each path in turn with the threads that took the other path switched off, so the two paths are serialized rather than run in parallel. This is why divergence inside a warp costs real time, and why it costs nothing when an entire warp takes the same branch.

![The if-path and the else-path cannot run at the same time, so a divergent branch costs you the sum of both.](/images/gemm/warp-divergence.png)

The qualifier that matters here is *inside a warp*. Divergence is a warp-level cost, not a program-level one. If warp 0 takes the if-path and warp 1 takes the else-path, nothing is serialized - they are separate scheduling units, and each one runs its own branch at full width. The expensive case is only when the condition comes out differently across the 32 lanes of a single warp.

That distinction is worth carrying into the kernel we are about to write. A bounds guard like `if (row < M && col < N)` looks like a branch sitting on every thread, but when the matrix dimensions are multiples of the block size, every lane in a warp evaluates it identically and it costs a comparison and nothing more. Only the warps hanging off the edge of the matrix actually diverge, and there are very few of those.

The other half of the model is latency hiding. Many warps sit resident on an SM at once, and the warp scheduler issues each cycle from whichever ones are ready. When one warp stalls waiting on a global memory load, the scheduler simply issues from another. A CPU spends enormous transistor budget on caches and out-of-order execution to avoid stalling; a GPU accepts the stall and keeps enough warps in flight that something is always runnable.

![No single warp runs faster here - the memory latency is the same. The SM just always has some other warp ready to issue.](/images/gemm/latency-hiding.png)

The scale is what makes this work at all. A load that misses every cache and goes out to device memory takes many hundreds of cycles, while the FMA waiting on it takes a handful. Nothing you do inside one warp closes a gap that wide. The only thing that closes it is having other warps ready to issue in the meantime.

And switching between them is free, which is the part worth sitting with. A CPU context switch has to save and restore registers, so it costs real time. A GPU never does that, because every resident warp already owns its registers in the register file for as long as it lives on the SM. The scheduler picks a different warp and issues on the next cycle. That is why the register file is so enormous, and it is also why asking for too many registers per thread hurts you - it does not make any individual thread slower, it means fewer warps fit on the SM, and warps are the only mechanism you have for covering memory latency.

How many warps can be resident on an SM at once is not a number you choose - it falls out of what each block asks for. Registers per thread come out of that SM's register file, shared memory per block comes out of that SM's shared memory, and no block may exceed 1024 threads. Ask for more of any one of them and fewer blocks, and so fewer warps, fit. This ratio - how many warps are resident relative to the maximum the SM could hold - is what's called **occupancy**. Note what that does and does not tell you. It is the *capacity* to hide latency, not proof that you are hiding any - which is why chasing occupancy on its own is not the same thing as chasing performance.[^4] We will run into that distinction directly once register pressure starts costing us residency.

Nearly every optimization in this article follows from the warp. Coalescing is about what one warp's 32 addresses look like to the memory system, bank conflicts are about what they look like to shared memory, and tiling is about giving each warp enough arithmetic to chew on while other warps wait.

This section of the article covers the most bare-bones aspects of CUDA programming and CUDA hardware architecture. It should be just enough to follow every kernel that comes after it. However, if you want to dive deeper into what happens when you run a CUDA kernel, check out Fergus Finn's blog.[^5]

[^1]: [How CUDA Programming Works - Stephen Jones, CUDA Architect](https://www.youtube.com/watch?v=QQceTDjA4f4&t=75s)
[^2]: [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
[^3]: [Programming Massively Parallel Processors: A Hands-on Approach (5th Edition)](https://www.amazon.com/dp/0443439001)
[^4]: [Understanding Latency Hiding on GPUs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/Archive/EECS-2016-143.pdf)
[^5]: [What happens when you run a CUDA kernel](https://fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/#an-interposition-hook)
[^6]: [NVIDIA's PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=Cache#cache-operators)
