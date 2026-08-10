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

There are additional components within an SM, but the ones listed above are the most important for understanding the performance optimizations we'll make throughout this article.

In knowing this, you should immediately be aware that a CUDA kernel is not executing on "the GPU" operating as a monolithic processor like the CPU. Instead, work is distributed across all of the SMs within the GPU. 

## Execution Units
CUDA Cores are the general-purpose arithmetic execution units used for many ordinary CUDA operations. 

Tensor Cores, on the other hand, are specialized hardware designed to accelerate matrix multiplication and tensor operations, making them particularly important for deep learning workloads. 

Essentially, both are execution units that receive instructions and perform computations on data. The results of these computations are typically written back to registers, while Load/Store Units are responsible for moving data between registers and the memory hierarchy. CUDA Cores and Tensor Cores are both execution units within the SM, but they are specifized for different types of operations. 

## Load/Store Units 

The Load/Store Units (LSUs) are the hardware components responsible for issuing memory operations that move data between registers and the memory hierarchy. 

Practically speaking, when CUDA code is compiled into SASS instructions, a load or store instruction specifies the memory address involved and the registers used as the source or destination. 

For a load, the LSU initiates a request to the memory hierarchy. The memory system first checks the appropriate cache levels, such as L1 and L2. If the data is found in L1, it is returned to the destination register. If there is an L1 cache miss but an L2 cache hit, the data is retrieved from L2 and returned to the register. If both caches miss, the data must ultimately be fetched from device memory (VRAM). 

Since we are working with a consumer grade GPU, the device memory (VRAM) refers to GDDR7. ON data-center GPUs like the H100, the device memory (VRAM) is HBM.

## Warp Schedulers

The next big question following the introduction of Streaming Multiprocessor should be "how is work distributed across all of the SMs?". That job does not belong to the warp schedulers at all. It belongs to the **GigaThread Engine**, the block sitting next to the host interface in the die diagram above, and its whole job is to take the work you launched and hand it out to SMs that have room for it.

The warp schedulers work one level down, after that handout has already happened. Each sub-partition owns one, and it decides, every cycle, which instruction that sub-partition issues next. We have not defined a warp yet, so we will come back to what these actually do once the execution model is on the table.

## Memory Hierarchy

The LSU walks a ladder of memories, and that ladder is the single most important thing to hold in your head for the rest of this article. Each level trades capacity for latency, and each one is visible to a different scope of the program:

![Every level down costs an order of magnitude more, the key is to stay higher memory region for as long as possible.](/images/gemm/memory-hierarchy.png)

Registers are private to a single thread and are the only storage the execution units read operands from directly. They are also finite: the register file is a fixed budget — physically one per sub-partition, as we saw above — split across every thread resident on it, so a kernel that asks for more registers per thread gets fewer threads resident at once.

Shared memory sits one step down. It is carved out of the same physical storage as L1 and it is private to a thread block, which makes it the one level the programmer manages by hand — you decide what goes in it and when. This is the level the entire optimization ladder is built around: if a value is going to be read many times by many threads, you want it in shared memory, read from device memory exactly once.

L2 is shared by every SM on the die and is managed entirely by the hardware. You do not place data in it, but you can still influence your hit rate by controlling which blocks touch which addresses at roughly the same time.

Device memory is the bottom of the ladder and the top of the capacity chart. It is where your matrices live when the kernel launches, it is an order of magnitude slower than anything above it, and every optimization in this article is ultimately an argument about how few times you can afford to touch it.

## CUDA Programming Model

That is the hardware half of the co-design. The other half is what you actually write.

The CUDA programming model is the abstraction you write to harness the powerful GPU you have on hand. CUDA exposes a hierarchical model enabling CUDA practioners to express parallel execution through the division of grid, blocks, threads. 

A grid is the complete collection of blocks. A block is a group of threads that execute and cooperate with one another. And a thread is the smallest unit of execution exposed by the CUDA programming model. Every thread in the grid runs the same kernel body, but each one runs it over its own piece of the data. 

![Zooming in one level at a time: the grid is every block, a block is a group of threads that share memory and can synchronize, and a thread is the smallest unit of execution.](/images/gemm/programming-model.png)

Which raises the obvious question. If every thread is handed the same code, what actually makes thread 3 do something different from thread 4?

Its position, and nothing else. CUDA hands every thread four built-ins to read that position with. `threadIdx` and `blockIdx` tell a thread where it sits — which thread it is inside its block, and which block that is inside the grid. `blockDim` and `gridDim` tell it the shape of the two things it sits in.

A thread turns that position into a piece of work with a single line of arithmetic:

```cuda
const int row = blockIdx.y * blockDim.y + threadIdx.y;
```

Identical source text in all of them, a different value in each. That one line is the entire mechanism, and it is easier to watch than to describe — so let's zoom in on a single block.

![Zooming in one more level. Every thread in the block issues the same instructions; the index it computes from those built-ins is the only thing that sends it to a different element of the data.](/images/gemm/inside-thread-block.png)

The diagram draws eight threads because eight of them fit on a page, but a real block is hundreds. What matters is that every one of them is running the same instruction stream. There is no per-thread program. There is one kernel body, and the index each thread computes is the only thing that sends it somewhere different in memory. This is what people mean when they call CUDA an **SIMT** model: single instructions, multiple threads.

Look at what that buys you in the line we just wrote. `blockIdx.y * blockDim.y + threadIdx.y` is identical source text in every thread, but `threadIdx.y` holds a different value in each one, so every thread lands on its own `row`. You never write a loop over the rows. The loop is the launch itself, and if you want to cover twice as much of the matrix you launch twice as many threads without touching the kernel.

The last line of the diagram is what makes a block worth having as a concept. Threads in the same block can hand data to each other through shared memory, and they can line up at a `__syncthreads()` barrier, where no thread moves past it until every thread in the block has reached it. Threads in two *different* blocks get neither. Our naive kernel will use neither, and every optimization from shared memory tiling onward is built on both — which is why the block, not the thread, ends up being the unit you do most of your thinking in.

## How the Program Maps to the Hardware

Now that we have established what the GPU hardware does and what the CUDA software is, we next question to ask is which piece of the software abstraction (grid, blocks, threads) lands on which piece of silicon. 

An SM can host multiple blocks at the same time. The amount of blocks that's resident on the SM is not a number you choose. Now, a block can hold at most 1024 threads at a time, and each Block is assigned to exactly one SM and it stays resident until every thread has finished its execution. During execution, the threads within the block can write to the on-chip shared memory. Each thread within a block can also write to a register within the register file as well. 

What the mapping does not yet explain is what an SM does with a block once it has one. It does not run 1024 threads independently. It divides the block into **warps**.

## CUDA Execution Model

On the software level, the thread is the smallest unit of execution. But on the hardware level, the smallest unit of execution is the **warp**: a group of 32 threads that the SM issues instructions for as one. A block is made up of multiple warps — a 1024-thread block is 32 of them.

![The SM never schedules a thread on its own. It schedules one of these.](/images/gemm/block-splits-into-warps.png)

You never declare a warp anywhere in your code. The hardware forms them for you by walking the block's threads in linear order — `threadIdx.x` first, then `y`, then `z` — and cutting every 32. Threads 0 through 31 become warp 0, threads 32 through 63 become warp 1, and so on down the block.

Which also means a block whose size is not a multiple of 32 still pays for whole warps. Launch 1000 threads and you do not get 31.25 warps, you get 32, and the last one runs with 8 lanes doing work and 24 doing nothing at all. This is the first place the number 32 shows up in this article, and it is why the block dimensions in real kernels are almost always a multiple of it.

When a block arrives at an SM, the SM splits it into warps and schedules those. A warp issues one instruction at a time on behalf of all 32 of its threads, so they advance in lockstep. When a branch sends them different ways, the warp walks each path in turn with the threads that took the other path switched off, so the two paths are serialized rather than run in parallel. This is why divergence inside a warp costs real time, and why it costs nothing when an entire warp takes the same branch.

![The if-path and the else-path cannot run at the same time, so a divergent branch costs you the sum of both.](/images/gemm/warp-divergence.png)

The qualifier that matters here is *inside a warp*. Divergence is a warp-level cost, not a program-level one. If warp 0 takes the if-path and warp 1 takes the else-path, nothing is serialized — they are separate scheduling units, and each one runs its own branch at full width. The expensive case is only when the condition comes out differently across the 32 lanes of a single warp.

That distinction is worth carrying into the kernel we are about to write. A bounds guard like `if (row < M && col < N)` looks like a branch sitting on every thread, but when the matrix dimensions are multiples of the block size, every lane in a warp evaluates it identically and it costs a comparison and nothing more. Only the warps hanging off the edge of the matrix actually diverge, and there are very few of those.

The other half of the model is latency hiding. Many warps sit resident on an SM at once, and the warp scheduler issues each cycle from whichever ones are ready. When one warp stalls waiting on a global memory load, the scheduler simply issues from another. A CPU spends enormous transistor budget on caches and out-of-order execution to avoid stalling; a GPU accepts the stall and keeps enough warps in flight that something is always runnable.

![No single warp runs faster here — the memory latency is the same. The SM just always has some other warp ready to issue.](/images/gemm/latency-hiding.png)

The scale is what makes this work at all. A load that misses every cache and goes out to device memory takes many hundreds of cycles, while the FMA waiting on it takes a handful. Nothing you do inside one warp closes a gap that wide. The only thing that closes it is having other warps ready to issue in the meantime.

And switching between them is free, which is the part worth sitting with. A CPU context switch has to save and restore registers, so it costs real time. A GPU never does that, because every resident warp already owns its registers in the register file for as long as it lives on the SM. The scheduler picks a different warp and issues on the next cycle. That is why the register file is so enormous, and it is also why asking for too many registers per thread hurts you — it does not make any individual thread slower, it means fewer warps fit on the SM, and warps are the only mechanism you have for covering memory latency.

This is what occupancy, which we met a moment ago, is actually measuring: how many warps are resident on an SM relative to the maximum it could hold. Note what that does and does not tell you. It is the *capacity* to hide latency, not proof that you are hiding any — which is why chasing occupancy on its own is not the same thing as chasing performance.[^4] We will run into that distinction directly once register pressure starts costing us residency.

Nearly every optimization in this article follows from the warp. Coalescing is about what one warp's 32 addresses look like to the memory system, bank conflicts are about what they look like to shared memory, and tiling is about giving each warp enough arithmetic to chew on while other warps wait.

This section of the article covers the most bare-bones aspects of CUDA programming and CUDA hardware architecture. It should be just enough to get us started with writing the actual naive kernel. However, if you want to dive deeper into what happens when you run a CUDA kernel, check out Fergus Finn's blog.[^5]

[^1]: [How CUDA Programming Works - Stephen Jones, CUDA Architect](https://www.youtube.com/watch?v=QQceTDjA4f4&t=75s)
[^2]: [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
[^3]: [Programming Massively Parallel Processors: A Hands-on Approach (5th Edition)](https://www.amazon.com/dp/0443439001)
[^4]: [Understanding Latency Hiding on GPUs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/Archive/EECS-2016-143.pdf)
[^5]: [What happens when you run a CUDA kernel](https://fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/#an-interposition-hook)


