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

Tensor Cores, on the other hand, are specialized hardware designed to accelerate matrix multiplication and tensor operations, making them particularly important for deep learning workloads. 

Essentially, both are execution units that receive instructions and perform computations on data. The results of these computations are typically written back to registers, while Load/Store Units are responsible for moving data between registers and the memory hierarchy. CUDA Cores and Tensor Cores are both execution units within the SM, but they are specifized for different types of operations. 

## Load/Store Units 

The Load/Store Units (LSUs) are the hardware components responsible for issuing memory operations that move data between registers and the memory hierarchy.
Practically speaking, when CUDA code is compiled into SASS instructions through the nvcc compiler, there exists either a load or store instruction specifying the memory address involved and the registers used as the source or destination.

For a **load instruction**, the LSU initiates a request to the memory hierarchy. The memory system first checks the L1 cache. If the data is found within the L1 cache (L1 cache hit), the data is returned to the requesting thread. If the data is not found in (L1 cache miss), the request proceeds to L2 cache. If the data is found in L2, it is retrieved and returned to the thread. If not, this implies that there are two cache misses, where the data must ultimately be fetched from the device memory.

Since we are working with a consumer grade GPU, the device memory (VRAM) refers to GDDR7. In a data-center the GPUs such as the H100s’ (VRAM) is HBM.

Now for a **store instruction**, the process is very similar to the load instruction. The LSU first initiates a request to write data from the register to the memory hierarchy. How the data is stored is dependent entirely on the cache policy thats associated with the store instruction. 

For example, a write-back store instruction can cache data at multiple coherent cache level, but a cache-at-global-level store instruction will bypass the L1 and cache the data straight into the L2. Again, it all depends on the cache policy and the type of memory access performed by the store instruction. 

We won't cover the details of cache operators here but if you want to understand how store instructions interact with the cache hierarchy, I recommend reading NVIDIA's Cache Operators for Memory Store Instructions section in the PTX ISA documentation[^6]. Its just beyond the scope of this blog. 

## Warp Schedulers

Now a linger question you should have in the back of your head should be: "How does work actually get distributed across all of the SMs?". 

This responsibility does not actually belong to the warp scheduler, instead it belongs to the **GigaThread Engine**. Its whole job is to take the work you launched with your kernel and hand it to an avaliable SMs. 

Once a block is assigned to a SM by the GigaThread Engine, the threads within the block is divided into 32 threads groups called a warp.  At this point, know that the GigaThread engine determines which SM execuates a block, and the warp scheduler determines which warp executes an instruction next. 

As stated earlier in this chapter, each SM contains 4 sub-partitions, and each sub-partition contains one **warp scheduler**. During each clock cycle, the warp scheduler selects an eligble warp and issues its next instructions to the execution units available to that sub-partition.   

We will cover more detail when we introduce the CUDA execution model later.

## Memory Hierarchy

The GPU memory hierarchy is also a very important concept to understand when it comes to optimizations. As you move down the hierarchy, the capacity of memory and latency both increases. 

At the highest level of the memory hierarchy, the registers are private to each thread which provides teh fasts look-up/storage. Shared memory is shared within a block which gives the programmer explicit control over frequently used data. L2 cache is shared across all SMs and is managed automatically by the hardware.The device memory is the largest memory but has the highest latency. 

![Every level down costs an order of magnitude more, the key is to stay higher memory region for as long as possible.](/images/gemm/memory-hierarchy.png)

The goal of optimizations is to keep data as high in the memory hierarchy as possible and aim for data resue before accessing the slower memory for the same data. 

## CUDA Programming Model

Everything discussed above is around the CUDA hardware, the first half of the CUDA co-design. The second half of the CUDA co-design is what you actually write, the CUDA software.

The CUDA programming model is the software abstraction you write to harness the powerful GPU you have on hand. The CUDA Programming Model exposes a hierarchical model enabling CUDA practioners to express parallel execution through the division of grid, blocks, threads. 

A grid is the complete collection of blocks. A block is a group of threads that execute and can cooperate with one another. And a thread is the smallest unit of execution exposed by the CUDA programming model. Every thread within a block executes the same instruction but just with different data. 

![Zooming in one level at a time: the grid is every block, a block is a group of threads that share memory and can synchronize, and a thread is the smallest unit of execution.](/images/gemm/programming-model.png)

Now, if every thread is executes the same instruction, what makes a thread within a warp different from every other threads? 

The answer is the threads position. CUDA has 4 built-in APIs that helps determine the position of a thread. Each API also has 3 possible variables `x`, `y`, or  `z` which determines the dimension direction. Writing CUDA kernels within the context of deep learning generally takes place around the `x` and `y` plane, within the 2-dimensional therefore the `z` variable is rarely used. 

The `threadIdx` and `blockIdx` determines where a thread sits within its block and the block's position within the grid. `blockDim` and `gridDim` determines the shape of the block or grid, respectively.

![Zooming in one more level. Every thread in the block issues the same instructions; the index it computes from those built-ins is the only thing that sends it to a different element of the data.](/images/gemm/inside-thread-block.png)

```cuda
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;
```

The diagram above has eight threads shown because that's all I could reasonably fit on a page, but a real block can hold upwards of **1024 threads**. What matters is that every thread executes the same instructions. There is no separate program for each thread; there is one kernel body, and each thread's unqiue index determines which data it accesses. This is what people mean when they call CUDA an **SIMT** model: single instructions, multiple threads.

Also understand that the threads in the same block can hand data to each other through shared memory and they can line up at a `__syncthreads()` barrier, where no thread moves past it until every thread in the block has reached it. Note that this only applies to threads that share the same block. Note that threads in two *different* blocks do not share the same shared memory and cannot `__syncthreads()`. This isn't a software restriction CUDA is choosing to enforce - it's physical. A block is assigned to exactly one SM and stays resident there until every one of its threads finishes, so the shared memory it writes to and the barrier it synchronizes on are both hardware sitting on that specific SM. A block on a different SM has no way to reach either one.

In our first GEMM kernel, the naive kernel implementation will use neither, but every optimization moving forward, starting with shared memory tiling, is built on both ideas. The key insight is to recognize that the block, not the thread, ends up being the unit you do most of your thinking in, which ultimately dictates how you write your kernel.

## CUDA Execution Model

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

## The Roofline Model

The roofline model is a way to guage the performance of a kernel relative to the hardware the kernel is running on. It is the tool this article uses throughout each page decide what to optimize next.

The idea is that a kernel can be bounded by compute, memory, or some overhead[^1]. It is often limited compute or memory, but never by both at once. 

The first limitation is arithmetic. The RTX 5080 can at most perform 56.3 TFLOP/s of FP32, and no kernel can exceed this no matter how little memory it touches. 

The second limitation is bandwidth. The VRAM delivers at most 960 GB/s, so a kernel that needs a lot of bytes per unit of arithmetic runs out of bandwidth long before it runs out of FP32 units.

To determine if you're compute bound or memory bound is entirely dependent on the ratio of floating point operations per byte; and that ratio is called **arithmetic intensity**. It is the number of floating point operations a kernel performs for every byte it moves.

``` latex
P_{\text{attainable}} =
\min\left(P_{\text{peak}},\ AI \times BW\right)
```

Plotting that gives the shape the model is named for. Performance rises along a slope as intensity increases, because a kernel with more arithmetic per byte gets more out of the same bandwidth, and then it flattens into a roof once the FP32 units saturate. The two pieces meet at one intensity, the **ridge point**, where the machine's arithmetic and its bandwidth are exactly in balance:

``` latex
AI_{\text{ridge}} =
\frac{56.3\text{ TFLOP/s}}
{960\text{ GB/s}}
\approx 58.6\text{ FLOP/byte}
```

![The roofline for the RTX 5080, with arithmetic intensity in FLOP per byte on the x axis and attainable performance in TFLOP/s on the y axis, both log scale. The sloped line rises at 960 GB/s until it meets the flat 56.3 TFLOP/s ceiling at the ridge point of 58.6 FLOP/byte. Everything left of the ridge is the memory bound region, where DRAM is the limit and extra FLOPs are free, and the way out is to raise intensity by reusing data through shared memory and register tiling. Everything right of it is the compute bound region, where the SMs are the limit and only more FLOP/s helps.](/images/gemm/roofline-model.png "full")

So on this card a kernel has to perform roughly 59 floating point operations for every byte it pulls from VRAM just to keep the arithmetic units fed. Put in terms of the FP32 numbers we are actually multiplying, that is about 234 operations for every 4 byte float loaded. A kernel below that intensity is **memory bound** and its ceiling is the sloped one; a kernel above it is **compute bound** and its ceiling is flat. Note that intensity is a property of the kernel and the ridge point is a property of the card, which means the ridge point is fixed and the only thing we get to move is the kernel.

It is worth knowing where this card sits. 58.6 FLOP/byte is a demanding ridge point, because the 5080 pairs a great deal of FP32 throughput with a fairly narrow 256 bit memory bus. A GPU with more bandwidth per FLOP has a lower ridge and forgives a sloppier kernel. That is also why performance numbers from an article written on a different card do not transfer to this one, and why every figure in this article was measured here rather than copied.

One caveat to carry forward, because we will run into it almost immediately. The bandwidth in the model is VRAM bandwidth, so the model quietly assumes every byte a kernel asks for is a byte that travels all the way from VRAM. Caches break that assumption. A value that is still sitting in L2 costs a fraction of what the model charges for it, so a kernel can measure *faster* than its own roofline says is possible. When that happens it is not a broken measurement, it is the model telling you that the traffic you counted is not the traffic that actually reached memory. That distinction turns out to be the difference between the first optimization in this article working and the second one being necessary.

## What is GEMM

Before we actually begin writing our GEMM kernels and go through an iterative process on optimizing it, let's first establish what a General Matrix Multiplication (GEMM) is so that we understand what's going on behind the scenes with every implementation. Every kernel in this article computes the exact same thing; an output matrix C computed by matrices A and B. The **GEMM** follow the same mathematical formula listed below:

``` latex
C \leftarrow \alpha A B + \beta C
```

Note that variables alpha and beta are just scalar coefficients that scale the matrix multiplication and the existing output matrix C.

**A is (M x K)**, **B is (K x N)**, and **C is (M x N)**. All three of these matrices are stored in row-major order in memory. Each optimized kernel thats introduced within the article will use the following dimensions: **M = N = K = 4096**, **alpha = 1**, **beta = 0**. 

This ensures that very kernel performs the same 2 * M * N * K floating point operations, which is roughly 137 billion operation in total, against the same three matrices. It's really just a basic matrix multiplication.

[^1]: [How CUDA Programming Works - Stephen Jones, CUDA Architect](https://www.youtube.com/watch?v=QQceTDjA4f4&t=75s)
[^2]: [NVIDIA RTX Blackwell GPU Architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
[^3]: [Programming Massively Parallel Processors: A Hands-on Approach (5th Edition)](https://www.amazon.com/dp/0443439001)
[^4]: [Understanding Latency Hiding on GPUs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/Archive/EECS-2016-143.pdf)
[^5]: [What happens when you run a CUDA kernel](https://fergusfinn.com/blog/what-happens-when-you-run-a-gpu-kernel/#an-interposition-hook)
[^6]: [NVIDIA's PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=Cache#cache-operators)

