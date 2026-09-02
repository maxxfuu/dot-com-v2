## Global Memory Coalescing

From the naive kernel implementation, we have established that the kernel loads the right values in a inefficient order. Right now the way we are assigning the global thread index to the row and col variables will result in all of the threads accessing data from the global memory in a non-contiguous manner. 

From a warp perspective of matrix A, each thread accesses a different row in memory during 1 lockstep. A thread would access `A[0][k]`, the next thread will access `A[1][k]`, and the next would be `A[2][k]`; each thread within the warp accesses a new row that's K x 4-bytes apart. Thats roughly 16 KB of data. 

From a thread perspective of matrix A, a thread walks along its row and accesses a new element. Suppose thread 0's first iteration accesses `A[0][0]`, the second iteration the thread will access `A[0][1]`. Each time the warp goes through a new iteration it fetches a new element, and each of those fetches pulls in a whole 32 byte sector from the VRAM only to use a 4 byte float. 

With matrix A, we can conclude that each thread fetching 32 bytes from the VRAM just to use 4 bytes of memory per iteration is wasteful. To enable global memory coalescing we can swap the global thread index assignment of the row and col variables so that `threadIdx.x` maps to col, making col the fastest changing index across a warp. 

Changing the global thread index between the row and col would change how A is accessed. Instead of 32 threads accessing 32 separate rows, all 32 threads will access the same row and same element within a single lockstep. As a result, the hardware serves that one address out of a single sector and broadcasts the value to all 32 lanes/registers. 

![Left: the naive mapping puts threadIdx.x on row, so a warp's 32 lanes land on 32 different rows of A, each K·4 bytes apart in memory - nothing to merge into one transaction. Right: swapping the mapping puts threadIdx.x on col, so all 32 lanes sit on the same row of A and their accesses collapse into a single 128 B segment instead of 32 scattered ones.](/images/gemm/coalescing-access-pattern.png "large")

Matrix B behaves differently with the naive kernel implementation. Initially, the `col` variable is assigned with `threadIdx.y`, which is a constant value across all 32 threads within a warp. What differs across the lanes is row variable with the `threadIdx.x`. Every lane computes the same column element and asks for the same address. The hardware loads one 32 byte sector, uses 4 bytes of it and broadcasts that single float to all 32 registers. Each iteration the threads step down one row on the same column.

With the coalesced kernel for matrix B, all 32 threads read the same row of B during a single lockstep, one thread per adjacent column. Because matrix B is stored in row-major order, those adjacent cells sit sequentially in physical memory. The hardware packs the 128 bytes into 4 consecutive 32 byte sectors. This makes a perfectly coalesced access when matrix B access a whole row. 

![Left: the naive mapping takes col from threadIdx.y, so all 32 lanes of a warp resolve to the same column of B - one address, one 32 B sector loaded, 4 B of it used and the rest of the warp served by broadcast, and each k step walks one row down that same column. Right: taking col from threadIdx.x puts the 32 lanes on 32 adjacent columns of row k - 128 contiguous bytes packed into 4 x 32 B sectors with every byte used, and the whole warp drops one row per k step.](/images/gemm/coalescing-matrix-b.png "large")

### The Kernel

The launch configuration is unchanged from kernel 1. It is `32 x 32` threads per block, one thread per output element, `CEIL_DIV(N, 32) x CEIL_DIV(M, 32)` blocks:

```cuda
dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
```

The kernel body is unchanged too. The entire delta is which component of `threadIdx` feeds which variable:

```cuda
// kernel 1: naive
const int row = blockIdx.y * BLOCKSIZE + threadIdx.x;
const int col = blockIdx.x * BLOCKSIZE + threadIdx.y;

// kernel 2: coalesced
const int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
const int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
```

Two lines. Same loop, same arithmetic, same number of loads.

### Mechanics

1. **Why `threadIdx.x` is the index that matters.** A block's threads are linearized as `threadIdx.x + blockDim.x * threadIdx.y` before being cut into warps of 32. With `blockDim.x = 32`, a warp is exactly one row of the block: 32 consecutive values of `threadIdx.x` at a single `threadIdx.y`. Whichever variable `threadIdx.x` feeds is the one that varies across a warp, and that variable decides whether the warp's addresses are contiguous. Kernel 1 fed it to `row`; kernel 2 feeds it to `col`.

2. **C moves with the loads.** The swap is usually described in terms of A and B, but `C[row * N + col]` is indexed by the same variables. Its read-back for `beta` and its store both go from 32 sectors to 4. C is touched once per thread rather than once per K iteration, so it barely shows up in the total, but it is the same fix.

Kernel 2 runs in **47.213 ms at 2911.0 GFLOP/s, which is 8.1% of cuBLAS at `M = N = K = 4096`**, against 334.793 ms and 1.1% for the naive kernel. A 7.1x speedup for a two line change.

### Counting Transactions Instead of Accesses

The access count did not move, so the speedup has to come out of the transactions. Counting sectors per warp per step of the K loop:

| load | naive | coalesced |
|---|---|---|
| A | 32 lanes, stride `K x 4` = 16 KB apart, **32 sectors** | one address, broadcast, **1 sector** |
| B | one address, broadcast, **1 sector** | 32 adjacent floats, 128 B, **4 sectors** |
| **total** | **33 sectors = 1056 B** | **5 sectors = 160 B** |

That predicts **6.6x fewer bytes moved**, and the measured speedup is 7.1x. The two matrices trade places: A was the scattered one and becomes the broadcast, B was the broadcast and becomes the contiguous one. Only B ends up in the textbook coalesced case, and it is worth seeing what that case looks like from the memory system's side.

Naive, each lane claims its own 32 byte sector for a single 4 byte value: 32 sectors x 32 bytes = 1024 bytes moved, only 128 bytes of it used, 12.5% efficiency. Once the lanes of a warp are contiguous, one 32 byte sector covers 8 lanes instead of 1: 4 sectors x 32 bytes = 128 bytes moved, all 128 bytes used, 100% efficiency. Same hardware, same 32 byte sector size. The only thing that changed is how many lanes share one sector. 

<!-- TODO: profiler callout for kernel 2, the measured version of the table above.
     `sudo ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./bench 4096`
     on kernels 1 and 2. Sectors per request should read ~32 for the naive kernel
     and 4 for this one. Counters are admin only on this box right now. -->

![Naive: one 32 B sector per lane, 1 of 8 slots used. 32 sectors x 32 B = 1024 B moved, only 128 B used, 12.5% efficiency. Coalesced: one 32 B sector per 8 lanes, all 8 slots used. 4 sectors x 32 B = 128 B moved, all 128 B used, 100% efficiency.](/images/gemm/sector-utilization.png "large")

### What Could Be Improved

The main difference between the naive kernel implementation and the coalesced kernel is that the coalesced version reduces the total hardware transaction making the kernel 7.1% times faster than the naive kernel. But the catch is that the total work done the byte is still the same; the algorithmic workload hasn't changed. To compute one cell in matrix C, we are still doing a `2K` load just to perform the mulitplication and addition to create the dot product.

Since the coalesced kernels request and work done is the same compared to the naive kernel, the arithmetic intensity has not changed; it is stuck at `0.25 FLOP/byte`. However the upside to global memory coalescing is that we are increasing the throughput. 

By aligning the 32 thread memory accesses into adjacent physical addresses, we reduce the number of hardware transactions. A naive kernel reads 32 seperate 32 byte sectors. This is 1024 total bytes moved through the cache path just to deliver 128 useful bytes. Coalescing fixes this by packing 128 useful bytes into 4 perfectly aligned 32-byte sectors. So instead of choking on the Load/Store units with 32 separate transactions we only issue 4 transactions. This relieves the pressure
on teh L1/L2 cache path. As a reuslt we actaully increase the GFLOP/s since memory requests are now processed in a fraction of a time. Therefore the warp spends less time waiting for data and spends more time computing; which is an increase in computational throughput even though the workload hasn't changed a bit. 





