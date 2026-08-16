## Global Memory Coalescing

From the naive kernel implementation, we have established that the kernel loads the right values in a inefficient order. Right now the way we are assigning the global thread index to the row and col variables will result in all of the threads accessing data from the global memory in a non-contiguous manner. 

From a warp perspective of matrix A, each thread accesses a different row in memory during 1 lockstep. A thread would access `A[0][k]`, the next thread will access `A[1][k]`; each thread within the warp accesses a new row. Consecutive lanes land K x 4 bytes apart, so there is nothing for the hardware to merge. 

From a thread perspective of matrix A, a thread walks along its row and accesses a new element. Suppose thread 0's first iteration accesses `A[0][0]`, the second iteration the thread will access `A[0][1]`. Each time the warp goes through a new iteration it fetches a new element, and each of those fetches pulls in a whole 32 byte sector only to use a 4 byte float. 

With matrix A, we can conclude that each thread fetching 32 bytes just to use 4 bytes of memory is wasteful. To enable global memory coalescing we can swap the global thread index assignment to the row and col variables so that `threadIdx.x` maps to col, making col the fastest changing index across a warp. 

Changing the global thread index between the row and col would change how A is accessed. Instead of 32 threads accessing 32 separate rows, all 32 threads will access the same row — and within a single lockstep the same element. As a result, the hardware serves that one address out of a single sector and broadcasts the value to all 32 lanes. 

![Left: the naive mapping puts threadIdx.x on row, so a warp's 32 lanes land on 32 different rows of A, each K·4 bytes apart in memory - nothing to merge into one transaction. Right: swapping the mapping puts threadIdx.x on col, so all 32 lanes sit on the same row of A and their accesses collapse into a single 128 B segment instead of 32 scattered ones.](/images/gemm/coalescing-access-pattern.png "large")

Matrix B behaves differently with the naive kernel implementation. The `col` variable is assigned with threadIdx.y, which is a constant value across all 32 threads within a warp. What differs across the lanes is `threadIdx.x`, and that index is feeding row, not col. Every lane computes the same column element and asks for the same address. The hardware loads one 32 byte sector, uses 4 bytes of it and broadcasts that single float to all 32 registers. Each iteration the threads step down one row on the same column.

In matrix B, all 32 threads read the same row of B during a single lockstep, one thread per adjacent column. Because matrix B is stored in row-major order, those adjacent cells sit sequentially in physical memory. The hardware packs the 128 bytes into 4 consecutive 32 byte sectors. This is a perfectly coalesced access. 

![Left: the naive mapping takes col from threadIdx.y, so all 32 lanes of a warp resolve to the same column of B - one address, one 32 B sector loaded, 4 B of it used and the rest of the warp served by broadcast, and each k step walks one row down that same column. Right: taking col from threadIdx.x puts the 32 lanes on 32 adjacent columns of row k - 128 contiguous bytes packed into 4 x 32 B sectors with every byte used, and the whole warp drops one row per k step.](/images/gemm/coalescing-matrix-b.png "large")

### The Code

The launch configuration is unchanged from kernel 1 — `32 x 32` threads per block, one thread per output element, `CEIL_DIV(N, 32) x CEIL_DIV(M, 32)` blocks:

```cuda
dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
```

The kernel body is unchanged too. The entire delta is which component of `threadIdx` feeds which variable:

```cuda
// kernel 1 — naive
const int row = blockIdx.y * BLOCKSIZE + threadIdx.x;
const int col = blockIdx.x * BLOCKSIZE + threadIdx.y;

// kernel 2 — coalesced
const int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
const int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
```

Two lines. Same loop, same arithmetic, same number of loads.

### Mechanics

1. **Why `threadIdx.x` is the index that matters.** A block's threads are linearized as `threadIdx.x + blockDim.x * threadIdx.y` before being cut into warps of 32. With `blockDim.x = 32`, a warp is exactly one row of the block: 32 consecutive values of `threadIdx.x` at a single `threadIdx.y`. Whichever variable `threadIdx.x` feeds is the one that varies across a warp, and that variable decides whether the warp's addresses are contiguous. Kernel 1 fed it to `row`; kernel 2 feeds it to `col`.

2. **C moves with the loads.** The swap is usually described in terms of A and B, but `C[row * N + col]` is indexed by the same variables. Its read-back for `beta` and its store both go from 32 sectors to 4. C is touched once per thread rather than once per K iteration, so it barely shows up in the total, but it is the same fix.

3. **The guard is not doing anything here.** `if (row < M && col < N)` never fires at `M = N = K = 4096`, because 4096 is exactly 128 blocks of 32. It exists so the kernel stays correct when the dimensions are not multiples of `BLOCKSIZE`.

Kernel 2 runs in **43.532 ms at 3157.2 GFLOP/s — 8.1% of cuBLAS at `M = N = K = 4096`**, against 300.088 ms and 1.2% for the naive kernel. A 6.9x speedup for a two line change.

### Counting Transactions Instead of Accesses

The access count did not move, so the speedup has to come out of the transactions. Counting sectors per warp per step of the K loop:

| load | naive | coalesced |
|---|---|---|
| A | 32 lanes, stride `K x 4` = 16 KB apart — **32 sectors** | one address, broadcast — **1 sector** |
| B | one address, broadcast — **1 sector** | 32 adjacent floats, 128 B — **4 sectors** |
| **total** | **33 sectors = 1056 B** | **5 sectors = 160 B** |

That predicts **6.6x fewer bytes moved**, and the measured speedup is 6.9x. The two matrices trade places: A was the scattered one and becomes the broadcast, B was the broadcast and becomes the contiguous one. Only B ends up in the textbook coalesced case, and it is worth seeing what that case looks like from the memory system's side.

Naive, each lane claims its own 32 byte sector for a single 4 byte value: 32 sectors x 32 bytes = 1024 bytes moved, only 128 bytes of it used, 12.5% efficiency. Once the lanes of a warp are contiguous, one 32 byte sector covers 8 lanes instead of 1: 4 sectors x 32 bytes = 128 bytes moved, all 128 bytes used, 100% efficiency. Same hardware, same 32 byte sector size — the only thing that changed is how many lanes share one sector. 

![Naive: one 32 B sector per lane, 1 of 8 slots used — 32 sectors x 32 B = 1024 B moved, only 128 B used, 12.5% efficiency. Coalesced: one 32 B sector per 8 lanes, all 8 slots used — 4 sectors x 32 B = 128 B moved, all 128 B used, 100% efficiency.](/images/gemm/sector-utilization.png "large")

### What Is Still Wrong

Every byte this kernel fetches now gets used — B's sectors by the whole warp at once, A's by the next seven steps of the K loop — and it is still at 8.1% of cuBLAS. The reason is the number that never changed: `2K` = 8192 global loads per output element, one from A and one from B on every trip through the K loop. We made each of those loads cheap; we did not make any of them go away. Every element of A is still fetched once for every thread that needs it, and so is every element of B.

Before attacking that, there is one piece of index arithmetic worth isolating while the kernel is still this small. Every kernel from here on computes a thread's position inside a tile from a flat `threadIdx.x` rather than from a 2D block shape, and getting that mapping backwards silently undoes everything this section just bought.
