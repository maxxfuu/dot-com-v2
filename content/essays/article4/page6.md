## Shared Memory Tiling

In the previous chapter, we concluded that the coalesced kernel loads data more efficiently by reducing the number of hardware transactions. From the perspective of Matrix B, you pack the memory such that each thread accesses memory in a row major fashion where each thread accesses the matrix in a sequential and adjacent manner; so the hardware only needs 4 transactions instead of 32 to serve the whole warp. And from the persepctive of Matrix A, you optimized it so that all 32 threads access the same data, bringing the load down to a single transaction, which gets broadcasted to all 32 lanes/registers. 

You should by now realize that global memory coalescing optimizes on the thread level, however we can optimize the kernel even further. If you take a step back and view the kernel from a warp level, you will be able to spot the redundancies that allows further optimizations. 

The whole premise of the shared memory tiling technique that is introduced in this chapter is to move a block size matrix, (`BLOCKSIZE * BLOCKSIZE`), to the shared memory thats located on the streaming multi-processor. This is because accessing global memory takes longer than accessing shared memory, and since a lot of data is being resued, we can temporarily store the data inside the shared memory for faster access.

Looking at matrix B as a whole, understand that each row corresponds to a whole warp. And each warp accesses the same coalesced data within B. In a single warp, the threads accesses 32 floats that walks along the whole row. Warp 0 needs 32 floats that span `B[k][0-31]` during k lockstep, warp 1 needs 32 floats that also span `B[k][0-31]` on k lockstep, essentially all 32 warps needs the same row, the same 32 floats. Therefore 32 distinct warps hits the L1 cache asking for the same 32 floats of data or 128 bytes. This means the hardware executes 32 redundant reads for data that has already been fetched by one of the 32 warps. 

Matrix A's inefficiency is a little tougher to spot. The warps load a new row. Warp 0 needs `A[0][k]`, warp 1 needs `A[1][k]`, and warp 2 needs `A[2][k]`. So collectively, all the warps during one lockstep reads down a whole column `A[0-31][k]`. Therefore you have 32 independent warps accessing strided memory within the VRAM. Even though global memory coalescing broadcasts the data between all of the threads, this optmization only exists within the scope of a warp. So do not
mistake strided access as another oportunity for coalescing. Instead, just acknoledge that all the threads within a warp accesses the same element. And all of the warps accesses the element on the same column space, thus memory access is strided. So whole block tile contains 32 warps that generates 32 uncoalesced transactions such that each wrap fetches one byte sector for a single lockstep. 

Shared Memory Tiling fixes both inefficiencies in two different ways: 

Firstly, for the matrix B, a corresponding tile within Matrix B is loaded into the shared memory once. Now, when warps 0 through 31 need to read the same row of B, they can read it from the tile stored within the shared memory. Instead of 32 redundant reads from the global memory, they are reduced to just 1 read from global memory and 32 shared memory loads. 

![Two panels. Load: each of the 1024 threads copies one element of the A tile and one of the B tile out of global memory into As and Bs, so thread (tx, ty) owns exactly one cell of each, and then calls syncthreads, which means the tile is whole before anyone reads it. Compute: thread (tx, ty) walks row ty of As against column tx of Bs for 32 multiply accumulates with no global memory traffic at all, summing into C[ty][tx], and syncs again before the next tile overwrites shared memory.](/images/gemm/smem-load-sync-compute.png "full")

Secondly, for matrix A, to load the corresponding 32x32 tile into shared memory, the threads must change their global memory access pattern. Instead of all 32 threads in a warp fetching the same float, and all warps are accesses data in a strided manner forcing the block to make 32 scattered broadcast transactions down a column, the threads inside each warp are reassigned to fetch adjacent elements across a contiguous row of Matrix A. This turns the scattered vertical reads from a warp perspective into perfectly coalesced 128-byte memory accesses. Now, each warp collectively reads an entire row of the tile into shared memory in just 4 hardware transactions, completely eliminating the scattered memory bottleneck.

![One thread block owns a 32 x 32 tile of C, named by cRow and cCol, with M = N = K = 4096. Each trip around the outer loop copies a 32 x 32 tile of A and a 32 x 32 tile of B from global memory into shared memory as As and Bs, 8 KB resident per block, then slides A right by 32 columns and B down by 32 rows for the next trip.](/images/gemm/smem-tiling-overview.png "full")

Here is how you launch the kernel:

```cuda
const int BLOCKSIZE = 32;

// note the order: x walks M here, not N
dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

sgemm_smem_tiling<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

This is the SMEM tiling kernel: 
``` cuda 
template <const int BLOCKSIZE>
__global__ void sgemm_smem_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  // block position within the grid
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  // the two tiles this block stages in shared memory
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // thread position within the block's tile of C
  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  // move each pointer to the top left corner of this block's tile
  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float acc = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // one element per thread, GMEM -> SMEM
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block all threads until the shared memory is fully populated
    __syncthreads();

    // advance A right by one tile, B down by one tile
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // partial dot product over the cached tile
    for (int k = 0; k < BLOCKSIZE; ++k) {
      acc += As[threadRow * BLOCKSIZE + k] * Bs[k * BLOCKSIZE + threadCol];
    }

    // sync again at the end, so a fast thread cannot overwrite the tile
    // a slow thread is still reading
    __syncthreads();
  }

  C[threadRow * N + threadCol] =
      alpha * acc + beta * C[threadRow * N + threadCol];
}

```

### Memory Access Pattern

Step back from the indexing and there are three separate movements in this kernel, each with its own shape.

The first is global to shared, and it happens once per tile per block. All 1024 threads take part, and the mapping is chosen so that a warp's 32 lanes land on 32 adjacent columns of a single row: `threadCol` is `threadIdx.x % 32`, the fast moving index, so each warp pulls 128 contiguous bytes of A and 128 of B in four sectors apiece. The block repeats this 128 times, sliding A right by 32 columns and B down by 32 rows, and every element of the two tiles is fetched from VRAM exactly once for the whole block rather than once per thread that wants it.

The second is the barrier, and it is a memory movement even though it moves nothing. `__syncthreads()` is what turns 1024 independent loaders into one cooperative one. After it, no thread knows or cares which element it personally fetched, because all 2048 floats are equally available to all of them. That handoff is the entire reason the first movement is worth making.

The third is shared to register, and it is the one that does not scale. Every thread reads a full row of `As` and a full column of `Bs`, 32 elements out of each, to feed the 32 multiply accumulates it owns. Across the block that is 1024 threads each pulling 64 floats out of a 2048 float working set, so every element of `As` is read by the 32 threads sitting in its row and every element of `Bs` by the 32 in its column. The reuse is real, and all of it now runs through the shared memory pipe.

Those three movements are the win and the next problem in one picture. Traffic to VRAM fell by a factor of 32 because the block cooperates on the load. Traffic to the FMA units did not move at all, because the compute mapping never changed: one thread, one output, two reads per multiply.

### Key Mechanics

1. The pointers `A`, `B`, and `C` are advanced to the block's top left corner before the loop starts. Each pointer marks the relative starting point for a pointer within the block tile. To move the pointers, `A += BLOCKSIZE` has to step 32 columns right and `B += BLOCKSIZE * N` has to step 32 rows down.

2. The first `__syncthreads()` holds every thread in the block until all 1024 of them have written their element into `As` and `Bs`. It is a block-wide barrier, not a warp-level one, and that is the point: a thread that reaches the dot product early would otherwise read cells of `As` and `Bs` that another warp has not written yet, and it would read whatever was there before. The second `__syncthreads` guards read before overwrite. Without it, a faster warp comes around the loop and starts writing the next tile into `As` while a slow warp is still multiplying with the current one. Removing either of the `__syncthreads` introduces a race condition.

3. The block is 1024 threads, and that on its own caps residency at one block per SM. An SM holds 1536 threads, so a second block of 1024 does not fit and 512 thread slots go unused, which is 66.7% occupancy. `ptxas` reports 40 registers per thread and the two tiles cost 8192 bytes of shared memory, so neither of those is what binds here; the block size is. Note this is not a defect the next kernel repairs. It is a number worth having on hand before occupancy starts moving in the sections that follow.

### The Arithmetic

In the previous kernel, every thread that walks a row has to load the entire row `K = 4096` and the entire column `K = 4096`. Therefore each cell output in matrix C requires `4096 + 4096 = 8192 Loads`. With the optimized shared memory tiling kernel, we have to look at a whole `32 x 32` block tile instead of a thread. The block walks along the dimension `K = 4096` and the tile size is `32`, therefore it takes `4096 / 32 = 128` block tiles per matrix to write to the tile in matrix C. Given that we have matrix A and B, we have `256` loads per tile and each tile has `1024 floats`, `256 x 1024 = 262,144 total floats`. Given that we have `1024` total output cells: 

```latex
\frac{262{,}144 \ \text{total loads}}{1024 \ \text{output cells}} = 256 \ \text{loads per cell}
```

We reduced the loads per cell from 8192 to 256. This is a decrease of global memory traffic by a factor of 32. And the runtime went from 47.0 ms to 33.3 ms, which is 1.41 times faster. The shared memory tiling kernel runs in **33.268 ms at 4131.3 GFLOP/s, which is 11.5% of cuBLAS at `M = N = K = 4096`**, up from 47.007 ms and 8.1%.

### Where You Are On The Roofline

The shared memory tiling kernel moves the arithmetic intensity based on the roofline model. The previous intensity of `0.25` moves to `8 FLOP/byte`. This is a `32` times increase. However, it's still `7.3` times short of the theoretical ridge point based on the RTX 5080. 

![The same RTX 5080 roofline as before, with the naive kernel sitting at an arithmetic intensity of 0.25 FLOP/byte and 0.24 TFLOP/s. Shared memory tiling moves the kernel 32 times to the right along the 960 GB/s slope, to 8 FLOP/byte, and the attainable performance climbs with it. It is still on the sloped part of the roof: the ridge point is at 58.6 FLOP/byte, so the kernel is 7.3 times short of the intensity it would need to leave the memory bound region and hit the 56.3 TFLOP/s FP32 ceiling.](/images/gemm/roofline-smem-tiling.png "full")

### What Could Be Improved

Shared memory tiling dropped the global memory traffic by 32 times. However, against the previous kernel we've only increased the speed by a factor of 1.41. This makes sense because we simply shifted the traffic from the global memory to shared memory, and as a result we are now bounded by shared memory bandwidth. 

Now purely from the arithmetic side of things, there are 2 memory reads for every fused multiply add (FMA), precisely `2K = 8192` reads per output element. This is the same number that the previous kernel achieved prior to the shared memory tiling optimization, meaning we are still reading the same amount of data just from a different memory. So instead of `2K` reads out of global memory, we do `2K` reads out of shared memory. On the RTX 5080's GB203 die, a shared memory load returns in roughly 33 cycles[^1], against 358 cycles for an L2 hit and several hundred more if the line has to come all the way from GDDR7.[^2] Latency is not really the point here, though, because the previous kernel already had enough warps in flight to cover it. The point is traffic: those same 8192 reads no longer cross the memory bus. 

Nsight Compute confirms this. The kernel spends 40.5 warp cycles per issued instruction, and 23.9 of those cycles (58.9%) are stalled on `Stall MIO Throttle`, which is a warp waiting for a shared memory instruction to issue. The previous kernel stalled 56.1% on `Stall LG Throttle`, the same queue for global memory. What this tells us is that threads now waste time waiting on data to be loaded from the shared memory region instead of the global memory. 

`Stall MIO Throttle` names the queue, but it does not tell us how far off we are. For that we need something to measure the inner loop against, and the shared memory pipe has a number. An SM can issue four warp-wide `FFMA` per cycle, one from each of its four sub-partitions. Shared memory is 32 banks four bytes wide, so it serves 128 bytes per cycle for the entire SM, and 128 bytes is exactly one warp-wide `LDS`. The hardware is therefore balanced at **four FMAs for every shared memory load**, and an inner loop below that ratio leaves FP32 units idle however many warps are resident.

Our inner loop issues two loads per FMA, which is 0.5. Against a balance point of 4 we are **eight times short**, so seven cycles out of every eight the arithmetic units have nothing to do but wait for the MIO queue to drain. That idle fraction is what the profiler is reporting from the other side.

This is the roofline argument again, one level down the hierarchy. Up on the DRAM roof the ratio was FLOP per byte and the ridge point was 58.6; down here the ratio is FMA per load and the balance point is 4. The kernel climbed under the first ceiling and landed straight into the second, which is the whole reason that 32 times less traffic bought only 1.41 times the speed.

And nothing about shared memory is what pins that ratio at 0.5. The thread mapping is. One thread owns one element of C, so every value it pulls out of `As` or `Bs` feeds exactly one FMA and is then thrown away. Two loads per FMA is not a cost of using shared memory, it is simply what one output per thread costs, and at one output per thread the ratio cannot be anything else. Making the loads cheaper cannot help either, because the loads are not what is expensive. Issuing that many of them is.

Four FMAs per load is the number to get to, and getting there means changing what a thread owns.

[^1]: [Dissecting the SM_120 Microarchitecture: Cycle-Level Characterization of Blackwell Consumer GPUs](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html)
[^2]: [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789)
