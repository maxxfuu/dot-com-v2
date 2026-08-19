## Shared Memory Tiling

The coalesced kernel loads data more efficiently by reducing the amount of hardware transactions. For Matrix B, you packed the memory such that each thread accesses memory in a sequential and adjacent manner, so the hardware only needs 4 transactions instead of 32 to serve the whole warp. For Matrix A, you optimized it so that all 32 threads access the same element, bringing the load down to a single transaction, which gets broadcasted to all 32 lanes/registers. 

But you can actually make this even more efficient. Global memory coalescing optimizes on the thread level, but if you take a step back and view the kernel from a warp level, you can see the redundancies. 

Matrix B's inefficiency is a duplicate data redundancy. Each row within the C tile corresponds to a warp, and each warp accesses the same coalesced data within B. Warp 0 needs 32 floats that span `B[k][0-31]`, warp 1 needs 32 floats that also span `B[k][0-31]`, and all 32 warps need the same 32 floats. Therefore 32 warps hit the L1 cache asking for the same 128 bytes, which makes the hardware execute 32 redundant reads for data another warp has already fetched. 

Matrix A's inefficiency is a little tougher to spot. The warps load different data since each warp corresponds to a new row. Warp 0 needs `A[0][K]`, warp 1 needs `A[1][K]`, warp 2 needs `A[2][K]`. Each warp reads down a column of A for each lockstep. Therefore you have 32 independent warps accessing uncoalesced memory within the VRAM. Even though the broadcasts exist, each broadcast only handles the threads inside the warp. However, the block as a whole with 32 warps generates 32 hardware transactions for a single lockstep. 

Shared Memory Tiling fixes both problems in two different ways: 

For Matrix B, a corresponding tile within Matrix B is loaded into the shared memory once. Now, when warps 0, 1, k-31 need to read the same row of B, they can read it from the tile stored within the shared memory. 32 redundant reads from L2 are reduced to just 1 as the data can be read from the L1 cache now. 

![Two panels. Load: each of the 1024 threads copies one element of the A tile and one of the B tile out of global memory into As and Bs, so thread (tx, ty) owns exactly one cell of each, and then calls syncthreads, which means the tile is whole before anyone reads it. Compute: thread (tx, ty) walks row ty of As against column tx of Bs for 32 multiply accumulates with no global memory traffic at all, summing into C[ty][tx], and syncs again before the next tile overwrites shared memory.](/images/gemm/smem-load-sync-compute.png "full")

For Matrix A, to load the corresponding 32x32 tile into shared memory, the threads must change their global memory access pattern. Instead of all 32 threads in a warp fetching the same float, which forced the block to make 32 scattered broadcast transactions down a column, the threads inside each warp are reassigned to fetch adjacent elements across a contiguous row of Matrix A. This turns the scattered vertical reads into perfectly coalesced 128-byte memory accesses. Now, each warp pulls an entire row of the tile into shared memory in just 4 hardware transactions, completely eliminating the scattered memory bottleneck.

![One thread block owns a 32 x 32 tile of C, named by cRow and cCol, with M = N = K = 4096. Each trip around the outer loop copies a 32 x 32 tile of A and a 32 x 32 tile of B from global memory into shared memory as As and Bs, 8 KB resident per block, then slides A right by 32 columns and B down by 32 rows for the next trip.](/images/gemm/smem-tiling-overview.png "full")

Here is how you launch the kernel:

```cuda
const int TILESIZE = 32;

// note the order: x walks M here, not N
dim3 gridDim(CEIL_DIV(M, TILESIZE), CEIL_DIV(N, TILESIZE));
dim3 blockDim(TILESIZE * TILESIZE);

sgemm_shared_mem_block<TILESIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

This is the SMEM tiling kernel: 

```cuda
template <const int TILESIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  // Define block position within the grid
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;
  
  // Define thread position relative to a tile shape in matrix C
  const int threadRow = threadIdx.x / TILESIZE;
  const int threadCol = threadIdx.x % TILESIZE;
  
  // Shift the pointers of the matrices within the global memory. Move them to always point at the top left corner of a tile. 
  A += cRow * TILESIZE * K ;
  B += cCol * TILESIZE;
  C += cRow * TILESIZE * N + cCol * TILESIZE ;
 
  // declare 2D array in SMEM
  __shared__ float As[TILESIZE * TILESIZE];
  __shared__ float Bs[TILESIZE * TILESIZE];

  float temp = 0.0f;

  // load data and advance the global pointers
  for (int blockIdx = 0; blockIdx < K; blockIdx += TILESIZE) {
    // load 1 element per thread from GMEM into SMEM 
    As[threadRow * TILESIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * TILESIZE + threadCol] = B[threadRow * N + threadCol];
    
    // wait for all threads to finish loading values from GMEM to SMEM
    __syncthreads();
    
    // advance A to the right by 1 tile. advance B down by one tile. 
    A += TILESIZE;
    B += TILESIZE * N;
    
    // compute partial dot product within the tile 
    for (int k = 0; k < TILESIZE; ++k) {
      temp += As[threadRow * TILESIZE + k] * Bs[k * TILESIZE + threadCol];
    }

    // wait for all threads to finish computing before next iteration loads data
    __syncthreads();
  }

  C[threadRow * N + threadCol] = alpha * temp + beta * C[threadRow * N + threadCol];
}

```

### Key Mechanics


1. The pointers `A`, `B`, and `C` are advanced to the blocks's top left corner before the loop starts. Each pointer marks the relative starting point for a tile. To move the pointers `A += TILESIZE` has to step 32 columns right and B `B += TILESIZE * N` has to step 32 rows down.

2. **Both `__syncthreads()` are load bearing, for different reasons.** The first one guards fill before read. A thread that reaches the dot product early would otherwise read cells of `As` and `Bs` that another warp has not written yet, and it would read whatever was there before. The second one guards read before overwrite. Without it a fast warp comes around the loop and starts writing the next tile into `As` while a slow warp is still multiplying with the current one. Removing either is a race, and neither will fail every time, which is what makes them nasty.

The shared memory tiling kernel runs in **33.268 ms at 4131.3 GFLOP/s, which is 11.5% of cuBLAS at `M = N = K = 4096`**, up from 47.007 ms and 8.1%.

### The Arithmetic

In the previous kernel, every thread that walks a row has to load the entire row `K = 4096` and the entire column `K = 4096`. Therefore each cell output in matrix C requires `4096 + 4096 = 8192 Loads`. With the optimized SMEM tiling kernel, we have to look a whole `32 x 32` tile instead of a thread. The block walks along the dimension `K = 4096` and the tile size if `32`, therefore it takes `4096 / 32 = 128` block tiles per matrix to write to the tile in matrix C. Given that we have matrix A and B we have `256` loads per tile and each tile has `1024 floats`, `256 x 1024 = 262,144 total floats`. Given that we have `1024` total output cells: 

```latex
\frac{262{,}144 \ \text{total loads}}{1024 \ \text{output cells}} = 256 \ \text{loads per cell}
```

We reduced the loads per cell from 8192 to 256. This is a decrease of global memory traffic by a factor of 32. 

And the runtime went from 47.0 ms to 33.3 ms, which is 1.41 times faster.

### Where You Are On The Roofline

This is the first kernel that moves on the roofline at all. Intensity goes from `0.25` to `8 FLOP/byte`, a 32 times move to the right, and it is still `7.3` times short of the 58.6 ridge point, so the kernel is still nominally memory bound.

What changed is that the model finally applies. The sloped ceiling at 8 FLOP per byte sits at `8 x 960 GB/s = 7680 GFLOP/s`, and you measured 4131.3, which is **54% of the roof**. Kernels 1 and 2 both ran above their roof because the caches were quietly serving most of the requests. Kernel 4 is the first one to sit underneath it, and the reason is that 17.2 GB in 33.3 ms is only 516 GB/s of requested traffic, which is a number a 960 GB/s bus can plausibly supply on its own.

Read as a floor, the same arithmetic says 17.2 GB at 960 GB/s costs `17.9 ms`, and you are at 33.3. So there is still 1.9 times of headroom before global bandwidth becomes the thing stopping you, and the compute floor of 2.44 ms is further away still. You are not bound by either ceiling on the roofline. Whatever is costing you the other 15 ms is inside the SM.

### What Is Still Wrong

That gap is the whole point of this section. Global traffic fell by a factor of 32 and the kernel got 1.41 times faster, so global traffic was not what the kernel was waiting on any more. The work did not disappear; it moved.

Look at what the inner loop actually costs. Each of the 32 steps reads one float out of `As` and one out of `Bs` to feed a single multiply accumulate, so the shared memory access count per output element is `2K = 8192`, which is exactly the number global memory used to carry. You swapped a 500-cycle load for a 25-cycle load and kept the count identical. Every FMA in this kernel is still paying for two loads to feed it; they are just cheaper loads now.

<!-- TODO: profiler callout for kernel 4. Needs `sudo ncu --section WarpStateStats ./bench 4096`.
     Expect Stall MIO Throttle and Stall Short Scoreboard to dominate. Counters are
     admin only on this box right now, so the numbers are not in yet. -->

There is a second cost that is easy to miss. Each block runs 1024 threads and the SM caps out at 1536, so only one block is resident per SM and a third of the thread slots go unused. Those two `__syncthreads()` then stall the entire SM, because there is no second block sitting there with work to issue while this one waits at the barrier.

The fix for both is the same, and it is to stop giving each thread exactly one output. If a thread computes several results instead of one, a single value read out of shared memory can feed several multiply accumulates rather than just one, and the ratio of loads to arithmetic finally starts to move.
