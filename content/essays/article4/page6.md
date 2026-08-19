## Shared Memory Tiling

The coalesced kernel loads data more efficiently by reducing the number of hardware transactions. For Matrix B, you packed the memory such that each thread accesses memory in a sequential and adjacent manner, so the hardware only needs 4 transactions instead of 32 to serve the whole warp. For Matrix A, you optimized it so that all 32 threads access the same element, bringing the load down to a single transaction, which gets broadcasted to all 32 lanes/registers. 

But you can actually make this even more efficient. Global memory coalescing optimizes on the thread level, but if you take a step back and view the kernel from a warp level, you can see the redundancies. 

Matrix B's inefficiency is a duplicate data redundancy. Each row within the C tile corresponds to a warp, and each warp accesses the same coalesced data within B. Warp 0 needs 32 floats that span `B[k][0-31]`, warp 1 needs 32 floats that also span `B[k][0-31]`, and all 32 warps need the same 32 floats. Therefore 32 warps hit the L1 cache asking for the same 128 bytes, which makes the hardware execute 32 redundant reads for data another warp has already fetched. 

Matrix A's inefficiency is a little tougher to spot. The warps load different data since each warp corresponds to a new row. Warp 0 needs `A[0][k]`, warp 1 needs `A[1][k]`, and warp 2 needs `A[2][k]`. Each warp collectively reads down a column that spans the column space `A[0-31][k]`. Therefore you have 32 independent warps accessing strided memory within the VRAM. Even though the broadcasts exist, each broadcast only handles the threads inside the warp. However, the block as a whole with 32 warps generates 32 hardware transactions for a single lockstep. 

Shared Memory Tiling fixes both problems in two different ways: 

For Matrix B, a corresponding tile within Matrix B is loaded into the shared memory once. Now, when warps 0 through 31 need to read the same row of B, they can read it from the tile stored within the shared memory. Instead of 32 redundant reads from the global memory, they are reduced to just 1 read from global memory and 32 shared memory loads. 

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

### Mechanics

1. The pointers `A`, `B`, and `C` are advanced to the block's top left corner before the loop starts. Each pointer marks the relative starting point for a tile. To move the pointers, `A += TILESIZE` has to step 32 columns right and `B += TILESIZE * N` has to step 32 rows down.

2. The first `__syncthreads()` guards fill before read. A thread that reaches the dot product early would otherwise read cells of `As` and `Bs` that another warp has not written yet, and it would read whatever was there before. The second one guards read before overwrite. Without it a fast warp comes around the loop and starts writing the next tile into `As` while a slow warp is still multiplying with the current one. Removing either is a race, and neither will fail every time, which is what makes them nasty.

### The Arithmetic

In the previous kernel, every thread that walks a row has to load the entire row `K = 4096` and the entire column `K = 4096`. Therefore each cell output in matrix C requires `4096 + 4096 = 8192 Loads`. With the optimized SMEM tiling kernel, we have to look at a whole `32 x 32` tile instead of a thread. The block walks along the dimension `K = 4096` and the tile size is `32`, therefore it takes `4096 / 32 = 128` block tiles per matrix to write to the tile in matrix C. Given that we have matrix A and B, we have `256` loads per tile and each tile has `1024 floats`, `256 x 1024 = 262,144 total floats`. Given that we have `1024` total output cells: 

```latex
\frac{262{,}144 \ \text{total loads}}{1024 \ \text{output cells}} = 256 \ \text{loads per cell}
```

We reduced the loads per cell from 8192 to 256. This is a decrease of global memory traffic by a factor of 32. And the runtime went from 47.0 ms to 33.3 ms, which is 1.41 times faster. The shared memory tiling kernel runs in **33.268 ms at 4131.3 GFLOP/s, which is 11.5% of cuBLAS at `M = N = K = 4096`**, up from 47.007 ms and 8.1%.

### Where You Are On The Roofline

The shared memory tiling kernel moves the arithmetic intensity based on the roofline model. The previous intensity of `0.25` moves to `8 FLOP/byte`. This is a `32` times increase. However, it's still `7.3` times short of the theoretical ridge point based on the RTX 5080. 

![The same RTX 5080 roofline as before, with the naive kernel sitting at an arithmetic intensity of 0.25 FLOP/byte and 0.24 TFLOP/s. Shared memory tiling moves the kernel 32 times to the right along the 960 GB/s slope, to 8 FLOP/byte, and the attainable performance climbs with it. It is still on the sloped part of the roof: the ridge point is at 58.6 FLOP/byte, so the kernel is 7.3 times short of the intensity it would need to leave the memory bound region and hit the 56.3 TFLOP/s FP32 ceiling.](/images/gemm/roofline-smem-tiling.png "full")

### What Is Still Wrong

Shared memory tiling dropped the global memory traffic by 32 times. However, against the previous kernel we've only increased the speed by a factor of 1.41x. This makes sense because we simply shifted the traffic from the global memory to shared memory, and as a result we are now bounded by shared memory bandwidth. 

There are 2 memory reads for every FMA, precisely `2K = 8192` reads per output element. This is the same number that the previous kernel achieved prior to the shared memory optimization, meaning we are still reading the same amount of data just from a different memory. So instead of reading from the slow global memory that takes 500 cycles to load, we are doing the loading from the shared memory, which takes 25 cycles to load. 

Nsight Compute confirms this. The kernel spends 40.5 warp cycles per issued instruction, and 23.9 of those cycles (58.9%) are stalled on `Stall MIO Throttle`, which is a warp waiting for a shared memory instruction to issue. The previous kernel stalled 56.1% on `Stall LG Throttle`, the same queue for global memory. What this tells us it that threads now waste time waiting on data to be loaded from the shared memory region instead of the global memory. 

Based on our GPU architecture, each SM has 1536 active threads. Since a block uses 1024 threads, the SM doesn't have enough space to fit another block of 1024 threads. This means there are 512 threads that sit idle when the block is launched on the SM. This tells a story of how threads are being underutilized; therefore, this is a thread mapping issue. 

There are two key insights. Firstly, the wait between the threads has now moved to shared memory region. Secondly, we can fit more blocks on a thread. In the next kernel optimization, our main objective would be thread coarsening: optimizing the SGEMM kernel so that one thread can compute multiple cell outputs.
