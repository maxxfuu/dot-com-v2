## Shared Memory Tiling

The last two kernels fixed the order the values arrive in and left the number of values alone. Every element of A is still pulled out of global memory once for every thread that needs it, and so is every element of B. Counting it from the matrix side, an element of A is read `N` times over the whole kernel and an element of B is read `M` times, which at 4096 is four thousand trips to global memory for a float that never changes.

The waste is easiest to see inside a single block. One block owns a `32 x 32` tile of C, and to compute that tile it needs a 32 row strip of A and a 32 column strip of B. Within that block, the same row of A is read by the 32 threads sitting in that row of the tile, and the same column of B is read by the 32 threads sitting in that column. That is 32 threads in the same block asking global memory for the same float, at the same time, and the hardware has no way to know they are the same request.

Shared memory is the fix. It is a small block of memory that lives on the SM, it is visible to every thread in the block, and it costs roughly 20 to 30 cycles to read instead of the several hundred a global load costs. So instead of every thread fetching what it needs, the block fetches the strip once, parks it in shared memory, and then every thread reads it from there.

The strips do not fit. A full 32 row strip of A at `K = 4096` is `32 x 4096 x 4 B = 512 KB`, and we only have 100 KB of shared memory per SM. So we walk the strips in chunks. The block loads a `32 x 32` tile of A and a `32 x 32` tile of B, does all the multiplying it can with those two tiles, then slides both windows along K and does it again. `K / 32 = 128` trips around that loop and the tile of C is finished.

![One thread block owns a 32 x 32 tile of C, named by cRow and cCol, with M = N = K = 4096. Each trip around the outer loop copies a 32 x 32 tile of A and a 32 x 32 tile of B from global memory into shared memory as As and Bs, 8 KB resident per block, then slides A right by 32 columns and B down by 32 rows for the next trip.](/images/gemm/smem-tiling-overview.png "full")

Inside one trip the block does four things in order. Every thread copies one element of A and one element of B into shared memory, then the block syncs, then every thread runs a 32 step dot product entirely out of shared memory, then the block syncs again. There are 1024 threads and each tile has exactly 1024 elements, so the load is one element per thread with nothing left over and no loop needed. The figure below calls a thread's position `tx` and `ty`; the code calls them `threadCol` and `threadRow` and recovers them from the flat `threadIdx.x` with the `%` and `/` from the previous kernel, which is the same position by a different name.

![Two panels. Load: each of the 1024 threads copies one element of the A tile and one of the B tile out of global memory into As and Bs, so thread (tx, ty) owns exactly one cell of each, and then calls syncthreads, which means the tile is whole before anyone reads it. Compute: thread (tx, ty) walks row ty of As against column tx of Bs for 32 multiply accumulates with no global memory traffic at all, summing into C[ty][tx], and syncs again before the next tile overwrites shared memory.](/images/gemm/smem-load-sync-compute.png "full")

The launch is one block per `32 x 32` tile of C, 1024 flat threads per block, the same shape kernel 3 used. The one oddity is the grid: this kernel takes its row from `blockIdx.x`, so the grid dimensions are swapped relative to every other kernel in the series.

```cuda
const int TILESIZE = 32;

// note the order: x walks M here, not N
dim3 gridDim(CEIL_DIV(M, TILESIZE), CEIL_DIV(N, TILESIZE));
dim3 blockDim(TILESIZE * TILESIZE);

sgemm_smem_block<TILESIZE><<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
```

```cuda
template <const int TILESIZE>
__global__ void sgemm_smem_block(float *A_gmem, float *B_gmem, float *C_gmem, int M, int N, int K, float alpha, float beta) {
  // Define block position within the grid
  const int C_Row = blockIdx.x;
  const int C_Col = blockIdx.y;

  // Define thread position relative to a tile shape in matrix C
  const int threadRow = threadIdx.x / TILESIZE;
  const int threadCol = threadIdx.x % TILESIZE;

  // Shift the pointers into global memory so each one points at the top left
  // corner of this block's tile.
  A_gmem += C_Row * TILESIZE * K;
  B_gmem += C_Col * TILESIZE;
  C_gmem += C_Row * TILESIZE * N + C_Col * TILESIZE;

  __shared__ float A_smem[TILESIZE * TILESIZE];
  __shared__ float B_smem[TILESIZE * TILESIZE];

  float temp = 0.0f;

  for (int block_k = 0; block_k < K; block_k += TILESIZE) {
    // one element per thread, GMEM -> SMEM
    A_smem[threadRow * TILESIZE + threadCol] = A_gmem[threadRow * K + threadCol];
    B_smem[threadRow * TILESIZE + threadCol] = B_gmem[threadRow * N + threadCol];

    // the tile must be whole before anyone reads it
    __syncthreads();

    // advance A right by one tile, B down by one tile
    A_gmem += TILESIZE;
    B_gmem += TILESIZE * N;

    // partial dot product, entirely out of shared memory
    for (int dot_k = 0; dot_k < TILESIZE; ++dot_k) {
      temp += A_smem[threadRow * TILESIZE + dot_k] * B_smem[dot_k * TILESIZE + threadCol];
    }

    // everyone must finish reading before the next trip overwrites SMEM
    __syncthreads();
  }

  C_gmem[threadRow * N + threadCol] = alpha * temp + beta * C_gmem[threadRow * N + threadCol];
}
```

### Mechanics

1. **The pointers move, the indices do not.** `A`, `B` and `C` are advanced to the block's own corner before the loop starts, so every index inside the loop is written relative to that corner and stays small. Sliding the window is then two lines, `A += TILESIZE` to step 32 columns right and `B += TILESIZE * N` to step 32 rows down. It is the same reason the loads stay readable: `A[threadRow * K + threadCol]` is a position inside the tile, not a position inside the matrix.

2. **The loads are still coalesced, and that is not an accident.** `threadCol` is `threadIdx.x % TILESIZE`, so consecutive threads still differ only in the column, and both `A[threadRow * K + threadCol]` and `B[threadRow * N + threadCol]` hand a warp 32 adjacent floats. This is exactly the mapping kernel 3 isolated, and it is why that kernel was worth writing down before this one.

3. **Both `__syncthreads()` are load bearing, for different reasons.** The first one guards fill before read. A thread that reaches the dot product early would otherwise read cells of `As` and `Bs` that another warp has not written yet, and it would read whatever was there before. The second one guards read before overwrite. Without it a fast warp comes around the loop and starts writing the next tile into `As` while a slow warp is still multiplying with the current one. Removing either is a race, and neither will fail every time, which is what makes them nasty.

4. **`blockIdx` is shadowed.** The loop counter is named `blockIdx`, which hides the built in variable of the same name for the rest of the loop body. It compiles and runs correctly here only because `cRow` and `cCol` were read out before the loop, so nothing inside the loop needs the real one. It is worth renaming to `bk` before this kernel gets copied into the next one.

5. **`cRow` comes from `blockIdx.x`, and the launch compensates.** Every other kernel in this series takes its row from `blockIdx.y` and launches `gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM))`. This one is the exception in both places at once, so the two mistakes cancel and the kernel is correct. The figure above draws the conventional orientation, `cRow` from `blockIdx.y`, because that is what the rest of the series uses; read it as the tile geometry rather than as the literal index assignment. Two consequences worth knowing. It only survives because `M = N` here, so the two grid extents are equal and nothing indexes past the end; at `M != N` this kernel reads out of bounds. And the next kernel quietly switches back to the conventional mapping, which is worth noticing rather than absorbing.

Kernel 4 runs in **33.268 ms at 4131.3 GFLOP/s, which is 11.5% of cuBLAS at `M = N = K = 4096`**, up from 47.007 ms and 8.1%.

### The Arithmetic

The general form for a block tile of `BM x BN` walked over K in steps of `BK` is that each trip loads `BM x BK` elements of A and `BK x BN` elements of B to produce `BM x BN` results, and there are `K / BK` trips. The `BK` cancels:

```
GMEM accesses per result = (K / BK) * (BM*BK + BK*BN) / (BM*BN)
                         = K * (1/BM + 1/BN)
```

At `BM = BN = 32` that is `4096 / 16 = 256` global loads per output element, against `2K = 8192` for every kernel so far. **32 times fewer.** Counted as bytes over the whole matrix it is 549.8 GB of global loads down to **17.2 GB**. Arithmetic intensity goes from 0.25 FLOP per byte to 8, because the same 8192 FLOP per output now sit behind 1024 bytes of global traffic instead of 32768.

And the runtime went from 47.0 ms to 33.3 ms, which is 1.41 times faster.

### Where We Are On The Roofline

This is the first kernel that moves on the roofline at all. Intensity goes from `0.25` to `8 FLOP/byte`, a 32 times move to the right, and it is still `7.3` times short of the 58.6 ridge point, so the kernel is still nominally memory bound.

What changed is that the model finally applies. The sloped ceiling at 8 FLOP per byte sits at `8 x 960 GB/s = 7680 GFLOP/s`, and we measured 4131.3, which is **54% of the roof**. Kernels 1 and 2 both ran above their roof because the caches were quietly serving most of the requests. Kernel 4 is the first one to sit underneath it, and the reason is that 17.2 GB in 33.3 ms is only 516 GB/s of requested traffic, which is a number a 960 GB/s bus can plausibly supply on its own.

Read as a floor, the same arithmetic says 17.2 GB at 960 GB/s costs `17.9 ms`, and we are at 33.3. So there is still 1.9 times of headroom before global bandwidth becomes the thing stopping us, and the compute floor of 2.44 ms is further away still. We are not bound by either ceiling on the roofline. Whatever is costing us the other 15 ms is inside the SM.

### What Is Still Wrong

That gap is the whole point of this section. Global traffic fell by a factor of 32 and the kernel got 1.41 times faster, so global traffic was not what the kernel was waiting on any more. The work did not disappear, it moved.

Look at what the inner loop actually costs. Each of the 32 steps reads one float out of `As` and one out of `Bs` to feed a single multiply accumulate, so the shared memory access count per output element is `2K = 8192`, which is exactly the number global memory used to carry. We swapped a 500 cycle load for a 25 cycle load and kept the count identical. Every FMA in this kernel is still paying for two loads to feed it, they are just cheaper loads now.

<!-- TODO: profiler callout for kernel 4. Needs `sudo ncu --section WarpStateStats ./bench 4096`.
     Expect Stall MIO Throttle and Stall Short Scoreboard to dominate. Counters are
     admin only on this box right now, so the numbers are not in yet. -->

There is a second cost that is easy to miss. Each block runs 1024 threads and the SM caps out at 1536, so only one block is resident per SM and a third of the thread slots go unused. Those two `__syncthreads()` then stall the entire SM, because there is no second block sitting there with work to issue while this one waits at the barrier.

The fix for both is the same, and it is to stop giving each thread exactly one output. If a thread computes several results instead of one, a single value read out of shared memory can feed several multiply accumulates rather than just one, and the ratio of loads to arithmetic finally starts to move.
