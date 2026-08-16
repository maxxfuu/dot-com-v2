## Shared Memory Tiling

The last two kernels fixed the order the values arrive in and left the number of values alone. Every element of A is still pulled out of global memory once for every thread that needs it, and so is every element of B. Counting it from the matrix side, an element of A is read `N` times over the whole kernel and an element of B is read `M` times, which at 4096 is four thousand trips to global memory for a float that never changes.

The waste is easiest to see inside a single block. One block owns a `32 x 32` tile of C, and to compute that tile it needs a 32 row strip of A and a 32 column strip of B. Within that block, the same row of A is read by the 32 threads sitting in that row of the tile, and the same column of B is read by the 32 threads sitting in that column. That is 32 threads in the same block asking global memory for the same float, at the same time, and the hardware has no way to know they are the same request.

Shared memory is the fix. It is a small block of memory that lives on the SM, it is visible to every thread in the block, and it costs roughly 20 to 30 cycles to read instead of the several hundred a global load costs. So instead of every thread fetching what it needs, the block fetches the strip once, parks it in shared memory, and then every thread reads it from there.

The strips do not fit. A full 32 row strip of A at `K = 4096` is `32 x 4096 x 4 B = 512 KB`, and we only have 100 KB of shared memory per SM. So we walk the strips in chunks. The block loads a `32 x 32` tile of A and a `32 x 32` tile of B, does all the multiplying it can with those two tiles, then slides both windows along K and does it again. `K / 32 = 128` trips around that loop and the tile of C is finished.

![One thread block owns a 32 x 32 tile of C, named by cRow and cCol. Each trip around the outer loop copies a 32 x 32 tile of A and a 32 x 32 tile of B from global memory into shared memory as As and Bs, 8 KB resident per block, then slides A right by 32 columns and B down by 32 rows for the next trip.](/images/gemm/smem-tiling-overview.png "full")

Inside one trip the block does four things in order. Every thread copies one element of A and one element of B into shared memory, then the block syncs, then every thread runs a 32 step dot product entirely out of shared memory, then the block syncs again. There are 1024 threads and each tile has exactly 1024 elements, so the load is one element per thread with nothing left over and no loop needed.

![Load: each of the 1024 threads copies one element of the A tile and one element of the B tile into As and Bs, using tx and ty taken from its flat thread index, and then calls syncthreads. Compute: thread (tx, ty) walks row ty of As against column tx of Bs for 32 multiply accumulates with no global memory traffic at all, and syncs again before the next tile overwrites shared memory.](/images/gemm/smem-load-sync-compute.png "full")

```cuda
template <const int TILESIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  __shared__ float As[TILESIZE * TILESIZE];
  __shared__ float Bs[TILESIZE * TILESIZE];

  const int threadRow = threadIdx.x / TILESIZE;
  const int threadCol = threadIdx.x % TILESIZE;

  A += cRow * TILESIZE * K;                    
  B += cCol * TILESIZE;                        
  C += cRow * TILESIZE * N + cCol * TILESIZE; 

  float temp = 0.0;
  for (int blockIdx = 0; blockIdx < K; blockIdx += TILESIZE) {
    As[threadRow * TILESIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * TILESIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += TILESIZE;
    B += TILESIZE * N;

    for (int k = 0; k < TILESIZE; ++k) {
      temp += As[threadRow * TILESIZE + k] *  Bs[k * TILESIZE + threadCol];
    }

    __syncthreads();
  }

  C[threadRow * N + threadCol] =
      alpha * temp + beta * C[threadRow * N + threadCol];
}
```

### Mechanics

1. **The pointers move, the indices do not.** `A`, `B` and `C` are advanced to the block's own corner before the loop starts, so every index inside the loop is written relative to that corner and stays small. Sliding the window is then two lines, `A += TILESIZE` to step 32 columns right and `B += TILESIZE * N` to step 32 rows down. It is the same reason the loads stay readable: `A[threadRow * K + threadCol]` is a position inside the tile, not a position inside the matrix.

2. **The loads are still coalesced, and that is not an accident.** `threadCol` is `threadIdx.x % TILESIZE`, so consecutive threads still differ only in the column, and both `A[threadRow * K + threadCol]` and `B[threadRow * N + threadCol]` hand a warp 32 adjacent floats. This is exactly the mapping kernel 3 isolated, and it is why that kernel was worth writing down before this one.

3. **Both `__syncthreads()` are load bearing, for different reasons.** The first one guards fill before read. A thread that reaches the dot product early would otherwise read cells of `As` and `Bs` that another warp has not written yet, and it would read whatever was there before. The second one guards read before overwrite. Without it a fast warp comes around the loop and starts writing the next tile into `As` while a slow warp is still multiplying with the current one. Removing either is a race, and neither will fail every time, which is what makes them nasty.

4. **`blockIdx` is shadowed.** The loop counter is named `blockIdx`, which hides the built in variable of the same name for the rest of the loop body. It compiles and runs correctly here only because `cRow` and `cCol` were read out before the loop, so nothing inside the loop needs the real one. It is worth renaming to `bk` before this kernel gets copied into the next one.

Kernel 4 runs in **30.201 ms at 4550.9 GFLOP/s, which is 11.6% of cuBLAS at `M = N = K = 4096`**, up from 43.320 ms and 8.1%.

### The Arithmetic

The general form for a block tile of `BM x BN` walked over K in steps of `BK` is that each trip loads `BM x BK` elements of A and `BK x BN` elements of B to produce `BM x BN` results, and there are `K / BK` trips. The `BK` cancels:

```
GMEM accesses per result = (K / BK) * (BM*BK + BK*BN) / (BM*BN)
                         = K * (1/BM + 1/BN)
```

At `BM = BN = 32` that is `4096 / 16 = 256` global loads per output element, against `2K = 8192` for every kernel so far. **32 times fewer.** Counted as bytes over the whole matrix it is 549.8 GB of global loads down to **17.2 GB**. Arithmetic intensity goes from 0.25 FLOP per byte to 8, because the same 8192 FLOP per output now sit behind 1024 bytes of global traffic instead of 32768.

And the runtime went from 43.3 ms to 30.2 ms, which is 1.43 times faster.

### What Is Still Wrong

That gap is the whole point of this section. Global traffic fell by a factor of 32 and the kernel got 1.43 times faster, so global traffic was not what the kernel was waiting on any more. The work did not disappear, it moved.

Look at what the inner loop actually costs. Each of the 32 steps reads one float out of `As` and one out of `Bs` to feed a single multiply accumulate, so the shared memory access count per output element is `2K = 8192`, which is exactly the number global memory used to carry. We swapped a 500 cycle load for a 25 cycle load and kept the count identical. Every FMA in this kernel is still paying for two loads to feed it, they are just cheaper loads now.

<!-- TODO: profiler callout for kernel 4. Needs `sudo ncu --section WarpStateStats ./bench 4096`.
     Expect Stall MIO Throttle and Stall Short Scoreboard to dominate. Counters are
     admin only on this box right now, so the numbers are not in yet. -->

There is a second cost that is easy to miss. Each block runs 1024 threads and the SM caps out at 1536, so only one block is resident per SM and a third of the thread slots go unused. Those two `__syncthreads()` then stall the entire SM, because there is no second block sitting there with work to issue while this one waits at the barrier.

The fix for both is the same, and it is to stop giving each thread exactly one output. If a thread computes several results instead of one, a single value read out of shared memory can feed several multiply accumulates rather than just one, and the ratio of loads to arithmetic finally starts to move.
