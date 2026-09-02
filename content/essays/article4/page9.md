## 2D Register Tiling: The Outer Product

Kernel 5 tiled one dimension and got a one sided result. A thread owning a column of C shares a B value across all `TM` of its outputs and shares nothing on the A side, so `1 + TM` loads feed `TM` FMAs and the ratio bottoms out at one load per FMA no matter how tall the column gets.

Give the thread a rectangle instead. If a thread owns a `TM x TN` patch of C, then for one step of K it needs `TM` values of A, the rows of its patch, and `TN` values of B, the columns of its patch. Every one of the `TM x TN` outputs is a product of one of those A values with one of those B values, so `TM + TN` loads feed `TM x TN` FMAs.

That is an outer product. The thread loads a short column vector from `As` and a short row vector from `Bs`, forms every pairwise product between them, and accumulates the whole `TM x TN` grid at once. At `TM = TN = 8` that is 16 loads feeding 64 FMAs, four FMAs per load, against kernel 5's 9 loads feeding 8.

![The block owns a 128 x 128 tile of C divided into 256 patches, one per thread, with As held as 128 x 8 and Bs as 8 x 128. One thread owns an 8 x 8 patch, fed by the 8 rows of A and 8 columns of B that intersect at it. The inset compares the two shapes directly: kernel 5's TM x 1 column takes 1 + TM = 9 loads to feed 8 FMAs, which is 0.9 FMAs per load, while kernel 6's TM x TN patch takes TM + TN = 16 loads to feed 64 FMAs, which is 4 FMAs per load.](/images/gemm/register-2d-patch.png "full")

The ratio is `(TM + TN) / (TM x TN)`, which falls as the patch grows, and that is the first place in this series where a knob exists that keeps paying. It is also the first knob with a real price, and the price is registers: those `TM x TN` accumulators live in registers for the entire lifetime of the kernel.

### The Kernel

The block tile doubles again in both dimensions, to `128 x 128`, and each thread produces an `8 x 8` patch, so the block needs `(128 x 128) / (8 x 8) = 256` threads. Note that the thread count is falling as the tiles grow, 1024 to 512 to 256, because the work per thread is rising faster than the tile is:

```cuda
const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
dim3 blockDim((BM * BN) / (TM * TN));   // 256

sgemm_register_tiling_2d<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

```cuda
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_register_tiling_2d(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

 const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  A += cRow * BM * K;                   
  B += cCol * BN;                       
  C += cRow * BM * N + cCol * BN;       

  const int innerColA = threadIdx.x % BK;             
  const int innerRowA = threadIdx.x / BK;             
  constexpr int ROW_STRIDE_A = NUM_THREADS / BK;                

  const int innerColB = threadIdx.x % BN;             
  const int innerRowB = threadIdx.x / BN;             
  constexpr int ROW_STRIDE_B = NUM_THREADS / BN;                

  float threadResults[TM * TN] = {0.0f};

  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

    for (int offset = 0; offset < BM; offset += ROW_STRIDE_A) {
      As[(innerRowA + offset) * BK + innerColA] =
          A[(innerRowA + offset) * K + innerColA];
    }

    for (int offset = 0; offset < BK; offset += ROW_STRIDE_B) {
      Bs[(innerRowB + offset) * BN + innerColB] =
          B[(innerRowB + offset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {

      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }

      for (int i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }

      for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
      const int cIdx = (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN;
      C[cIdx] = alpha * threadResults[resIdxM * TN + resIdxN] + beta * C[cIdx];
    }
  }
}
```

### Mechanics

1. **The loads need loops now, and that is a consequence of the arithmetic.** `As` holds `BM x BK = 1024` floats and there are only 256 threads, so each thread copies four elements rather than one. The loop steps by `ROW_STRIDE_A = NUM_THREADS / BK = 32`, which is how many complete rows of the A tile 256 threads cover in one pass, and it runs `BM / ROW_STRIDE_A = 4` times. `Bs` is also 1024 floats but its rows are 128 wide, so `ROW_STRIDE_B = NUM_THREADS / BN = 2` and the loop again runs four times over `BK = 8` rows. Both loops keep `innerCol` on the fast moving index, so every pass is still a fully coalesced global read.

2. **Three loops, and only the last one is arithmetic.** The `dotIdx` body is a load of `TM` values into `regM`, a load of `TN` values into `regN`, and then a `TM x TN` product grid. Splitting it this way is what makes the reuse explicit to the compiler: the two small loops are the only shared memory traffic in the kernel, and the nested pair underneath them touches registers exclusively. Fusing the loads into the product loops would read the same values `TM` and `TN` times over and undo the entire section.

3. **`regM` reads down a column of `As` with stride `BK`.** `As[(threadRow * TM + i) * BK + dotIdx]` walks `i` across rows of the tile, so consecutive `i` are 8 floats apart in shared memory. Shared memory has no coalescing requirement, but it does have banks, and a strided read of this shape is the thing kernel 10 eventually has to look at. It is not costing us yet at `BK = 8`.

4. **The register budget is the real ceiling on `TM` and `TN`.** A thread holds `TM x TN = 64` accumulators plus `TM + TN = 16` staging values, and `ptxas` reports 93 registers per thread for this kernel with nothing spilled. At 256 threads that is 2 blocks per SM. Doubling `TM` and `TN` to 16 would want 256 accumulators, which is past the 255 register hard limit per thread, so the loop would spill to local memory and the kernel would collapse. The knob does keep paying, right up until it does not pay at all.

Kernel 6 runs in **6.996 ms at 19644.4 GFLOP/s, which is 54.6% of cuBLAS at `M = N = K = 4096`**, up from 10.506 ms and 36.3%. Half of cuBLAS, from a kernel whose inner loop is nine lines of scalar arithmetic.

### The Arithmetic

Both counts move this time, and it is worth writing the general form down once because it explains every tile shape decision left in the series. For a block tile of `BM x BN` walked over K in steps of `BK`, each trip loads `BM x BK` elements of A and `BK x BN` elements of B to produce `BM x BN` results, over `K / BK` trips:

```
GMEM accesses per result = (K / BK) * (BM*BK + BK*BN) / (BM*BN)
                         = K * (1/BM + 1/BN)
```

`BK` cancels. Global traffic per output depends only on the two block tile dimensions, and it is a sum of reciprocals, which means that for a fixed tile area a square tile minimizes it. At `BM = BN = 128` that is `K / 64 = 64` loads per output element, half of kernel 5's 128, for a total of **4.29 GB**. Remember that this says square is best, because kernel 8 is going to pick a rectangle and win.

The shared memory count has exactly the same shape one level down, with the thread tile in place of the block tile:

```
SMEM accesses per result = K * (1/TM + 1/TN)
                         = 4096 / 4
                         = 1024
```

That is **4.5 times fewer** than kernel 5's 4608, and 8 times fewer than kernel 4's 8192. Runtime moved 1.50 times. As with kernel 5 the count and the clock disagree, but this time the count over-predicts, which is the signal that shared memory reads have stopped being the dominant cost and something else is taking over.

### Where We Are On The Roofline

Arithmetic intensity is now `137.44 GFLOP / 4.29 GB = 32 FLOP/byte`, up from 8 at kernel 4 and 16 at kernel 5. The ridge point is 58.6, so for the first time the kernel is within striking distance of the compute bound region rather than three orders of magnitude away from it.

More useful than the position is what happens to the two floors. The memory floor for 4.29 GB at 960 GB/s is `4.47 ms`, and the compute floor has been 2.44 ms all along. Those two numbers are now within a factor of two of each other, where at kernel 1 they were 573 ms against 2.44 ms. The model has stopped being a story about bandwidth and started being a story about both resources at once, which is what a well tiled GEMM is supposed to look like.

We measured 6.996 ms against a 4.47 ms memory floor and a 2.44 ms compute floor, so we are 1.56 times off the binding one. The sloped ceiling at 32 FLOP/byte sits at `32 x 960 = 30720 GFLOP/s` and we are at 19644, which is 64% of it. Every kernel from here on is trying to close that last 1.56 times, and none of them will do it by reducing traffic further.

### The Occupancy Trap

There is one number in this section that looks like a regression. Kernel 5 ran at 66.7% occupancy. Kernel 6 uses 93 registers per thread, which at 256 threads per block allows 2 blocks per SM, 8 warps out of a possible 48, or **33.3% occupancy**. We halved the occupancy and the kernel got 1.5 times faster.

This is the point in the series where "higher occupancy is better" has to be retired. Occupancy is not a goal, it is one of two ways to hide latency. Many resident warps hide it by having something else to run whenever one warp stalls. Instruction level parallelism hides it by giving a single warp a long run of independent work, and 64 independent FMAs accumulating into 64 separate registers is exactly that: none of them depends on the result of another, so the scheduler can keep the pipeline full from one warp alone.

Register tiling deliberately trades the first mechanism for the second. Every kernel from here spends registers to buy independent work per thread, occupancy keeps falling, and performance keeps rising. The final kernel in this series runs at 16.7%.

### What Could Be Improved

The remaining shared memory reads are few, but each one is its own instruction. The inner loop issues 16 separate 32 bit `LDS` instructions per `dotIdx`, eight down a column of `As` and eight along a row of `Bs`, to feed 64 FMAs. The data volume is no longer the problem; the number of instructions carrying it is.

That distinction has a name in the profiler, and the measurement is more interesting than a confirmation would have been. `Stall MIO Throttle` is a warp waiting for a shared memory instruction to *issue* rather than for the data it asked for, and it is what kernel 4 was drowning in: 23.9 of its 40.5 warp cycles per issued instruction, 58.9%. Kernel 5 cut that to 6.6 of 21.0, 31.4%. This kernel comes in at **7.49 warp cycles per issued instruction, and Nsight names no dominant stall reason for it at all**. Kernel 5's 31.4% was large enough to trip the rule; nothing here is.

So MIO throttle is not what bounds this kernel, and the case for the next one does not rest on it. It rests on the count above. Sixteen `LDS` per `dotIdx` to feed 64 FMAs is a lot of issue slots spent carrying very little data, and issue slots are now the scarce resource. The fix does not move a single byte less. It moves the same bytes in a quarter of the instructions.

<!-- TODO: smsp__inst_executed.sum on kernels 6 and 7, to state the instruction
     reduction as measured rather than derived. WarpStateStats is done. -->
