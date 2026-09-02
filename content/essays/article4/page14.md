## Double Buffering: Software Pipelining the K-Loop

Kernel 10 ended with almost nothing left to remove. Roughly 95% of the instructions issued per K-tile are `FFMA`, the loads are vectorized, the tile shape is the best of a measured sweep, and the bank conflicts are gone. And it is still at 81.9% of cuBLAS.

What is left is not work, it is waiting. The K loop has had the same four beats since kernel 4:

```
load a tile into shared memory
__syncthreads()
compute 16 steps out of shared memory
__syncthreads()
```

Those two barriers make the two halves mutually exclusive. While the block is issuing global loads, the FMA pipes have nothing to do, because the data they would need is exactly what is being fetched. While the block is computing, the load pipes have nothing to do. Every SM alternates between two idle resources when it could be using both.

Residency has been papering over this. With 5 blocks per SM, one block stalling at a barrier leaves four others with work to issue, which is a large part of why kernel 8's small blocks beat kernel 7's large ones. But that is latency hiding across blocks, and it only works while there are spare blocks.

The direct fix is to overlap the two phases within a single block. Keep two tiles in shared memory instead of one. Compute on tile `n` while the loads for tile `n + 1` are already in flight, then swap. The loads still cost what they cost; they just happen underneath the arithmetic instead of in front of it.

### Two Stages, One Barrier

Doubling the buffers is what makes the barrier count fall, and the reason is worth stating precisely, because it is the entire correctness argument for the kernel.

With one buffer, the second `__syncthreads()` exists to stop a fast warp from overwriting the tile a slow warp is still reading. With two buffers that hazard is gone by construction: warps read from `As[curStage]` and write into `As[nextStage]`, which are different memory. The only remaining hazard is the original one, that a warp might read a stage before every warp has finished filling it, and that needs one barrier per trip rather than two.

The other half of the mechanism is `cp.async`, which is a Blackwell-era instruction that copies from global memory straight into shared memory without the data passing through registers or through the warp at all. Issue it, keep going, and check later that it landed:

```cuda
__device__ __forceinline__ void cp_async16(float *smemDst, const float *gmemSrc) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smemDst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmemSrc));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
}
```

`cp.async` copies bytes verbatim, which is precisely why only B can use it. `Bs` is stored in the same orientation it is read from global memory, so a straight byte copy is correct. `As` is stored transposed, and a transpose is not a copy: the four floats of a `float4` have to end up in four different rows, `A_STRIDE` apart. That has to pass through a register. So A is prefetched into a `float4` register array early in the loop and written into shared memory late, and B goes global to shared directly with no register involvement.

### The Kernel

The configuration is unchanged from kernels 8 and 10, including the `A_STRIDE` padding inherited from the previous section:

```cuda
const int BM = 128, BN = 64, BK = 16;
const int WM = 64, WN = 32, WNITER = 2;
const int TM = 4, TN = 4, NUM_THREADS = 128;
```

```cuda
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_double_buffered(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  
  constexpr int A_STRIDE = BM + (32 / BK) % 8;

  __shared__ float As[2][BK * A_STRIDE];
  __shared__ float Bs[2][BK * BN];

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpCol = warpIdx % (BN / WN);
  const int warpRow = warpIdx / (BN / WN);

  constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;

  const int threadIdxInWarp = threadIdx.x % WARPSIZE;
  const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  constexpr int ROW_STRIDE_A = (NUM_THREADS * 4) / BK;
  constexpr int A_PASSES = BM / ROW_STRIDE_A;

  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  constexpr int ROW_STRIDE_B = NUM_THREADS / (BN / 4);

  float4 aFrag[A_PASSES];

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

#pragma unroll
  for (int pass = 0; pass < A_PASSES; ++pass)
    aFrag[pass] = reinterpret_cast<const float4 *>(
        &A[(innerRowA + pass * ROW_STRIDE_A) * K + innerColA * 4])[0];
#pragma unroll
  for (int pass = 0; pass < A_PASSES; ++pass) {
    const int row = innerRowA + pass * ROW_STRIDE_A;
    As[0][(innerColA * 4 + 0) * A_STRIDE + row] = aFrag[pass].x;
    As[0][(innerColA * 4 + 1) * A_STRIDE + row] = aFrag[pass].y;
    As[0][(innerColA * 4 + 2) * A_STRIDE + row] = aFrag[pass].z;
    As[0][(innerColA * 4 + 3) * A_STRIDE + row] = aFrag[pass].w;
  }
#pragma unroll
  for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B)
    cp_async16(&Bs[0][(innerRowB + offset) * BN + innerColB * 4],
               &B[(innerRowB + offset) * N + innerColB * 4]);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const int numTiles = K / BK;
  int curStage = 0;
  for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
    const int nextStage = curStage ^ 1;
    const bool hasNextTile = (tileIdx + 1) < numTiles;

    if (hasNextTile) {

      const float *aNext = A + (tileIdx + 1) * BK;
      const float *bNext = B + (long)(tileIdx + 1) * BK * N;
#pragma unroll
      for (int pass = 0; pass < A_PASSES; ++pass)
        aFrag[pass] = reinterpret_cast<const float4 *>(
            &aNext[(innerRowA + pass * ROW_STRIDE_A) * K + innerColA * 4])[0];
#pragma unroll
      for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B)
        cp_async16(&Bs[nextStage][(innerRowB + offset) * BN + innerColB * 4],
                   &bNext[(innerRowB + offset) * N + innerColB * 4]);
      cp_async_commit();
    }

#pragma unroll
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
#pragma unroll
      for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
#pragma unroll
        for (int i = 0; i < TM; ++i)
          regM[wSubRow * TM + i] =
              As[curStage][dotIdx * A_STRIDE + warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i];
#pragma unroll
      for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
#pragma unroll
        for (int i = 0; i < TN; ++i)
          regN[wSubCol * TN + i] =
              Bs[curStage][dotIdx * BN + warpCol * WN + wSubCol * WSUBN + threadColInWarp * TN + i];
#pragma unroll
      for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
#pragma unroll
        for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
#pragma unroll
          for (int m = 0; m < TM; ++m)
#pragma unroll
            for (int n = 0; n < TN; ++n)
              threadResults[(wSubRow * TM + m) * (WNITER * TN) + wSubCol * TN + n] +=
                  regM[wSubRow * TM + m] * regN[wSubCol * TN + n];
    }

    if (hasNextTile) {
      
#pragma unroll
      for (int pass = 0; pass < A_PASSES; ++pass) {
        const int row = innerRowA + pass * ROW_STRIDE_A;
        As[nextStage][(innerColA * 4 + 0) * A_STRIDE + row] = aFrag[pass].x;
        As[nextStage][(innerColA * 4 + 1) * A_STRIDE + row] = aFrag[pass].y;
        As[nextStage][(innerColA * 4 + 2) * A_STRIDE + row] = aFrag[pass].z;
        As[nextStage][(innerColA * 4 + 3) * A_STRIDE + row] = aFrag[pass].w;
      }
      cp_async_wait<0>();
      __syncthreads();   
    }
    curStage = nextStage;
  }

  for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
    for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
      float *cSub = C + (wSubRow * WSUBM) * N + wSubCol * WSUBN;
      for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; n += 4) {
          float *cPtr = &cSub[(threadRowInWarp * TM + m) * N + threadColInWarp * TN + n];
          float4 tmp = reinterpret_cast<float4 *>(cPtr)[0];
          const int accIdx = (wSubRow * TM + m) * (WNITER * TN) + wSubCol * TN + n;
          tmp.x = alpha * threadResults[accIdx + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[accIdx + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[accIdx + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[accIdx + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(cPtr)[0] = tmp;
        }
      }
    }
  }
}
```

### Mechanics

1. **The prologue is the loop body with the compute removed.** Before the loop can compute on stage 0, something has to fill stage 0, and that is what the three blocks above `numTiles` do. This is the structural cost of software pipelining: the steady state is efficient, and you pay for it with a prologue that has no arithmetic to overlap with, once per block rather than once per tile.

2. **The A prefetch and the A store sit on opposite sides of the compute.** `aFrag` is loaded from global memory at the top of the iteration and written into `As[nextStage]` at the bottom, with 2048 FMAs in between. That gap is the point. A global load has a latency of several hundred cycles, and the arithmetic in the middle is what covers it. Moving those two loops next to each other would compile, run, produce identical results, and lose most of the benefit of the section.

3. **`cp_async_wait<0>()` waits for all outstanding groups, and the placement matters more than the argument.** It sits immediately before the barrier at the end of the iteration, so the B copies for tile `n + 1` have the entire compute phase to complete in. The `commit_group` right after issuing them is what makes them a group that can be waited on.

4. **`cur ^ 1` rather than `(cur + 1) % 2`.** Both work with two stages. The XOR makes it obvious that the two indices are a pair being swapped rather than a counter being advanced, which matters if the stage count ever goes to 3 and the XOR stops being correct.

5. **`#pragma unroll` on everything.** Every loop in the hot path has a compile-time trip count, and unrolling them is what lets `ptxas` interleave the independent global loads, shared loads and FMAs into a single scheduled block. Without it the pipelining is expressed in the source and then thrown away by the scheduler.

Kernel 11 runs in **4.281 ms at 32107.5 GFLOP/s, which is 89.2% of cuBLAS at `M = N = K = 4096`**, up from 4.660 ms and 81.9%. That is 8.9% faster.

### What The Overlap Cost

This is the first optimization in the series that does not reduce anything. Global accesses per output element are still 96 and shared accesses are still `K / 4`, exactly as they were for kernels 8 and 10. The instruction mix is nearly identical. The only thing that changed is when the instructions issue relative to one another, so the accounting for this section has to be a cost accounting rather than a saving.

| | kernel 10 | kernel 11 |
|---|---|---|
| barriers per K-tile | 2 | 1 |
| shared memory per block | 12416 B | 24832 B |
| registers per thread | 94 | **150** |
| blocks per SM | 5 | **3** |
| binding limit | registers | registers |
| occupancy | 41.7% | **25.0%** |

Two stages of shared memory doubles the shared memory, which on its own would still allow 4 blocks per SM inside the 100 KB budget. It is the registers that bind. 150 registers per thread rounds up to 4864 per warp, 19456 per block of four warps, and `65536 / 19456 = 3`. The `aFrag` staging array and the deeper scheduling window the unrolled loop needs are what pushed 94 up to 150.

So the trade is explicit: **this kernel spends the residency that made barriers cheap in order to have fewer barriers.** It goes from 5 resident blocks with 2 barriers each to 3 resident blocks with 1 barrier each, and it wins by 8.9%. Both mechanisms were hiding the same latency, and the in-block one turns out to hide it better than the across-block one, which is the same lesson the occupancy discussion at kernel 6 reached from a different direction.

The profiler agrees the barriers stopped mattering. This kernel runs at **3.94 warp cycles per issued instruction against kernel 10's 6.81**, and its largest named stall is `Stall Not Selected` at 38.0%, warps that were ready and simply were not chosen. Nothing is waiting on memory or on a barrier any more; there is more eligible work than there are issue slots.

It is worth being careful here, because it would be easy to write that this kernel trades shared memory for barriers. It does not. It trades registers for barriers, and the shared memory increase is free. The distinction is measurable in one command, `nvcc -Xptxas -v`, and it changes which knob you would reach for next.

### What Could Be Improved

Nothing, on its own terms. This kernel does what it set out to do, and every technique the series has to offer is now in it: coalesced global access, shared memory tiling, two levels of register tiling, warp tiling, vectorized loads, a padded shared buffer, and a software pipeline. There is no eighth technique waiting.

What is wrong is older than this kernel. Every configuration decision behind it was made against a kernel that no longer exists.

The tile shape came from a sweep over the non-pipelined kernel, where barriers were expensive and residency was the way to hide them, and that sweep duly picked the small blocks that pack five to an SM. The padding came from a profiler reading taken when the thread tile was `4 x 4` and the load side was narrow, so the store conflicts really were a meaningful share of shared memory traffic. Both decisions were correct when they were made. The pipeline changed what is scarce, and neither of them was revisited.

Nothing in the code signals that. A stale tuning decision does not throw an error or show up as a stall. It just sits there being a slightly wrong constant, and the only way to find it is to run the search again against the kernel you actually have.
