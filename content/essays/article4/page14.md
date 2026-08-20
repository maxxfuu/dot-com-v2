## The Final Kernel: Retuning Against the Pipeline

Kernel 11 closed on a problem with no code in it. Its tile shape was chosen by a sweep over the non-pipelined kernel, and its shared memory padding came from a profiler reading taken when the thread tile was `4 x 4`. Both decisions were correct when they were made, and pipelining the K loop invalidated both without changing a line of either.

This section has no new technique in it at all. There is no new instruction, no new level of tiling, no stage the previous kernel did not already have. What changes is four constants and one compiler declaration, and together they close most of the remaining gap to cuBLAS.

That is worth saying plainly, because it is the least satisfying and most transferable result in the article: at this point in the ladder, the returns are no longer in the ideas.

### What Pipelining Changed About Scarcity

Before the pipeline, the K loop was `load, sync, compute, sync`, and every barrier stalled the whole block. The way to survive that is to have other blocks resident, so the search that produced the `128 x 64` shape was implicitly optimizing for occupancy, and it correctly picked small blocks that pack five to an SM.

After the pipeline there is one barrier per tile and the loads for the next tile are already in flight during this tile's arithmetic. Occupancy stops paying for itself, because the latency it was hiding is now hidden by the pipeline instead. What matters in its place is how much arithmetic each thread has available to issue between its shared memory loads, which is instruction level parallelism per thread, and that is bought with registers.

The two are in direct competition, since registers are what limits how many blocks fit. So the same search run against the pipelined kernel walks in the opposite direction: fewer, fatter threads.

Widening the thread tile from `4 x 4` to `16 x 4` and doubling `BN` back to 128 changes the inner loop ratio:

| config | SMEM loads per `dotIdx` | FMAs per `dotIdx` | FMA per float loaded |
|---|---|---|---|
| `T4x4`, WNITER 2 (kernel 11) | 8 + 8 = 16 | 64 | 4.0 |
| `T16x4`, WNITER 2 (here) | 16 + 8 = 24 | 128 | 5.3 |

And because `TM = 16` reads 16 *contiguous* floats out of the transposed `As`, that side of the load is four `LDS.128` rather than sixteen `LDS.32`. The inner loop ends up issuing six vector shared memory loads to feed 128 FMAs.

### The Padding Has To Go

The same widening un-does the previous section. The bank conflict on the transposed store is still real, and the derivation in that section is still correct: `BM = 128` is a multiple of 32, the column term drops out of the bank index, and a warp's stores land on 8 banks.

What changed is the ratio between the two paths. Per K-tile this kernel issues:

| instruction | count per K-tile |
|---|---|
| `STS`, the conflicting transposed stores | 16 |
| `LDS.128`, the vector loads feeding the FMAs | 96 |
| `FFMA` | 2048 |

The conflicting stores are 0.7% of the instruction stream, which is what the previous section already established. The new problem is that the padding is not free on the other side. `A_STRIDE = 130` is not a multiple of 4, so `dotIdx * A_STRIDE` is only 8 byte aligned on odd `dotIdx`, and a `LDS.128` cannot issue against an 8 byte aligned address. Half of those 96 vector loads split into pairs of `LDS.64`. Kernel 11's SASS shows exactly that: 96 `LDS.128` and an extra 64 `LDS.64` that should not exist.

So the padding pays a tax on the hot path, 96 loads per tile, to fix a conflict on the cold one, 16 stores per tile. Removing it is measurably faster, and `A_STRIDE` becomes `BM`:

```cuda
constexpr int A_STRIDE = BM;
```

This is the most useful single fact in the article. A profiler-driven fix, correctly derived and correctly measured, became a pessimization four kernels later, and nothing about the padded kernel itself changed to signal it. Its premise expired.

### Skipping The Read Of C

There is one saving here that is not a retune. Every kernel since the 2D register tiling section ends with the same epilogue:

```cuda
float4 tmp = reinterpret_cast<float4 *>(cPtr)[0];   // read C back
tmp.x = alpha * threadResults[accIdx + 0] + beta * tmp.x;
```

When `beta == 0` that read is pure waste. The value loaded is multiplied by zero and thrown away, and it costs a full pass over C: `M x N x 4 = 67 MB` of DRAM reads that cannot hit in L2, because C is 67 MB against a 64 MB cache and is streamed exactly once. At 960 GB/s that is roughly 70 microseconds on a 4 ms kernel, and the measured gain is larger than that, because those loads also occupy issue slots in the epilogue.

The choice is made once on the host rather than once per thread, with the kernel templated on a `bool` and both instantiations compiled:

```cuda
if constexpr (BETA_IS_ZERO) {
  // build the float4 and store it, no read of C at all
} else {
  // the kernel 11 epilogue
}
```

This is not a benchmark artifact. `beta == 0` is the case in essentially every GEMM inside a neural network forward pass, and cuBLAS special cases it for the same reason. The kernel stays fully general in `beta`; the fast path is a fast path, not a restriction.

### The Declaration That Was Worth More Than Any Of It

Every kernel in this series from warp tiling onwards is declared `__launch_bounds__(NUM_THREADS)`. This one is declared `__launch_bounds__(NUM_THREADS, 1)`, and that second argument is the single largest change in the file:

| declaration | registers | throughput |
|---|---|---|
| `__launch_bounds__(NT)` | 202 | 90.0% |
| `__launch_bounds__(NT, 1)` | **227** | **95.7 to 95.9%** |

The one argument form only tells `ptxas` the maximum block size. It is then free to apply its own heuristic about how many blocks it would like to fit per SM, and that heuristic is written for kernels that want occupancy. It duly holds this kernel to 202 registers trying to fit a third block on each SM. The second argument, `minBlocksPerMultiprocessor = 1`, tells it the truth: one block per SM is fine here, so stop rationing.

Note the direction. We are asking for lower occupancy on purpose, and the flag that delivers it is the one that sounds like it is asking for less. The 227 registers that make the wide thread tile work are not something the configuration produces on its own. They have to be permitted.

`ptxas` reports 0 bytes spilled at 227 registers, and that headroom is the only reason this configuration is legal. It is worth re-checking with `-Xptxas -v` after any edit, because the cliff on the other side of it is very steep.

### The Kernel

```cuda
const int BM = 128, BN = 128, BK = 16, WM = 64, WN = 64, WNITER = 2;
const int TM = 16, TN = 4, NT = 128;

dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
dim3 blockDim(NT);
```

```cuda
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          const bool BETA_IS_ZERO>
__global__ void __launch_bounds__(NUM_THREADS, 1)
sgemm_final(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  constexpr int A_STRIDE = BM;

  __shared__ float As[2][BK * A_STRIDE];
  __shared__ float Bs[2][BK * BN];

  // ... prologue, main loop and staging are kernel 11's, unchanged ...

  if constexpr (BETA_IS_ZERO) {
    for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
      for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
        float *cSub = C + (wSubRow * WSUBM) * N + wSubCol * WSUBN;
        for (int m = 0; m < TM; ++m) {
          for (int n = 0; n < TN; n += 4) {
            float *cPtr = &cSub[(threadRowInWarp * TM + m) * N + threadColInWarp * TN + n];
            const int accIdx = (wSubRow * TM + m) * (WNITER * TN) + wSubCol * TN + n;
            float4 out;
            out.x = alpha * threadResults[accIdx + 0];
            out.y = alpha * threadResults[accIdx + 1];
            out.z = alpha * threadResults[accIdx + 2];
            out.w = alpha * threadResults[accIdx + 3];
            reinterpret_cast<float4 *>(cPtr)[0] = out;
          }
        }
      }
    }
  } else {
    // kernel 11's epilogue, which reads C back for the beta term
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
void launch_sgemm_final(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
  dim3 blockDim(NUM_THREADS);
  if (beta == 0.0f)
    sgemm_final<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, true>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  else
    sgemm_final<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, false>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
```

### Mechanics

1. **`WMITER` collapses to 1.** With `WM = WN = 64`, `TM = 16`, `TN = 4` and `WNITER = 2`, the derived `WMITER = (64 x 64) / (32 x 16 x 4 x 2) = 1`, so `WSUBM = WM = 64` and the warp covers its rectangle in a single pass down the M dimension. The per warp thread grid is `(WSUBM / TM) x (WSUBN / TN) = 4 x 8 = 32`. The three level tiling machinery from the warp tiling section is all still there, with one of its loops having degenerated to a single iteration.

2. **128 accumulators per thread.** `WMITER x TM x WNITER x TN = 1 x 16 x 2 x 4 = 128` floats live in registers for the whole kernel, exactly the ceiling the autotuner's `cfg_valid()` allows before spilling is certain. This is the config sitting on that boundary rather than near it.

3. **The block is still 128 threads, and the grid is now `32 x 32`.** That is 1024 blocks over 84 SMs, about 12 waves deep. Small grids are where scheduling effects and tail effects show up, and this is the point at which block rasterization schemes start being suggested. It was tried here and did nothing, because at 12 waves L2 already captures the reuse those schemes are designed to create.

Kernel 12 runs in **3.929 ms at 34978.3 GFLOP/s, which is 97.1% of cuBLAS at `M = N = K = 4096`**, up from 4.281 ms and 89.2%. That is 9.0% faster than the double buffered kernel and 85 times faster than the naive one.

### Where The Points Came From

Measured cumulatively, adding one change at a time to the double buffered kernel:

| | share of cuBLAS |
|---|---|
| kernel 11 as shipped | 85.5 to 86.4% |
| plus the `beta == 0` epilogue | 88.6% |
| plus the retune to `BN = 128`, `T8x8` | 90.3 to 91.4% |
| plus the retune to `T16x4`, unpadded | 94.2 to 95.1% |
| plus `__launch_bounds__` min-blocks = 1 | 95.7 to 95.9% |

Those percentages come from an earlier benchmark run than the results table at the top of this article, so the absolute values are a point or two below the 97.1% quoted above and should be read as a shape rather than as figures. The clocks were not pinned, so any gap under about one point in that column is noise.

The resource budget the winning configuration actually lands on:

| | kernel 11 | kernel 12 |
|---|---|---|
| thread tile | 4 x 4 | 16 x 4 |
| block tile | 128 x 64 | 128 x 128 |
| accumulators per thread | 64 | 128 |
| registers per thread | 150 | **227** |
| shared memory per block | 24832 B | 32768 B |
| blocks per SM | 3 | **2** |
| occupancy | 25.0% | **16.7%** |

Occupancy has now fallen for four kernels in a row while throughput rose every time, from 41.7% at warp tiling to 16.7% here. That trend is the clearest single answer this article has to the question of what these optimizations are actually doing: they are converting warp level parallelism into instruction level parallelism, one register at a time.

### A Measurement That Nearly Shipped

One thing in this kernel's history is worth more than any of the optimizations in it.

An earlier version of the `beta == 0` work claimed that `if constexpr` was worth four points over an equivalent runtime `if (beta == 0.0f)` inside the kernel, and had numbers to back it up, and a mechanism to explain it: both epilogues live in one function, so the register allocator has to budget for the worse one. That is a plausible story. It is also wrong.

The two variants had been built with different `__launch_bounds__`. The comparison moved two variables at once and attributed the entire difference to the one that was interesting. Re-run with `__launch_bounds__` held fixed:

| epilogue selection | throughput | registers |
|---|---|---|
| `if constexpr` plus host dispatch | 95.08 / 95.16 / 96.18% | 227 |
| runtime `if (beta == 0.0f)` | 95.05 / 95.12 / 95.91% | 221 |

Indistinguishable. All four points belonged to the launch bounds argument.

The `if constexpr` form is kept, because it is the honest way to say "these are two different kernels" and it costs nothing, but it is not why this kernel is fast. The lesson is about method rather than about GEMM: an A/B that changes two things at once produces a real, repeatable and entirely false number, and a plausible mechanism will always suggest itself for whichever of the two changes you were already interested in.

### What Could Be Improved, And What It Would Take

We are at 3.929 ms against cuBLAS at 3.817 ms, so the remaining gap is 2.9% at `M = N = K = 4096`. Run to run spread on this kernel is 1.17%, so that gap is real but only just.

Several things were tried against it and did not work, and they are worth recording because a negative result measured is more useful than a technique listed:

- **Block rasterization**, grouping blocks into columns so that consecutive blocks share tiles of B, swept over four group sizes. Landed inside noise in both directions. The grid is only 1024 blocks over 84 SMs and L2 is 64 MB, so the reuse the swizzle is designed to create is already there.
- **Shared memory to register double buffering**, a second pipeline stage inside the `dotIdx` loop. Slower. The loop is fully unrolled at compile time, so `ptxas` already schedules those loads ahead of their FMAs, and hand rolling it only adds indexing.
- **`BK = 8` and `BK = 32`.** The first halves the reuse of every loaded element and lost around ten points. The second does not fit the 48 KB static shared memory cap once double buffered.
- **256 thread blocks** in three shapes. All fell to one block per SM and lost six to eleven points.

What would actually close 2.9% is not on the list, because it is not in this series' scope. cuBLAS at this size is running a hand tuned assembly kernel with a schedule no compiler will reproduce from CUDA C, and beyond that it can select tensor core paths that this article deliberately never touches. Every kernel here runs on FP32 CUDA cores, and the 3.817 ms reference is `cublasSgemm` on the same units, which is what makes the comparison fair. Switching to TF32 or FP16 with FP32 accumulation would make this kernel several times faster and would be answering a different question.

One measurement makes the size of that remaining gap concrete. Nsight puts this kernel at **2.55 warp cycles per issued instruction with `Stall Not Selected` at 32.5%**, and cuBLAS's cutlass kernel at **2.55 cycles with Not Selected at 33.3%**. The two stall profiles are indistinguishable. Whatever the last 2.9% is, it is not a stall this kernel suffers and cuBLAS avoids. Both sit in the regime where the scheduler has more eligible warps than it can issue, and what separates them is the instruction schedule itself.

The more honest closing note is about the shape of the ladder rather than its last rung. Twelve kernels moved us from 410.5 GFLOP/s to 34978.3, which is 85 times, and the techniques responsible are not exotic: read memory in the order it is laid out, load each value once per block instead of once per thread, give each thread enough work that its loads amortize, hand the hardware its widest instructions, keep each warp's footprint compact, overlap loading with arithmetic, and then measure everything again because the earlier decisions have gone stale. Almost all of it is data movement, and almost none of it is arithmetic.

The last one deserves the final word. Three separate optimizations in this article were undone or overturned by later ones: the square tile the access count formula recommended, the padding the profiler recommended, and the occupancy that every introductory guide recommends. None of those were mistakes at the time. They stopped being right when the kernel around them changed, and the only way that was ever going to surface was by measuring again.
