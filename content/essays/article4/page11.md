## Autotuning: Searching the Tile-Shape Space

Kernel 8 ended with an admission: its tile shape came from measurement, and the argument for it was written afterwards. The access count formula would have picked a square block tile, the register argument would have picked an `8 x 8` thread tile, and both would have been slower than the `128 x 64` with `4 x 4` that actually won.

If reasoning picks the wrong configuration, stop reasoning about the configuration. There are nine numbers in kernel 8's template parameter list, they interact through at least five different hardware resources, and the kernel body does not care which values it gets. That is the shape of a search problem, not an analysis problem.

The interesting part is not the search. It is that most of the space is not slow, it is **illegal**.

### Most Configurations Are Not Wrong By A Little

Kernel 8's body makes about a dozen unstated assumptions. It assumes the block divides into whole warps, that the block tile divides into whole warp tiles, that `WMITER` comes out as an integer, that each sub-tile is covered by exactly 32 threads with none left over, that `BK` and `TN` are multiples of 4 so the `float4` casts are aligned, that the load loop strides divide their tiles, and that every element of C is owned by exactly one thread.

Violate any of them and the kernel still compiles and still launches. It reads past the end of `As`, or leaves part of the tile unwritten, or has two threads accumulate into the same output. The failure is silent and the result is garbage, which at `alpha = 1, beta = 0` looks exactly like a working kernel that got the wrong answer.

So the legality check has to happen before the kernel exists, which means at compile time:

```cuda
constexpr bool cfg_valid(int BM, int BN, int BK, int WM, int WN, int WNITER,
                         int TM, int TN, int NT) {
  // block must be a whole number of warps, tiled evenly by warp tiles
  if (NT % WARPSIZE != 0) return false;
  if (BM % WM != 0 || BN % WN != 0) return false;
  if ((BM / WM) * (BN / WN) != NT / WARPSIZE) return false;

  // a warp's WM x WN rectangle must decompose into whole WMITER x WNITER steps
  const int denom = WARPSIZE * TM * TN * WNITER;
  if (denom <= 0 || (WM * WN) % denom != 0) return false;
  const int WMITER = (WM * WN) / denom;
  if (WMITER < 1 || WM % WMITER != 0 || WN % WNITER != 0) return false;

  // each sub-tile must be covered by exactly 32 threads
  const int WSUBM = WM / WMITER, WSUBN = WN / WNITER;
  if (WSUBM % TM != 0 || WSUBN % TN != 0) return false;
  if ((WSUBM / TM) * (WSUBN / TN) != WARPSIZE) return false;

  // float4 vectorization requires 4-element alignment everywhere
  if (BK % 4 != 0 || BN % 4 != 0 || TN % 4 != 0) return false;

  // the SMEM load loops must tile each cache exactly, with no thread
  // computing an out-of-range innerRow
  if ((NT * 4) % BK != 0) return false;
  const int rowStrideA = (NT * 4) / BK;
  if (rowStrideA < 1 || BM % rowStrideA != 0) return false;
  if (NT % (BN / 4) != 0) return false;
  const int rowStrideB = NT / (BN / 4);
  if (rowStrideB < 1 || BK % rowStrideB != 0) return false;

  // every output element must be owned by exactly one thread
  if ((long)NT * WMITER * TM * WNITER * TN != (long)BM * BN) return false;

  // static __shared__ is capped at 48KB without an opt-in
  if ((BM * BK + BK * BN) * 4 > 48 * 1024) return false;
  // accumulators alone above ~128 registers guarantees spilling
  if (WMITER * TM * WNITER * TN > 128) return false;

  return true;
}
```

```cuda
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NT>
void tryConfig(const char *name) {
  if constexpr (!cfg_valid(BM, BN, BK, WM, WN, WNITER, TM, TN, NT)) {
    printf("  %-46s  skipped (illegal shape)\n", name);
    return;
  } else {
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    dim3 block(NT);
    // ... launch, verify against cuBLAS, time, record ...
  }
}
```

### Mechanics

1. **Every check is an assumption the kernel body makes, written down.** That is the useful way to read `cfg_valid()` and the useful way to write one: walk the kernel, and each time an expression assumes a division comes out whole, add the guard. Two of the checks are not legality at all but hard limits, the 48 KB static shared memory cap and the accumulator count above which spilling is certain. They belong in the same function because from the search's point of view "will not compile", "will corrupt memory" and "will spill and be useless" are the same answer.

2. **`if constexpr` rather than `if`.** A plain runtime `if` would not help, because both branches of a runtime conditional are instantiated. `sgemm_warp_tiled<...>` would be compiled for the illegal config whether or not the branch ever executes, and it would either fail to compile or sit there ready to be called. `if constexpr` discards the untaken branch before instantiation, so an illegal tuple never becomes a kernel at all. The legality check and the compile become the same event.

3. **The order of the checks matters.** Each guard protects the divisions in the guards below it, so `denom <= 0` is tested before anything divides by it and `WM % WMITER` is tested only after `WMITER` is known to be at least 1. A `constexpr` function evaluated at compile time will happily hand you a division by zero diagnostic if the checks are written out of order.

### What The Sweep Found

The space here is 16 hand-picked points rather than an exhaustive grid, chosen to vary one or two dimensions at a time around kernel 7's shape. Two are rejected at compile time. Ranked by throughput, the fourteen that survive:

| config | ms | GFLOP/s |
|---|---|---|
| **128x64x16 W64x32 I2 T4x4 128t** | **4.55** | **30197.9** |
| 128x128x16 W64x64 I4 T8x4 128t | 4.66 | 29487.5 |
| 128x128x16 W64x64 I2 T8x8 128t | 4.67 | 29425.9 |
| 128x128x16 W64x64 I4 T4x8 128t | 4.76 | 28880.1 |
| 64x128x16 W32x64 I4 T4x4 128t | 4.90 | 28037.8 |
| 128x128x32 W64x64 I4 T8x4 128t | 4.91 | 27967.5 |
| 128x128x8 W64x64 I4 T8x4 128t | 5.00 | 27481.1 |
| 256x64x16 W64x32 I2 T8x4 256t | 5.02 | 27375.8 |
| 128x128x16 W32x64 I2 T4x4 256t | 5.05 | 27207.5 |
| 128x256x16 W64x64 I4 T8x4 256t | 5.27 | 26071.4 |
| 128x128x8 W32x64 I2 T4x4 256t | 5.33 | 25785.8 |
| 256x128x16 W128x32 I2 T8x4 256t | 5.47 | 25134.4 |
| 64x64x16 W32x32 I2 T4x4 128t | 5.72 | 24027.2 |
| 256x128x16 W64x32 I2 T8x4 512t | 6.17 | 22259.9 |

These come from the autotuner's own harness, which uses a shorter warm-up and fewer trials than the benchmark the rest of this article quotes, so read them against each other rather than against the results table.

Three things fall out of that ranking. The winner is the non-square `128 x 64` with the small `4 x 4` thread tile, ahead of every square configuration, but only by 2.4%, so the reasoning of the last two sections was close to right and confidently wrong about the final step. The spread between the best and worst legal config is **1.36 times**, which is a larger factor than several of the optimizations in this series delivered individually, for the same kernel body and no new ideas at all. And `BK` is not monotone: holding `128 x 128` fixed it runs 5.00 ms at `BK = 8`, 4.66 at 16 and 4.91 at 32, so there is a real interior optimum that no formula in this article predicts.

The last row deserves one more look. `256x128x16 ... 512t` has the largest block tile in the sweep, which by `K x (1/BM + 1/BN)` moves the least global traffic of anything here, and it finishes last by a wide margin. At 512 threads and 96 registers per thread it fits one block per SM, so it spends the entire kernel with nothing to switch to whenever it reaches a barrier.

### What Is Still Wrong

A sweep hands back a configuration and no explanation. That is what it is for, but it means every kernel from here inherits a shape that was validated rather than derived, and it is worth being precise that the next sections optimize `128 x 64 x 16, W64x32, WNITER 2, 4 x 4, 128 threads` specifically rather than warp tiling in general. That distinction looks pedantic now. Four kernels from now it turns out to be the most important sentence in this section.

The search is exhausted and the shape is fixed, so the remaining question is what the winning configuration actually spends its time on. Nsight has an unambiguous answer, and it points at the one piece of the kernel that has been quietly getting worse since the transpose was introduced: **4.2-way bank conflicts on every shared memory store, 68% of store wavefronts affected, with an estimated 46% speedup available if they were removed.**
