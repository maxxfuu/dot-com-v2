## Warp Tiling: A Third Level of Tiling

Kernel 7 left a warp smeared across the block tile. `threadRow = threadIdx.x / 16` cuts the 256 threads into rows of 16, so one warp of 32 consecutive threads covers two of those rows, which is 128 columns and 16 rows of the output. Every shared memory instruction that warp issues reaches across the full width of `Bs`, and nothing chose that shape. It is a side effect of how `threadIdx.x` happened to divide.

This matters because the warp, not the thread, is the unit that executes. When a warp issues one `LDS.128`, the hardware resolves 32 lane addresses together, and what it costs depends on how those 32 addresses are spread across banks and how much of the data they pull in is shared. A warp reading a compact rectangle asks for a small, dense range. A warp reading a wide strided band asks for a large sparse one, gets the same answer, and pays more for it.

So add a level. The block tile already splits into thread tiles; now put a warp tile between them. Each of the block's warps takes a contiguous `WM x WN` rectangle of the output, and the threads within that warp tile the rectangle among themselves. The decomposition becomes block, then warp, then thread, and every level is now an explicit parameter rather than an accident of integer division.

### The Warp Tile Is Not Contiguous Either

There is one twist that the parameter names give away. A warp has 32 threads each producing `TM x TN = 16` outputs, which is 512 outputs, but the warp tile is `WM x WN = 64 x 32 = 2048`. The warp must therefore cover its rectangle in four passes, and the code expresses that as `WMITER x WNITER` sub-tiles of size `WSUBM x WSUBN`.

The sub-tiles are interleaved rather than laid end to end. Within one sub-tile of `32 x 16`, the warp's 32 threads form a compact `8 x 4` grid of `4 x 4` patches, so the warp's read out of `Bs` for that sub-tile spans exactly `WSUBN = 16` consecutive floats and its read out of `As` spans `WSUBM = 32`. Compare that with kernel 7, where the equivalent read spanned all 128 columns of the tile. The instruction count per warp is unchanged; the footprint of each instruction is eight times narrower.

`WMITER` is not a parameter you pass. It is whatever is left once the others are fixed, which is why it is a `constexpr` derived inside the kernel:

```
WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER)
       = (64 * 32) / (32 * 4 * 4 * 2)
       = 2
```

If that division does not come out whole, the config is not slow, it is wrong, and threads either overlap or leave holes in C. Nothing in this kernel checks it. The next section is about that.

### The Code

The config changes shape in a way worth flagging before the listing. The block tile is `128 x 64`, which is not square for the first time in the series, `BK` doubles to 16, the thread tile *shrinks* from `8 x 8` to `4 x 4`, and the block runs on 128 threads, one warp fewer than a quarter of kernel 7's:

```cuda
const int BM = 128, BN = 64, BK = 16;
const int WM = 64, WN = 32, WNITER = 2;
const int TM = 4, TN = 4, NUM_THREADS = 128;

dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
dim3 blockDim(NUM_THREADS);
```

Every one of those numbers came out of a search rather than an argument, which is the subject of the next section and the closing point of this one.

```cuda
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_warp_tiled(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

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
  const int rowStrideA = (NUM_THREADS * 4) / BK;

  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  const int rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    
    for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<const float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }
    for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {

      for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
        for (int i = 0; i < TM; ++i)
          regM[wSubRow * TM + i] =
              As[dotIdx * BM + warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i];

      for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
        for (int i = 0; i < TN; ++i)
          regN[wSubCol * TN + i] =
              Bs[dotIdx * BN + warpCol * WN + wSubCol * WSUBN + threadColInWarp * TN + i];

      for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
        for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
          for (int m = 0; m < TM; ++m)
            for (int n = 0; n < TN; ++n)
              threadResults[(wSubRow * TM + m) * (WNITER * TN) + wSubCol * TN + n] +=
                  regM[wSubRow * TM + m] * regN[wSubCol * TN + n];
    }
    __syncthreads();
  }

  for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
    for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
      float *C_sub = C + (wSubRow * WSUBM) * N + wSubCol * WSUBN;
      for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; n += 4) {
          float *cPtr = &C_sub[(threadRowInWarp * TM + m) * N + threadColInWarp * TN + n];
          float4 tmp = reinterpret_cast<float4 *>(cPtr)[0];
          const int i = (wSubRow * TM + m) * (WNITER * TN) + wSubCol * TN + n;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(cPtr)[0] = tmp;
        }
      }
    }
  }
}
```

### Mechanics

1. **The index chain now has three links, and each one is a pointer offset or an added term.** The block's corner is folded into `A`, `B` and `C` as before. The warp's corner is folded into `C` as well, `(cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN`, so the epilogue never mentions the block again. Inside the loop, `warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i` reads as exactly what it is: block tile, then warp tile, then sub-tile, then thread patch, then element.

2. **The accumulators are the same 64 as kernel 7, indexed differently.** `threadResults[WMITER * TM * WNITER * TN]` is `2 * 4 * 2 * 4 = 64` values, the same register footprint as an `8 x 8` patch, but laid out as four `4 x 4` patches scattered across the warp tile rather than one contiguous `8 x 8`. `ptxas` reports 96 registers per thread against kernel 7's 94, so the third level of tiling costs two registers.

3. **The load loops come back, with different trip counts.** `rowStrideA = (NUM_THREADS * 4) / BK = 32`, and `BM / rowStrideA = 4` passes to fill `As`. `rowStrideB = NUM_THREADS / (BN / 4) = 8`, and `BK / rowStrideB = 2` passes for `Bs`. The loop bound is written `offset + rowStrideA <= BM` rather than `offset < BM`, which matters only when the stride does not divide the tile, and in that case it silently loads less than the full tile instead of overrunning it. That is a config bug rather than a runtime one, and again, nothing here checks.

4. **`__launch_bounds__` appears for the first time.** It promises the compiler that no more than `NUM_THREADS` threads will ever be launched in a block, which lets `ptxas` size the register allocation against a known block size. It is doing modest work here. In the final kernel, the second argument to this same declaration turns out to be the largest single win in the series.

Kernel 8 runs in **4.679 ms at 29376.6 GFLOP/s, which is 81.6% of cuBLAS at `M = N = K = 4096`**, up from 5.469 ms and 69.8%. That is 1.17 times faster.

### The Arithmetic, As A Negative Result

Now the uncomfortable part. Put this kernel's tile shape through the formula from the 2D register tiling section:

```
GMEM accesses per result = K * (1/BM + 1/BN)
                         = 4096 * (1/128 + 1/64)
                         = 96
```

against 64 for kernel 7's square `128 x 128` tile. Total global load traffic goes from **4.29 GB to 6.44 GB**, an increase of 50%. The shared memory count does not improve either: `WMITER * TM + WNITER * TN = 16` loads per `dotIdx` feeding `8 x 8 = 64` results is `K / 4 = 1024` per output element, identical to kernels 6 and 7.

Neither access count improved. One of them got materially worse. The kernel is 1.17 times faster.

This is worth sitting with, because the formula that says this config is bad is the same formula that has been correct for four kernels running. It says square tiles minimize traffic for a given area, and it is right about that. What it does not model is anything else on the machine:

- **Residency.** Kernel 7 ran 256 threads with 93 registers each, giving 2 blocks per SM and 33.3% occupancy. This kernel runs 128 threads with 96 registers, giving **5 blocks per SM** and 41.7%. Smaller blocks pack more of them onto an SM, and more resident blocks means the barriers cost less, because when one block stalls at `__syncthreads()` there are four others with work to issue.
- **Warp locality.** Each shared memory instruction now touches a 16 or 32 float range instead of a 128 float one, which was the entire point of the section.
- **L2.** The extra 2.15 GB of requests is only expensive if it reaches DRAM, and on this card it does not have to. A block row band of A is `BM x K x 4 = 2 MB`, and the L2 is 64 MB.

That last point deserves a number, because it is where the roofline stops being useful.

### Retiring The Roofline

Arithmetic intensity falls this time, from 32 to `137.44 GFLOP / 6.44 GB = 21.3 FLOP/byte`, so the model says we have moved backwards. Its predictions:

- The sloped ceiling at 21.3 FLOP/byte sits at `21.3 x 960 GB/s = 20486 GFLOP/s`. We measured **29376.6**, which is 143% of the roof.
- The memory floor for 6.44 GB at 960 GB/s is **6.71 ms**. We measured **4.679 ms**, which is faster than the floor.

The kernel beats its own roofline in both forms, and by now the reason is familiar, because kernel 1 did the same thing for the same reason. The model assumes every requested byte comes from VRAM. Here it does not: at 4096 the working set fits the 64 MB L2 well enough that the extra requests this tile shape generates are served on chip, and the DRAM never sees them.

That is the third and final time this model earns its keep in the series, and it earns it by failing. It was the right tool at kernel 1, where it explained a 234 times intensity deficit; it was the right tool at kernel 4, where the kernel first sat underneath it; and it is the wrong tool from here on, because every remaining bottleneck is inside the SM. The traffic accounting that would still be informative is against L2 rather than DRAM, and `lts__t_sector_hit_rate.pct` is the metric that would settle it.

### What Is Still Wrong

There is nothing wrong with this kernel that a better argument would fix, which is precisely the problem.

The tile shape `128 x 64` with `BK = 16` and a `4 x 4` thread tile is not what any of the reasoning in this series would have chosen. The access count formula picks square. The register argument from the 2D tiling section picks the largest thread tile that does not spill, which is `8 x 8`, not `4 x 4`. Both arguments are sound, both were load bearing when they were made, and both would have landed on kernel 7's config, which is 17% slower.

The honest summary of this section is that the shape came from measurement, and the reasoning was reconstructed afterwards to explain a result it did not predict. That is not a failure of the reasoning so much as a statement about how many interacting resources are in play: registers, residency, bank behaviour, L2, and instruction issue all move when a tile dimension moves, and no closed form covers all five.

So stop arguing about the configuration and search it instead.
