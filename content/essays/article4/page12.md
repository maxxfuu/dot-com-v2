## Padded Shared Memory: Eliminating Bank Conflicts

Nsight's reading on the configuration the sweep chose is specific: **4.2-way bank conflicts on every shared memory store, 68% of store wavefronts affected, and an estimated 46% speedup available if they went away.** That is the largest single number any profiler has produced about any kernel in this series, and it points at four lines of code.

This section is one derivation and one honest result, and the two do not agree with each other.

### How Shared Memory Is Addressed

Shared memory is not one bank of memory, it is 32 of them, interleaved every 4 bytes. The bank holding a float is:

``` latex
\text{bank} = \left\lfloor \frac{\text{byte address}}{4} \right\rfloor \bmod 32
```

which for an array of floats is just the array index mod 32. The hardware services one address per bank per cycle. If the 32 lanes of a warp touch 32 different banks the whole access completes in one cycle, and if they all touch the same address the hardware broadcasts, which is also one cycle. The bad case is in between: several lanes wanting different addresses that happen to live in the same bank. Those serialize, and an `n` way conflict costs `n` cycles.

Now look at where the transposed store puts things. From kernel 7 onwards, A is written into shared memory transposed:

```cuda
As[(innerColA * 4 + j) * BM + innerRowA + offset] = tmp.<x,y,z,w>;
```

The row stride of `As` is `BM = 128`, and 128 is a multiple of 32. So in the bank calculation the entire first term vanishes:

``` latex
\text{bank} = \big((\text{innerColA} \cdot 4 + j) \cdot 128 + \text{innerRowA}\big) \bmod 32 = \text{innerRowA} \bmod 32
```

The column part of the address contributes nothing at all. Every lane's bank is decided entirely by which row of A it is carrying, and that is exactly the index that varies slowest across a warp. With `BK = 16` the load index is `innerRowA = threadIdx.x / (BK / 4) = threadIdx.x / 4`, so the 32 lanes of a warp cover only **8 distinct rows**, four lanes each. Eight banks, four lanes per bank, a 4-way conflict on each of the four stores. Nsight measured 4.2-way, which is that plus the ragged edge of the last warp.

Worth noticing that this got worse as the kernel got better. At kernel 7's `BK = 8` the divisor was 2, so a warp covered 16 rows and the conflict was only 2-way. Doubling `BK` halved the number of distinct rows a warp touches and doubled the conflict.

### The Fix Is One Constant

If the row stride is a multiple of 32 the column term disappears, so make it not a multiple of 32. Pad each row of `As` by a couple of floats, leaving a small hole at the end of every row, and the column term comes back into the bank index:

```cuda
constexpr int A_STRIDE = BM + (32 / BK) % 8;   // 128 + 2 = 130

__shared__ float As[BK * A_STRIDE];
```

With `A_STRIDE = 130` and `130 mod 32 = 2`:

``` latex
\text{bank} = \big((\text{innerColA} \cdot 4 + j) \cdot 130 + \text{innerRowA}\big) \bmod 32 = \big(2(\text{innerColA} \cdot 4 + j) + \text{innerRowA}\big) \bmod 32
```

Within a warp `innerColA = threadIdx.x % 4` takes the values 0 through 3, so for a fixed `j` the column term takes the four values `{2j, 8 + 2j, 16 + 2j, 24 + 2j}`, and each of those is offset by one of the 8 values of `innerRowA`. Four groups of eight, 32 distinct banks, no two lanes colliding. The conflict is gone.

The rest of the kernel is kernel 8 with `BM` replaced by `A_STRIDE` in the three places that index `As`, plus the extra 128 bytes of shared memory per block that the padding costs.

```cuda
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_padded_smem(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  constexpr int A_STRIDE = BM + (32 / BK) % 8;

  __shared__ float As[BK * A_STRIDE];
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
  constexpr int ROW_STRIDE_A = (NUM_THREADS * 4) / BK;

  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  constexpr int ROW_STRIDE_B = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int offset = 0; offset + ROW_STRIDE_A <= BM; offset += ROW_STRIDE_A) {
      float4 tmp = reinterpret_cast<const float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * A_STRIDE + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * A_STRIDE + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * A_STRIDE + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * A_STRIDE + innerRowA + offset] = tmp.w;
    }

    for (int offset = 0; offset + ROW_STRIDE_B <= BK; offset += ROW_STRIDE_B) {
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
              As[dotIdx * A_STRIDE + warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i];

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

1. **`(32 / BK) % 8` is a formula, not a magic number.** It asks how many rows of `As` a warp's stores span and pads by enough to walk the bank index off a multiple of 32. At `BK = 16` it gives 2, at `BK = 32` it gives 1, at `BK = 8` it gives 4, and the outer `% 8` keeps the pad small when `BK` is tiny. It is worth re-deriving by hand for any new `BK` rather than trusting it, because the argument above depends on how many distinct values `innerColA` takes, and that moves with `BK` too.

2. **Padding costs shared memory and nothing else.** `As` goes from `BK x BM x 4 = 8192` bytes to `BK x 130 x 4 = 8320`, so the block total goes from 12288 to 12416 bytes. At 5 blocks per SM that is still far inside the 100 KB budget, and `ptxas` reports 94 registers against kernel 8's 96, so residency stays at 5 blocks per SM and occupancy stays at 41.7%. Nothing was traded away for this.

3. **Only the store was conflicted, and the loads were already fine.** `regM` reads `As[dotIdx * A_STRIDE + ...]` over consecutive `i`, which are consecutive floats and therefore consecutive banks. This is worth saying out loud, because it is both the reason the fix is cheap and, as the next paragraphs show, the reason the fix does not pay. The load path runs six times more often than the store path and was never what needed fixing.

Kernel 10 runs in **4.660 ms at 29490.7 GFLOP/s, which is 81.9% of cuBLAS at `M = N = K = 4096`**, against 4.679 ms and 81.6% for kernel 8. That is a gain of **0.4%**.

### 46% Estimated, 0.4% Delivered

Both of those numbers are real, and reconciling them is the point of this section.

The conflicts were genuinely there. The derivation is arithmetic rather than conjecture: `128 mod 32 = 0` collapses the bank index onto `innerRowA`, and `innerRowA = threadIdx.x / 4` gives 8 rows per warp. The profiler measured 4.2-way against a predicted 4. And the fix genuinely removed them, by the same arithmetic run in reverse.

What was wrong was the estimate, and specifically the assumption underneath it. Nsight's estimated speedup figures answer a narrow question: if this stall disappeared entirely, how much faster would the unit reporting it run. It has no way to know whether that unit is on the critical path. Here it is not. Count the instructions in one trip around the K loop:

| instruction | count per K-tile | share |
|---|---|---|
| `STS`, the conflicting transposed stores | 16 | 0.7% |
| `LDS`, the vector loads feeding the FMAs | 96 | 4.4% |
| `FFMA` | 2048 | 94.9% |

The conflicting stores are under 1% of the instruction stream. Removing a 4-way conflict makes them roughly four times cheaper, which removes something like half a percent of the total, and half a percent is what we measured. The 46% was never on the table because the store path was never what the kernel was waiting on.

The general lesson is about how to read a profiler. It reports where time goes inside each unit accurately, and it estimates the benefit of fixing a stall on the assumption that the stall is what bounds you. That second part is a hypothesis you are supplying, not a measurement it is making. The cheap way to test it before writing any code is the table above: count how often the instruction you are about to optimize actually executes.

<!-- TODO: re-run the conflict measurement now the fix is in, to show it landing
     at 1.0-way: `sudo ncu --metrics
     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bench 4096` on
     kernels 8 and 10. Also want smsp__inst_executed to replace the derived
     instruction table above with a measured one. Counters are admin only on this
     box right now. -->

Keep the derivation anyway. It is correct, it is the only place in this series where a hardware detail as small as a bank index reaches into a line of code, and padding is the right default for a transposed shared memory buffer. It is also, as the final section of this article discovers, a fix with an expiry date.

### What Could Be Improved

The instruction table says something else worth reading twice. `FFMA` is 95% of what this kernel issues per K-tile, and around 83% of everything it executes once loop overhead and the epilogue are counted. There is essentially no overhead left to remove. Every instruction that is not arithmetic has been hoisted, vectorized, tiled or padded away across the last five kernels.

So the remaining gap is not instructions the kernel executes. It is cycles in which it executes nothing. The K loop is still `load, sync, compute, sync`, and both barriers stall every warp in the block until the slowest one arrives. While the block is loading, the FMA pipes have nothing to do; while it is computing, the load pipes have nothing to do. The work is already minimal and it is still serialized against itself.
