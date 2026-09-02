## Vectorized Memory Access: 128-bit Loads and Stores

Kernel 6 ended with a distinction that matters from here on. Its inner loop reads 16 floats out of shared memory to feed 64 FMAs, which is a small amount of data carried by a large number of instructions: eight separate 32 bit loads down a column of `As` and eight more along a row of `Bs`, every `dotIdx`, in every thread. A warp waiting for those to issue rather than for their data is stalled on `MIO Throttle`, and that is a queue for the load store pipe, not a bandwidth problem. Kernel 6 is no longer dominated by that stall (at 7.49 warp cycles per issued instruction, Nsight names no dominant reason for it), but the instruction count is real either way, and it is what this kernel removes.

Nothing about the data needs to change. The GPU can move 128 bits in a single instruction, and a `float4` is exactly four contiguous floats, so if the four values a thread wants next to each other in memory really are next to each other, one `LDS.128` replaces four `LDS.32`. Same bytes, quarter the instructions, and the queue drains four times faster.

The whole section is therefore about making the values contiguous in the right direction, because two of the three places we touch memory are already contiguous and one is not.

### Why `As` Has To Be Transposed

Look at what a thread reads out of `As` in kernel 6: `As[(threadRow * TM + i) * BK + dotIdx]`. Consecutive `i` are `BK` floats apart, because `As` is stored row major with a row length of `BK`. That is a strided read, and a strided read cannot be a `float4` no matter how it is spelled.

The fix is to store `As` transposed, as `BK` rows of `BM` floats instead of `BM` rows of `BK`. Then a thread's `TM` values sit at `As[dotIdx * BM + threadRow * TM + i]`, consecutive in `i`, and `TM = 8` becomes two 128 bit loads instead of eight 32 bit ones.

The transpose has to happen on the store side, not the load side, and that is the constraint the whole indexing scheme is built around. The read from global memory has to stay coalesced, which means a warp must read along a row of A, along the K dimension. So the thread reads four contiguous floats of A, then writes them into four different rows of `As`, one element each, scattering as it goes. We accept a scattered write into shared memory to preserve a contiguous read from global memory and to buy a contiguous read out of shared memory later. `Bs` needs none of this; it is already stored with `BN` as the fast dimension, which is the direction its readers want.

### The Kernel

The tile shape and thread count are unchanged from kernel 6, so this is a pure instruction level change with the same launch:

```cuda
const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
dim3 blockDim((BM * BN) / (TM * TN));   // 256, same as kernel 6
```

```cuda
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  A += cRow * BM * K;                   
  B += cCol * BN;                       
  C += cRow * BM * N + cCol * BN;       

  
  const int innerRowA = threadIdx.x / (BK / 4);   
  const int innerColA = threadIdx.x % (BK / 4);   

  const int innerRowB = threadIdx.x / (BN / 4);   
  const int innerColB = threadIdx.x % (BN / 4);   

  float threadResults[TM * TN] = {0.0f};

  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

    
    float4 tmp = reinterpret_cast<const float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {

      for (int i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
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
    for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float *cPtr = &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
      float4 tmp = reinterpret_cast<float4 *>(cPtr)[0];

      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;

      reinterpret_cast<float4 *>(cPtr)[0] = tmp;
    }
  }
}
```

### Mechanics

1. **`As` and `Bs` swap their declared shapes, and only one of them is a real transpose.** `As` is now `BK * BM` and is indexed `[k][m]`, which is the transpose of kernel 6's `[m][k]`. `Bs` is still `BK * BN` indexed `[k][n]`, unchanged; it was already oriented the way its readers want. If you only remember one thing about this kernel, it is that A gets transposed and B does not, and the reason is that A is the operand whose thread tile runs down the M dimension.

2. **The load loops disappear.** Each thread now moves a `float4`, and 256 threads times 4 floats is exactly the 1024 elements in each tile, so kernel 6's four pass loops collapse into a single statement each. `innerColA` ranges over `BK / 4 = 2` values and `innerRowA` over 128; `innerColB` ranges over `BN / 4 = 32` and `innerRowB` over 8. Both still put the column index on the fast moving lane, so both global reads are still perfectly coalesced.

3. **The A store is four scalar stores and cannot be anything else.** `tmp.x` through `tmp.w` go to four addresses `BM` floats apart, so there is no vector store to be had on that side. This is the price of the transpose, paid once per tile per thread, against a saving collected `BK` times per tile in the inner loop.

4. **The epilogue reads C back as a `float4` and writes it back as one.** `beta * C` still requires reading the old value, but a thread's `TN = 8` outputs in a row are contiguous in C, so the read modify write goes four elements at a time. This is also the first appearance of a requirement that becomes load bearing two kernels from now: everything here assumes `TN`, `BN` and `BK` are multiples of 4 and that the pointers are 16 byte aligned. Nothing checks it yet, and a config that violates it does not fail loudly, it computes garbage.

5. **The listing reads `regM` and `regN` with scalar loops on purpose.** Because the addresses are now contiguous and 16 byte aligned, `ptxas` fuses each group of four into a single `LDS.128` by itself. Writing the `float4` cast by hand here changes nothing in the generated SASS, and the scalar form survives a change of `TM` without editing. The transpose is what produced the win; the vector instruction is what the compiler does once the transpose makes it legal.

Kernel 7 runs in **5.469 ms at 25130.5 GFLOP/s, which is 69.8% of cuBLAS at `M = N = K = 4096`**, up from 6.996 ms and 54.6%. That is 1.28 times faster with no change to the tiling, the thread count, or the number of bytes touched.

This kernel also has the widest run to run spread in the series, 3.24% against well under 1% for its neighbours, so treat the last decimal place as noise rather than signal.

### Instructions, Not Bytes

The counts from kernel 6 all carry over exactly. Global memory is still 64 accesses per output element and 4.29 GB in total. Shared memory is still `K x (1/TM + 1/TN) = 1024` floats read per output element. Arithmetic intensity is still 32 FLOP per byte, the roofline position has not moved, and the memory and compute floors are the 4.47 ms and 2.44 ms they were before.

What changed is the instruction count carrying those bytes:

| per thread, per `dotIdx` | kernel 6 | kernel 7 |
|---|---|---|
| `LDS` from `As` | 8 x 32 bit | 2 x 128 bit |
| `LDS` from `Bs` | 8 x 32 bit | 2 x 128 bit |
| `FFMA` | 64 | 64 |
| shared loads per FMA | 1 per 4 | 1 per 16 |

Counted per output element, shared memory load instructions go from `K / 4` to `K / 16`, four times fewer, and the global side sees the same fourfold reduction in `LDG` and `STG` instructions for identical traffic. If a description of this kernel says vectorization moves less data, it is wrong. It moves precisely the same data with a quarter of the instructions, and the thing that was scarce was instruction issue.

Nsight puts numbers on the issue side of that. Kernel 6 runs at 7.49 warp cycles per issued instruction and this kernel at **6.37**, and neither has a stall reason large enough for the profiler to name one. The instruction count fell fourfold and the cost of issuing what remains fell with it.

<!-- TODO: smsp__inst_executed.sum on kernels 6 and 7, to state the instruction
     reduction as measured rather than derived. WarpStateStats is done. -->

### What Could Be Improved

The transpose bought the loads and quietly created a problem on the stores. Shared memory is divided into 32 banks by `(byte address / 4) mod 32`, and two lanes of the same warp hitting different addresses in the same bank serialize. The store is `As[(innerColA * 4 + j) * BM + innerRowA]`, and with `BM = 128`, which is a multiple of 32, the entire first term vanishes from the bank index. Every lane's bank is just `innerRowA mod 32`.

With `BK = 8` a warp's 32 lanes cover `innerRowA = threadIdx.x / 2`, which is 16 distinct rows, so the warp's stores land on 16 banks and every bank takes two of them. That is a 2-way conflict on all four of the A stores, and it gets worse rather than better as `BK` grows, because a wider `BK` means fewer distinct rows per warp. The next kernel raises `BK` to 16, and the same expression then covers only 8 rows.

There is a second and larger problem, and it is about which values a warp reads rather than where it writes them. A warp is 32 consecutive threads, and `threadRow = threadIdx.x / 16` with `threadCol = threadIdx.x % 16`, so one warp spans 16 columns and 2 rows of the thread grid, which is `16 x 8 = 128` columns and `2 x 8 = 16` rows of the output tile. Its reads out of `Bs` therefore span the entire 128 wide tile, and its reads out of `As` span 16 rows sitting in the middle of a 128 row structure. The warp is smeared across the block tile, and every shared memory instruction it issues touches a wide strided range instead of a compact one.

Nothing in the code decides that shape deliberately. It falls out of `threadIdx.x` being cut into rows of 16, which was never a decision at all. The next kernel makes it one.
