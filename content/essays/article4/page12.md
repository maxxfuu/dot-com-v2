## Double Buffering: Software Pipelining the K-Loop

```cuda
__device__ __forceinline__ void cp_async16(float *smemDst, const float *gmemSrc) {
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smemDst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmemSrc));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_double_buffered(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  
  constexpr int ASTRIDE = BM + (32 / BK) % 8;

  __shared__ float As[2][BK * ASTRIDE];
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
  constexpr int rowStrideA = (NUM_THREADS * 4) / BK;
  constexpr int APASSES = BM / rowStrideA;

  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  constexpr int rowStrideB = NUM_THREADS / (BN / 4);

  float4 aReg[APASSES];

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

#pragma unroll
  for (int p = 0; p < APASSES; ++p)
    aReg[p] = reinterpret_cast<const float4 *>(
        &A[(innerRowA + p * rowStrideA) * K + innerColA * 4])[0];
#pragma unroll
  for (int p = 0; p < APASSES; ++p) {
    const int r = innerRowA + p * rowStrideA;
    As[0][(innerColA * 4 + 0) * ASTRIDE + r] = aReg[p].x;
    As[0][(innerColA * 4 + 1) * ASTRIDE + r] = aReg[p].y;
    As[0][(innerColA * 4 + 2) * ASTRIDE + r] = aReg[p].z;
    As[0][(innerColA * 4 + 3) * ASTRIDE + r] = aReg[p].w;
  }
#pragma unroll
  for (int o = 0; o + rowStrideB <= BK; o += rowStrideB)
    cp_async16(&Bs[0][(innerRowB + o) * BN + innerColB * 4],
               &B[(innerRowB + o) * N + innerColB * 4]);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const int nTiles = K / BK;
  int cur = 0;
  for (int tile = 0; tile < nTiles; ++tile) {
    const int next = cur ^ 1;
    const bool hasNext = (tile + 1) < nTiles;

    if (hasNext) {

      const float *An = A + (tile + 1) * BK;
      const float *Bn = B + (long)(tile + 1) * BK * N;
#pragma unroll
      for (int p = 0; p < APASSES; ++p)
        aReg[p] = reinterpret_cast<const float4 *>(
            &An[(innerRowA + p * rowStrideA) * K + innerColA * 4])[0];
#pragma unroll
      for (int o = 0; o + rowStrideB <= BK; o += rowStrideB)
        cp_async16(&Bs[next][(innerRowB + o) * BN + innerColB * 4],
                   &Bn[(innerRowB + o) * N + innerColB * 4]);
      cp_async_commit();
    }

#pragma unroll
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
#pragma unroll
      for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
#pragma unroll
        for (int i = 0; i < TM; ++i)
          regM[wSubRow * TM + i] =
              As[cur][dotIdx * ASTRIDE + warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i];
#pragma unroll
      for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
#pragma unroll
        for (int i = 0; i < TN; ++i)
          regN[wSubCol * TN + i] =
              Bs[cur][dotIdx * BN + warpCol * WN + wSubCol * WSUBN + threadColInWarp * TN + i];
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

    if (hasNext) {
      
#pragma unroll
      for (int p = 0; p < APASSES; ++p) {
        const int r = innerRowA + p * rowStrideA;
        As[next][(innerColA * 4 + 0) * ASTRIDE + r] = aReg[p].x;
        As[next][(innerColA * 4 + 1) * ASTRIDE + r] = aReg[p].y;
        As[next][(innerColA * 4 + 2) * ASTRIDE + r] = aReg[p].z;
        As[next][(innerColA * 4 + 3) * ASTRIDE + r] = aReg[p].w;
      }
      cp_async_wait<0>();
      __syncthreads();   
    }
    cur = next;
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
