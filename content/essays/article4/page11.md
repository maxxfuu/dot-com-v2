## Padded Shared Memory: Eliminating Bank Conflicts

```cuda
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_padded_smem(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  constexpr int ASTRIDE = BM + (32 / BK) % 8;

  __shared__ float As[BK * ASTRIDE];
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
      As[(innerColA * 4 + 0) * ASTRIDE + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * ASTRIDE + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * ASTRIDE + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * ASTRIDE + innerRowA + offset] = tmp.w;
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
              As[dotIdx * ASTRIDE + warpRow * WM + wSubRow * WSUBM + threadRowInWarp * TM + i];

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
