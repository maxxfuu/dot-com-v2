## 2D Register Tiling: The Outer Product

```cuda
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_register_2d_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int numThreads = (BM * BN) / (TM * TN);

 const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  A += cRow * BM * K;                   
  B += cCol * BN;                       
  C += cRow * BM * N + cCol * BN;       

  const int innerColA = threadIdx.x % BK;             
  const int innerRowA = threadIdx.x / BK;             
  const int strideA = numThreads / BK;                

  const int innerColB = threadIdx.x % BN;             
  const int innerRowB = threadIdx.x / BN;             
  const int strideB = numThreads / BN;                

  float threadResults[TM * TN] = {0.0f};

  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }

    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
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
