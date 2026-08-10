## Vectorized Memory Access: 128-bit Loads and Stores

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
