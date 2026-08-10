## 1D Register Tiling: One Thread, TM Outputs

```cuda
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_register_1d_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  
  
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  A += cRow * BM * K;                   
  B += cCol * BN;                       
  C += cRow * BM * N + cCol * BN;       

  const int innerColA = threadIdx.x % BK;  
  const int innerRowA = threadIdx.x / BK;  
                                           
  const int innerColB = threadIdx.x % BN;  
  const int innerRowB = threadIdx.x / BN;  

  float threadResults[TM] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol]; 
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}
```
