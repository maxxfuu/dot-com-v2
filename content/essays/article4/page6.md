## Shared Memory Tiling

```cuda
template <const int TILESIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  __shared__ float As[TILESIZE * TILESIZE];
  __shared__ float Bs[TILESIZE * TILESIZE];

  const int threadRow = threadIdx.x / TILESIZE;
  const int threadCol = threadIdx.x % TILESIZE;

  A += cRow * TILESIZE * K;                    
  B += cCol * TILESIZE;                        
  C += cRow * TILESIZE * N + cCol * TILESIZE; 

  float temp = 0.0;
  for (int blockIdx = 0; blockIdx < K; blockIdx += TILESIZE) {
    As[threadRow * TILESIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * TILESIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += TILESIZE;
    B += TILESIZE * N;

    for (int k = 0; k < TILESIZE; ++k) {
      temp += As[threadRow * TILESIZE + k] *  Bs[k * TILESIZE + threadCol];
    }

    __syncthreads();
  }

  C[threadRow * N + threadCol] =
      alpha * temp + beta * C[threadRow * N + threadCol];
}
```
