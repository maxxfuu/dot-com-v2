## Flattening the Block

```cuda
template <const int BLOCKSIZE>
__global__ void sgemm_coalesced_1d(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  const int row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row < M && col < N) {
    float temp = 0.0f;
    for (int i = 0; i < K; ++i) {
      temp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}
```
