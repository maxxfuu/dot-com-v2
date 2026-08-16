## Flattening the Block

Kernel 2 got its warp layout for free: `blockDim` was 2D, so `threadIdx.x` already ran along the row of the block and the hardware handed us contiguous lanes. Every kernel after this one launches a flat block and rebuilds that 2D position by hand. This kernel does the mapping and nothing else, so the arithmetic can be looked at once on its own before tiling is stacked on top of it.

The launch is the only thing that changes on the host side:

```cuda
dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
dim3 blockDim(BLOCKSIZE * BLOCKSIZE);   // 1024 flat threads, not 32 x 32
```

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

### Mechanics

1. **The `%` has to land on the column.** `threadIdx.x % BLOCKSIZE` is the term that changes for every consecutive thread, and `threadIdx.x / BLOCKSIZE` is the term that stays fixed across 32 of them. Threads 0 through 31 are one warp, so putting `%` on `col` reproduces exactly the layout kernel 2 got from a 2D `blockDim` — one row, 32 adjacent columns. Swap the two operators and every warp reads a column again; the kernel still computes the right answer at a sixth of the speed.

2. **Why not keep the 2D block.** Once tiles arrive, the same set of threads is decomposed two different ways: one shape to cooperatively load a tile from global memory, another to compute with it. Neither decomposition can be the launch shape, so both get derived from a flat index. Starting that here keeps the next kernel about tiling rather than about indexing.

Kernel 3 runs in **43.320 ms at 3172.7 GFLOP/s**, against kernel 2's 43.532 ms — the same 8.1% of cuBLAS at `M = N = K = 4096`, and the 0.5% difference is inside run-to-run clock variation. It is a no-op, which is the point: the mapping is free, and now it is written down.

What it does not fix is the count. Every element of A is still loaded once for every thread in that column of C, and every element of B once for every thread in that row — the same value pulled from global memory over and over by threads sitting in the same block, which could have shared it.
