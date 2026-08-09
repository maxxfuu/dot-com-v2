## The Naive Kernel: Establishing a Baseline

### Mapping the Programming Model onto GEMM

For GEMM, we can think of each thread as being responsible for computing some portion of the output matrix C.

For example:

```
              Matrix C

       ┌────┬────┬────┬────┐
       │ T0 │ T1 │ T2 │ T3 │
       ├────┼────┼────┼────┤
       │ T4 │ T5 │ T6 │ T7 │
       ├────┼────┼────┼────┤
       │... │... │... │... │
       └────┴────┴────┴────┘
```

A naive GEMM might assign one thread to one output element:

```
C[row][col] = dot(A[row], B[col]);
```

This gives us a simple programming abstraction: many threads independently compute many elements of C in parallel. Nothing is shared, nothing is coordinated, and every thread walks the full K dimension by itself. Written out against our actual signature, the entire kernel is this:

```cuda
template <const int BLOCKSIZE>
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const int row = blockIdx.y * BLOCKSIZE + threadIdx.x;
  const int col = blockIdx.x * BLOCKSIZE + threadIdx.y;

  if (row < M && col < N) {
    float temp = 0.0f;
    for (int i = 0; i < K; ++i) {
      temp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}
```

The launch side decides how that index space is carved up. We want one thread per element of C, so we ask for `32x32` threads per block and enough blocks to cover the matrix:

```cuda
const int BLOCKSIZE = 32;

// one threadblock per 32x32 tile of C; grid.x walks N, grid.y walks M
dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

sgemm_naive<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

Two details in there matter more than they look. The `CEIL_DIV` exists because the matrix dimensions are not required to be multiples of the block size, so we round the grid up and let the last blocks hang over the edge — which is what the `if (row < M && col < N)` guard inside the kernel is cleaning up. At `M = N = K = 4096` with `BLOCKSIZE = 32` the division is exact, but writing the kernel as though it isn't costs one comparison and saves you from a class of bug that only appears on someone else's matrix.

The second detail is the bound this kernel is actually working against. Every thread reads an entire row of A and an entire column of B out of global memory to produce one output value: `2*M*N*K` FLOPs against `2*M*N*K` loaded floats. That is roughly one FLOP per byte moved, and a 5080 can move nowhere near enough bytes per second to keep its arithmetic units busy at that ratio. The kernel is memory bound before it has executed a single instruction, and 460 GFLOP/s against cuBLAS's 38,216 is what that looks like in practice.

Every optimization in the rest of this article is an attack on that one ratio.

:::check
The programming model says thread blocks must be able to run in any order, yet threads inside a block may cooperate freely. Why does CUDA draw the line there?
---
Because the block is the unit the hardware assigns to a single SM. Threads in one block are resident on the same SM at the same time, so they have a shared memory and a barrier to coordinate through. Two different blocks may be on different SMs, or may not even be resident simultaneously, so no such mechanism can exist between them.

That restriction is what makes a kernel portable across GPUs. Since blocks are independent, the runtime is free to schedule as many concurrently as the device has room for — 84 SMs' worth or 16 SMs' worth — without the kernel being written any differently.
:::
