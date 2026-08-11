## The Naive Kernel: Establishing a Baseline

### What is GEMM 

Before we begin, let's establish what a General Matrix Multiplication (GEMM) is so that we understand what's going on behind the scenes with every kernel implementation. Every kernel in this article computes the exact same thing; an output matrix C computed by matrices A and B. The **GEMM** follow the same mathematical formula listed below:

Note that variables alpha and beta are just scalar coefficients that scale the matrix multiplication and the existing output matrix C. 

``` latex
C \leftarrow \alpha A B + \beta C
```

**A is (M x K)**, **B is (K x N)**, and **C is (M x N)**. All three of these matrices are stored in row-major order in memory. Each optimized kernel thats introduced within the article will use the following dimensions: **M = N = K = 4096**, **alpha = 1**, **beta = 0**. 

This ensures that very kernel performs the same 2 * M * N * K floating point operations, which is roughly 137 billion operation in total, against the same three matrices.

### Naive Implementation

For the first kernel, we are really just trying to express GEMM in CUDA. We'll start off by using the grid, block, and thread to assign one thread to each output cell in the output matrix C. Start by expressing each thread with its respective global thread indicies: 

![Each thread computes one element of C. Thread t[0][0] owns the top-left corner, and moving one thread over or one thread down moves you exactly one column or row over in the output.](/images/gemm/matrix-c-indexing.png)

``` cuda 
const int BLOCKSIZE = 32;

const int row = blockIdx.x * BLOCKSIZE + threadIdx.x;
const int col = blockIdx.y * BLOCKSIZE + threadIdx.y;
```

Given this mapping, a single thread computes the dot product between a row of A and a columns of B, then writes the product into its corresponding output cell in C. 

Therefore, we need to write the function for a single thread as everything will be executed in parallel within a warp. For each thread that writes to its own output cell in matrix C, the thread will walk the full K dimension between its corresponding row and column. Written out against our actual function signature, the entire kernel looks like this:

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

To visualize this naive kernel:

![Two neighbouring threads and the memory each one touches. Both walk the full K dimension, thread 0 reading row 0 of A against column 0 of B, thread 1 reading row 1 against the same column, to produce one element of C each.](/images/gemm/naive-access-pattern.png "large")

To launch this kernel, we map `blockIdx.x` and `threadIdx.x` to col and `blockIdx.y` and `threadIdx.y` to row, so the `gridDim.x` walks along the `N` dimension and `gridDim.y` walks along the `M` dimension. For M = N = 4096 with BLOCKSIZE = 32, that's gridDim = (128, 128, 1) and blockDim = (32, 32, 1), or 16,384 blocks of 1024 threads each, one thread per output element.

``` cuda 
const int BLOCKSIZE = 32;

// one threadblock per 32x32 tile of C; grid.x walks N, grid.y walks M
dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

sgemm_naive<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

Three details worth noticing. 

Firstly, the CUDA hardware linearizes the threads in a block when forming warps. Even though we define the block with dimensions `x` and `y`, the CUDA hardware flattens everything into a single linear index. Therefore the `x` dimension is the fastest-changing dimension within the thread block. This matters because it dictates how each cell is accessed within the C matrix. In this case we will walk down column 0, such that the row value is the one incrementing: `c[0][0]`, `c[1][0]`, `c[n][0]`.

Secondly, the `CEIL_DIV` represents ceiling division, exists because the matrix dimensions are not required to be multiples of the block size, so we round the grid up and let the last blocks hang over the "edge". This is also why we have to have the boundary check in place; the `if (row < M && col < N)` guard inside the kernel prevents out-of-bounds threads from accessing garbage memory. In our kernel implementation, `M = N = K = 4096` with `BLOCKSIZE = 32` makes the division exact, but writing the kernel as though it isn't costs one comparison and further prevents bugs related to boundedness.

Lastly, every thread reads an entire row of A and an entire column of B out of global memory to produce one output value. That is `2*M*N*K` FLOPs against `2*M*N*K` loaded floats. That is roughly one FLOP per float loaded, or, since each float is 4 bytes, about 0.25 FLOP per byte moved, and a 5080 can move nowhere near enough bytes per second to keep its arithmetic units busy at that ratio. The kernel is memory bound before it has executed a single instruction, and 460 GFLOP/s against cuBLAS's 38,216 is what that looks like in practice. 

