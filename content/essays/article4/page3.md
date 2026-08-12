## The Naive Kernel: Establishing a Baseline

For the first kernel, we are really just trying to express GEMM in CUDA. We'll start off by using the grid, block, and thread to assign one thread to each output cell in the output matrix C. Start by expressing each thread with its respective global thread indicies: 

![Each thread computes one element of C. Thread t[0][0] owns the top-left corner, and moving one thread over or one thread down moves you exactly one column or row over in the output.](/images/gemm/matrix-c-indexing.png)

``` cuda 
const int BLOCKSIZE = 32;

const int row = blockIdx.y * BLOCKSIZE + threadIdx.x;
const int col = blockIdx.x * BLOCKSIZE + threadIdx.y;
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

To launch this kernel, we map `blockIdx.x` and `threadIdx.y` to col and `blockIdx.y` and `threadIdx.x` to row, so the `gridDim.x` walks along the `N` dimension and `gridDim.y` walks along the `M` dimension. For M = N = 4096 with BLOCKSIZE = 32, that's gridDim = (128, 128, 1) and blockDim = (32, 32, 1), or 16,384 blocks of 1024 threads each, one thread per output element.

``` cuda 
const int BLOCKSIZE = 32;

// one threadblock per 32x32 tile of C; grid.x walks N, grid.y walks M
dim3 gridDim(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE), 1);
dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

sgemm_naive<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

Two details worth noticing. 

Firstly, the CUDA hardware linearizes the threads in a block when forming warps. Even though we define the block with dimensions `x` and `y`, the CUDA hardware flattens everything into a single linear index. Therefore the `x` dimension is the fastest-changing dimension within the thread block. This matters because it dictates how each cell is accessed within the C matrix. In this case we will walk down column 0, such that the row value is the one incrementing: `c[0][0]`, `c[1][0]`, `c[n][0]`.

Secondly, the `CEIL_DIV` represents ceiling division, exists because the matrix dimensions are not required to be multiples of the block size, so we round the grid up and let the last blocks hang over the "edge". This is also why we have to have the boundary check in place; the `if (row < M && col < N)` guard inside the kernel prevents out-of-bounds threads from accessing garbage memory. In our kernel implementation, `M = N = K = 4096` with `BLOCKSIZE = 32` makes the division exact, but writing the kernel as though it isn't costs one comparison and further prevents bugs related to boundedness.

Running this kernel at M = N = K = 4096 takes **299.1 ms**, which works out to **459.5 GFLOP/s**. This be our first baseline that we will try to optimize against cuBLAS's `cublasSgemm` implementation. Note that the cuBLAS implementation computes the same product in 3.616 ms at 38,014 GFLOP/s. This puts our FP32 SGEMM naive kernel **performance at 1.2%** of cuBLAS.



### Lower Bounding the Fastest Possible Runtime

<!-- ============================================================
     BLOCK B - THE ROOFLINE                          (~30 min)

     Answers: "how bad is this, and is 90% of cuBLAS even reachable?"

     Write in this order:

     1. Compute floor
          2*M*N*K = 137.44 GFLOP  /  56.3 TFLOP/s  =  2.44 ms

     2. Memory floor (compulsory traffic only)
          read A + B + C = 201.3 MB,  write C = 67.1 MB
          268.4 MB  /  960 GB/s  =  0.28 ms

     3. *** THE THESIS SENTENCE OF THE ARTICLE ***
        Compute floor is 8.7x the memory floor, so GEMM at 4096 is
        FUNDAMENTALLY COMPUTE BOUND. The naive kernel is memory
        bound by construction - by how it was written - not by
        anything about the problem. Everything in chapters 4-8 is
        raising arithmetic intensity until compute binds.

     4. Where this kernel sits
          2K loads per result = 0.25 FLOP/byte
          machine balance     = 58.6 FLOP/byte   (56.3 TFLOP / 960 GB)
          off by 234x

     5. Scale, so the reader can trust the destination
          cuBLAS 3.509 ms = 70% of the 2.44 ms floor -> not magic,
          just a well-fed kernel. You are at 300 ms = 123x off.

     6. Honesty beat
          The 2*M*N*K count implies 549.8 GB and a 573 ms floor,
          but it runs in 300 ms. B[i*N+col] is warp-invariant, so
          the caches already absorb much of it. Access counts are a
          model, not a measurement.

     TRAP: do NOT plot this on a roofline chart at AI = 0.25. That
     roof predicts 0.25 * 960 = 240 GFLOP/s and the kernel measures
     458, so it would sit above its own memory roof and look like an
     error. Lower-bound arithmetic only, no plot. siboehm does the
     same, for the same reason.
     ============================================================ -->


### Memory Access Pattern of the Naive Kernel

<!-- ============================================================
     BLOCK C - THE DEFECT                            (~30 min)

     Answers: "what specifically is wrong with THIS kernel?"
     B named the class (memory bound). C names the mechanism.

     Write in this order:

     1. Call back to "Firstly" above - x is the fastest-changing
        dimension, so warp 0 is threadIdx.x 0..31 at threadIdx.y = 0,
        which spans 32 consecutive ROWS of a single column.

     2. Three access sites, three different behaviours:

          C[row*N + col]  store  32 addrs, 4096 floats apart  -> 32 sectors
          A[row*K + i]    load   strided by K                 -> 32 sectors
          B[i*N + col]    load   col is warp-invariant,
                                 all 32 lanes -> one address  ->  1 sector (broadcast)

     3. Ideal is 4 sectors (32 lanes * 4 B = 128 B). Two of the three
        sites are at 32: the warp moves 1024 bytes to deliver 128.

     4. [DIAGRAM - checklist step 9] one warp's 32 addresses: strided
        fan-out on A and C against the single broadcast address on B.
        Build with the cuda-viz skill. NOT the same figure as the one
        above - that one shows what the kernel computes, this one
        shows what the memory system sees.

     5. [PROFILER - checklist step 8, currently blocked on counter
        permission] sectors/request from
          l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
          l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
        This is confirmation, not the argument. Ship without it if
        the permission fix drags.

     6. CLOSE ON THIS DEFECT, not on block B's roofline. Chapter 3
        fixes the mechanism and leaves arithmetic intensity at
        exactly 0.25 FLOP/byte - so a close on "memory bound" would
        be contradicted on arrival.
     ============================================================ -->


<!-- ============================================================
     THE SEAM - write both sides in one sitting     (step 6, 10 min)

     LAST sentence of this page:  the 1024-bytes-to-deliver-128 cost,
       plus the note that fixing it changes nothing about the
       arithmetic - still 2K loads per result.

     FIRST sentence of page4:     picks up that exact number.

     Page 4 must NOT re-teach warps. Chapter 1 already covered warps,
     SIMT, divergence and latency hiding. siboehm opens his kernel 2
     with "we need to learn about the concept of a warp" only because
     he has no chapter 1. Page 4 teaches sectors and transaction
     sizes, and nothing else.
     ============================================================ -->

