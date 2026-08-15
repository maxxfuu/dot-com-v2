## The Naive Kernel: Establishing a Baseline

For the first kernel, we are really just trying to express GEMM in CUDA. We'll start off by using the grid, block, and thread to assign one thread to each output cell in the output matrix C. Start by expressing each thread with its respective global thread indices: 

![Each thread computes one element of C, labelled here by the cell it owns as t[row][col]. Thread t[0][0] takes the top-left corner, and stepping one cell right or one cell down moves you exactly one column or one row through the output.](/images/gemm/matrix-c-indexing.png)

``` cuda 
const int BLOCKSIZE = 32;

const int row = blockIdx.y * BLOCKSIZE + threadIdx.x;
const int col = blockIdx.x * BLOCKSIZE + threadIdx.y;
```

Given this mapping, a single thread computes the dot product between a row of A and a column of B, then writes the dot product into its corresponding output cell in C. 

Therefore, we need to write the kernel for a single thread as everything will be executed in parallel within a warp. For each thread that writes to its own output cell in matrix C, the thread will walk the full K dimension between its corresponding row and column. Written out against our actual function signature, the entire kernel looks like this:

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

Running this kernel at M = N = K = 4096 takes **300.1 ms**, which works out to **458.0 GFLOP/s**. This will be our first baseline that we will try to optimize against cuBLAS's `cublasSgemm` implementation. Note that the cuBLAS implementation computes the same product in 3.509 ms at 39,168 GFLOP/s. This puts our FP32 SGEMM naive kernel **performance at 1.2%** of cuBLAS.



### Lower Bounding the Fastest Possible Runtime on a Theoretical GEMM

A lower bound identifies the fastest runtime achievable under ideal conditions; we are going to establish the compute floor and the memory floor as theoretical lower bounds on how quickly a **theoretical GEMM** can execute, based on the GPU's compute throughput and memory bandwidth.

Given that the output matrix `C` is in the shape of `M x N` output elements, and there are `K` elements per dot-product, and there exists `2` operations (multiplication & addition) when computing the dot-product, the SGEMM has to perform `2 x M x N x K` floating point operations.

Strictly speaking, for `M = N = K = 4096`:

``` latex
2(4096)^3 = 137.44\text{ GFLOP}
```

Now throughput in GPU computing is the rate at which a system can complete work. The RTX 5080's peak throughput with FP32 is 56.3 Tera FLOP/s; this means it can retire at most 56.3 trillion floating point operations every second. Pushing this GPU to its absolute limit gives us a theoretical compute floor: 

``` latex 
T_{\text{compute}} =
\frac{137.44\text{ GFLOP}}
{56.3\text{ TFLOP/s}}
\approx 2.44\text{ ms}

```

Similarly, let's establish the memory floor. Every byte the kernel uses has to come from VRAM, GDDR7 in our case, at least once. For our SGEMM in FP32, each element is `4 bytes`, so a single `4096 x 4096` matrix occupies: 

``` latex
4096 \times 4096 \times 4\text{ B} = 67.1\text{ MB}
```

The `C = alpha*AB + beta*C` accumulate form reads `A`, reads `B`, reads `C`, and
writes `C` back. The scalars cost no traffic of their own, so this is four
matrix-sized trips through VRAM:

``` latex
4 \times 67.1\text{ MB} = 268.4\text{ MB}
```

The RTX 5080 has `960 GB/s` of memory bandwidth, so even at perfect bandwidth
utilization the traffic alone costs:

``` latex
T_{\text{memory}} =
\frac{268.4\text{ MB}}
{960\text{ GB/s}}
\approx 0.28\text{ ms}
```

The GPU overlaps arithmetic with memory traffic rather than serializing them, so these two floors do not add. The kernel is bounded by whichever one is higher:

``` latex
T \geq \max(T_{\text{compute}}, T_{\text{memory}}) = 2.44\text{ ms}
```

For now, know that there is no ideal SGEMM kernel with FP32 where dimensions M = N = K = 4096 with this specific GPU can finish in less than **2.44 ms**. With our hand written kernel, SGEMM took 300.1 ms, against a floor of 2.44 ms. It is 123 times slower than the ideal.

So really, there's only one question. Where did all this additional time come from and how are we 123 times slower than the theoretical? No single mistake accounts for the whole gap, but the first and the largest of them lies in **transaction**. How many separate chunks the memory system has to fetch to satisfy a single instruction. A memory instruction does not belong to a thread, it belongs to a warp. 

### Inefficient Warp Memory Access Pattern

Knowing that `threadIdx.x` is the fastest-changing dimension, warp 0 is `threadIdx.x` 0 through 31 at `threadIdx.y = 0`. And because we mapped `row` to `threadIdx.x`, those 32 lanes hold 32 consecutive **rows** at a single column. The warp maps onto a vertical strip of C, and onto A the same way.

With B, `col` is warp-invariant, so all 32 threads ask for the same element of B. The hardware then broadcasts that one value into all 32 registers for free data reuse.

So, seen from the warp rather than from the thread, the kernel touches memory in three places: two loads on every iteration of the K loop, and a read-modify-write of C once the loop is done.

```
A[row * K + i]     load    32 addresses, strided by K        -> 32 sectors
B[i * N + col]     load    col is warp-invariant,
                           all 32 lanes -> one address       ->  1 sector (broadcast)
C[row * N + col]   load    32 addresses, 4096 floats apart   -> 32 sectors
                   store   the same 32 addresses             -> 32 sectors
```

C shows up twice because `beta * C[row * N + col]` has to read the old value back before the assignment overwrites it, and the read is strided exactly like the write.

![One warp of the block, lanes 0 through 31 of `threadIdx.x`, and the memory each site touches. The load from A walks 32 rows at a stride of K, so the hardware fetches 32 separate cache lines; the load from B is uniform across all 32 lanes, so one value is broadcast into every register.](/images/gemm/inefficient-warp-access.png "large")

Memory comes back from the L1 in `32-byte` sectors. A warp asking for 32 contiguous floats wants `128 bytes`, which is 4 sectors, and 4 is the best any warp can do. Two of our three arrays need 32, and C needs it twice.

Strictly speaking, the traffic to C and the load from A each move **1024 bytes to deliver the 128** the warp actually asked for. Seven eighths of every transaction is fetched, paid for, and thrown away, and A pays it on every one of the `K` iterations. Only the load from B escapes, and it escapes by accident: `col` depends on `blockIdx.x` and `threadIdx.y`, both constant across warp 0, so the hardware hands one sector to all 32 lanes.

One caveat before we carry that number too far. Sectors are traffic between the L1 and the L2, not traffic out of VRAM. Counted as accesses, this kernel asks for `2 x M x N x K x 4 B = 549.8 GB` of loads, and at 960 GB/s that alone would take 573 ms, yet the kernel finishes in 300.1 ms. Running the bound backwards, VRAM cannot have supplied more than `300.1 ms x 960 GB/s = 288 GB` of that, so a little under half of what we ask for never reaches VRAM at all; the cache is already absorbing it. The waste is real, but it is waste in requests, not in DRAM bytes.

It is worth being precise about what this defect is not. We are not loading too many *values*. The access count stays at exactly `2K` **global loads** per output element, one from A and one from B on every trip through the K loop, and the next kernel will not change it by a single load. We are loading the right values in the wrong *order*, and paying 8x the bytes to get them.
