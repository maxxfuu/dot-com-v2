## 1D Register Tiling: One Thread, TM Outputs

Kernel 4 moved the traffic instead of removing it. Global loads per output element fell 32 times, runtime fell 1.41 times, and the count that did not move at all was the shared memory one: `2K` = 8192 reads per output element, one from `As` and one from `Bs` for every multiply accumulate. We swapped a load that crosses the memory bus for a roughly 33 cycle shared memory load[^1], and kept the number of loads identical.

The reason that count is stuck is the thread mapping, not the memory. One thread owns one element of C, so every value it pulls out of shared memory feeds exactly one FMA and is then discarded. There is no way to amortize a load across arithmetic when there is only one piece of arithmetic to amortize it against.

So give a thread more than one output. Let one thread own a short column of C, `TM` elements tall, all in the same column and in consecutive rows. Those `TM` outputs all need the same element of B, because they share a column, and they need `TM` different elements of A. The thread loads the B value once into a register and multiplies it against `TM` values of A. One shared memory read now feeds `TM` FMAs instead of one.

![One block owns a 64 x 64 tile of C and walks K in steps of BK = 8, loading a 64 x 8 tile of A into As and an 8 x 64 tile of B into Bs. The zoom on a 16 x 16 corner of the C tile shows the change in mapping: kernel 4 gave one thread one cell, and now one thread owns a column of 8 consecutive cells in the same column, with the next thread taking the column beside it.](/images/gemm/register-1d-overview.png "full")

That is the whole idea, and the accounting for it is immediate. Per step of the K loop a thread reads `1` value of B and `TM` values of A, so `1 + TM` reads produce `TM` results instead of `2` reads producing `1`.

![One thread at one step of the BK loop. Kernel 4's two shared memory reads per result become 1 + TM reads for TM results: a single read from the Bs row lands in a register and is reused TM times, while TM values are read down a column of As, and the two feed TM = 8 fused multiply-adds accumulating into 8 registers. The figure names the broadcast register Btmp; the listing below calls it regN.](/images/gemm/register-1d-inner-loop.png "large")

### The Kernel

The tile shape changes along with the thread mapping. The block now owns a `64 x 64` tile of C and walks K in steps of `BK = 8`, and because each thread produces `TM = 8` outputs the block needs `(64 x 64) / 8 = 512` threads rather than 4096:

```cuda
const int BM = 64, BN = 64, BK = 8, TM = 8;

dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
dim3 blockDim((BM * BN) / TM);   // 512

sgemm_register_tiling_1d<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

``` cuda 
#include <cuda_runtime.h>

// Template accepts block dimensions followed by a thread mulitiplier
template <const int block_M, const int block_N, const int block_K, const int thread_multiplier>
__global__ void sgemm_1d_register_tiling(int M, int N, int K, float alpha, float beta, float *A, float *B, float *C) {
  const int C_Row = blockIdx.y;
  const int C_Col = blockIdx.x;

  // shared memory allocation
  __shared__ float A_smem[block_M * block_K];
  __shared__ float B_smem[block_K * block_N];

  // Set pointer at top left coner of the block tile
  A += C_Row * block_M * K;
  B += C_Col * block_N;
  C += C_Row * block_M * N * C_Col * block_N;

  // calculate coordinates for computation within the block tile
  const int thread_col = threadIdx.x % block_N;
  const int thread_row = threadIdx.x % block_N;

  // calculate coordinates for block tile
  const int inner_col_A = threadIdx.x % block_K;
  const int inner_row_A = threadIdx.x / block_K;

  const int inner_col_B = threadIdx.x % block_N;
  const int inner_row_B = threadIdx.x / block_N;

  // static array allocated within the on-chip registers
  float thread_results[thread_multiplier] = {0.0f};

  for (int block_idx = 0; block_idx < K; block_idx += block_K) {
    // load data from global memory into shared memory
    A_smem[inner_row_A * block_K + inner_col_A] = A[inner_row_A * K + inner_col_A];
    B_smem[inner_row_B * block_K + inner_col_B] = B[inner_row_B * K + inner_col_B];

    __syncthreads();

    // advance global pointers to the next block iteration
    A += block_K;
    B += block_K * N;

    // compute thread_multiplier elelments per thread
    for (int dot_product_idx = 0; dot_product_idx < block_K; ++dot_product_idx) {
      // load one temporary B element into a register per iteration
      float temp_B = B_smem[dot_product_idx * block_N + thread_col];

      for (int res_idx = 0; res_idx < thread_multiplier; ++res_idx) {
        thread_results[res_idx] += A_smem[(thread_row * thread_multiplier + res_idx) * block_K + dot_product_idx] * temp_B;
      }
    }

    __syncthreads();
  }

  for (int res_idx = 0; res_idx < thread_multiplier; ++res_idx) {
    C[(thread_row * thread_multiplier + res_idx) * N + thread_col] =
        alpha * thread_results[res_idx] +
        beta * C[(thread_row * thread_multiplier + res_idx) * N + thread_col];
  }
}

```

### Mechanics

1. **One hoisted line is the entire kernel.** `float regN = Bs[dotIdx * BN + threadCol];` sits outside the `resIdx` loop, so the B value is read once and reused `TM` times from a register. Move that line inside the inner loop and you have written kernel 4 again with extra steps: same results, same arithmetic, eight times the shared memory reads. Everything else in this file is bookkeeping to make that hoist possible.

2. **The same 512 threads are decomposed two different ways.** For loading, `innerRowA`/`innerColA` cut the block into a `64 x 8` grid matching the shape of the A tile, and `innerRowB`/`innerColB` cut it into `8 x 64` matching the B tile. For computing, `threadRow`/`threadCol` cut it into `8 x 64`, where the row index is scaled by `TM` on use. This is what kernel 3 was for. None of these three shapes is the launch shape, and all three come out of the same flat `threadIdx.x`.

3. **The loads still land exactly, with no loop.** `As` holds `BM x BK = 512` floats and `Bs` holds `BK x BN = 512`, and there are 512 threads, so each thread copies one element of each and the load is a single statement. This is a coincidence of the chosen tile shape, not a property of the technique, and the next kernel loses it.

4. **`threadRow * TM + resIdx`, not `threadRow + resIdx * something`.** A thread's `TM` outputs are consecutive rows, so its rows are `threadRow * TM` through `threadRow * TM + TM - 1`. That keeps `threadCol` as the fast moving index across a warp, which keeps the store to C coalesced. The read from `As` marches down a column of the tile with a stride of `BK`, which shared memory does not mind the way global memory would.

Kernel 5 runs in **10.506 ms at 13081.8 GFLOP/s, which is 36.3% of cuBLAS at `M = N = K = 4096`**, up from 33.268 ms and 11.5%. That is 3.17 times faster than kernel 4, and the largest single jump in the series.

### The Arithmetic

Global memory first, because the tile grew. `K x (1/BM + 1/BN)` with `BM = BN = 64` gives `K / 32 = 128` loads per output element, down from 256, so doubling each tile dimension halved global traffic to **8.59 GB**. That is a real improvement and it is not where the speedup came from.

Shared memory is where the section lives. Per step of the K loop a thread now issues `1 + TM = 9` reads to produce `TM = 8` results, so:

```
SMEM accesses per result = K * (1 + TM) / TM
                         = 4096 * 9 / 8
                         = 4608
```

against `2K = 8192` for kernel 4. That is a factor of **1.78**, and the runtime moved by a factor of **3.17**. The count under-predicts the win by nearly twice, which is worth stopping on because it is the opposite of kernel 4's problem, where the count over-predicted.

The reason is that a shared memory read is not only a latency to be hidden, it is an instruction to be issued. Kernel 4's inner loop issued two `LDS` and one `FFMA` per output; this one issues nine `LDS` and eight `FFMA` per eight outputs. The loads that disappeared took issue slots with them, and the ratio of memory instructions to arithmetic instructions went from 2:1 to 9:8. Access counts measure bytes touched. They do not measure the instruction stream, and from here on the instruction stream is increasingly what we are fighting.

Nsight measures that shift directly. Kernel 4 sat at 40.5 warp cycles per issued instruction with 23.9 of them, 58.9%, stalled on `Stall MIO Throttle`, the queue a warp waits in for a shared memory instruction to issue. This kernel runs at **21.0 warp cycles per issued instruction with MIO throttle down to 6.6 cycles, 31.4%**. The stall did not merely shrink in proportion to the loads removed; the whole issue pipeline got shorter.

The occupancy table says the same thing from another direction. This kernel uses 48 registers per thread against kernel 4's 40, and at 512 threads per block that is 2 blocks per SM and 66.7% occupancy, identical to kernel 4's. We bought an extra 8 registers per thread and paid nothing for them.

### What Could Be Improved

The reuse is one sided. A thread reads a value of A once and uses it once; it reads a value of B once and uses it `TM` times. Written as a ratio, `1 + TM` loads feed `TM` FMAs, so as `TM` grows the loads per FMA approach 1 and stop there. Even at `TM = 64` every FMA would still cost roughly one shared memory read, because the A side never amortizes at all.

The fix is symmetric with the problem. A thread owns a column of C, which is why only B is shared among its outputs. Give it a rectangle instead and both operands amortize at once.

[^1]: [Dissecting the SM_120 Microarchitecture: Cycle-Level Characterization of Blackwell Consumer GPUs](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html)
