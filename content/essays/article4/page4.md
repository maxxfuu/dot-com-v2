## Global Memory Coalescing

The naive kernel implementation loads the right values in the wrong order. Right now the way we are assigning the global thread index to the row and col variables will result in all of the threads accessing data from the global memory in a unconsecutive manner. 

From a warp perspective of A, each thread access a different row at 1 lockstep. A thread would access the row C[0][n], the next thread will access c[1][n]; each thread within the warp accesses a new row. From a thread persepctive, a thread walks along row and access a new element. Suppose the thread 0's first iteration access c[0][0], the second iteration the thread will access c[0][1]. Each time the warp fetches a new element is needs to load the whole 32 byte row. 

Within A, we can conclude that each thread accessing a new row for a single element is very wasteful. Each thread fetching a 32 bytes just to use 4 bytes of memory is wasteful. If instead we can have all threads within the same warp access data sequentially such that each element is adjacenet to each other, every thread would load the value from one 32 byte loads with nothing wasted. 

![Left: the naive mapping puts threadIdx.x on row, so a warp's 32 lanes land on 32 different rows of A, each K·4 bytes apart in memory - nothing to merge into one transaction. Right: swapping the mapping puts threadIdx.x on col, so all 32 lanes share one row and hit 32 adjacent columns instead - one contiguous 128 B segment the hardware can coalesce into a single transaction.](/images/gemm/coalescing-access-pattern.png "large")

``` latex
1024\text{ B} = 32 \text{ sectors} \times 32\text{ B/sector}, \qquad 128\text{ B} = 32 \text{ threads} \times 4\text{ B/float}
```

That's the accounting. Here's what it looks like at the sector level.

![Four 32 B sectors, eight 4 B slots each. Naive: each lane claims a whole sector for its one 4-byte value, so only 1 of 8 slots per sector is ever used — 32 sectors moved, 1024 B, 128 B of it useful. Coalesced: all 32 lanes pack into the same four sectors, every slot filled — 4 sectors moved, 128 B, all of it useful.](/images/gemm/sector-utilization.png "large")

Coalesced never asks the memory system for more than it needs — four sectors, and every slot inside them is a lane that's actually waiting on that value. Naive asks for the same four-sector footprint thirty-two separate times, once per lane, and each time only one of the eight slots inside is doing anything. Same hardware, same 32 B granularity — the only thing that changed is whether the requests line up well enough to share.


```cuda
template <const int BLOCKSIZE>
__global__ void sgemm_coalesced_2d(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
  
  const int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
  const int col = blockIdx.x * BLOCKSIZE + threadIdx.x;

  if (row < M && col < N) {
    float temp = 0.0f;
    for (int i = 0; i < K; ++i) {
      temp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}
```
