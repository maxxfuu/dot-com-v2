## The Roofline Model and GEMM

## The Roofline Model

The roofline model is a way to guage the performance of a kernel relative to the hardware the kernel is running on. It is the tool this article uses throughout each page decide what to optimize next.

The idea is that a kernel can be bounded by compute, memory, or some overhead[^1]. It is often limited compute or memory, but never by both at once. 

The first limitation is arithmetic. The RTX 5080 can at most perform 56.3 TFLOP/s of FP32, and no kernel can exceed this no matter how little memory it touches. 

The second limitation is bandwidth. The VRAM delivers at most 960 GB/s, so a kernel that needs a lot of bytes per unit of arithmetic runs out of bandwidth long before it runs out of FP32 units.

To determine if you're compute bound or memory bound is entirely dependent on the ratio of floating point operations per byte; and that ratio is called **arithmetic intensity**. It is the number of floating point operations a kernel performs for every byte it moves.

``` latex
P_{\text{attainable}} =
\min\left(P_{\text{peak}},\ AI \times BW\right)
```

Plotting that gives the shape the model is named for. Performance rises along a slope as intensity increases, because a kernel with more arithmetic per byte gets more out of the same bandwidth, and then it flattens into a roof once the FP32 units saturate. The two pieces meet at one intensity, the **ridge point**, where the machine's arithmetic and its bandwidth are exactly in balance:

``` latex
AI_{\text{ridge}} =
\frac{56.3\text{ TFLOP/s}}
{960\text{ GB/s}}
\approx 58.6\text{ FLOP/byte}
```

![The roofline for the RTX 5080, with arithmetic intensity in FLOP per byte on the x axis and attainable performance in TFLOP/s on the y axis, both log scale. The sloped line rises at 960 GB/s until it meets the flat 56.3 TFLOP/s ceiling at the ridge point of 58.6 FLOP/byte. Everything left of the ridge is the memory bound region, where DRAM is the limit and extra FLOPs are free, and the way out is to raise intensity by reusing data through shared memory and register tiling. Everything right of it is the compute bound region, where the SMs are the limit and only more FLOP/s helps.](/images/gemm/roofline-model.png "full")

So on this card a kernel has to perform roughly 59 floating point operations for every byte it pulls from VRAM just to keep the arithmetic units fed. Put in terms of the FP32 numbers we are actually multiplying, that is about 234 operations for every 4 byte float loaded. A kernel below that intensity is **memory bound** and its ceiling is the sloped one; a kernel above it is **compute bound** and its ceiling is flat. Note that intensity is a property of the kernel and the ridge point is a property of the card, which means the ridge point is fixed and the only thing we get to move is the kernel.

It is worth knowing where this card sits. 58.6 FLOP/byte is a demanding ridge point, because the 5080 pairs a great deal of FP32 throughput with a fairly narrow 256 bit memory bus. A GPU with more bandwidth per FLOP has a lower ridge and forgives a sloppier kernel. That is also why performance numbers from an article written on a different card do not transfer to this one, and why every figure in this article was measured here rather than copied.

One caveat to carry forward, because we will run into it almost immediately. The bandwidth in the model is VRAM bandwidth, so the model quietly assumes every byte a kernel asks for is a byte that travels all the way from VRAM. Caches break that assumption. A value that is still sitting in L2 costs a fraction of what the model charges for it, so a kernel can measure *faster* than its own roofline says is possible. When that happens it is not a broken measurement, it is the model telling you that the traffic you counted is not the traffic that actually reached memory. That distinction turns out to be the difference between the first optimization in this article working and the second one being necessary.

## What is GEMM

Before we actually begin writing our GEMM kernels and go through an iterative process on optimizing it, let's first establish what a General Matrix Multiplication (GEMM) is so that we understand what's going on behind the scenes with every implementation. Every kernel in this article computes the exact same thing; an output matrix C computed by matrices A and B. The **GEMM** follow the same mathematical formula listed below:

``` latex
C \leftarrow \alpha A B + \beta C
```

Note that variables alpha and beta are just scalar coefficients that scale the matrix multiplication and the existing output matrix C.

**A is (M x K)**, **B is (K x N)**, and **C is (M x N)**. All three of these matrices are stored in row-major order in memory. Each optimized kernel thats introduced within the article will use the following dimensions: **M = N = K = 4096**, **alpha = 1**, **beta = 0**. 

This ensures that very kernel performs the same 2 * M * N * K floating point operations, which is roughly 137 billion operation in total, against the same three matrices. It's really just a basic matrix multiplication.

[^1]: [How CUDA Programming Works - Stephen Jones, CUDA Architect](https://www.youtube.com/watch?v=QQceTDjA4f4&t=75s)
