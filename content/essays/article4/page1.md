---
title: "To write a CUDA Matmul Kernel That Reaches 97% of cuBLAS Performance"
date: "2026-08-10"

summary: "Understanding GPU hardware, CUDA's execution and memory models, and progressively optimizing a matmul kernel from naive implementation to 97% of cuBLAS performance."

---

## Introduction 

In this article, I will walk you through how to write a CUDA matrix multiplication naive kernel and optimizing it to reach ~97% of SGEMM cuBLAS kernel. 

![Every kernel in this article, benchmarked on an RTX 5080 at M=N=K=4096, alpha=1 and beta=0. Each row is the median of 7 trials of 5 reps after a 300 ms warm-up, with the run-to-run spread and the max relative error against cuBLAS alongside it. The ladder runs from the naive kernel at 410.5 GFLOP/s and 1.1% of cuBLAS up to the final tuned kernel at 34978.3 GFLOP/s and 97.1%. cuBLAS was re-timed after the sweep and drifted 0.13%, so the percentage column is sound.](/images/gemm/kernel-benchmark.png)

Learning CUDA has been an incredible and exciting journey. While learning CUDA has become easier over the past few years, most resources remain difficult to digest. 

The intended audience for this article is someone who knows *absolutely nothing* about CUDA or GPU optimization. My goal is to bring them up to speed and teach them how to write a performant CUDA kernels from scratch.

This article teaches you how to optimize a matrix multiplication kernel; it is the single most important operation within modern deep learning. Therefore there is no better way to learn about CUDA than to write an optimized version of matrix multiplication that performs near performance of cuBLAS. 

I've previously thought that writing performant CUDA kernels was about coming up with most very clever algorithms to squeeze out all of the performance of a GPU. I was very wrong. While learning how to optimize a the kernel myself, I've noticed the majority of optimization comes from understanding the GPU memory hierachy at the most fundamental level and learning how to feed data into computation units efficiently without wasting any resources.

Its all about the memory hierachy and the movement of data with the CUDA software. The idea is simple, but the execution can be difficult. Within this article, we will iteratively walk through the hardware components of a CUDA architecture all the way to optimizing a CUDA kernel. 

Below is the table of contents, feel free to skip around although reading this sequentially is advised. 

### To write a CUDA Matmul Kernel That Reaches 97% of cuBLAS Performance
1. [From CUDA Hardware Architecture to CUDA Kernels](/writings/article4/page2)
2. [The Naive Kernel: Establishing a Baseline](/writings/article4/page3)
3. [Global Memory Coalescing](/writings/article4/page4)
4. [Flattening the Block](/writings/article4/page5)
5. [Shared Memory Tiling](/writings/article4/page6)
6. [1D Register Tiling: One Thread, TM Outputs](/writings/article4/page7)
7. [2D Register Tiling: The Outer Product](/writings/article4/page8)
8. [Vectorized Memory Access: 128-bit Loads and Stores](/writings/article4/page9)
9. [Warp Tiling: A Third Level of Tiling](/writings/article4/page10)
10. [Autotuning: Searching the Tile-Shape Space](/writings/article4/page11)
11. [Padded Shared Memory: Eliminating Bank Conflicts](/writings/article4/page12)
12. [Double Buffering: Software Pipelining the K-Loop](/writings/article4/page13)
13. [The Final Kernel: Retuning Against the Pipeline](/writings/article4/page14)


This writing was inspired by blogs such as Simons writing on this exact topic[^1]. Abhik's[^2] and Robert's[^3] on this topic is what cemented these concepts for me. The reset of my knowledge gap was filed with reading PMPP [^4], Modals' GPU Glossary[^5], and a lot of tokens.

[^1]: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
[^2]: [CUDA Matrix Multiplication Optimization: From Naive to Near-cuBLAS](https://abhik.ai/articles/cuda-matrix-multiplication-optimization)
[^3]: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://robertzhang.me/blog/cuda-mmm)
[^4]: [Programming Massively Parallel Processors: A Hands-on Approach (5th Edition)](https://www.amazon.com/dp/0443439001)
[^5]: [GPU Glossary](https://modal.com/gpu-glossary)
