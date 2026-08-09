---
title: "To write a CUDA Matmul Kernel That Reaches 90% of cuBLAS Performance"
date: "2026-07-31"

summary: "Understanding GPU hardware, CUDA's execution and memory models, and progressively optimizing a matmul kernel from naive implementation to 90% of cuBLAS performance."

---

## Introduction 

In this article, I will walk you through how to write a CUDA matrix multiplication naive kernel and optimizing it to reach ~90% of SGEMM cuBLAS kernel. 

![Every kernel in this article, benchmarked on an RTX 5080 at M=N=K=4096.](/images/gemm/kernel-benchmark.png)

Learning CUDA has been an incredible and exciting journey. While learning CUDA has be easier within the past few years, most of the resources remain difficult to digest. My intended audience for this article is to teach someone that knows *absolutely nothing* about CUDA and GPU optimization and bring them up to speed on writing a performant kernel from scratch.

While learning how to optimize a SGEMM kernel myself, I've noticed the majority of optimization comes from understanding the GPU memory hierachy at the most fundamental level and learning how to feed computation units correctly. I've previously thought that writing performant CUDA kernels was about coming up with most very clever algorithms to squeeze out all of the performance of a GPU, I was wrong. 

Its all about the memory hierachy and the movement of data with the CUDA software. The idea is simple, but the execution can be difficult. We will iteratively walk through the hardware components of a CUDA architecture all the way to optimizing a CUDA kernel. Below is the table of contents, feel free to skip to skip around although reading this sequentially is advised. 

### To write a CUDA Matmul Kernel That Reaches 90% of cuBLAS Performance
1. [From GPU Hardware to High-Performance Kernels](/essays/article4/page2)
2. [The Naive Kernel: Establishing a Baseline](/essays/article4/page3)
3. [Global Memory Coalescing](/essays/article4/page4)
4. [Flattening the Block](/essays/article4/page5)
5. [Shared Memory Tiling](/essays/article4/page6)
6. [1D Register Tiling: One Thread, TM Outputs](/essays/article4/page7)
7. [2D Register Tiling: The Outer Product](/essays/article4/page8)
8. [Vectorized Memory Access: 128-bit Loads and Stores](/essays/article4/page9)
9. [Padded Shared Memory: Eliminating Bank Conflicts](/essays/article4/page10)
10. [Double Buffering: Software Pipelining the K-Loop](/essays/article4/page11)
11. [Warp Tiling: A Third Level of Tiling](/essays/article4/page12)
12. [Autotuning: Searching the Tile-Shape Space](/essays/article4/page13)


This writing was inspired by blogs such as Simons writing on this exact topic[^1]. Abhik's[^2] and Robert's[^3] on this topic is what cement these concepts for me. The reset of my knowledge gap was filed with reading PMPP [^4] and Modals' GPU Glossary[^5] and a lot of tokens.

[^1]: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
[^2]: [CUDA Matrix Multiplication Optimization: From Naive to Near-cuBLAS](https://abhik.ai/articles/cuda-matrix-multiplication-optimization)
[^3]: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://robertzhang.me/blog/cuda-mmm)
[^4]: [Programming Massively Parallel Processors: A Hands-on Approach (5th Edition)](https://www.amazon.com/dp/0443439001)
[^5]: [Modal, GPU Glossary](https://modal.com/gpu-glossary)
