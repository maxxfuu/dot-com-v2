---
title: "A Small cuBLAS Detail With a Deep Lesson"
date: "2026-07-23"
summary: "The reasoning behind cuBLAS's operand swap, and the layout lesson hiding inside it."
---

# A Small cuBLAS Detail With a Deep Lesson 

When I first called cuBLAS's cublasSgemm(), I learned that the input matrices needed to be swapped to get the correct output. But when you're starting out, it's easy to gloss over the little detail that makes cublasSgemm() work. Instead of focusing on the fix, I want to share the reasoning behind why the swap works!

To define this "problem", we first need to understand that all of the matrices are stored in memory as a contiguous array. When we write matmul ourselves as CUDA practitioners, we internalize the data in a row-major layout, where neighboring cells run from left to right, row by row. However, cuBLAS interprets the matrices in memory in a column-major layout, meaning it reads the data top to bottom, column by column. This difference lies in the interpretation; cublasSgemm() reads the matrices as if they were transposed.

![Difference in interpretation.](/videos/cublas/cublas-1-two-interpretations.mp4)

Our objective is to perform a matrix multiply, a GEMM.

Suppose matrix A is (m, n) and matrix B is (n, k), so matrix C = A · B with the shape (m, k). Because cuBLAS reads our row-major buffers in a column-major fashion, it sees A as Aᵀ, now shaped (n, m) instead of (m, n), and B as Bᵀ, shaped (k, n) instead of (n, k). Keeping our original order, cuBLAS would try to compute Aᵀ · Bᵀ; that's (n, m) @ (k, n), and the inner dimensions (m and k) don't match, so the multiply is invalid.

![Transposed shapes.](/videos/cublas/cublas-2-transposed-shapes.mp4)

To fix this, we swap the matrices (along with the dimension and leading-dimension arguments that are passed into cublasSgemm()). cuBLAS then computes Bᵀ · Aᵀ with the correct shapes, (k, n) @ (n, m), resulting in the output matrix Cᵀ with shape (k, m).

![Swap the operands.](/videos/cublas/cublas-3-operand-swap.mp4)

Again, cuBLAS writes Cᵀ to memory in column-major fashion, which lays down the identical bytes, in the identical order, as writing C in row-major would. So when any row-major code reads the buffer back, it reads C directly, without the transpose, meaning we never need to worry about Cᵀ.

![Identical bytes.](/videos/cublas/cublas-4-identical-bytes.mp4)

Afterthoughts: While this concept is trivial, I believe understanding it provides intuition for how GEMM actually gets optimized. Throughout writing CUDA and learning about GPU hardware, I've noticed a pattern where a lot of CUDA optimizations boil down to how data is laid out rather than some clever algorithm.

