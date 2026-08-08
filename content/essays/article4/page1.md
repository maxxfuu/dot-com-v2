---
title: "To write a CUDA Matmul Kernel That Reaches 90% of cuBLAS Performance"
date: "2026-07-31"
summary: "Work in progress... Check again during late September!"
---

## Introduction 



In this article I will walk you through how to write a CUDA matrix multiplication naive kernel to optimizing it to reach a performance of 90% comparitively to that of a GEMM cuBLAS kernel. Learning CUDA has been an incredible and exciting journey, but something that helped me build my CUDA muslces was optiming a SGEMM. 

This writing is an taken inspirtaion from other blogs such as[^1] and this,[^2] and link down to the sources below.[^3]

:::check
During the forward pass, why do we store every `z` and `a` instead of discarding them?
---
The backward pass reuses them: computing each layer's gradients requires the activations produced during the forward pass, so recomputing them would double the work.
:::

[^1]: https://siboehm.com/articles/22/CUDA-MMM (@Si_Boehm)
[^2]: https://abhik.ai/articles/cuda-matrix-multiplication-optimization (@abhiksark)
[^3]: https://robertzhang.me/blog/cuda-mmm (@robdobflob)
