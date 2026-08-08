---
title: "To write a CUDA Matmul Kernel That Reaches 90% of cuBLAS Performance"
date: "2026-07-31"
summary: "Work in progress... Check again during late September!"
---

## Introduction 

Placeholder — inline references render as superscript numbers, like this[^1] and this,[^2] and link down to the sources below.[^3]

:::check
During the forward pass, why do we store every `z` and `a` instead of discarding them?
---
The backward pass reuses them: computing each layer's gradients requires the activations produced during the forward pass, so recomputing them would double the work.
:::

[^1]: https://siboehm.com/articles/22/CUDA-MMM (@Si_Boehm)
[^2]: https://abhik.ai/articles/cuda-matrix-multiplication-optimization (@abhiksark)
[^3]: https://robertzhang.me/blog/cuda-mmm (@robdobflob)
