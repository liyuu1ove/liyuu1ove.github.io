+++
title = 'CuTe学习7-SM90 WGMMA'
date = '2026-08-27T10:29:37+08:00'
draft = true
description = '基于CuTe封装讲解SM90 WGMMA'
readingTimeText = '阅读此文大概需要 分钟'
tags = []
categories = ['Technical Blog']
+++


# WGMMA
WGMMA（Warp-Group Matrix Multiply-Accumulate，线程束组矩阵乘加是NVIDIA在Hopper架构中引入的第四代Tensor Core核心PTX指令。其新特性为更大的矩阵尺寸，异步计算和Shared Memory操作数。


## Warp Group and Larger Tile
前代Ampere通常使用以一个Warp32个线程为单位的 mma.sync 指令。而 WGMMA 引入了 Warp Group 概念，由 4 个连续 Warp（共 128 个线程） 共同组成一个执行单元，跨 128 个线程协同调用 Tensor Core 发起矩阵乘加计算。
### Warp Specialize and Warp Group


## Async pipeline

## stmatrix
