+++
date = '2026-05-28T16:29:15+08:00'
draft = false
title = 'CuTe学习0-CuTe简介'
description = '简单介绍CuTe抽象'
readingTimeText = ''
tags = ['CuTe']
categories = ['Technical Blog']
+++
# CuTe

CuTe（全称 CUDA Templates）是NVIDIA从CUTLASS 3.0 版本开始推出的一个C++模版库。博主认为，CuTe主要作用在如下三个方面，本文将按此顺序介绍CuTe的抽象。
- 提供数据从内存排布到逻辑索引的映射原语
- 提供数据在各内存层级之间的搬运原语
- 提供数据做矩阵乘法的计算原语

# Layout & Tensor 