+++
title = 'CuTe学习7-CuTe swizzle and TMA swizzle'
date = '2026-09-01T16:12:34+08:00'
draft = true
description = ''
readingTimeText = '阅读此文大概需要 分钟'
tags = []
categories = ['Technical Blog']
+++

# Why Swizzle
swizzle在算子开发中主要有两层意思，一个是从Global Memory读到Shared Memory时，用于提高L2缓存命中率的优化。另一种是写入/读取Shared Memory时防止出现Bank Conflict的优化，本篇主要介绍后者。

我们需要Swizzle有两方面原因，首先是要防止出现Bank Conflict，其次是ldmatrix/wgmma/tcgen05从shared Memory取操作数要求特定的排列。这种特定的排列会出现，也是为了防止bank conflict，所以这两个原因本质上是一样的。

本篇会先介绍bank conflict，然后介绍CuTe
## Shared Memory and Bank Conflict
Bank Conflict这节我有点懒得写了，用AI代劳了，后面两节是手搓的。
TODO：介绍Shared Memory Bank物理结构，然后介绍Bank Conflict。

# CuTe swizzle


# TMA swizzle



