+++
title = 'CuTe学习6-SM80 CuTe GEMM Walk Through'
date = '2026-07-13T17:29:09+08:00'
draft = true
description = '以SM80 HGEMM为例，详细介绍CuTe GEMM中的层次化结构'
readingTimeText = '阅读此文大概需要38分钟'
tags = ['CuTe','GEMM','CUDA']
categories = ['Technical Blog']
+++

# Overview
本文分析基于Ampere架构的dense HGEMM。

{{< image src="gemm-hierarchy.png" alt="Gemm Hierarchy" maxWidth="500px" caption="GEMM Hierarchy">}}
*Source from [CUTLASS](https://docs.nvidia.com/cutlass/latest/overview.html), modify applied*

从流程上来讲，使用CuTe编写的代码和Raw CUDA进行的HGEMM没有本质区别，也没有用到全新的优化方式，仍然是SM80优化的那一套。CuTe把曾经几乎没有可读性的，复杂的坐标计算重构为了CuTe的Tensor分块和切片，操作tensor的每个函数都有明确的语义。本文会重点解释参与运算的矩阵是如何被TiledMMA和TiledCopy重新组织，分块和切片的，而不是讲解如何在SM80上面优化一个HGEMM，较为基础的优化部分本文会略过。本文代码基于[CuTe Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/)。

本文会按照GEMM tiling的顺序进行讲解。本文会使用d表示动态整数，s表示静态整数。



# function calling structure

```CPP
int main(){//初始化参数，parsing，profiling
     gemm_tn(){// warpper，实例化TiledMMA，TiledCopy等，并且launch kernel
          gemm_device() // device kernel，处理核心逻辑（Tiling，Copy，Compute）
     }
}
```
我们在行文时会用gemm_tn()和 gemm_device()来区分代码处在的层级。

## gemm_device() signature
``` CPP
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)

```
其中一些参数的含义和类型如下：

1. `ProblemShape` 矩阵乘法的形状，直接使用make_shape(m,n,k)封装了三个dynamic int。本文使用d5120，d5120，d4096

2. `CtaTiler` 每个CTA(block)处理的形状，有的教程里面也叫 CTA tile，静态layout。本文使用128，128，64 

3.  `ASmemLayout, BSmemLayout, CSmemLayout` Shared Memory的layout，静态layout。

4. `TiledCopy` and `TiledMMA` Copy和MMA的形状和TV layout，静态。

5. `S2RAtom` 从Shared Memory搬运到Register时用的CopyAtom，本代码中用的是ldmatrix。

## gemm_tn() signature

``` CPP
template <class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        cute::half_t const* A, int ldA,
        cute::half_t const* B, int ldB,
        Beta beta,
        cute::half_t      * C, int ldC,
        cudaStream_t stream = 0)
```
值得吐槽的是，作者至今未理解为什么alpha和beta类型要用泛型表示，感觉完全没有意义...

## CuTe variable naming convention

CuTe和CUTLASS的GEMM代码变量名有自己的规范。这些变量名表示Tensor所在的数据层级、采用的线程partition以及GEMM operand。

没有经过线程partition的Tensor：
| 形式 | 含义 | 本文中的例子 |
|---|---|---|
| `mX` | 完整matrix的Tensor视图 | `mA`表示完整的A矩阵 |
| `gX` | 从matrix中切出的global-memory tile | `gA`表示当前CTA看到的A tile |
| `sX` | shared-memory Tensor | `sA`表示shared memory中的A tile |
| `rX` | register fragment | `rA`表示寄存器中的A fragment |

最后的大写`X`通常是`A`、`B`或`C`，表示这个Tensor属于哪个GEMM operand。

经过线程partition之后，常见形式是`tXgY`。左边的`tX`表示使用哪套thread partitioning pattern，右边的`gY`表示被partition的Tensor。本文中的`tA`和`tB`表示A/B的`ThrCopy`，`tC`表示`ThrMMA`。`tX`表示s2r copy。

| 变量 | 拆分 | 含义 |
|---|---|---|
| `tAgA` | `tA + gA` | 用A的GMEM-to-SMEM copy线程布局partition global-memory A tile |
| `tAsA` | `tA + sA` | 用同一个`tA`布局partition shared-memory A tile |
| `tBsB` | `tB + sB` | 用B的copy线程布局partition shared-memory B tile |
| `tCgC` | `tC + gC` | 用MMA线程布局partition global-memory C tile |
| `tCrA` | `tC + rA` | MMA线程看到的register A fragment |
| `tCrC` | `tC + rC` | MMA线程看到的register C accumulator |
| `tXsA` | `tX + sA` | 用SMEM-to-RMEM Copy的辅助布局partition shared-memory A |
| `tXrA` | `tX + rA` | 同一个`tX`布局看到的register A fragment |

# HGEMM
接下来我们按照数据在各个层级的内存之间流动的顺序来讲解HGEMM。
# GMEM->SMEM
## CTA Tiling

我们先建立整个tensor的视图。
```cpp
Tensor mA = make_tensor(make_gmem_ptr(A),select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B),select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C),select<0,1>(shape_MNK), dC); // (M,N)
```
我们处理的是一个tn形状的gemm，dA/dB/dC 是Stride，定义在gemm_tn()中
```CPP
auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
```

接下来我们使用CtaTiler来选出这个CTA要处理的CTA tile
``` CPP
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)
```
{{< image src="CtaTiler.png" alt="CTA Tiler" maxWidth="500px">}}

`cta_tiler`有M/N/K三个mode，但A、B、C都只使用两个mode。`Step`描述目标Tensor的mode使用tiler中的哪些mode，其中`X`表示该tiler mode不参与这个Tensor。A与N无关，所以同一个`blockIdx.x`下，不同`blockIdx.y`的CTA会读取相同的A区域；B则相反。这正是GEMM的正常数据复用关系。

## Shared Memory Buffer
```CPP
template <class ElementA,class ElementB,class SmemLayoutA,class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};
```
```CPP
extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
```
其中 sA_layout sB_layout定义在gemm_tn()中
```CPP
auto swizzle_atom = composition(Swizzle<3,3,3>{},
                              Layout<Shape <_8,Shape <_8, _8>>,
                              Stride<_8,Stride<_1,_64>>>{} );
auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
auto sC = make_layout(make_shape(bM, bN));
```
这里使用了swizzle来防止bank conflict。
## Tiling
TiledCopy定义在gemm_tn()中
```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},
    Layout<Shape< _1,_8>>{}); 
TiledCopy copyB = ... // same as copyA
```
在gemm_device()中，我们取出当前线程负责的slice
```CPP
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);           
// (CPY,CPY_M,CPY_K,k) ((_8,_1),_8,_1,64)
Tensor tAsA = thr_copy_a.partition_D(sA);           
// (CPY,CPY_M,CPY_K,PIPE) ((_8,_1),_8,_1,(_1,_3))
```
这里mode-0是CPY，这个维度包含了一次copy PTX能搬运的元素，我们使用128b的copy.async搬运half，每次可以搬运8个，所以得到CPY(_8,_1)。
mode-1和mode-2表示Copy需要重复的次数，CTA的A tile是`128 x 64`,TiledCopy是`16 x 64`，所以这个copy tile还要沿M方向重复8次，沿K方向重复1次。得到 _8,_1。k和gA中的k完全相同，我们传入Kd4096，bK取s64，这里就得到d64。

# SMEM->Register
## TiledMMA tiling

```cpp
TiledMMA mmaC = make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},
    Layout<Shape<_2,_2>>{},
    Tile<_32,_32,_16>{});
```

最底层MMA Atom的shape是`16 x 8 x 16`，由一个32线程warp执行。这里128个线程单次覆盖实际tile`32 x 16 x 16`，代码重新组织逻辑tile为`32 x 32 x 16`。这里是如何组织的在MmaAtom已经详细讲解过了。

我们要根据MMA的Shape划分gC，并且创建MMA fragment。
```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);
// (MMA,MMA_M,MMA_N) ((_2,_2),_4,(_2,_4))
```
这里使用ThrMMA去切分gC，得到gC的分块方式。mode-0表示MMA内部的Layout。mode-1和2表示TiledMMA在gC上的排列方式，bM是128，容纳4个TiledMMA实际tile。bN是128，容纳8个TiledMMA实际tile，但是我们对N进行了重排，实际组织为(_2,_4)。
```cpp
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
Tensor tCrC = thr_mma.make_fragment_C(tCgC);
```
这里我们创建mma fragment，`sA(_,_,0)`和`sB(_,_,0)`中的`0`只是表示第一个pipe的二维shape来创建fragment。`partition_fragment_A/B`在这里创建的是线程私有register fragment，之后会从任意pipe向其中填数据，并不意味着A/B永远读取pipe 0。

## Copy Atom retiling
从SMEM->Register我们使用ldmatrix进行搬运，所以这里就出现了第二套映射。mma和ldmatrix处理的是同一批逻辑元素，但其tiling方式并不相同。
这里使用的Atom是：
```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
```
它对应`ldmatrix.x4`风格的shared-to-register加载。

代码用`make_tiled_copy_A`把LDSM Copy Atom适配到既有TiledMMA：
```cpp
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);

Tensor tXsA = s2r_thr_copy_a.partition_S(sA);// (CPY,MMA_M,MMA_K,PIPE)
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);// (CPY,MMA_M,MMA_K)
```
`retile_D(tCrA)`则很关键：它不会新建一份register，也不会移动数据，而是为`tCrA`的同一Engine建立一个适合Copy destination的Layout视图。
copy按照`tX*`视图把数据写入register；随后的`gemm`按照`tC*`视图读取相同register。Retile的本质是协调两套Layout，而不是一次额外的数据重排。




# Pipelined main loop

前面建立的Tensor包含所有K tile、所有SMEM pipe和所有register K block。主循环每一时刻只操作其中一个切片。

## Prefetch to shared memory
```CPP
auto K_PIPE_MAX = size<3>(tAsA); //s3
int k_tile_count = size<3>(tAgA);//d64
int k_tile_next = 0;
CUTE_UNROLL
for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) { // prefetch 2 k_tiles
     copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
     copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
     cp_async_fence();
     --k_tile_count;
     if (k_tile_count > 0) { ++k_tile_next; }
}
```
做了两次Tensor slicing：
```text
tAgA(_,_,_,k_tile_next) 固定global K tile，保留当前线程的所有copy value
tAsA(_,_,_,k_pipe)      固定SMEM pipeline stage，保留对应的所有目标位置
```
两侧得到相同的逻辑shape`(CPY,CPY_M,CPY_K)`，所以一次`copy`就能表达当前线程从一个global K tile到一个shared memory stage的全部搬运。

## Prefetch to Register
在切分tXsA时我们在K维度上切出了4份，我们在K维度上进行pipeline。
```CPP
// Current pipe index in smem to read from
int smem_pipe_read  = 0;
// Current pipe index in smem to write to
int smem_pipe_write = K_PIPE_MAX-1;

// Pipe slice
Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

// Size of the register pipeline
auto K_BLOCK_MAX = size<2>(tCrA);
CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

// PREFETCH register pipeline
if (K_BLOCK_MAX > 1) {
// Wait until our first prefetched tile is loaded in
cp_async_wait<K_PIPE_MAX-2>();
__syncthreads();

// Prefetch the first rmem from the first k-tile
copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
}
```
## main loop
主循环伪代码如下：

```cpp
while(k_tile_count > -(K_PIPE_MAX-1)){//loop along K on SMEM
  cp_async_wait<K_PIPE_MAX-2>(); // 2 SMEM block on the flight
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)//loop along bK on Register
  {
  // Load A, B shmem->regs for k_block+1
  auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
  copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
  copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
  gemm();
  }
  // load bK tiles
  copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
  copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
}
```
外层循环在K上循环，从global memory预取到shared memory，这部分是真正硬件异步的。内层在k_block上循环，即mma内部，从shared memory预取到register。ldmatrix是阻塞的，我们在这里利用软件流水复用SM单元，在warp0 ldmatrix被阻塞时，warp1可以发射mma指令，如此保证Tensor Core永远处于活跃状态。

# Epilogue
在计算完成之后，结果还储存在tCrC中，这里我们使用cute的tensor algorithm axpby。
```cpp
axpby(alpha, tCrC, beta, tCgC);
```
`tCrC`和`tCgC`具有相同的逻辑输出shape，但Engine分别位于register和global memory。`axpby`逐元素完成：
```text
tCgC = alpha * tCrC + beta * tCgC
```
这个例子的epilogue直接会把fragment写回gC。

# 从一个CTA的视角串起整个过程

假设`blockIdx = (bm,bn)`，可以把一个CTA的执行过程概括为：

1. `local_tile`固定M/N tile坐标，得到`gA(128,64,k)`、`gB(128,64,k)`和`gC(128,128)`；
2. `copyA/copyB.get_slice(tid)`固定copy线程坐标，`partition_S/D`留下每线程的value与外层重复坐标；
3. 前两个global K tile通过`cp.async`进入三个SMEM stage中的前两个；
4. `mma.get_slice(tid)`固定warp和lane坐标，建立该线程的A/B/C MMA fragment；
5. `make_tiled_copy_A/B`为相同A/B fragment建立LDSM视图，`retile_D`让Copy和MMA共享同一批register；
6. 每个SMEM stage再切成4个`k_block`，一边加载下一个block，一边计算当前block；
7. 处理一个stage时，同时把后续global K tile写进空闲stage；
8. 所有K tile归约完成后，`axpby`把register accumulator写回当前线程的`gC`切片。

贯穿整个过程，真正的数据搬运只有三处：

```text
copy(copy_a/copy_b)       GMEM -> SMEM
copy(s2r_atom_a/b)        SMEM -> RMEM
axpby                     RMEM -> GMEM
```
# 总结

这段GEMM可以看成四次坐标分解：

```text
Problem MNK
  -> CTA tile (128,128,64) x CTA coordinate
  -> Thread-Value partition x copy/MMA outer repeat
  -> SMEM PIPE stage
  -> Register MMA_K block (16)
```

Tiling负责建立这些层级，partitioning负责把层级映射到线程，slicing负责在某一时刻固定CTA、thread、pipe或K block坐标。CuTe没有让程序员手算地址，而是把每一步都表达成Tensor Layout变换，再让`copy`和`gemm`在分割好的Tensor上面进行搬运和计算。

理解这套代码时，可以始终问三个问题：

1. 这个Tensor的Engine现在指向GMEM、SMEM还是线程私有register？
2. 每个mode是元素坐标、外层tile坐标、线程内value坐标，还是pipeline坐标？
3. 当前语句是在构造视图，还是在真正搬运/计算数据？

只要这三个问题有明确答案，像`tAgA(_,_,_,k_tile_next)`、`tXsA_p(_,_,k_block_next)`这样的表达式就不再是一串下划线，而是在精确地描述“哪一个线程、在哪一级tile、操作哪一段数据”。

# Reference

1. [CuTe GEMM Tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)
2. [CuTe Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html)
3. [CuTe MMA Atom](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html)
4. [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
