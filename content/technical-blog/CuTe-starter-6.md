+++
title = 'CuTe学习6-SM80 CuTe GEMM Walk Through'
date = '2026-07-13T17:29:09+08:00'
draft = true
description = '以一个SM80 HGEMM为例，详细介绍CuTe GEMM中的tiling、slicing'
readingTimeText = '阅读此文大概需要38分钟'
tags = ['CuTe','GEMM','CUDA']
categories = ['Technical Blog']
+++

# Overview
本文分析基于Ampere架构的dense HGEMM，A和B通过`cp.async`进行GMEM到SMEM的搬运，再通过`ldmatrix`从SMEM搬运到register，最后使用`SM80_16x8x16_F16F16F16F16_TN` MMA PTX计算。

从流程上来讲，使用CuTe编写的代码和Raw CUDA进行的HGEMM没有本质区别，也没有用到全新的优化方式，仍然是SM80优化的那一套。CuTe把曾经几乎没有可读性的，复杂的坐标计算重构为了CuTe的Tensor分块和切片，操作tensor的每个函数都有易于理解的语义。

本文会重点解释参与运算的矩阵是如何被TiledMMA和TiledCopy重新组织，分块和切片的。而不是讲解如何在SM80上面优化一个HGEMM，较为基础的优化部分本文会略过。本文代码基于CuTe官方教程，略有一些修改，完整代码在github上。

//TODO github连接
# Tiling Partitioning Slicing

## Tiling：按照tiler完成tiling，再选择一个tile

``` CPP
local_tile(Tensor    && tensor,
           Tiler const& tiler,   // tiler to apply
           Coord const& coord)   // coord to slice into "remainder"
{
  return inner_partition(static_cast<Tensor&&>(tensor),
                         tiler,
                         coord);
}
```

在代码中，我们经常使用local tile进行CTA切片，我们传入的是希望得到的Tile的形状，得到的是按照这个形状切出来的，再通过coord选择的一片。这在CTA的层级上非常常见，举个例子，用户传入的Problem Shape是128，128，64，我们就要处理一个128，128的C，我们选择32，32的tile去切分C，这时候用到的就是local tile，得到（（32，32），（4，4））的shape，在通过coord选出一片。

## Partitioning：按照参与者布局完成tiling，再选择一个参与者

``` CPP
local_partition(Tensor                     && tensor,
                Layout<LShape,LStride> const& tile,    // coord -> index
                Index                  const& index)   // index to slice for
{
  static_assert(is_integral<Index>::value);
  return outer_partition(static_cast<Tensor&&>(tensor),
                         product_each(shape(tile)),
                         tile.get_flat_coord(index));
}
```

在代码中，我们使用Partitioning分配线程任务，我们传入的是线程布局，得到的是工作平均按照布局分配，再按照线程序号选出来的一片。继续上面的例子，我们得到了一个（32，32）的CTA tile，这个CTA由32个线程参与，形状是（16，2），我们使用local——partition，切出（（2，8），（16，2））shape，这时一个线程的工作就是（2，8）。

## Slicing：切片


# 静态Tilesize、Copy和MMA 

```cpp
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int< 64>{};
auto cta_tiler = make_shape(bM, bN, bK);
auto bP = Int<3>{};
```
这里我们选择三级流水线，bM，bN，bK为128 128 64，

## Shared Memory Layout

```cpp
auto swizzle_atom = composition(
    Swizzle<3,3,3>{},
    Layout<Shape <_8,Shape <_8, _8>>,
           Stride<_8,Stride<_1,_64>>>{});

auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
auto sC = make_layout(make_shape(bM, bN));
```

两者合计98304 bytes，也就是96 KiB动态shared memory。host侧的`sC`只提供静态`(128,128)` Layout，让kernel模板检查它与CTA的M/N shape一致；代码没有为C分配shared memory，真正的C视图是后面的`gC`。

## GMem到SMem的TiledCopy

```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16,_8>,Stride<_8,_1>>{},
    Layout<Shape< _1,_8>>{});
```

`copyB`使用相同配置。三个参数分别表示：

1. Copy Atom：每次执行128-bit的`cp.async`，即8个half；
2. Thread Layout：128个线程排成逻辑`16 x 8`；
3. Value Layout：每线程负责逻辑`1 x 8`个value。

Thread Layout和Value Layout共同覆盖一个`16 x 64`的copy tile：

```text
M/N方向：16 threads * 1 value  = 16
K方向：   8 threads * 8 values = 64
```

一次协同copy搬运：

```text
128 threads * 16 bytes = 2048 bytes
16 * 64 half           = 2048 bytes
```

CTA的A tile是`128 x 64`，所以这个copy tile还要沿M方向重复8次。B同理沿N方向重复8次。

## TiledMMA

```cpp
TiledMMA mmaC = make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},
    Layout<Shape<_2,_2>>{},
    Tile<_32,_32,_16>{});
```

最底层MMA Atom的shape是`16 x 8 x 16`，由一个32线程warp执行。`Layout<Shape<_2,_2>>`把warp在M/N方向各重复2次，因此共有4个warp，也就是128个线程。

只看这个warp阵列，它覆盖的是：

```text
(16 * 2) x (8 * 2) x 16 = 32 x 16 x 16
```

第三个参数`Tile<_32,_32,_16>`要求TiledMMA的逻辑tile为`32 x 32 x 16`。N方向还差一个2，因此CuTe在每个线程的value空间中加入`1 x 2 x 1`的value group。换句话说，4个warp构成线程组，而这个线程组还会在N方向处理两个MMA Atom位置，最终得到`32 x 32 x 16`。

这个细节非常重要：`MMAThrLayout`描述“安排多少组MMA线程”，`Tile`还可以描述“每组线程处理多少组value”。两者共同决定TiledMMA，而不是只把MMA Atom shape与warp数相乘。

这也解释了kernel的launch配置：

```cpp
dim3 dimBlock(size(mmaC));  // 128 threads
```

注意CTA tile和TiledMMA不是同一层：CTA负责`128 x 128 x 64`，一次TiledMMA覆盖`32 x 32 x 16`。`TiledMMA`定义线程和MMA Atom如何协作，CTA tile则定义这个线程块最终要遍历多大的问题区域。

# 从完整矩阵切出CTA Tile

进入`gemm_device`之后，首先为A/B/C建立完整的global memory Tensor视图：

```cpp
Tensor mA = make_tensor(make_gmem_ptr(A),
                        select<0,2>(shape_MNK), dA); // (M,K)
Tensor mB = make_tensor(make_gmem_ptr(B),
                        select<1,2>(shape_MNK), dB); // (N,K)
Tensor mC = make_tensor(make_gmem_ptr(C),
                        select<0,1>(shape_MNK), dC); // (M,N)
```

host侧传入的stride是：

```cpp
dA = (ldA, 1); // offset_A(m,k) = m * ldA + k
dB = (ldB, 1); // offset_B(n,k) = n * ldB + k
dC = (1, ldC); // offset_C(m,n) = m + n * ldC
```

所以A/B在逻辑K方向连续，便于128-bit加载；C是column-major，M方向连续。`m`前缀表示完整matrix视图，`g`前缀表示从中切出的global memory tile，它们的Engine都仍然指向global memory。

## `local_tile`中的Step

```cpp
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)

Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                       Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                       Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord,
                       Step<_1,_1, X>{}); // (BLK_M,BLK_N)
```

`cta_tiler`有M/N/K三个mode，但A、B、C都只有两个mode。`Step`描述目标Tensor的mode使用tiler中的哪些mode：

| Tensor | 使用的CTA mode | 跳过的mode | 结果 |
|---|---|---|---|
| A(M,K) | M、K | N | `(128,64,k_tile)` |
| B(N,K) | N、K | M | `(128,64,k_tile)` |
| C(M,N) | M、N | K | `(128,128)` |

其中`X`表示该tiler mode不参与这个Tensor。A与N无关，所以同一个`blockIdx.x`下，不同`blockIdx.y`的CTA会读取相同的A区域；B则相反。这正是GEMM的正常数据复用关系。

`cta_coord`中的`blockIdx.x`和`blockIdx.y`固定M/N tile坐标，`_`保留K tile坐标。因此：

```text
gA(m_inner, k_inner, k_tile)
gB(n_inner, k_inner, k_tile)
```

可以遍历完整K方向，而`gC`不保留K tile mode，因为K是归约维度，CTA只产生一块`128 x 128`输出。

把`gA`的地址关系写开就是：

```text
gA(i, j, q) = mA(blockIdx.x * 128 + i,
                  q * 64 + j)
```

其中`i in [0,128)`、`j in [0,64)`、`q`是K tile编号。`local_tile`返回的是视图，并没有把数据从global memory搬到任何地方。

# 用ThrCopy把CTA Tile分给线程

建立完整的shared memory Tensor后：

```cpp
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);
```

代码从`TiledCopy`取出当前线程的slice：

```cpp
ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)
```

这里有两个不同层次的“slice”：

1. `get_slice(threadIdx.x)`固定TiledCopy中的thread坐标，得到描述当前线程映射规则的`ThrCopy`对象；
2. `partition_S/D`把这套线程映射应用到具体Tensor，得到当前线程负责的source/destination Tensor视图。

`partition_S`与`partition_D`不是普通的内存空间判断。S表示copy的source，D表示copy的destination。它们保证两侧保留下来的value和外层重复mode可以一一对应。

对于当前配置，实际类型打印为`tAgA: ((_8,_1),_8,_1,k)`、`tAsA: ((_8,_1),_8,_1,(_1,_3))`。保留层次结构有利于CuTe继续做Layout运算；只看每个mode的size，可以具体理解为：

```text
tAgA: (8, 8, 1, K_TILE_COUNT)
tAsA: (8, 8, 1, 3)
        ^  ^  ^  ^
        |  |  |  +-- global K tile / SMEM pipe
        |  |  +----- copy tile沿K方向的外层重复，当前为1
        |  +-------- copy tile沿M方向重复8次
        +----------- 每次128-bit copy中的8个half value
```

每个线程一次搬8个half；128个线程一次覆盖`16 x 64`；沿M重复8次后覆盖完整`128 x 64`的A tile。B侧把`CPY_M`换成`CPY_N`，推导完全相同。

于是下面这条语句：

```cpp
copy(copy_a,
     tAgA(_,_,_,k_tile_next),
     tAsA(_,_,_,k_pipe));
```

做了两次Tensor slicing：

```text
tAgA(_,_,_,k_tile_next) 固定global K tile，保留当前线程的所有copy value
tAsA(_,_,_,k_pipe)      固定SMEM pipeline stage，保留对应的所有目标位置
```

两侧得到相同的逻辑shape`(CPY,CPY_M,CPY_K)`，所以一次`copy`就能表达当前线程从一个global K tile到一个shared memory stage的全部搬运。

# 用ThrMMA切出线程的计算视图

Copy的线程映射只负责搬运，MMA有另一套Thread-Value映射：

```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC); // (MMA,MMA_M,MMA_N)

Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
Tensor tCrC = thr_mma.make_fragment_C(tCgC);
```

`mma.get_slice(threadIdx.x)`先确定当前线程属于`2 x 2` warp阵列中的哪个warp，以及它是warp中的哪个lane。`partition_A/B/C`再根据MMA Atom的Thread-Value Layout切出这个lane应当看到的元素。

`sA(_,_,0)`和`sB(_,_,0)`中的`0`只是用第一个pipe的二维shape来创建fragment。`partition_fragment_A/B`在这里创建的是线程私有register fragment，之后会从任意pipe向其中填数据，并不意味着A/B永远读取pipe 0。

## 当前配置下的MMA fragment shape

一个`m16n8k16` Atom中，每个lane逻辑上持有：

```text
A: 16 * 16 / 32 = 8个half
B:  8 * 16 / 32 = 4个half
C: 16 *  8 / 32 = 4个half
```

实际打印当前CuTe类型，shape为：

```text
tCrA: ((2,2,2),4,(2,2)) // 展平size为(8,4,4)
tCrB: ((2,2),8,(2,2))   // 展平size为(4,8,4)
tCrC: ((2,2),4,8)       // 展平size为(4,4,8)
tCgC: ((2,2),4,8)       // 与tCrC相同的输出坐标，但指向GMEM
```

层次化的`(2,2,2)`和`(2,2)`保留了MMA Atom内部寄存器布局的结构。讨论每个mode的元素数量时，可以暂时把它们分别看成8和4。

第一个mode不是warp编号，而是一个lane在单个MMA Atom中持有的value mode。warp身份已经在`get_slice(threadIdx.x)`时固定了。对于C，每线程最终负责`4 * 4 * 8 = 128`个输出，恰好满足：

```text
128 threads * 128 outputs/thread = 128 * 128 CTA outputs
```

`MMA_M/MMA_N/MMA_K`也不是矩阵元素坐标。它们是完成线程partition后保留下来的MMA重复坐标，其中还可能包含前面提到的value group。这里`MMA_M=4`，`MMA_N=8`而不是4，正是因为`Tile<_32,_32,_16>`在N方向加入的2倍value group被合并到了`MMA_N`。例如`tCrA(_,2,1)`表示当前线程在第2个M重复位置、第1个K block上需要的全部A operand value。

几个静态检查正是在验证这些投影能够相乘：

```cpp
size<1>(tCgC) == size<1>(tCrA); // C的M重复 == A的M重复
size<2>(tCgC) == size<1>(tCrB); // C的N重复 == B的N重复
```

需要特别注意，`tCrA`不是“C中来自A的部分”。变量名最后的`rA`明确表示register中的A operand；`tCrC`才是register中的C accumulator。

# 从SMEM到Register：为什么需要Retile

现在出现了两套不同的线程映射：

```text
MMA映射：规定每个lane最终必须在哪些register位置拿到A/B
LDSM映射：规定每个lane用ldmatrix从SMEM加载哪些地址
```

两者处理的是同一批逻辑元素，但最方便的mode组织方式并不相同。代码用`make_tiled_copy_A/B`把LDSM Copy Atom适配到既有TiledMMA：

```cpp
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);

Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
```

B同理。这里使用的Atom是：

```cpp
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
```

它对应`ldmatrix.x4`风格的shared-to-register加载。

`partition_S(sA)`用LDSM的Thread-Value Layout切shared memory，结果为：

```text
tXsA: (CPY,MMA_M,MMA_K,PIPE)
tXsB: (CPY,MMA_N,MMA_K,PIPE)
```

例如当前A侧的实际shape是`tXsA: ((_8,_1),4,(2,2),(_1,_3))`，各mode展平后的size为`(8,4,4,3)`。

`retile_D(tCrA)`则很关键：它不会新建一份register，也不会移动数据，而是为`tCrA`的同一Engine建立一个适合Copy destination的Layout视图：

```text
tCrA -- MMA眼中的register布局
tXrA -- LDSM Copy眼中的同一批register
```

因此：

```cpp
copy(s2r_atom_a,
     tXsA_p(_,_,k_block_next),
     tXrA(_,_,k_block_next));
```

copy按照`tX*`视图把数据写入register；随后的`gemm`按照`tC*`视图读取相同register。Retile的本质是协调两套Layout，而不是一次额外的数据重排。

变量名也体现了这种关系：

| 名称 | 含义 |
|---|---|
| `tAgA` | copy-A线程视图中的global A |
| `tAsA` | copy-A线程视图中的shared A |
| `tCgC` | MMA线程视图中的global C |
| `tCrA/B/C` | MMA线程视图中的register A/B/C |
| `tXsA/B` | S2R Copy线程视图中的shared A/B |
| `tXrA/B` | S2R Copy线程视图中的register A/B |

这些字母是CUTLASS/CuTe代码约定，不是C++类型系统的一部分。最可靠的阅读方法仍然是同时看：谁创建了这个Tensor、Engine指向哪种内存、shape注释是什么。

# Pipeline中的动态Slicing

前面建立的Tensor包含所有K tile、所有SMEM pipe和所有register K block。主循环每一时刻只操作其中一个切片。

## Shared Memory三阶段流水

```cpp
auto K_PIPE_MAX = size<3>(tAsA); // 3
int k_tile_count = size<3>(tAgA);
int k_tile_next = 0;
```

预取阶段先填`K_PIPE_MAX - 1 = 2`个stage：

```cpp
for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
  copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
  copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
  cp_async_fence();
  ...
}
```

`cp_async_fence()`提交一个async copy group，但不保证数据已经可见。消费者在使用相应stage前还需要`cp_async_wait`和`__syncthreads()`：前者等待异步事务，后者保证整个CTA中的生产者/消费者线程都到达正确的阶段。

## 把PIPE mode切掉

```cpp
Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);
```

`tXsA`是四维Tensor，最后一维是PIPE。固定`smem_pipe_read`后，`tXsA_p`成为三维视图：

```text
(CPY,MMA_M,MMA_K,PIPE) --slice PIPE--> (CPY,MMA_M,MMA_K)
```

这里只复制了一个很小的Tensor描述对象，没有把整个stage复制到新内存。之后pipe索引变化时，代码重新赋值`tXsA_p`，让它指向新的stage。

## 把MMA_K mode切成register K block

```cpp
auto K_BLOCK_MAX = size<2>(tCrA); // 64 / 16 = 4
```

一个SMEM K tile含64个K元素，一次TiledMMA消费16个，所以有4个`k_block`。进入主循环前先把block 0加载到register：

```cpp
copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}),
                   tXrA  (_,_,Int<0>{}));
```

这里同时固定source和destination的`MMA_K = 0`，保留`(CPY,MMA_M)`。主循环中采用环形寄存器预取：

```cpp
auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;

copy(s2r_atom_a, tXsA_p(_,_,k_block_next),
                   tXrA  (_,_,k_block_next));
copy(s2r_atom_b, tXsB_p(_,_,k_block_next),
                   tXrB  (_,_,k_block_next));

gemm(mma, tCrA(_,_,k_block),
          tCrB(_,_,k_block),
          tCrC);
```

同一次循环迭代中：

```text
S2R copy 预取 k_block_next
MMA      计算 k_block
```

当`k_block == 3`时，`k_block_next`回到0。代码先切换`tXsA_p/tXsB_p`到下一个SMEM pipe，等待该pipe就绪，再把下一个pipe的block 0覆盖到register block 0。此时旧block 0已经在当前pipe更早的迭代中使用完毕，因此可以安全复用。

## 三层坐标不要混淆

这段循环同时维护三种K坐标：

| 坐标 | 粒度 | 所在Tensor | 作用 |
|---|---:|---|---|
| `k_tile_next` | 64 | `tAgA/tBgB`最后一维 | 选择global K tile |
| `smem_pipe_read/write` | 1个K tile | `tAsA/tXsA`最后一维 | 选择SMEM物理stage |
| `k_block` | 16 | `tCrA/tXrA`第三维 | 选择当前MMA K block |

`k_tile_next`是问题空间坐标；`smem_pipe_*`是循环缓冲区坐标；`k_block`是一个CTA K tile内部的计算坐标。三者恰好都与K有关，但语义完全不同。

主循环实现的重叠关系可以写成：

```text
GMEM[k_tile_next]  --cp.async--> SMEM[smem_pipe_write]
SMEM[read pipe]    --ldmatrix--> RMEM[k_block_next]
RMEM[k_block]      --mma.sync--> accumulator
```

这就是分层tiling带来的直接好处：每一级都有自己的逻辑坐标和buffer生命周期，代码可以只切出当前时刻需要的视图，让三种硬件操作并行推进。

# MMA与Epilogue

主循环中的：

```cpp
gemm(mma,
     tCrA(_,_,k_block),
     tCrB(_,_,k_block),
     tCrC);
```

固定`MMA_K`后：

```text
A: (MMA_A,MMA_M)
B: (MMA_B,MMA_N)
C: (MMA_C,MMA_M,MMA_N)
```

`cute::gemm`根据Tensor rank继续dispatch，在M/N重复mode上展开，最终对当前线程的value fragment调用`m16n8k16` MMA Atom。K方向的累加由外层四个`k_block`以及所有global K tile共同完成；`tCrC`在整个过程中一直驻留在线程私有register中。

计算开始前：

```cpp
clear(tCrC);
```

所有K tile完成后：

```cpp
axpby(alpha, tCrC, beta, tCgC);
```

`tCrC`和`tCgC`具有相同的逻辑输出shape，但Engine分别位于register和global memory。`axpby`逐元素完成：

```text
tCgC = alpha * tCrC + beta * tCgC
```

这个例子没有shared memory epilogue。MMA的线程布局已经保证每个输出元素只由一个lane负责，因此每个线程可以直接把自己的fragment写回C。

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

`local_tile`、`get_slice`、`partition_*`、Tensor的`operator()`切片以及`retile_D`都只是在构造或变换视图。区分“视图变换”和“真实copy”是阅读CuTe代码最重要的习惯之一。

# 当前实现的边界条件

host侧grid使用了向上取整：

```cpp
dim3 dimGrid(size(ceil_div(M, bM)),
             size(ceil_div(N, bN)));
```

但kernel的copy和epilogue没有predicate Tensor，也没有对越界元素做条件判断。因此当前实现要求：

```text
M是128的整数倍
N是128的整数倍
K是64的整数倍
```

默认的`M=5120, N=5120, K=4096`满足这些条件。若要支持任意shape，需要为GMEM load和C store构造identity/predicate Tensor，或对边缘tile采用clear-fill与条件写回。`ceil_div`只保证能启动足够多的CTA，本身不会自动阻止越界访问。

此外，这个kernel使用`SM80_16x8x16_F16F16F16F16_TN`，累加器和输出都是FP16；验证时cuBLAS也显式选择了`CUBLAS_COMPUTE_16F`。这不是常见的FP32 accumulate配置，比较数值误差或性能时需要保持计算类型一致。

# 总结

这段GEMM可以看成四次坐标分解：

```text
Problem MNK
  -> CTA tile (128,128,64) x CTA coordinate
  -> Thread-Value partition x copy/MMA outer repeat
  -> SMEM PIPE stage
  -> Register MMA_K block (16)
```

Tiling负责建立这些层级，partitioning负责把层级映射到线程，slicing负责在某一时刻固定CTA、thread、pipe或K block坐标。CuTe没有让程序员手算地址，而是把每一步都表达成Tensor Layout变换，再让`copy`和`gemm`消费shape兼容的视图。

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
