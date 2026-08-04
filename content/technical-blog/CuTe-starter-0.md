+++
date = '2026-05-28T16:29:15+08:00'
draft = false
title = 'CuTe学习0-CuTe简介'
description = '简单介绍CuTe抽象'
readingTimeText = '阅读此文大概需要38分钟'
tags = ['CuTe']
categories = ['Technical Blog']
+++
# CuTe

CuTe（CUDA Templates）是NVIDIA从CUTLASS 3.0版本开始推出的一个C++模板库。和传统意义上的矩阵库不同，CuTe并不是简单地提供一个`gemm()`接口，然后让用户快乐地把A、B、C丢进去等结果。它更像是一套用于描述GPU程序的元语言。我们用它描述数据如何排布，线程如何访问数据，数据如何在不同内存层级之间流动，最后如何映射到硬件矩阵乘法指令。

如果说CUDA C++给了我们操控GPU庞大数量运算单元的能力，那么CuTe试图回答的是另一个问题：如何把一个高性能kernel里那些重复、复杂、容易写错的索引和搬运逻辑，抽象成可以组合、可以检查、可以在编译期优化掉的组件。

博主认为，CuTe主要作用在如下三个方面，本文将按此顺序介绍CuTe的抽象。
- 提供数据从内存排布到逻辑索引的映射原语
- 提供数据在各内存层级之间的搬运原语
- 提供数据做矩阵乘法的计算原语

本文对CuTe中的概念均浅尝辄止，目的是提供一个总览，而不是具体的深度到某个函数的用法。具体的用法和逻辑会在后面的文章中进一步介绍。

# WHY CuTe?

在学习CuTe之前，我们先看一下一个简单的GEMM kernel。

``` C++
// C = A * B
for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = acc;
    }
}
```

从数学角度看，这段代码已经可以正确完成矩阵运算了。但是从GPU性能角度看，这段代码几乎什么都没说清楚。

1. `A[m * K + k]` 是行主序还是列主序？有没有padding？
2. 一个CTA算C矩阵的哪一块？一个warp算哪一块？一个thread算哪几个元素？
3. 全局内存读取是否合并访存？shared memory中是否会出现bank conflict？
4. 数据从global memory到shared memory，再到register，该由哪些线程搬？每个线程搬几个元素？
5. 最后的乘加是普通FMA，还是Tensor Core上的MMA指令？

手写CUDA kernel时，这些问题通常会变成一大堆`threadIdx.x`、`blockIdx.x`、魔法数字、位运算和手算偏移。它们看起来很硬核，实际也确实很硬核，但是也很脆弱。改一个tile size，可能十几个地方都要跟着改；换一个数据布局，可能整个kernel的索引逻辑都得重写；想从SIMT FMA换成Tensor Core MMA，则又要重新组织线程和寄存器片段。

``` C++
// instead of raw cuda... 
int offset = (blockIdx.x * 128 + threadIdx.x / 4) * lda + (threadIdx.x % 4) * 8;

// We can use CuTe!
auto layout1 = make_layout(make_shape(_2{}, _4{}), make_stride(_1{}, _2{}));
int offset = layout1(make_coord(1, 3));
```

CuTe要解决的核心问题，就是把这些索引和搬运从手写整数表达式中抽出来，用类型系统和模板元编程来描述。这就是CuTe和普通工具库最大的区别。CuTe并不只是帮你少写几行代码，它希望你把索引计算分解为逻辑形状和内存排布的组合。然后这些组合在编译期被实例化，最终生成没有抽象开销的CUDA代码。

# Layout & Tensor 

CuTe中最基础，也是最重要的抽象是`Layout`和`Tensor`。他们描述了数据的逻辑布局和数据映射，是张量操作的最底层逻辑。

## Layout

`Layout`描述的是从逻辑坐标到线性偏移的映射。更直白一点，Layout就是一个函数，输入逻辑坐标，输出物理地址偏移

``` C++
offset = layout(coord);
```

以一个二维行主序矩阵为例

``` plain text
matrix shape = (2, 3)

逻辑坐标:
(0,0) (0,1) (0,2)
(1,0) (1,1) (1,2)

线性内存:
0 1 2 3 4 5
```

它的映射关系如下。

``` plain text
layout(0, 0) = 0
layout(0, 1) = 1
layout(0, 2) = 2
layout(1, 0) = 3
layout(1, 1) = 4
layout(1, 2) = 5
```

如果我们把它写成一个公式，就是：

``` C++
offset = row * 3 + col;
```

在CuTe里，这个“`row * 3 + col`”不会被散落在kernel的每个角落，而是被封装成`Layout`。一个Layout由`Shape`和`Stride`组成。

``` C++
using namespace cute;

auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          make_stride(Int<3>{}, Int<1>{}));
```

这里`Shape=(2,3)`表示逻辑上这是一个2行3列的对象，`Stride=(3,1)`表示第0维走一步在线性内存中跳3个元素，第1维走一步跳1个元素。所以：

``` C++
layout(1, 2) == 1 * 3 + 2 * 1 == 5
```

这件事看起来非常普通，但是它是CuTe世界观的起点。因为一旦Layout变成一个独立对象，我们就可以组合它、切分它、重排它、把它映射到线程上，而不是到处手写下标，或者使用冗长的表达式。Layout为索引计算提供了清晰的逻辑。

### Shape & Stride

CuTe在layout上做的另外一件事情就是将逻辑形状与索引解耦，即 Shape & Stride。相同的数据可以被排列分割为不同形状，而保持数据在内存中的排列不变。让我么看一个例子。

``` C++
auto row_major = make_layout(make_shape(Int<2>{}, Int<3>{}),
                             make_stride(Int<3>{}, Int<1>{}));

auto col_major = make_layout(make_shape(Int<2>{}, Int<3>{}),
                             make_stride(Int<1>{}, Int<2>{}));
```

同样是`Shape=(2,3)`，行主序和列主序只是Stride不同。

对于不同的stride，相同的逻辑坐标会落在不同的内存地址上。
``` plain text
row_major(0, 1) = 1
col_major(0, 1) = 2
```

这说明CuTe的Layout不是在描述“有一个二维数组”，而是在描述“一个逻辑二维坐标如何落到一段线性内存上”。这就是为什么CuTe可以非常自然地表达行主序、列主序、分块布局甚至是嵌套布局。

## Tensor

如果说Layout是坐标到偏移的函数，那么Tensor就是把这个坐标映射到内存上。一个Tensor自己通常不拥有内存，它只是一个view。它知道底层数据从哪里开始，也知道逻辑坐标应该如何被翻译成线性偏移。

``` C++
float* ptr = some_data[6];

auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          make_stride(Int<3>{}, Int<1>{}));

auto tensor = make_tensor(make_gmem_ptr(ptr), layout);
```

于是我们可以用逻辑坐标访问数据。

``` C++
tensor(1, 2);
```

这比`ptr[row * lda + col]`多了一层抽象，但这层抽象在编译期大概率会被优化掉。更重要的是，它把“这块数据长什么样”和“这块数据在哪里”分开了。Layout提供这块数据的逻辑形状与排布信息。Pointer指定这块数据所在的内存位置(SMEM，GMEM，register,etc.)。Tensor则是Pointer + Layout，可以直接通过Tensor访问到内存。

Tensor可以被分割和重新组织，可以切给CTA，之后还可以继续切给warp，然后切给thread，我们都不需要手算原始矩阵的全局偏移，CuTe可以重新组织它的起点和Layout。
``` C++
auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K));

// 取出当前CTA负责的tile
auto tile_A = local_tile(gA,
                         make_shape(Int<128>{}, Int<64>{}),
                         make_coord(block_m, block_k));
```

Tensor的另一层抽象是解耦了数据搬运，在手写kernel的时候，我们总是需要在使用数据之前开辟内存再搬运数据。但是Tensor通过`make_gmem_ptr()`等函数为pointer打上tag。这个抽象为cute自动寻找数据搬运的最优算法提供了可能性，并且是在使用的时候再搬运，自动管理内存的生命周期。我们将在下一节里介绍。

# TiledCopy

理解Layout和Tensor之后，我们就可以进入第二个抽象：数据搬运。

GPU上的矩阵乘法并不是直接从global memory里读两个数，然后立刻乘一下这么简单。一个典型的SM80高性能GEMM kernel中的数据搬运大概会经历如下路径。

``` plain text
global memory -> shared memory -> register -> MMA -> register -> global memory
```

如果手写这部分代码，通常会看到很多这样的表达式。

``` C++
int lane = threadIdx.x % 32;
int row  = lane / 8;
int col  = lane % 8;

smem[row * smem_stride + col] = gmem[(global_row + row) * lda + global_col + col];
```

这当然能写，而且很多经典CUDA教程都是这么写的。但问题是，当tile形状、线程数量、每线程搬运元素数量、数据类型、目标架构发生变化时，这种代码会快速膨胀。

CuTe用`TiledCopy`来描述一组线程如何协作完成一次搬运。它的核心思想仍然是Layout：线程也可以有Layout，数据也可以有Layout，搬运就是把线程Layout映射到数据Layout上。为了防止代码过于混乱，作者在这里省略了很多变量的声明。具体的用法会在后面的文章中详细讲解。

``` C++
auto tiled_copy = make_tiled_copy(copy_atom, //单次搬运使用的指令
                                  thr_layout,//线程如何组织
                                  val_layout);//每个线程内部搬哪些元素
```

这里可以先不用纠结每个参数的具体写法，只要理解它们的语义。然后我们可以把一个Tensor按照这个`TiledCopy`切给每个线程。

``` C++
//按照tiled_copy描述的线程分工
auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

auto src = thr_copy.partition_S(gmem_tensor);
auto dst = thr_copy.partition_D(smem_tensor);
//从src tensor搬运到dst tensor
copy(tiled_copy, src, dst);
```

也就是说，我们不再直接写“第几个线程搬第几个元素”，而是先描述线程和数据的布局关系，再让CuTe帮我们展开成具体的每线程访问。当然，CuTe不是魔法棒。写出`TiledCopy`并不代表访存一定高效。它只是把访存模式变成了一个可组合、可检查、可替换的对象。真正的性能仍然取决于你选择的tile形状、线程布局、向量化宽度和目标硬件特性。

# TiledMMA

第三个抽象是计算，也就是`TiledMMA`。

在现代NVIDIA GPU上，高性能GEMM通常不会使用普通的逐元素FMA，而是使用Tensor Core提供的MMA指令。比如在某些架构上，一个MMA指令可以完成一个小矩阵块的乘加。但是硬件MMA指令并不是一个简单的函数调用。它有固定的数据类型、固定的矩阵形状、固定的寄存器分片方式和固定的线程协作方式。手写时我们需要关心：

1. 一个warp内哪些lane持有A片段？
2. 哪些lane持有B片段？
3. 累加结果C分布在哪些寄存器里？
4. 多个MMA atom如何拼成更大的CTA tile？

CuTe把最小硬件MMA指令抽象成`MMA_Atom`，再把多个atom按照线程和数值布局组合成`TiledMMA`。

``` C++
using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

auto tiled_mma = make_tiled_mma(MmaAtom{},
                                Layout<Shape<_2, _2, _1>>{});
```

这段代码大概表达的是：底层使用SM80架构上的某个计算 `16x8x16`tile的MMA指令，然后把这些atom平铺成更大的计算结构。

实际写kernel时，我们通常会从`TiledMMA`中取出当前线程负责的寄存器片段。

``` C++
auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

auto tCrA = thr_mma.partition_fragment_A(sA);
auto tCrB = thr_mma.partition_fragment_B(sB);
auto tCrC = thr_mma.partition_fragment_C(c_shape);

gemm(tiled_mma, tCrA, tCrB, tCrC);
```

这段代码和`TiledCopy`很像。我们不是手写每个lane拿哪些寄存器，而是先描述MMA的组织方式，再让CuTe根据这个组织方式切分Tensor和寄存器片段。

讲到这里，我们就拥有了CuTe中组合出GEMM的所有积木，CuTe中的GEMM kernel大概是如下这种结构。

``` C++
// 1. 用Layout/Tensor描述global memory中的A/B/C
auto gA = make_tensor(make_gmem_ptr(A), ...);
auto gB = make_tensor(make_gmem_ptr(B), ...);
auto gC = make_tensor(make_gmem_ptr(C), ...);

// 2. 切出当前CTA负责的tile
auto tile_A = local_tile(gA, ...);
auto tile_B = local_tile(gB, ...);
auto tile_C = local_tile(gC, ...);

// 3. 用TiledCopy把global memory搬到shared memory
copy(tiled_copy_g2s, tile_A, smem_A);
copy(tiled_copy_g2s, tile_B, smem_B);

// 4. 用TiledMMA在register中计算
gemm(tiled_mma, smem_A, smem_B, accum);

// 5. 把结果写回global memory
copy(tiled_copy_r2g, accum, tile_C);
```

这个框架看起来没有比手写CUDA简单，但是它的优势在于每个部分都变成了语义清晰的对象。我们可以换Layout，可以换Copy Atom，可以换MMA Atom，可以换tile shape，而不是在一堆整数下标中做考古。我们可以很简单地实验不同tile shape对性能的影响，而不用更改很多数值计算的逻辑。同时这样写出来的代码可读性极高，省去了阅读手写CUDA中无规则运算的困难。

# CuTe的编程思想

到这里我们可以总结一下CuTe的编程思想。CuTe表面上是一个C++模板库，实际上它强迫我们用一种更结构化的方式思考CUDA kernel。

# 学习建议

CuTe的学习曲线非常陡。它难的地方不是单个概念，而是多个概念叠在一起之后，类型系统会把你带到一个非常抽象的地方。博主会按如下顺序来组织整个教程。

1. Layout
2. layout algebra
3. Tensor
4. SM80 `TiledMMA`和`TiledCopy`。
5. SM80 Dense HGEMM WalkThrough
6. SM90 GMMA and TMA
7. SM100 tcgen05 and TMEM
8. SM100 Dense fp8 GEMM WalkThrough
9. SM90 Fused Multi-head Attention
10. SM100 Multi-Latent Attention
11. SM100 Deepseek Sparse Attention

# 写在最后

CuTe的本质是一套面向GPU高性能计算的编译期抽象。它把过去手写CUDA kernel中最容易混乱的部分，也就是索引、布局、切分、搬运和硬件MMA映射，统一放进了类型系统和模板系统里。

这带来的好处非常明显：代码可以组合，抽象可以优化，很多错误可以在编译期暴露，同一套逻辑也更容易适配不同tile shape和硬件指令。但代价同样明显：代码更抽象，类型更复杂，报错更难看，学习门槛也更高。不过学习CuTe仍然非常promising，CuTe可以说是现代算子开发皇冠上的明珠，基本所有基模厂都用CuTe来写sota算子(DeepGemm, KDA, etc.)，其能提供的粒度控制，自由度和性能不是其他任何一种语言能够比拟的。

# 最后的最后

有的读者可能会有些疑惑，使用CuTe可以彻底消灭运行时计算索引的开销吗？这其实是一个美丽的误会。我们必须直面硬件的物理现实：threadIdx.x 是一个纯粹的运行时变量（只有当 GPU 核函数在硬件核心上跑起来、具体的线程被调度时，它才知道自己是谁）。既然 threadIdx.x 是动态的，那么它衍生出来的 lane, row, col 也必然只能在运行时由 GPU 的算术逻辑单元（CUDA core）现场计算。任何黑魔法都无法在编译期未卜先知。CuTe真正消灭的是Layout的编译期解算内存几何拓扑和防止更复杂的非二的幂次映射发生优化退化。但是CUDA core是可以与Tensor Core 并行计算的，这就是为什么cublas等高度优化的线性代数库仍然能几乎达到计算卡标称的吞吐。

简单来说，CuTe消灭了基本上所有在编译期已知的数值计算。但是CuTe最大价值在于消灭程序员的心智开销，或者说增加代码的可读性。使用CuTe不一定能写出来性能最好的代码（当然，据作者经验，经常是比raw CUDA性能好很多），但是一定能写出来可读性很好的代码。

# 最后的最后的最后
有的读者还可能会疑惑，CUTLASS和CuTe到底是什么关系呢，听起来好像我用CuTe也能组装出超高性能的GEMM，为什么还需要CUTLASS呢？简单的说，CuTe像一套高性能的赛车配件，但是你要亲手设计组装。CUTLASS像一个赛车工厂，用CuTe提供的高性能配件，自动地组装出高性能的赛车。回到计算上来，CuTe 是一个专注于“张量描述与硬件映射”的底层库。而 CUTLASS 是建立在 CuTe 之上、开箱即用的高性能矩阵乘法框架。CUTLASS会处理复杂的流水线与异步控制，边缘处理与对齐等一个实际的高性能的GEMM的kernel要考虑的所有事情。如果你要写一个规整的，传统的gemm-like (batched-gemm，grouped-gemm，etc.) kernel，那么CUTLASS将是你的不二之选。但是如果是实现Flash Attention这种访存模式和gemm不同的kernel，CuTe将是最强而有力的工具。

