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

CuTe（CUDA Templates）是NVIDIA从CUTLASS 3.0版本开始推出的一个C++模板库。和传统意义上的矩阵库不同，CuTe并不是简单地提供一个`gemm()`接口，然后让用户快乐地把A、B、C丢进去等结果。它更像是一套用于描述GPU程序的元语言。我们用它描述数据如何排布，线程如何访问数据，数据如何在不同内存层级之间流动，最后如何映射到硬件矩阵乘法指令。

如果说CUDA C++给了我们操控GPU庞大数量运算单元的能力，那么CuTe试图回答的是另一个问题：如何把一个高性能kernel里那些重复、复杂、容易写错的索引和搬运逻辑，抽象成可以组合、可以检查、可以在编译期优化掉的组件。

博主认为，CuTe主要作用在如下三个方面，本文将按此顺序介绍CuTe的抽象。
- 提供数据从内存排布到逻辑索引的映射原语
- 提供数据在各内存层级之间的搬运原语
- 提供数据做矩阵乘法的计算原语

# WHY CuTe?

在学习CuTe之前，我们先看一下一个GEMM kernel到底在忙什么。

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

从数学角度看，这段代码已经把矩阵乘法说清楚了。但是从GPU性能角度看，这段代码几乎什么都没说清楚。

1. `A[m * K + k]` 是行主序还是列主序？有没有padding？
2. 一个CTA算C矩阵的哪一块？一个warp算哪一块？一个thread算哪几个元素？
3. 全局内存读取是否合并访存？shared memory中是否会出现bank conflict？
4. 数据从global memory到shared memory，再到register，该由哪些线程搬？每个线程搬几个元素？
5. 最后的乘加是普通FMA，还是Tensor Core上的MMA指令？

手写CUDA kernel时，这些问题通常会变成一大堆`threadIdx.x`、`blockIdx.x`、魔法数字、位运算和手算偏移。它们看起来很硬核，实际也确实很硬核，但是也很脆弱。改一个tile size，可能十几个地方都要跟着改；换一个数据布局，可能整个kernel的索引逻辑都得重写；想从SIMT FMA换成Tensor Core MMA，则又要重新组织线程和寄存器片段。

CuTe要解决的核心问题，就是把这些索引和搬运从手写整数表达式中抽出来，用类型系统和模板元编程来描述。

``` C++
// 手写CUDA中经常出现的逻辑
int offset = (blockIdx.x * 128 + threadIdx.x / 4) * lda + (threadIdx.x % 4) * 8;

// CuTe更希望你描述这件事
// 这是一个什么形状的tile？
// 它的逻辑坐标如何映射到内存？
// 哪些线程负责哪些坐标？
```

这就是CuTe和普通工具库最大的区别。CuTe并不只是帮你少写几行代码，它希望你把GPU程序写成“形状、布局、切分、搬运、计算”的组合。然后这些组合在编译期被实例化，最终生成没有抽象开销的CUDA代码。

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

这件事看起来非常普通，但是它是CuTe世界观的起点。因为一旦Layout变成一个独立对象，我们就可以组合它、切分它、重排它、把它映射到线程上，而不是到处手写下标。

### Shape & Stride

CuTe在layout上做的另外一件事情就是将逻辑形状与索引解耦，即Shape & Stride。这将数据在内存中的排布与代码中的逻辑形状解耦，相同的数据可以被排列分割为不同形状，而保持数据在内存中的排列不变。让我么看一个例子。

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

这说明CuTe的Layout不是在描述“有一个二维数组”，而是在描述“一个逻辑二维坐标如何落到一段线性内存上”。这就是为什么CuTe可以非常自然地表达行主序、列主序、分块布局、嵌套布局，甚至线程布局。

## Tensor

如果说Layout是坐标到偏移的函数，那么Tensor就是“指针 + Layout”。

``` C++
Tensor = data pointer + layout
```

一个Tensor自己通常不拥有内存，它只是一个view。它知道底层数据从哪里开始，也知道逻辑坐标应该如何被翻译成线性偏移。

``` C++
float* ptr = ...;

auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          make_stride(Int<3>{}, Int<1>{}));

auto tensor = make_tensor(make_gmem_ptr(ptr), layout);
```

于是我们可以用逻辑坐标访问数据。

``` C++
tensor(1, 2);     // 等价于 ptr[layout(1, 2)]
```

这比`ptr[row * lda + col]`多了一层抽象，但这层抽象在编译期大概率会被优化掉。更重要的是，它把“这块数据长什么样”和“这块数据在哪里”分开了。

Layout提供这块数据的逻辑形状与排布信息。Pointer指定这块数据所在的内存位置(SMEM，GMEM，register,etc.)。Tensor则是Pointer + Layout，可以直接通过Tensor访问到内存。

这个抽象在写复杂kernel时非常有用。比如一个CTA只处理大矩阵中的一个tile，我们不想重新创建数据，也不想复制数据，只想得到一个局部视图。

``` C++
auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K));

// 取出当前CTA负责的tile
auto tile_A = local_tile(gA,
                         make_shape(Int<128>{}, Int<64>{}),
                         make_coord(block_m, block_k));
```

从语义上讲，`tile_A`仍然是Tensor，只是它的起点和Layout已经被CuTe重新组织好了。之后不管是继续切给warp，还是切给thread，我们都不需要手算原始矩阵的全局偏移。

# TiledCopy

理解Layout和Tensor之后，我们就可以进入第二个抽象：数据搬运。

GPU上的矩阵乘法并不是直接从global memory里读两个数，然后立刻乘一下这么简单。一个典型的高性能kernel中的数据搬运大概会经历如下路径。

``` plain text
global memory -> shared memory -> register -> MMA -> register -> global memory
```

这里的每一步都很讲究。

1. global memory读取要尽量合并访存。
2. shared memory写入要尽量避免bank conflict。
3. 每个线程搬运的数据量要均衡。
4. 搬运的布局要方便后续MMA读取。

如果手写这部分代码，通常会看到很多这样的表达式。

``` C++
int lane = threadIdx.x % 32;
int row  = lane / 8;
int col  = lane % 8;

smem[row * smem_stride + col] = gmem[(global_row + row) * lda + global_col + col];
```

这当然能写，而且很多经典CUDA教程都是这么写的。但问题是，当tile形状、线程数量、每线程搬运元素数量、数据类型、目标架构发生变化时，这种代码会快速膨胀。

CuTe用`TiledCopy`来描述一组线程如何协作完成一次搬运。它的核心思想仍然是Layout：线程也可以有Layout，数据也可以有Layout，搬运就是把线程Layout映射到数据Layout上。

``` C++
auto tiled_copy = make_tiled_copy(copy_atom,
                                  thr_layout,
                                  val_layout);
```

这里可以先不用纠结每个参数的具体写法，只要理解它们的语义。

``` plain text
copy_atom:  单次搬运使用什么指令或向量宽度
thr_layout: 线程如何组织
val_layout: 每个线程内部搬哪些元素
```

然后我们可以把一个Tensor按照这个`TiledCopy`切给每个线程。

``` C++
auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

auto src = thr_copy.partition_S(gmem_tensor);
auto dst = thr_copy.partition_D(smem_tensor);

copy(tiled_copy, src, dst);
```

这段代码的读法应该是：

``` plain text
请按照tiled_copy描述的线程分工，
从src tensor搬运到dst tensor。
```

也就是说，我们不再直接写“第几个线程搬第几个元素”，而是先描述线程和数据的布局关系，再让CuTe帮我们展开成具体的每线程访问。对于高性能CUDA来说，这个抽象非常关键。因为访存模式不只是代码风格问题，它直接决定了kernel是不是能跑满带宽。

当然，CuTe不是魔法棒。写出`TiledCopy`并不代表访存一定高效。它只是把访存模式变成了一个可组合、可检查、可替换的对象。真正的性能仍然取决于你选择的tile形状、线程布局、向量化宽度和目标硬件特性。

# TiledMMA

第三个抽象是计算，也就是`TiledMMA`。

在现代NVIDIA GPU上，高性能GEMM通常不会使用普通的逐元素FMA，而是使用Tensor Core提供的MMA指令。比如在某些架构上，一个MMA指令可以完成一个小矩阵块的乘加。

``` plain text
D = A * B + C
```

但是硬件MMA指令并不是一个简单的函数调用。它有固定的数据类型、固定的矩阵形状、固定的寄存器分片方式和固定的线程协作方式。手写时我们需要关心：

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

这段代码大概表达的是：底层使用SM80架构上的某个`16x8x16` MMA指令，然后把这些atom平铺成更大的计算结构。

实际写kernel时，我们通常会从`TiledMMA`中取出当前线程负责的寄存器片段。

``` C++
auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

auto tCrA = thr_mma.partition_fragment_A(sA);
auto tCrB = thr_mma.partition_fragment_B(sB);
auto tCrC = thr_mma.partition_fragment_C(c_shape);

gemm(tiled_mma, tCrA, tCrB, tCrC);
```

这段代码和`TiledCopy`很像。我们不是手写每个lane拿哪些寄存器，而是先描述MMA的组织方式，再让CuTe根据这个组织方式切分Tensor和寄存器片段。

于是CuTe中的GEMM kernel大概会变成下面这种结构。

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

这个框架看起来没有比手写CUDA短多少，但是它的优势在于每个部分都变成了可替换的对象。我们可以换Layout，可以换Copy Atom，可以换MMA Atom，可以换tile shape，而不是在一堆整数下标中做考古。我们可以很简单地实验不同tile shape对性能的影响，而不用更改很多数值计算的逻辑。同时这样写出来的代码可读性极高，省去了阅读手写CUDA中无规则运算的困难。

# CuTe的编程思想

到这里我们可以总结一下CuTe的编程思想。CuTe表面上是一个C++模板库，实际上它强迫我们用一种更结构化的方式思考CUDA kernel。

## 形状优先

CuTe代码里经常会出现`Shape`、`Stride`、`Layout`、`Tile`这些词。它们看起来比普通下标麻烦，但背后有一个统一思想：先描述对象的形状，再描述如何把形状映射到内存、线程或寄存器。

``` C++
auto cta_shape = make_shape(Int<128>{}, Int<128>{}, Int<64>{});
auto mma_shape = make_shape(Int<64>{},  Int<64>{},  Int<32>{});
```

这种写法比直接写`128`、`64`更啰嗦，但它保留了语义。读代码时我们知道这不是一个随机常数，而是某个计算层级的tile shape。

## 编译期优先

CuTe大量使用`Int<128>{}`、`_0`、`_1`、`Shape<_128, _64>`这样的编译期整数。原因在上一篇已经讲过：只要形状、步长、分支和循环能在编译期确定，编译器就可以展开、折叠、消除抽象。

``` C++
auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                          make_stride(Int<8>{}, Int<1>{}));
```

这里的`4`、`8`不是普通运行时变量，而是类型的一部分。编译器知道这个Layout的完整结构，也就更容易生成没有循环和分支的代码。

当然，CuTe并不要求所有东西都是静态的。真实GEMM里`M`、`N`、`K`经常来自运行时参数。但高性能kernel内部最关键的tile size、线程布局和MMA形状，通常会尽量放到编译期。

## 组合优先

CuTe中很多复杂行为不是靠继承层级实现的，而是靠小对象组合出来的。

``` plain text
Shape + Stride -> Layout
Pointer + Layout -> Tensor
Copy Atom + Thread Layout + Value Layout -> TiledCopy
MMA Atom + Thread Layout + Value Layout -> TiledMMA
```

这也是CuTe一开始难读的原因。它不会在一个地方把所有事情都说完，而是把一个kernel拆成很多层抽象。每一层都只是一个小函数，但组合起来之后，类型会变得非常长，报错也会非常壮观。

所以读CuTe代码时，不要一上来就试图把所有模板参数都看懂。更实用的方法是先问三个问题。

1. 这个对象描述的是内存、线程，还是寄存器？
2. 这个对象的Shape是什么？
3. 它把哪个逻辑坐标映射到了哪个线性偏移？

能回答这三个问题，大多数CuTe代码就不会那么吓人了。


# 学习建议

CuTe的学习曲线非常陡。它难的地方不是单个概念，而是多个概念叠在一起之后，类型系统会把你带到一个非常抽象的地方。博主建议学习时按如下顺序来。

1. 先理解`Shape`、`Stride`、`Layout`，只在CPU侧打印和测试，不急着写CUDA kernel。
2. 再理解`Tensor`，重点看“指针 + Layout”如何形成view。
3. 然后看`local_tile`、`local_partition`、`composition`这些切分和组合操作。
4. 接着学习`TiledCopy`，用小tile观察每个线程搬哪些元素。
5. 最后再碰`TiledMMA`和完整GEMM kernel。

这里最重要的是，不要一开始就读完整CUTLASS GEMM。那种代码同时包含模板元编程、内存层级、流水线、MMA、架构特化和调度策略，信息密度太高。先用小例子把CuTe的基本对象彻底理解，再去读大kernel，会轻松很多。

# 写在最后

CuTe的本质是一套面向GPU高性能计算的编译期抽象。它把过去手写CUDA kernel中最容易混乱的部分，也就是索引、布局、切分、搬运和硬件MMA映射，统一放进了类型系统和模板系统里。

这带来的好处非常明显：代码可以组合，抽象可以优化，很多错误可以在编译期暴露，同一套逻辑也更容易适配不同tile shape和硬件指令。但代价同样明显：代码更抽象，类型更复杂，报错更难看，学习门槛也更高。

所以学习CuTe时，不要把它当成一个普通API库。更好的方式是把它当成一种写CUDA kernel的新语言。`Layout`是它的坐标系统，`Tensor`是它的数据视图，`TiledCopy`是它的搬运语义，`TiledMMA`是它的计算语义。理解了这四个词，才算真正走进CuTe的大门。

# 最后的最后
有的读者可能会有些疑惑，使用CuTe可以彻底消灭运行时计算索引的开销吗？这其实是一个美丽的误会。我们必须直面硬件的物理现实：threadIdx.x 是一个纯粹的运行时变量（只有当 GPU 核函数在硬件核心上跑起来、具体的线程被调度时，它才知道自己是谁）。既然 threadIdx.x 是动态的，那么它衍生出来的 lane, row, col 也必然只能在运行时由 GPU 的算术逻辑单元（CUDA core）现场计算。任何黑魔法都无法在编译期未卜先知。CuTe真正消灭的是Layout的编译期解算内存几何拓扑和防止更复杂的非二的幂次映射发生优化退化。

简单来说，CuTe消灭了基本上所有在编译期已知的数值计算。但是CuTe最大价值在于消灭程序员的心智开销，或者说增加代码的可读性。使用CuTe不一定能写出来性能最好的代码，但是一定能写出来可读性很好的代码。

