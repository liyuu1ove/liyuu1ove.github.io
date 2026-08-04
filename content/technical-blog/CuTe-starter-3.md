+++
title = 'CuTe学习3-Tensor'
date = '2026-07-03T17:00:38+08:00'
draft = false
description = '介绍CuTe Tensor模板类'
readingTimeText = '阅读此文大概需要27分钟'
tags = ['CuTe','Tensor']
categories = ['Technical Blog']
+++

Tensor 是 Layout + Engine。Layout 负责把逻辑坐标映射成线性 offset，Engine 负责说明这段线性数据从哪里来、如何访问。我们接下来会先介绍 Engine，再介绍 Tensor。

# Engine
一个 Engine 对象的核心职责是管理数据的存储和访问机制，和CPP容器有些类似。传统 C++ Iterator 往往强调 `++it` 这样的顺序步进。但在 GPU kernel 里，我们更常见的访问模式时随机访问，CuTe 的 Engine 就服务于这种随机访问的模式。

Engine的接口是std::array简化版，只暴露迭代器及其相关类型。

```CPP
using iterator     =  // 迭代器类型
using value_type   =  // 迭代器读写的值类型
using reference    =  // 迭代器解引用后的引用类型
iterator begin()      // 起始迭代器
```

Engine 大体分为两类：

1. `ArrayEngine`：Tensor 自己持有数据。
2. `ViewEngine` / `ConstViewEngine`：Tensor 不持有数据，只保存一个指向已有内存的迭代器或指针。

一般来说，我们很少手写 Engine 类型。创建 Tensor 时，CuTe 会根据 `make_tensor` 的参数自动选择合适的 Engine。

## ArrayEngine
`ArrayEngine` 是持有数据的 Engine。它把数据作为 Tensor 对象自身的一部分保存下来，语义上接近 `std::array<T, N>`。根据元素类型是否按整字节存储，`ArrayEngine` 会选择不同的底层存储：

1. `array_aligned<T, N>`：用于 `sizeof_bits<T>` 是 8 的倍数的类型，例如 `float`、`half`、`int8_t` 等。
2. `array_subbyte<T, N>`：用于 sub-byte 类型，例如 FP4、INT4 这类不足8-bit/1-byte的元素。

```CPP
template <class T, size_t N>
struct ArrayEngine
{
  using Storage = typename conditional<(sizeof_bits<T>::value % 8 == 0),
                                       array_aligned<T,N>,
                                       array_subbyte<T,N>>::type;
  using iterator     = typename Storage::iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  Storage storage_;

  CUTE_HOST_DEVICE constexpr auto begin() const { return storage_.begin(); }
  CUTE_HOST_DEVICE constexpr auto begin()       { return storage_.begin(); }
};
```
## ViewEngine & ConstViewEngine
这两种是不持有数据的 Engine，类似于 `std::span` 或 `std::string_view`。它们只保存一个迭代器，不管理迭代器背后的内存生命周期。

```CPP
template <class Iterator>
struct ViewEngine
{
  using iterator     = Iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;
  iterator storage_;

  CUTE_HOST_DEVICE constexpr iterator const& begin() const { return storage_; }
  CUTE_HOST_DEVICE constexpr iterator      & begin()       { return storage_; }//ConstViewEngine没有这个方法
};
```

`ViewEngine` 对应可写视图，`ConstViewEngine` 对应只读视图。复制这类 Tensor 只会复制指针/迭代器本身，不会复制底层元素；析构时也不会释放底层内存。底层内存的生命周期由外部负责。

## Tagged Iterator
构造 nonowning Tensor 时，最常见的迭代器是 tagged pointer。当然，裸指针本身也是随机访问迭代器。但裸指针不携带内存空间信息，CuTe 就无法根据S/D的物理内存选择更特化的算法。

Tagged Iterator 的 tag 用于指示内存所在的物理空间，例如：

1. `gmem`：Global Memory。
2. `smem`：Shared Memory。
3. `rmem`：Register Memory，通常指线程私有 fragment。
4. `tmem`：Tensor Memory，Blackwell 引入的片上 Tensor Core 累加存储，用来降低寄存器压力。

其中 `gmem`、`smem`、`rmem` 是最常见的几类。`tmem` 的物理内存类型比较特殊，后面介绍 Blackwell 相关内容时再展开。

创建 tagged pointer 的方法是在已有指针上调用工厂函数 `make_gmem_ptr(ptr)`等。标记内存空间后，CuTe 的算法可以根据 Tensor 的物理内存类型选择更合适的实现。例如 `gmem -> smem` 的搬运在合适条件下可以使用 TMA 或其他异步 copy 路径。

# Tensor
CuTe Tensor 的核心价值是把物理内存和坐标映射统一成一个可组合对象。后续的 tiling、slicing、partitioning等操作，本质上都是在操作 Tensor 的 Layout。有了Tensor之后，写 kernel 时我们就不用手算线程访问哪个地址，而是先构造一组 Tensor，然后通过 CuTe 的布局代数把全局内存、共享内存、寄存器 fragment、MMA operand 之间的关系表达出来。

Tensor 分为 owning 和 nonowning 两类。

1. Owning Tensor 使用 ArrayEngine，Tensor 自己持有元素。复制 Tensor 会复制元素，析构 Tensor 时对象内的数组生命周期结束。
2. Nonowning Tensor 使用 ViewEngine，Tensor 只保存指针/迭代器。复制 Tensor 不会复制元素，析构 Tensor 也不会释放底层内存。

## Tensor 创建
创建 Tensor 最常用的是 `make_tensor`。CuTe 会通过参数模板判断该创建 owning Tensor 还是 nonowning Tensor：

1. `make_tensor<T>(layout_args...)`：没有传入指针/迭代器，创建 owning Tensor，内部使用 `ArrayEngine<T, N>`。构建时必须指定数据类型，因为内存分配时必须知道类型大小。
2. `make_tensor(ptr_or_iter, layout_args...)`：传入了指针/迭代器，创建 nonowning Tensor，内部使用 `ViewEngine` 或 `ConstViewEngine`。

### Nonowning Tensor
下面是一些创建 nonowning Tensor 的例子。它们都只是已有内存的视图。

```CPP
float* A = ...;

// 未标记的裸指针
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // Layout
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // Shape

// Global Memory
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{},16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape (      8  ,Int<16>{}),
                                                   make_stride(Int<16>{},Int< 1>{}));

// Shared Memory
Layout smem_layout = make_layout(make_shape(Int<4>{},Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value];
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
```

对这些 Tensor 调用 `print`，会看到类似输出：

```CPP
tensor_8     : ptr[32b](0x7f42efc00000) o _8:_1
tensor_8s    : ptr[32b](0x7f42efc00000) o _8:_1

gmem_8sx16d  : gmem_ptr[32b](0x7f42efc00000) o (_8,16):(_1,_8)
gmem_8dx16s  : gmem_ptr[32b](0x7f42efc00000) o (8,_16):(_16,_1)

smem_4x8_col : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_1,_4)
smem_4x8_row : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_8,_1)
```

在上面的例子中，我们可以看到前四个基于A构建的Tensor全部指向A，这说明他们共享同一块内存，但是逻辑形状不同，逻辑坐标到内存的映射不同。
### Owning Tensor

Tensor 也可以自己持有一段数组内存。owning Tensor 通过 `make_tensor<T>` 创建，其中 `T` 是元素类型，后面的参数是一个 Layout，或者是可以构造 Layout 的 shape/stride 参数。owning Tensor 的底层存储类似 `std::array<T, N>`，不会在 Tensor 内部做动态内存分配，在创建时一次性全部分配，因此 owning Tensor 必须使用静态 Layout。

下面是一些创建 owning Tensor 的例子：

```CPP
// 寄存器内存，只支持静态 Layout
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4,_8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4,_8>{},
                                         LayoutRight{});
Tensor rmem_4x8_pad = make_tensor<float>(Shape <_4, _8>{},
                                         Stride<_32,_2>{});
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_pad);
```

这里容易产生一个问题：代码里明明没有 `register` tagged pointer，为什么文档说这是寄存器内存？

关键在 `make_tensor<T>(...)` 这个重载。它没有传入指针/迭代器，所以会走 owning Tensor 分支，CuTe 会选择ArrayEngine作为底层数据储存，自己分配一段内存，也就是说，数据并不是来自外部内存，而是作为owning Tensor对象自身的一段静态数组保存下来。这个对象通常是线程私有的局部变量。在CUDA kernel语境下，这类线程私有的小型静态数组/fragment通常分配在register上。

不过要注意，这里说通常分配在register上，并不等于编译器强制每个元素物理上永远放在寄存器里。最终是否完全驻留寄存器还取决于编译器寄存器分配、索引方式、数组大小和寄存器压力。如果 fragment 太大或访问方式让编译器难以标量化，可能会 spill 到 global memory。CuTe 在这里保证的是owning Tensor 不绑定 gmem/smem pointer，不做动态分配，而是静态大小的线程私有对象内存储。

## Tensor 操作

Tensor 的大多数操作都围绕 Layout 展开。访问元素时使用 Layout 计算 offset，切片时把部分坐标固化成新的起始指针和子 Layout，分块和分区时则把 Layout 代数应用到 Tensor 上，得到更适合线程块、warp、线程或 MMA 指令使用的视图。

### Accessing Tensor

访问 Tensor 使用 `operator()`或`operator[]`。我们传入的是一个coord，得到Tensor内元素的一个引用。

```CPP
Tensor A = make_tensor(ptr, make_shape(8, 24));

auto x = A(make_coord(2, 3));      // 访问坐标 (2, 3) 对应的元素
A(make_coord(2, 3)) = 1.0f;       // 如果 Engine 可写，也可以直接写回
```
### Slicing Tensor

切片仍然使用 `operator()`。
`_` 的含义类似 Matlab 或 Python 里的 `:`：保留这一维所有元素

```CPP
Tensor A = make_tensor(ptr, make_shape(_8{}, _24{}));  // (_8,_24)

Tensor row = A(_2{}, _);    // 第0维取第2个元素，第1维取所有元素，shape 为 (_24)
Tensor col = A(_, _3{});    // 第0维取所有元素，第1维取第三个元素，shape 为 (_8)
```

切片是取原Tensor的一个nonowning tensor。因此切片不会复制数据。
### Tiling Tensor

CuTe 可以把很多 Layout 代数操作应用到 Tensor 上，例如：

```CPP
composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
zipped_divide(Tensor, Tiler)
tiled_divide(Tensor, Tiler)
flat_divide(Tensor, Tiler)
```

这些操作的作用是把一个 Tensor 重新组织成“tile 维度 + 剩余维度”的结构，方便后续按 CTA、warp、线程或 MMA atom 分配数据。

例如：

```CPP
Tensor A = make_tensor(ptr, make_shape(8, 24));  // (8,24)
auto tiler = Shape<_4, _8>{};                    // 4x8 tile

Tensor tiled_A = zipped_divide(A, tiler);        // ((_4,_8),(2,3))
```

当然，我们一般不会直接使用底层的divide函数，通常使用包装好的，具有语义的高阶API。

需要注意，Tensor 通常不做 `_product` 类操作，因为那可能增大 codomain size，让 Tensor 访问到原本边界之外的地址。

### Partitioning Tensor

分区通常是“分块或 composition 之后再切片”。CuTe 常见的分区模式有三种：inner partitioning、outer partitioning 和 Thread-Value partitioning。

#### Inner Partitioning

Inner partitioning 是在tensor上切出tiler这样的小块，得到的tile形状就是tiler的形状，常用于“每个 CTA/线程块拿到一个数据 tile”。

```CPP
Tensor A = make_tensor(ptr, make_shape(8, 24));  // (8,24)
auto tiler = Shape<_4, _8>{};                    // (_4,_8)

Tensor cta_A = inner_partition(A,tiler,make_coord(blockIdx.x, blockIdx.y));  // (_4,_8)
```

CuTe 里常用的 `local_tile(Tensor, Tiler, Coord)` 就是 inner partitioning 的封装。

#### Outer Partitioning

Outer partitioning 是把tensor平均划分为tiler的形状，得到的tile形状通常与tiler不同，常用于“每个线程负责每个 tile 中的某个位置”。

```CPP
Tensor thr_A = outer_partition(A,tiler,threadIdx.x); // (2,3)
```
实际 GEMM kernel 中更常看到 `local_partition(Tensor, Layout, Idx)`，它会根据线程布局把 `threadIdx.x` 转换成坐标，再完成对应的 partition。

#### Thread-Value Partitioning

Thread-Value partitioning 通常简称 TV partitioning。它用一个 Layout 同时描述“线程 id”和“每个线程持有的 value id”如何映射到目标 Tensor 的逻辑坐标。

```CPP
auto tv_layout = Layout<Shape <Shape <_2,_4>,Shape <_2, _2>>,
                        Stride<Stride<_8,_1>,Stride<_4,_16>>>{}; // (T8,V4)

Tensor A  = make_tensor<float>(Shape<_4,_8>{}, LayoutRight{});    // (4,8)
Tensor tv = composition(A, tv_layout);                            // (8,4)
auto value_T_V1=tv(threadIdx.x,1); //access V1 of this thread
Tensor this_T_V=tv.get_slice(threadIdx.x,_);
```

这里 `tv_layout` 描述 8 个线程、每个线程 4 个 value 如何覆盖一个 4x8 Tensor。经过 `composition` 后，第 0 维变成线程维，第 1 维变成该线程持有的 value 维。再用 `threadIdx.x` 切片，就能得到当前线程负责的寄存器 fragment。

# Warp-up
本篇介绍了Tensor的容器Engine以及Tensor常用操作，CuTe所有的操作都建立在Tensor和其层次化分块之上，结合GEMM的层次化计算来介绍Tensor的使用是下下篇文章的重点。