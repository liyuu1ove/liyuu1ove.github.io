+++
title = 'CuTe学习1-Layout'
date = 2026-05-29T12:00:00+08:00
draft = false
description = '介绍CuTe的layout模板类'
readingTimeText = '阅读此文大概需要21分钟'
tags = ['CuTe','Layout']
categories = ['Technical Blog']
+++
# 写在前面-张量的形式化表达
计算机中的内存是连续的一维空间，而在BLAS或者深度学习中我们需要二维、甚至更高维的张量描述。探究如何高效便捷的将逻辑张量映射到内存中经历了多个阶段。

1. BLAS的row/col-major + leading dimension描述阶段。这一阶段对现代硬件和BLAS库产生了深远影响，比如`wmma::row_major`等指令。
2. Tensor单调描述阶段。即使用shape + stride描述Tensor。PyTorch ATensor 就用的是这种设计模式。这种设计模式的问题是底层的储存是且必须是扁平且致密的，这个特性就不适应现代机器学习的张量形状。
3. Tensor层次化描述阶段。Hierarchy Tensor是在2023年，在[Graphene](https://dl.acm.org/doi/10.1145/3582016.3582018)这篇论文中提出的。Hierarchy Tensor是指非致密的、多层级嵌套的张量结构。在普通Tensor里，矩阵的每个元素必须是一个标量（e.g. a float）。而在 Hierarchy Tensor 逻辑中，张量的某个元素本身可能又是一个完整的张量(e.g. (2,3))，且各元素的储存在内存中不必连续。

CuTe利用Hierarchy Tensor进行Tensor描述和计算。阅读Graphene原文有助于理解CuTe Layout Algebra。

本文遵循CuTe的notation。使用带下划线的数字(e.g. _3)表示静态整数，不带下划线的数字(e.g. 3)表示动态整数，使用`Shape:Stride`表示`Layout`，使用嵌套的括号表示`Tuple`的层级(e.g. (2,(3,4)) )。本文的张量并不特指CuTe中的Tensor，绝大多数指layout描述形状对应的张量。

# CuTe 类型与概念
## 静态整型
CuTe的绝大多数逻辑都建立在静态整型上面(Static integers or “compile-time integers”)。注意，这里使用了整型而不是整数，CuTe的Int<>是一个类模板，是cute::integral_constant<int, N>的别名。Int<1>的是这个类模板的特化类型，而不是一个整数，它的本质是一个类型标签，用于模板参数传递、类型推导或者函数签名中。而如果需要作为函数参数传入，则需要将其实例化，实例化的语法是Int<1>{}。同时CuTe为类模板重载了强制类型转换运算符与()运算符，使类模板对象也可以参与整数计算。再一次把实现的代码贴在这里，理解其写法有助于巩固TMP的语法和理解类模板和对象的区别。

**include/cute/numeric/integral_constant.hpp**
``` C++ {title="include/cute/numeric/integral_constant.hpp"}
// A constant value: short name and type-deduction for fast compilation
template <auto v>
struct C {
  using type = C<v>;
  static constexpr auto value = v;
  using value_type = decltype(v);
  CUTE_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
  CUTE_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};
// alias
template <int v>
using Int = C<v>;

using _0  = Int<0>;
```

与静态整型相对的是动态整型，int size_t uint16_t等被std::is_integral<T>接受的均在CuTe中被当作动态整型。这里有一个十分经典的疑问。

``` C++
Layout s8 = make_layout(Int<8>{});
Layout d8 = make_layout(8);
```
这里的8不是在编写代码的时候就hardcode进去了吗，为什么需要Int<8>{}呢？实际上编译期看到的并不是8，而是
``` C++
Layout d8 = make_layout(int);
```
真正的值会在运行时传入寄存器，然后参与运算，这就造成了运行时性能的浪费。

## CuTe Tuple
CuTe::tuple 行为和 std::tuple 完全一致，只不过是在device端和host端都work。CuTe中的IntTuple被定义为一个整型，或者是IntTuple的Tuple。这就为Hierarchy Tensor的描述提供了可能。在CuTe中，Shape, Stride, Step, and Coord都被视作IntTuple。Layout被视为tuple (Shape, Stride)。
# CuTe Layout
## Shape & Stride
在cute体系下，逻辑空间可以被称作domain，而代表存储的物理空间称作codomain。`Shape`描述的就是逻辑空间，而`Stride`描述的是物理空间。具体来说，Shape描述了一个张量的形状，比如说Shape:(3,4)就描述了一个3*4的张量，我们只关心它的形状。Stride:(4,1)描述了张量中元素的步长，在第一个维度上（CuTe也称其为最外围，由左向右数第一个维度）每前进一个元素，其对应的物理内存前进4个元素。我们print一下这个形状，就得到了：
``` bash
(_3,_4):(_4,_1)
       0    1    2    3 
    +----+----+----+----+
 0  |  0 |  1 |  2 |  3 |
    +----+----+----+----+
 1  |  4 |  5 |  6 |  7 |
    +----+----+----+----+
 2  |  8 |  9 | 10 | 11 |
    +----+----+----+----+
```
这是一个经典的行主序，张量外面的序号代表逻辑坐标，张量里面的数字代表在内存中的偏移，组合起来便是Layout，逻辑坐标到内存偏移量的映射。我们简单修改Stride，便可以得到列主序的layout。
``` bash
(_3,_4):(_1,_3)
       0    1    2    3 
    +----+----+----+----+
 0  |  0 |  3 |  6 |  9 |
    +----+----+----+----+
 1  |  1 |  4 |  7 | 10 |
    +----+----+----+----+
 2  |  2 |  5 |  8 | 11 |
    +----+----+----+----+
```
这便是解耦与抽象的好处，我们在之后的介绍中会认识到CuTe这种层次化设计的诸多方便。
## Layout 
在这一节中，我们从简单的一维向量开始，介绍Stride控制的逻辑坐标到内存偏移的映射。

偏移量的计算很简单，是逻辑坐标和Stride的点积。CuTe为我们简化了手算该偏移的过程。

$ \text{Offset}(c) = \sum_{i=0}^{n-1} c_i \times D_i $
### 1D Layout
我们接下来看几个例子。注：由于print_layout只接受二维Layout，所以我们把第一维只有1个元素的二维张量叫做一维张量。
``` bash
(_1,_8):(_8,_1)
      0   1   2   3   4   5   6   7 
    +---+---+---+---+---+---+---+---+
 0  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    +---+---+---+---+---+---+---+---+
size of the tensor: _8
cosize of the tensor: _8
```
这是一个简单的一维张量的映射。步长为1。
```bash
(_1,_8):(_8,_2)
       0    1    2    3    4    5    6    7 
    +----+----+----+----+----+----+----+----+
 0  |  0 |  2 |  4 |  6 |  8 | 10 | 12 | 14 |
    +----+----+----+----+----+----+----+----+
size of the tensor: _8
cosize of the tensor: _15
```
这个张量的步长变成了2，所以内存offset也发生了变化。同时cosize（codomain占用的空间，即内存上需要分配的空间）也变成了15（因为元素8后面没有元素了，所以少1）。
``` bash
(_1,_8):(_8,_-1)
      0   1   2   3   4   5   6   7 
    +---+---+---+---+---+---+---+---+
 0  | 0 | -1 | -2 | -3 | -4 | -5 | -6 | -7 |
    +---+---+---+---+---+---+---+---+
size of the tensor: _8
cosize of the tensor: _8
```
更加神秘的是，layout支持负数步长，可以用于前向访问。二维矩阵的描述和一维没有本质差别，感兴趣的读者可以自己尝试变换步长打印。
### Hierarchy Tensor

``` bash
(_4,(_2,_3)):(_6,(_3,_1))
       0    1    2    3    4    5 
    +----+----+----+----+----+----+
 0  |  0 |  3 |  1 |  4 |  2 |  5 |
    +----+----+----+----+----+----+
 1  |  6 |  9 |  7 | 10 |  8 | 11 |
    +----+----+----+----+----+----+
 2  | 12 | 15 | 13 | 16 | 14 | 17 |
    +----+----+----+----+----+----+
 3  | 18 | 21 | 19 | 22 | 20 | 23 |
    +----+----+----+----+----+----+
```
这是一个比较简单的嵌套张量，第一维(i.e. _4)有四个元素，每个元素是一个(_2,_3):(_3,_1)的二维张量，打印的形状是这个二维张量展平（flatten）到一维的结果。注意，这个嵌套张量是二维的，而非一个三维的张量。这一定程度上也是一种文字游戏，不过作者认为认识清楚嵌套的层级有助于理解该概念。如此的嵌套张量在写算子时十分常见，比如说MHA的实现中，我们会定义 (S, (H, D), B) 这样的嵌套张量。所以感谢CuTe，在写代码的过程中我们不必知道我们的嵌套张量到底是如何排布在内存中的，我们只需要关注逻辑索引就好了。

### Layout Coordinates
Layout可以接受多种类型的坐标，只要坐标的Shape和layout一致(有关一致性可以参阅[layout compatibility](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#layout-compatibility)，简单来说，可以展平某些维度)。CuTe的mapping使用 colexicographical order，即从左往右增加。
#### Coordinate Mapping
The map from an input coordinate to the corresponding natural coordinate via the `Shape`. 我们称和原Shape维度完全一样的坐标为自然坐标。输入坐标的形式可以有很多种。可以参考下面这个例子。同时，这个例子也描述了 colexicographical order 是什么样的。值得注意的是，我们日常使用的十进制数是lexicographical order的，即按照01，02，03这个顺序递增。CuTe使用 colexicographical order 也是自然的，因为C++变长模板展开需要从最左侧开始展开，我们很难实现从最右侧展开的代码逻辑。

| 1-D | 2-D | Natural |
| :---: | :---: | :---: |
| 0 | `(0,0)` | `(0,(0,0))` |
| 1 | `(1,0)` | `(1,(0,0))` |
| 2 | `(2,0)` | `(2,(0,0))` |
| 3 | `(0,1)` | `(0,(1,0))` |
| 4 | `(1,1)` | `(1,(1,0))` |
| 5 | `(2,1)` | `(2,(1,0))` |
| 6 | `(0,2)` | `(0,(0,1))` |
| 7 | `(1,2)` | `(1,(0,1))` |
| 8 | `(2,2)` | `(2,(0,1))` |
| 9 | `(0,3)` | `(0,(1,1))` |
| 10 | `(1,3)` | `(1,(1,1))` |
| 11 | `(2,3)` | `(2,(1,1))` |
| 12 | `(0,4)` | `(0,(0,2))` |
| 13 | `(1,4)` | `(1,(0,2))` |
| 14 | `(2,4)` | `(2,(0,2))` |
| 15 | `(0,5)` | `(0,(1,2))` |
| 16 | `(1,5)` | `(1,(1,2))` |
| 17 | `(2,5)` | `(2,(1,2))` |

`idx2crd` 用法如下：

``` C++
auto shape = Shape<_3,Shape<_2,_3>>{};
print(idx2crd(   16, shape));                                // (1,(1,2))
print(idx2crd(_16{}, shape));                                // (_1,(_1,_2))
print(idx2crd(make_coord(   1,5), shape));                   // (1,(1,2))
print(idx2crd(make_coord(_1{},5), shape));                   // (_1,(1,2))
print(idx2crd(make_coord(   1,make_coord(1,   2)), shape));  // (1,(1,2))
print(idx2crd(make_coord(_1{},make_coord(1,_2{})), shape));  // (_1,(1,_2))
```

#### Index Mapping
The map from a natural coordinate to the index via the `Stride`. 这是写代码中最常用的功能，从逻辑坐标映射到内存的偏移。同样的，这个api也能接受任意和原shape兼容的坐标。其映射关系如下所示：
``` bash
       0     1     2     3     4     5     <== 1-D col coord
     (0,0) (1,0) (0,1) (1,1) (0,2) (1,2)   <== 2-D col coord (j,k)
    +-----+-----+-----+-----+-----+-----+
 0  |  0  |  12 |  1  |  13 |  2  |  14 |
    +-----+-----+-----+-----+-----+-----+
 1  |  3  |  15 |  4  |  16 |  5  |  17 |
    +-----+-----+-----+-----+-----+-----+
 2  |  6  |  18 |  7  |  19 |  8  |  20 |
    +-----+-----+-----+-----+-----+-----+
```

`crd2idx`用法如下：
``` C++
auto shape  = Shape <_3,Shape<  _2,_3>>{};
auto stride = Stride<_3,Stride<_12,_1>>{};
print(crd2idx(   16, shape, stride));       // 17
print(crd2idx(_16{}, shape, stride));       // _17
print(crd2idx(make_coord(   1,   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},   5), shape, stride));  // 17
print(crd2idx(make_coord(_1{},_5{}), shape, stride));  // _17
print(crd2idx(make_coord(   1,make_coord(   1,   2)), shape, stride));  // 17
print(crd2idx(make_coord(_1{},make_coord(_1{},_2{})), shape, stride));  // _17
```

# 写在最后

本文介绍了CuTe中的layout和其相关的概念。layout掌握起来并不复杂，读者可以利用cute::print来打印layout帮助理解。下次我们将介绍Layout Algebra，是重要且复杂的部分。

# Reference

https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html#layouts-coordinates

https://zhuanlan.zhihu.com/p/661182311
