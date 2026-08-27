+++
title = 'CuTe学习00-模板元编程'
date = 2026-05-25T12:00:00+08:00
draft = false
description = '简单介绍C++模板元编程'
readingTimeText = '阅读此文大概需要28分钟'
tags = ['CuTe', 'TMP','Modern C++']
categories = ['Technical Blog']
+++

``` plain text
未成年C请在成年C++陪同下观看!
```

# 模板元编程

C++模板元编程（Template Meta Programming）是C++中最强大、也最硬核的特性之一。简单来说，普通的C++代码是在运行期（Runtime）由CPU执行；而模板元编程则是让编译器在编译期（Compile-time）执行代码。它的本质是利用C++的模板系统，在编译阶段生成定制化的源代码或计算出具体的值。

现代C++模板元编程主要依赖在C++17中引入的特性，本文接下来将简单介绍模板元编程与这些特性，作为理解CuTe的基础。本实验代码全部使用x86-64 gcc 16.1编译，flag为-O3 -maxv2。汇编代码使用intel asm syntax，仅用于解释TMP优化，并且只对本文负责。`...`表示省略不重要的内存搬运等汇编。

## WHY TMP?

在开始今天的正题之前，我希望和读者达成几个共识，作为我们接下来讨论性能的基础。
1. 逻辑分支的控制流开销远超纯计算指令。在CPU上分支预测失败会带来数十个周期的流水线惩罚，而在GPU的SIMT架构下，分支发散导致的性能惩罚则更为剧烈。
2. 高频的函数调用或深层递归会引入频繁的栈帧构建与上下文切换，这类开销通常是不可接受的性能瓶颈。
3. 编译器的启发式优化并非万能。评估优化策略的实际成效依赖于微基准测试，而验证其底层优化行为则需深入审查生成的汇编代码。

## Ex.1-编译期求值
在高性能计算中我们会面临很多常数计算，比如固定大小的TileSize，BlockSize等。我们以计算阶乘为切入点，描述TMP在常数优化方面的优势。

### 递归
``` C++ {hl_lines=["2"]}
int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

int main(){
    int a = factorial(5);
}
```
``` asm {hl_lines=["6-8"]}
"factorial(int)":
        ...
        cmp     DWORD PTR [rbp-4], 1
        jle     .L2
        ...
        call    "factorial(int)"
        imul    eax, DWORD PTR [rbp-4]
        jmp     .L4
.L2:
        mov     eax, 1
.L4:
        leave
        ret
"main":
        ...
        call    "factorial(int)"
        mov     DWORD PTR [rbp-4], eax
        ...
```
这段代码执行时会进行递归，直到计算出5!为止。从数值计算来讲，这当然没有问题。但是在性能方面，观察汇编，我们发现每次递归都要执行函数调用和条件判断终止，这两个操作的硬件代价很高，并且这个函数有着线性复杂度。言而总之，递归的方式性能并不好。

### 模板计算
``` C++ {hl_lines=[12]}
template<int N>
struct Factorial {
    if constexpr(N==1) static constexpr int value = 1;
    else static constexpr int value = N * Factorial<N - 1>::value;
};

int main(){
    int result = Factorial<5>::value;
}
```
``` asm {hl_lines=[3]}
"main": 
        ...
        mov     DWORD PTR [rbp-4], 120
        ...
        ret
```
观察汇编，我们发现一个神奇的事情，阶乘的结果是一个立即数！这代表运行时完全没有开销。当然，这也不是没有代价的，实际的计算过程是在编译期。我们把这个过程称作模板实例化，以 `factorial<3>` 为例，实例化的过程如下。
``` C++
template<int 3>
struct Factorial {
    static constexpr int value = 3 * Factorial<2>::value;
};
//我们需要Factorial<2>，继续实例化
template<int 2>
struct Factorial {
    static constexpr int value = 2 * Factorial<1>::value;
};
//我们已经有特化的Factorial<1>了，模板递归终止！
template<>
struct Factorial<1> {
    static constexpr int value = 1;
};

```
最终，编译器会直接把 Factorial<3>::value 替换成常数6。程序运行的时候，没有任何循环，没有任何函数调用，开销为绝对的零。

从这个例子中我们可以看出，模板元编程可以把运行时开销转移到编译期。在高性能计算的语境下，我们总是希望运行时能更快，并且一个核函数通常被调用很多次，而编译只有一次。更短的运行时总是我们乐意看到的结果。

## Ex.2-泛型维度
在高性能计算中，我们通常要处理很多不同形状的矩阵或向量。下面我们举例说明TMP在处理变长数组方面的优势。

### 运行时变长维度循环
``` C++ {hl_lines=[3]}
void add_vectors(const float* a, const float* b, float* result, int dimensions) {
    for (int i = 0; i < dimensions; ++i) {
        result[i] = a[i] + b[i];
    }
}
```
``` asm {hl_lines=[3]}
.L7     
        ...
        addss   xmm0, xmm1
        movss   DWORD PTR [rax], xmm0
        add     DWORD PTR [rbp-8], 1
.L6:
        mov     eax, DWORD PTR [rbp-8]
        cmp     eax, DWORD PTR [rbp-44]
        jl      .L7
        add     DWORD PTR [rbp-4], 1
```
在这个例子中，我们把运行时参数维度作为循环变量，每次进行条件判断决定结束循环或者计算下一个元素。当然，从数值的角度来讲，这完全没有问题。但从性能角度来说，处理极少量元素的小循环，循环控制的开销甚至超过了加法本身。同时，这种方式很难让编译器自动进行向量化，因为编译器无法确定到底要循环几次，处理向量化余数和条件判断的开销远远大于少做几次加法的收益。那我们自然可以想到，如果我们能为每个维度定制代码，是不是就可以消灭循环控制和判断开销，并可以自动向量化了，这就是TMP的优势。

### 编译期模板展开
``` C++ {hl_lines=["9"]}
template <size_t N, size_t I = 0>
inline void static_for(const float* a, const float* b, float* result) {
    if constexpr (I < N) {
        result[I] = a[I] + b[I];
        static_for<N, I + 1>(a, b, result);
    }
}
int main(){
    static_for<16>(a,b,result);
}
```
``` asm {hl_lines=["5-6"]}
...

vaddps  ymm3, ymm1, YMMWORD PTR .LC2[rip]
vaddps  ymm1, ymm1, YMMWORD PTR .LC4[rip]
vaddps  ymm2, ymm2, ymm3
vaddps  ymm0, ymm0, ymm1
vmovups YMMWORD PTR [rax+32], ymm2
vmovups YMMWORD PTR [rax], ymm0
vzeroupper
...
```

我们很惊喜的发现，生成的汇编当中, 只保留了两条ymm寄存器上的`vaddps`，这是256位寄存器上的的向量加法，十六个float正好使用两条，完全没有跳转和分支判断，并且应用了向量化，这是十分高性能的代码。

## C++17与TMP原语
严格来说，C++从诞生之日起（C++98）就支持模板，而且当时就被发现具备图灵完备性。但早期的TMP完全不可用，开发者不得不借用类模板的偏特化来假装写if-else，用递归来假装写for循环，编程工作量完全没有减少。直到C++17，C++才引入了一系列高级语法，让TMP变得可用且优雅，下面介绍几个TMP重要的语法和特性，以便于理解、使用和编写TMP。同时举出CuTe中使用此种特性的例子，帮助理解和学习CuTe。

### 编译期常量：constexpr
`constexpr` 的价值在于“零开销抽象”（Zero-overhead abstraction），这是C++模板元编程的基石。下面我们讲一下constexpr在C++17中的功能
#### 常量 constexpr T

``` C++
#define SIZE_A 1000
constexpr int SIZE_B = 1000;

int main() {
    int arr1[SIZE_A];
    int arr2[SIZE_B];
}
``` 
这两种定义常量的方式的效果完全一样，在最终生成的汇编中，编译器不会为任意一个数在内存里开辟空间，而是直接把1000作为立即数硬编码到指令当中。但他们的逻辑完全不同，`#define` 发生在编译之前的预处理阶段，只是简单的文本替换。这导致它存在三个巨大的致命缺陷，而 constexpr 完美解决了这些缺陷。

1. #define不是类型安全的，没有携带类型信息，但是constexpr是类型安全的，它是一个真正的 C++ 变量，拥有严格的类型。
2. #define没有作用域控制，一旦在头文件中定义，污染所有包含了这个头文件的代码。constexpr会严格遵守C++的命名空间和作用域。
3. #define没有符号表，在报错或是debug的时候只能看到数字。constexpr保留了变量类型和值，在debug的时候和运行时变量完全一样。

#### 全局常量 inline constexpr T
全局常量 inline constexpr T 解决了单一定义规则的问题，被多个cpp文件包含的头文件不会再抛出重复定义的错误，并且类中的静态成员会直接初始化并赋值，不需要在cpp中再次初始化。这个特性的重要程度，比这里短短的描写有分量的多，可以说没有这个特性，现代 Header-only 高性能库基本不可能编写。在 CuTe 中，inline constexpr 最成功的实践就是催生了 cute::_0 等静态常数

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

using _0      = Int<0>;
```

CuTe中的静态常数是CuTe编译期进行常数折叠的基石，有关静态常数的更多内容在后面介绍layout时会进一步讲解。

#### 编译期分支 if constexpr
`if constexpr`是 C++17 引入的一项非常强大的特性。它允许在编译期根据条件测试的结果，决定是否编译某一段代码。与传统的`if`相比`if constexpr`的核心优势在于不满足条件的分支代码根本不会被编译，从而消灭了条件判断和跳转开销。

我们先看一个非常常见的例子，根据类型决定函数行为。

``` C++
template <class T>
auto abs_value(T x) {
    if constexpr (std::is_unsigned_v<T>) {
        return x;
    } else {
        return x < 0 ? -x : x;
    }
}

int main() {
    auto a = abs_value(3u);
    auto b = abs_value(-3);
}
```

这里最重要的不是`if`，而是`constexpr`。对于`abs_value<unsigned>`来说，`else`分支根本不会被编译，函数体会直接变成：

``` C++
auto abs_value(unsigned x) {
    return x;
}
```

这和运行时的`if`完全不同。运行时`if`的两个分支都必须是语法合法的C++代码，只是运行时选择其中一个执行；`if constexpr`则会在编译期丢掉不满足条件的分支。因此它不仅能优化性能，还能写出普通`if`根本写不出来的泛型代码。

``` C++ {hl_lines=[5]}
template <class T>
void print_or_size(T const& x) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << x << "\n";
    } else {
        std::cout << x.size() << "\n";
    }
}
```

如果`T=int`，`x.size()`这个分支不会被编译，所以不会报错。如果这里使用普通`if`，即使运行时永远不走`x.size()`这一支，编译器也会因为`int`没有`size()`成员函数而拒绝编译。

在CuTe中，`if constexpr`几乎无处不在。原因也很自然：CuTe要处理大量“形状不同、rank不同、类型不同、是否静态不同”的对象。很多判断不是运行时判断，而是类型层面的判断。

``` C++
template <class T>
CUTE_HOST_DEVICE constexpr
auto get_static_value(T const& x) {
    if constexpr (is_static<T>::value) {
        return T::value;
    } else {
        return x;
    }
}
```

这段代码是示意，不对应CuTe中的某个原函数，但它表达了CuTe中非常典型的写法：如果一个值是编译期常量，就在编译期取出；如果它是运行时变量，就保留运行时读取。对用户来说，它们可能都叫`size`、`stride`、`shape`，但对编译器来说，一个是类型信息，一个是普通变量。`if constexpr`就是连接这两个世界的桥。

再举一个和Layout相关的例子。我们可能希望对一维Layout和二维Layout做不同处理。

``` C++
template <class Layout>
CUTE_HOST_DEVICE constexpr
void print_layout_kind(Layout const& layout) {
    if constexpr (rank(layout) == 1) {
        printf("1D layout\n");
    } else if constexpr (rank(layout) == 2) {
        printf("2D layout\n");
    } else {
        printf("ND layout\n");
    }
}
```

如果`rank(layout)`是编译期常量，那么最终生成的代码只会保留一个分支。我们写的是泛型代码，编译器看到的却是定制代码。这就是现代TMP最有魅力的地方。

#### 变量模板

C++17中另一个非常重要的小语法是变量模板，最典型的例子就是`std::is_same_v`。

在C++17之前，判断两个类型是否相同一般要这样写：

``` C++
std::is_same<T, U>::value
```

C++17之后可以写成：

``` C++
template <typename T, typename U>
inline constexpr bool is_same_v = is_same<T, U>::value;

std::is_same_v<T, U>
```

虽然这个语法的名字叫变量模板，但实际上我们总是用它来定义常量，变量模板的本质是“以模板形式定义一个常量”。它可以接收类型参数，也可以接收非类型参数。比如：

``` C++
template <class T>
constexpr bool is_small_type_v = sizeof(T) <= 4;

template <int N>
constexpr bool is_power_of_two_v = (N > 0) && ((N & (N - 1)) == 0);

static_assert(is_small_type_v<float>);
static_assert(is_power_of_two_v<128>);
```

这在高性能计算中很常见。例如我们常常需要根据数据类型选择不同的计算路径。

``` C++
template <class Element>
void load(Element* ptr) {
    if constexpr (std::is_same_v<Element, half>) {
        // half向量化读取
    } else if constexpr (std::is_same_v<Element, float>) {
        // float向量化读取
    } else {
        // 通用读取
    }
}
```

在CuTe里，类似的类型判断用于区分静态整数、动态整数、不同内存指针、不同MMA atom、不同Layout结构。它们的共同点是：判断发生在编译期，结果决定生成哪一种代码。

CuTe中有很多看起来像普通变量的对象，比如`_0`、`_1`、`_2`，但它们本质上携带的是类型信息。我们在上一篇代码中已经看到过：

``` C++
template <int v>
using Int = C<v>;

using _0 = Int<0>;
```

因此，CuTe代码中经常会出现“值和类型纠缠在一起”的情况。变量模板和`if constexpr`配合起来，可以让这些类型判断保持相对可读。

``` C++
template <class T>
constexpr bool is_static_int_v = is_static<T>::value;

template <class T>
CUTE_HOST_DEVICE constexpr
auto unwrap(T x) {
    if constexpr (is_static_int_v<T>) {
        return T::value;
    } else {
        return x;
    }
}
```

这里依然是示意代码。真正需要记住的是：`_v`变量模板不是语法糖玩具，而是现代C++把类型计算写得像普通表达式的关键工具。

### 折叠表达式

模板元编程中另一个经典问题是变长参数。比如我们想写一个函数，接收任意数量的参数并求和。

在C++17之前，通常需要递归。

``` C++
template <class T>
auto sum(T x) {
    return x;
}

template <class T, class... Ts>
auto sum(T x, Ts... xs) {
    return x + sum(xs...);
}

int main() {
    auto x = sum(1, 2, 3, 4);
}
```

这个写法非常TMP：用函数递归假装循环。它能工作，但不优雅，而且递归终止条件、重载匹配、错误信息都很不友好。C++17引入折叠表达式后，这件事可以一行写完。

``` C++ {hl_lines=[3]}
template <class... Ts>
auto sum(Ts... xs) {
    return (xs + ...);
}
```

`(xs + ...)`会被编译器展开为：

``` C++
(((x1 + x2) + x3) + x4)
```

如果我们想指定初始值，也可以写：

``` C++
template <class... Ts>
auto sum_with_zero(Ts... xs) {
    return (0 + ... + xs);
}
```

这会被展开为：

``` C++
(((0 + x1) + x2) + x3)
```

折叠表达式支持很多运算符，比如`+`、`*`、`&&`、`||`、`,`等。对于模板元编程来说，最常用的是把一组类型或一组值“压扁”成一个结果。

``` C++
template <class... Ts>
constexpr bool all_trivial_v = (std::is_trivial_v<Ts> && ...);

static_assert(all_trivial_v<int, float, char>);
```

这会被展开为：

``` C++
std::is_trivial_v<int> &&
std::is_trivial_v<float> &&
std::is_trivial_v<char>
```

在CuTe的语境下，折叠表达式尤其适合处理任意rank的Shape。一个Shape可能是一维、二维、三维，也可能是嵌套的。我们经常想对它的每个维度做同一件事，比如计算总元素个数。

``` C++
template <class... Dims>
constexpr auto product(Dims... dims) {
    return (dims * ...);
}

int main() {
    constexpr int size = product(4, 8, 16);  // 512
}
```

如果这些`dims`都是编译期常量，那么`product(4, 8, 16)`在运行时不会产生任何乘法。编译器会直接把它折叠成立即数512。对于TileSize、ThreadCount、VectorWidth这类常量来说，这正是我们想要的结果。

折叠表达式还常被用来做批量调用。比如对一组对象依次打印：

``` C++
template <class... Ts>
void print_all(Ts const&... xs) {
    ((std::cout << xs << "\n"), ...);
}
```

这里使用的是逗号折叠。它会按顺序执行每个输出表达式。虽然看起来有点像黑魔法，但比递归模板清爽很多。

另外一个应用就是为有多种复杂选项的函数传入配置，比如fmha中的

``` C++
template<auto kTag, typename Default, typename... Options>
struct find_option;

template<auto kTag, typename Default>
struct find_option<kTag, Default> {
  using option_value = Default;
};

template<auto kTag, typename Default, typename Option, typename... Options>
struct find_option<kTag, Default, Option, Options...> :
  std::conditional_t<
    Option::tag == kTag,
    Option,
    find_option<kTag, Default, Options...>
  >
{};

template<auto kTag, typename Default, typename... Options>
using find_option_t = typename find_option<kTag, Default, Options...>::option_value;

enum class Tag {
  kIsPersistent,
  kNumMmaWarpGroups,
  kLoadsQSeparately,

  kIsMainloopLocked,
  kIsEpilogueLocked,

  kStagesQ,
  kStagesKV,

  kEpilogueKind,

  kBlocksPerSM,
  kClusterM,

  kAccQK
};

template<auto kTag, class Value>
struct Option {
  static constexpr auto tag = kTag;
  using option_value = Value;
};

```
借助变长模板，我们省去了写运行时switch，甚至是嵌套if的麻烦，并且能保持函数调用接口干净。同时这种写法有着良好的拓展性，如果需要增加新的参数/选项，只需要在Tag里面增加选项，而不需要再嵌套一层if。

### 类模板参数推导

类模板参数推导（Class Template Argument Deduction, CTAD）也是C++17引入的重要特性。它解决的问题很简单：构造类模板对象时，很多模板参数明明可以从构造函数参数推导出来，为什么还要用户手写？

我们先看一个普通例子。

``` C++
std::pair<int, double> p1(1, 2.0);
```

C++17之后可以写成：

``` C++
std::pair p2(1, 2.0);
```

编译器会根据`1`和`2.0`自动推导出`std::pair<int, double>`。这就是CTAD。

这个特性对普通业务代码只是少写一点类型，但对CuTe此类高性能header-only库非常重要。因为模板元编程里的类型通常无法由人类简单写出。

``` C++
Layout<Shape<Int<2>, Int<3>>, Stride<Int<3>, Int<1>>> layout{
    Shape<Int<2>, Int<3>>{},
    Stride<Int<3>, Int<1>>{}
};
```

如果每次都要这样写，基本没人愿意使用CuTe。我们真正希望写的是：

``` C++
auto layout = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          make_stride(Int<3>{}, Int<1>{}));
```

这里主要依赖的是工厂函数`make_layout`、`make_shape`、`make_stride`的类型推导，而CTAD解决的是同一类问题：让编译器从构造参数中推导类型，用户只表达逻辑含义。

这对CuTe这类库非常关键。因为CuTe的对象往往不是运行时值复杂，而是类型复杂。一个小小的Layout，其类型可能包含Shape、Stride、静态整数、嵌套tuple等信息。用户如果必须手写这些类型，代码会立刻不可维护。

CTAD和`auto`、工厂函数一起，构成了现代C++模板库的三件套。

``` C++
auto shape  = make_shape(Int<128>{}, Int<64>{});
auto stride = make_stride(Int<64>{}, Int<1>{});
auto layout = make_layout(shape, stride);
```

这段代码里，真正的信息仍然完整存在于类型系统中。只是我们不再把这些类型全部写在源码表面。换句话说，CTAD不是让类型消失了，而是让人类不用亲手写出那坨类型。编译器很擅长处理这种东西，人类不擅长。

## 小结

到这里，我们已经介绍了几个现代TMP最重要的原语：

1. `constexpr`：把值和函数放到编译期。
2. `if constexpr`：在编译期选择代码路径。
3. 变量模板：把类型判断写成普通布尔变量。用变量模板推断常量
4. 折叠表达式：优雅处理变长模板参数。
5. CTAD：让编译器从构造参数中推导复杂类型。

这些特性单独看都不算特别吓人，但是组合起来就构成了CuTe这类现代高性能C++库的语法基础。CuTe之所以能把Layout、Tensor、Copy、MMA这些复杂对象全部写成可组合的编译期抽象，很大程度上就是依赖这些C++17工具。

如果读者后面阅读CuTe源码时看到非常长的类型、很多`constexpr`函数、很多`if constexpr`分支，不要慌。它们不是为了炫技而存在，而是在用C++类型系统描述GPU程序的层次和结构。理解了这一点，CuTe源码会从“奇怪的模板咒语”慢慢变成“编译期的数据流图”。

# 写在最后

笔者认为，现代C++模板元编程是C++中最抽象，最晦涩难懂的一部分。但是使用TMP带来的代码可复用性和极致的性能优化是任何编程技巧都做不到的事情。如果你只是想了解一下模板元编程或者是现代高性能计算中代码方面的优化，那简单读完本篇就够了。但如果你是想掌握CUTLASS，CuTe这种现代高性能模板库，则需要大量的阅读源码和实践。

# PS
如果你直接拿本文中示例代码去编译，大概率得不到如此漂亮的汇编，这一方面和不同编译器行为有关，一方面-O3优化会折叠很多常数或者删去死代码。当然，这也不是说本文的例子都是凭空造出来的，本文的示例汇编均是编译产物，只不过源代码中有很多与TMP无关，只是为了控制编译器行为的代码。如果读者有兴趣实践（这当然是我最推荐的学习方式），大可以自己去写TMP，观察编译器的行为，思考编译器优化的方式，这样才能亲自写出风格优雅，性能卓越的代码。本文主要目的还是讲述编译器行为和优化，所以尽可能只保留了代码的骨干，高亮了优化的部分，删去了随机初始化或者print之类仅为了不让编译器优化常量的代码。
