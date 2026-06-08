+++
title = 'CuTe学习00-模板元编程'
date = 2026-05-25T12:00:00+08:00
draft = false
description = '简单介绍C++模板元编程'
readingTimeText = ''
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
1. 逻辑分支比计算昂贵得多。在CPU上是如此，GPU上更甚。
2. 函数递归或者是多次调用的开销是不可接受的。
3. 编译器优化很聪明，但是也没那么聪明。优化的效果要看实际性能，实现优化的原理要看汇编。

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
    static constexpr int value = N * Factorial<N - 1>::value;
};

template<>
struct Factorial<1> {
    static constexpr int value = 1;
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
在这个例子中，我们把维度作为运行时参数作为循环变量，每次进行条件判断决定结束循环或者计算下一个元素。当然，从数值的角度来讲，这完全没有问题。但从性能角度来说，处理极少量元素的小循环，循环控制的开销甚至超过了加法本身。同时，这种方式很难让编译器自动进行向量化，因为编译器无法确定到底要循环几次，处理向量化余数和条件判断的开销远远大于少做几次加法的收益。那我们自然可以想到，如果我们能为每个维度定制代码，是不是就可以消灭循环控制和判断开销，并可以自动向量化了，这就是MTP的优势。

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

## C++17与MTP原语
严格来说，C++从诞生之日起（C++98）就支持模板，而且当时就被发现具备图灵完备性。但早期的TMP完全不可用，开发者不得不借用类模板的偏特化来假装写 if-else，用递归来假装写 for 循环，编程工作量完全没有减少。直到C++17，C++才引入了一系列高级语法，让MTP变得可用且优雅，下面介绍几个MTP重要的语法和特性，以便于理解、使用和编写MTP。同时举出CuTe中使用此种特性的例子，帮助理解和学习CuTe。

### 编译期常量：constexpr
`constexpr` 的价值在于“零开销抽象”（Zero-overhead abstraction），这是C++模板元编程的基石。下面我们讲一下constexpr在C++17中的功能
#### 常量 constexpr T 与 全局常量 inline constexpr T

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

全局常量 inline constexpr T 解决了单一定义规则的问题，被多个cpp文件包含的头文件不会再抛出重复定义的错误，并且类中的静态成员会直接初始化并赋值，不需要在cpp中再次初始化。这个特性的重要程度，比这里短短的描写有分量的多，可以说没有这个特性，现代 Header-only 高性能库基本不可能编写。在 CuTe 中，inline constexpr 最成功的实践就是催生了 cute::_0 等静态变量

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

#### 编译期分支 if constexpr
`if constexpr` 是 C++17 引入的一项非常强大的特性。它允许在编译期根据条件测试的结果，决定是否编译某一段代码。与传统的 if 相比，if constexpr 的核心优势在于不满足条件的分支代码根本不会被编译，从而消灭了条件判断和跳转开销。

#### _v变量模板（is_same_v）



### 折叠表达式

### 类模板参数推导



# 写在最后

笔者认为，现代C++模板元编程是C++中最抽象，最晦涩难懂的一部分。但是使用MTP带来的代码可复用性和极致的性能优化是任何编程技巧都做不到的事情。如果你只是想了解一下模板元编程或者是现代高性能计算中代码方面的优化，那简单读完本篇就够了。但如果你是想掌握CUTLASS，CuTe这种现代高性能模板库，则需要大量的阅读源码和实践。同时，多问AI，遇到不懂的先问再实验，亲自对比性能上的差别，而不是AI说什么就相信什么。

# PS
如果你直接拿本文中示例代码去编译，大概率得不到如此漂亮的汇编，这一方面和不同编译器行为有关，一方面-O3优化会折叠很多常数或者删去死代码。当然，这也不是说本文的例子都是凭空造出来的，所有示例代码和.o文件可以在github上找到。本文主要目的还是讲述编译器行为和优化，所以尽可能只保留了代码的骨干，高亮了优化的部分，删去了随机初始化或者print之类仅为了不让编译器优化常量的代码。