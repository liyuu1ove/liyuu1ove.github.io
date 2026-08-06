+++
title = 'CuTe学习5-Copy Atom'
date = '2026-07-10T14:21:13+08:00'
draft = true
description = '介绍CuTe中Copy Atom的层次化结构'
readingTimeText = '阅读此文大概需要30分钟'
tags = ['CuTe','Copy Atom','CUDA']
categories = ['Technical Blog']
+++

# Copy Hierarchy

前面介绍MMA时，我们从一条PTX指令一路向上讲到了`MMA_Atom`、`TiledMMA`、`ThrMMA`和`cute::gemm`。CuTe中的copy也遵循类似的层次：最底层包装具体的数据搬运指令，中间层描述指令的Thread-Value映射，再向上把一条指令扩展成多个线程协作的copy tile，最后由`cute::copy`真正执行搬运。

{{< image src="copy.png" alt="CopyHierarchy" maxWidth="500px" caption="CuTe Copy Hierarchy">}}


1. `CopyOperation`：包装一条具体的copy指令，声明指令的source/destination寄存器接口。
2. `Copy_Traits`：描述该指令需要多少线程，以及source和destination两侧的bit级Thread-Value Layout。
3. `Copy_Atom`：把bit级Traits重解释为指定逻辑元素类型，形成一次最小copy操作。
4. `TiledCopy`：把Copy Atom铺到更大的逻辑tile上，描述多个线程分别负责哪些value。
5. `ThrCopy`：固定当前thread id，并用`partition_S/D`切出该线程的source/destination视图。
6. `cute::copy`：沿Tensor的外层mode循环，最终调用Copy Atom中的底层指令。

本文使用两条Ampere GEMM中常见的路径作为主例子：

``` plain text
GMEM --cp.async--> SMEM --ldmatrix--> RMEM --mma.sync--> Tensor Core
```

`cp.async`解决global memory到shared memory的异步、向量化搬运；`ldmatrix`解决shared memory中的矩阵tile如何按照Tensor Core需要的fragment布局进入每个lane的寄存器。理解这两种完全不同的指令之后，`Copy_Atom`为什么需要同时描述source和destination映射也就比较直观了。

# CopyOperation

`CopyOperation`是最靠近硬件的一层。和`MMAOperation`类似，它声明了PTX需要用到的寄存器数量和包装了PTX指令。

## ldmatrix

Tensor Core的MMA指令中，lane必须按照PTX规定的fragment格式持有A/B operand。Turing开始提供的`ldmatrix`正是为这个过程设计的：一个warp协作地从shared memory读取若干个`8 x 8`、元素宽度为16-bit的矩阵，并把结果分发到32个lane的寄存器中。

CuTe对`ldmatrix.sync.aligned.x1.m8n8.shared.b16`的包装如下：

``` C++
// Source from include/cute/arch/copy_sm75.hpp
struct SM75_U32x1_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#endif
  }
};
```

这里的`x1`表示整个warp读取一个`8 x 8`矩阵。矩阵一共有`8 * 8 * 16 bit = 1024 bit = 128 byte`。结果平均分给32个lane，因此每个lane得到一个32-bit寄存器，也就是两个16-bit元素。

`SRegisters = uint128_t[1]`容易引起误解：它并不表示每个lane都实际搬运128-bit到自己的寄存器。这个类型描述source侧传给指令的是一个能覆盖128-bit行数据的shared-memory地址视图。对于`x1`，只有lane 0到7提供的8个行地址会被硬件使用；整个warp仍然都要执行同一条`ldmatrix`，随后硬件把加载结果分发给全部32个lane。这个分发是`ldmatrix`指令本身的语义，不需要再写`__shfl_sync`。

CuTe还提供`x2`和`x4`版本：

``` plain text
x1: 读取1个8x8矩阵，lane 0..7提供地址，每个lane得到1个uint32_t
x2: 读取2个8x8矩阵，lane 0..15提供地址，每个lane得到2个uint32_t
x4: 读取4个8x8矩阵，lane 0..31提供地址，每个lane得到4个uint32_t
```

`SM75_U32x1_LDSM_N`名字中的`U32x1`表示每个lane产生一个32-bit destination register，`N`表示non-transpose版本。对应的`SM75_U16x2_LDSM_T`等类型包装了带`.trans`修饰的转置版本。实际选择哪个版本，取决于MMA operand的布局、shared-memory布局以及每次想加载多少fragment。

{{< image src="mma-ldmatrix-fragments.png" alt="ldmatrix" maxWidth="500px" caption="ldmatrix fragment layout for one 8x8 Matrix with 16-bit elements">}}

## cp.async

Ampere的`cp.async`用于从global memory直接向shared memory发起异步copy，其不需要在寄存器中转，因此可以降低中间寄存器压力，并允许copy和后续计算重叠。

CuTe中的`SM80_CP_ASYNC_CACHEALWAYS`包装如下：

``` C++
// Source from include/cute/arch/copy_sm80.hpp
template <class TS, class TD = TS>
struct SM80_CP_ASYNC_CACHEALWAYS
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD));
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16);

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst)
  {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    TS const* gmem_ptr    = &gmem_src;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr),
           "l"(gmem_ptr),
           "n"(sizeof(TS)));
#endif
  }
};
```

这条指令含义如下：

1. `cp.async`表示异步搬运，这条指令发射后不会阻塞 CUDA Core/ALU，SM 发射完指令后可以立即去执行后续的计算指令，数据的传输过程由专用的异步拷贝硬件逻辑在后台完成。
2. `.ca`（Cache Always）表示数据可以缓存在所有层级；CuTe还提供使用`.cg`（Cache Global）策略的`SM80_CP_ASYNC_CACHEGLOBAL`。
3. `shared.global`表示 目标空间.源空间。
4. `L2::128B`是L2 prefetch size hint，数据对齐与传输块大小为 128 字节，不表示当前线程一次搬运128 byte。真正的单线程copy宽度由`sizeof(TS)`决定，只能是4、8或16 byte。若`TS = uint128_t`，当前线程搬运的是16 byte。这是一条cp.async的最大搬运粒度。

最常见的写法是把128b访存和真实逻辑元素类型分开，即常说的128b合并访存：

``` C++
using GmemCopyAtom = Copy_Atom<
    SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,
    half_t>;
```

这里底层PTX每次搬一个`uint128_t`，而`Copy_Atom`把这128 bit解释为8个`half_t`。它只改变CuTe用来分区Tensor的value粒度，不会做`uint128_t`到`half_t`的数值转换。

<!-- 待补图3：cp.async单线程与warp协作copy示意图。画出每线程16-byte GMEM区间到SMEM区间的映射，并把L2::128B标成prefetch hint而不是单线程copy宽度。 -->

## cp.async同步原语

发出`cp.async`之后，数据不一定已经可以被后续指令使用。CuTe提供了与PTX group机制对应的接口：

``` C++
CUTE_HOST_DEVICE
void cp_async_fence()
{
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
CUTE_HOST_DEVICE
void cp_async_wait()
{
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
  }
#endif
}
```

`cp_async_fence()`实际上发出的是`cp.async.commit_group`，把当前线程此前尚未提交的`cp.async`组成一个group，但不等待group完成，类似于插桩。

`cp_async_wait<N>()`会阻塞当前线程，直到此前提交但尚未完成的group数量不大于`N`。因此：

``` plain text
cp_async_wait<0>()  等待所有旧group完成
cp_async_wait<1>()  允许至多1个较新的group继续在途
```

等待是per-thread的，而一个CTA通常会让不同线程copy不同的shared-memory位置，之后又由另一些线程消费这些位置。因此GEMM pipeline在等待目标stage之后通常还需要`__syncthreads()`，保证CTA内生产者和消费者在使用该stage前完成同步。`commit_group`、`wait_group`和CTA barrier解决的是不同层次的问题，不能互相替代。

# Copy_Traits

`Copy_Traits`补充三类信息：参与的逻辑线程、source侧映射、destination侧映射。

与MMA Traits直接用逻辑元素描述TV Layout不同，Copy Traits先用bit作为统一单位。这样同一条128-bit copy指令既可以服务8个`half_t`，也可以服务4个`float`，而不用为每一种元素类型重复定义Traits。

## ldmatrix Traits

`SM75_U32x1_LDSM_N`的Traits如下：

``` C++
template <>
struct Copy_Traits<SM75_U32x1_LDSM_N>
{
  using ThrID = Layout<_32>;

  using SrcLayout = Layout<Shape <Shape <  _8,_4>,_128>,
                           Stride<Stride<_128,_0>,  _1>>;

  using DstLayout = Layout<Shape <_32,_32>,
                           Stride<_32, _1>>;

  using RefLayout = DstLayout;
};
```

三个Layout的单位都是bit：

1. `ThrID = Layout<_32>`表示这是一个32线程的warp级操作。
2. `SrcLayout`描述`(src-thread, src-bit)`到source地址的映射。thread mode的`(_8,_4):(_128,_0)`中，第二层stride为0，表示32个lane只有8个不同的行地址；这正对应`x1`只使用lane 0到7提供地址的硬件行为。
3. `DstLayout = (_32,_32):(_32,_1)`表示32个lane各得到连续的32 bit。
4. `RefLayout`是CuTe协调source和destination映射时使用的参考TV Layout。这里结果分发方式就是参考布局。

source和destination的每线程value数量可以不同。以`half_t`解释该Atom时，提供有效行地址的lane需要让source视图覆盖一行8个half，而每个lane的destination只有2个half。`ldmatrix`的价值正是在一条warp级指令中完成这种collective load和fragment分发。

## cp.async Traits

`cp.async`的Traits简单得多：

``` C++
template <class S, class D>
struct Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<S,D>>
{
  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,Int<sizeof_bits<S>::value>>>;
  using DstLayout = Layout<Shape<_1,Int<sizeof_bits<D>::value>>>;
  using RefLayout = SrcLayout;
};
```

`ThrID = Layout<_1>`表示一个Atom只涉及一个线程。source和destination都是`(1, sizeof_bits<T>)`，所以它们是一一对应的bit copy。当`S = D = uint128_t`时，一个Atom覆盖1个线程和128 bit；随后`Copy_Atom<..., half_t>`会把它重解释为1个线程和8个half value。

# Copy_Atom

`Copy_Atom`把`CopyOperation`、`Copy_Traits`和用户指定的内部value类型组合在一起：

``` C++
template <class CopyOperation, class CopyInternalType>
struct Copy_Atom<CopyOperation, CopyInternalType>
  : Copy_Atom<Copy_Traits<CopyOperation>, CopyInternalType>
{};
```

在具体实现中，它会把Traits中的bit Layout recast为`CopyInternalType`对应的value Layout：

``` C++
using ValType = CopyInternalType;

using ValLayoutSrc = decltype(
    recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
using ValLayoutDst = decltype(
    recast_layout<uint1_t, ValType>(BitLayoutDst{}));
using ValLayoutRef = decltype(
    recast_layout<uint1_t, ValType>(BitLayoutRef{}));
```

因此，Copy Atom表达的是“一次最小指令中，哪些线程把哪些逻辑value从source映射到destination”。例如：

``` plain text
Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>
    一个线程，一次GMEM->SMEM搬运8个half

Copy_Atom<SM75_U32x4_LDSM_N, half_t>
    一个warp，一次SMEM->RMEM加载4个8x8 b16矩阵
    每个lane最终得到4个uint32_t，也就是8个half
```

`Copy_Atom::call()`要求source和destination都是rank-1 Tensor。当value数量满足指令要求时，它进入`copy_unpack`，把逻辑元素Tensor重新解释为`CopyOperation`声明的`SRegisters`和`DRegisters`类型，最终调用`CopyOperation::copy()`。这与`MMA_Atom::call()`先检查fragment、再unpack并调用PTX包装的结构基本一致。

# TiledCopy

一个Copy Atom只描述一次最小copy。GEMM中通常需要整个CTA协作搬运一个二维tile，因此还要定义：

1. `ThrLayout` 线程如何排列在tile上。
2. `ValLayout` 每个线程的多个value如何排列。

``` C++
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_8,_4>, Stride<_4,_1>>{}, // thread layout
    Layout<Shape< _1,_8>>{});               // value layout
```
{{< image src="tiledcopy.png" alt="tiledcopy" caption="这里的S和D layout完全一样，只展示了S">}}

这个例子中的thread layout把32个线程排成(8,4):(4,1),value layout表示每个线程在第一维取1个位置、第二维取连续8个half。两者通过`raked_product`组合后覆盖一个`8 x 32`的copy tile。因为每线程的8个half恰好是128 bits，所以每个线程执行一次128-bit `cp.async`。

# ThrCopy

进入kernel后，需要从`TiledCopy`中取出当前线程的视图：

``` C++
ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);

Tensor tAgA = thr_copy_a.partition_S(gA);
Tensor tAsA = thr_copy_a.partition_D(sA);
```

这里同样没有发生数据搬运：

1. `get_slice(threadIdx.x)`只把当前thread id保存在`ThrCopy`中。
2. `partition_S(gA)`使用Copy Atom的source映射切分`gA`。
3. `partition_D(sA)`使用Copy Atom的destination映射切分`sA`。

为什么不能对source和destination使用完全相同的普通partition？对于`cp.async`，两侧确实接近一一对应；但`ldmatrix`的source端是少量lane提供行地址，destination端却是32个lane各自接收fragment，两侧TV Layout并不相同。`partition_S/D`通过共同的`RefLayout`协调这两种视图，确保一次Atom调用的输入和输出正确对应。

假设`gA`表示`(128,64,K_TILE)`，`sA`表示三阶段pipeline中的`(128,64,3)`，前面的`16 x 64` TiledCopy会得到类似的线程私有视图：

``` plain text
tAgA: (CPY, CPY_M, CPY_K, K_TILE)
tAsA: (CPY, CPY_M, CPY_K, PIPE)
```

其中`CPY`是一次Copy Atom内的value mode，这里size为8；`CPY_M`的size为8，因为`16 x 64` copy tile还要沿M方向重复8次才能覆盖完整的`128 x 64` CTA tile；`CPY_K`在这个配置中size为1。

真正执行一个global K tile到一个shared-memory stage的copy时，再固定最外层坐标：

``` C++
copy(copyA,
     tAgA(_,_,_,k_tile),
     tAsA(_,_,_,k_pipe));
```

这里传给`copy`的是`copyA`而不是`thr_copy_a`。CuTe甚至显式删除了直接接收`ThrCopy`的`copy`重载，强调`ThrCopy`的职责只是partition。线程映射已经体现在`tAgA/tAsA`的Layout中，执行策略仍然由`TiledCopy`继承的Copy Atom提供。

## retile_S和retile_D

`ThrCopy`还提供`retile_S/D`。它们用于已有Tensor已经按照另一套逻辑布局完成线程分区、但底层Engine中的同一批数据需要被当前Copy Atom读写的场景。

典型例子是SMEM到RMEM的`ldmatrix`：

``` C++
TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);

Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
```

`tCrA`是MMA视角下的register fragment；`tXrA`是LDSM Copy视角下对同一批register的视图。`retile_D`不会分配第二份register，也不会重排数据，它只保留原Tensor的Engine并构造一个新的Layout。之后`ldmatrix`按照`tXrA`写入，`gemm`再按照`tCrA`读取同一批寄存器。

<!-- 待补图5：retile_D双视图图。中间画同一组物理register，左侧用tXrA/Copy Layout解释，右侧用tCrA/MMA Layout解释；注明retile不搬数据。 -->

# cute::copy

前面的对象都在描述映射，`cute::copy`才是真正执行copy的入口。指定Copy Atom时，其核心dispatch可以简化成：

``` C++
template <class... CopyArgs, class SrcTensor, class DstTensor>
void copy(Copy_Atom<CopyArgs...> const& copy_atom,
          SrcTensor const& src,
          DstTensor      & dst)
{
  if constexpr (rank(src) == 1) {
    copy_atom.call(src, dst);
  } else {
    // 保留第0个Atom value mode，合并其余mode并循环
    for (int i = 0; i < size_of_rest_modes; ++i) {
      copy_atom.call(src_v(_,i), dst_v(_,i));
    }
  }
}
```

因此传入：

``` plain text
src/dst: (CPY, CPY_M, CPY_K)
```

时，`CPY`是一次底层指令处理的value mode，`copy`会遍历`CPY_M * CPY_K`个外层位置。每次循环取出一个rank-1 `CPY` fragment，进入`Copy_Atom::call()`，最后落到`cp.async`、`ldmatrix`或其他CopyOperation。

如果直接调用不带policy的`copy(src, dst)`，CuTe会根据静态Layout、共同连续区间和对齐信息尝试自动向量化；无法向量化时则退回逐元素copy。显式传入Copy Atom的价值是把要使用的硬件指令及其TV映射固定下来，而不是让通用copy自行选择。

边界tile还可以使用`copy_if`配合predicate Tensor。需要注意，predicate不只是host侧少启动几个block：它必须和每个线程实际访问的逻辑元素对应。`cp.async`另有zero-fill Traits，可以让predicate为false的source不被读取，并把对应shared-memory目标补零。完整的边界处理取决于kernel如何构造identity Tensor和predicate，将在GEMM walk through中再展开。

# GMEM到SMEM

现在把前面的层次放回实际GEMM。host侧或kernel模板配置中先定义Copy Atom和CTA级TiledCopy：

``` C++
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_16,_8>, Stride<_8,_1>>{},
    Layout<Shape< _1,_8>>{});
```

进入kernel后，`gA`和`sA`分别是global-memory CTA tile与shared-memory pipeline Tensor：

``` C++
ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);
Tensor tAsA = thr_copy_a.partition_D(sA);

copy(copyA,
     tAgA(_,_,_,k_tile_next),
     tAsA(_,_,_,smem_pipe_write));
cp_async_fence();
```

这段代码的层次可以读成：

``` plain text
Copy Atom:      每条指令搬8个half
TiledCopy:      128个线程协作覆盖16x64
ThrCopy:        固定当前thread id并切出其视图
Tensor slicing: 固定当前global K tile和SMEM pipeline stage
cute::copy:     沿外层mode循环并发出cp.async
```

## Shared Memory Swizzle

为了让GMEM侧保持连续的128-bit访问，同时减少后续`ldmatrix`读取shared memory时的bank conflict，实际GEMM通常不会为`sA/sB`使用简单的线性Layout，而会加入swizzle：

``` C++
auto swizzle_atom = composition(
    Swizzle<3,3,3>{},
    Layout<Shape <_8,Shape <_8,_8>>,
           Stride<_8,Stride<_1,_64>>>{});

auto sA_layout = tile_to_shape(
    swizzle_atom, make_shape(_128{}, _64{}, _3{}));
auto sB_layout = tile_to_shape(
    swizzle_atom, make_shape(_128{}, _64{}, _3{}));
```

Swizzle改变的是逻辑坐标到shared-memory offset的映射，不改变Tensor的逻辑shape。`partition_D(sA)`会在这套swizzled Layout上计算每个线程的目标地址；之后`partition_S(sA)`又会为`ldmatrix`计算相应的source地址。因此两级copy仍然通过同一个逻辑`(M,K)`坐标衔接，不需要程序员手写异或地址公式。

这里不能只看到“swizzle可以消除bank conflict”就机械套用同一组参数。swizzle必须同时满足GMEM vector width、shared-memory对齐、LDSM寻址方式和目标MMA tile。上面的配置适用于本文后续使用的K-major half tile，换数据类型或tile shape时需要重新验证。

# SMEM到RMEM

SMEM到RMEM时，我们已经有一个`TiledMMA`，也已经用`ThrMMA`创建了MMA所需的register fragment。此时最重要的约束不是重新设计一个任意copy tile，而是让LDSM的结果与MMA fragment完全一致。因此CuTe提供了`make_tiled_copy_A/B`：

``` C++
Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_a;

TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);

Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
```

`make_tiled_copy_A`直接取`mma.get_layoutA_TV()`作为TiledCopy的参考TV映射，并使用MMA的M/K tile shape作为tiler。B侧则使用`mma.get_layoutB_TV()`和N/K shape。这样构造出来的copy不是“看起来与MMA大小相同”，而是在类型层面沿用MMA operand的Thread-Value覆盖关系。

真正加载一个K block时：

``` C++
copy(s2r_atom_a,
     tXsA(_,_,k_block),
     tXrA(_,_,k_block));
```

每个warp中的lane共同执行`ldmatrix.x4`。shared-memory source侧由LDSM Traits选出需要提供的地址，destination侧写入`tXrA`指向的register。随后的MMA从`tCrA`读取同一批register：

``` C++
gemm(mma,
     tCrA(_,_,k_block),
     tCrB(_,_,k_block),
     tCrC);
```

于是整个数据流中，以下操作会真正移动或计算数据：

``` plain text
copy(copyA, ...)       GMEM -> SMEM，发出cp.async
copy(s2r_atom_a, ...)  SMEM -> RMEM，执行ldmatrix
gemm(mma, ...)         RMEM -> accumulator，执行mma.sync
```

而下面这些都只创建或改变Tensor视图：

``` plain text
get_slice
partition_S / partition_D
retile_S / retile_D
Tensor的operator()切片
```

区分“Layout变换”和“真实数据搬运”，是阅读CuTe GEMM代码最重要的习惯之一。

<!-- 待补图6：完整GMEM->SMEM->RMEM->MMA数据流。每一级分别标出Copy Atom、TiledCopy/ThrCopy、Tensor视图名，以及cp.async fence/wait和CTA barrier的位置。 -->

# Wrap up

CuTe Copy体系解决的并不只是把一个赋值循环包装成`copy()`。它把硬件指令的寄存器接口、source/destination两侧不同的Thread-Value映射、CTA协作tile、线程私有视图以及最终dispatch统一在一套类型系统中。

本文最重要的几个结论是：

1. `CopyOperation`包装实际指令，`Copy_Traits`用bit级Layout描述指令语义，`Copy_Atom`再把它解释为逻辑元素。
2. `TiledCopy`描述多个线程如何覆盖更大的copy tile，`ThrCopy`只负责为某个线程partition或retile Tensor。
3. `get_slice`、`partition_*`和`retile_*`都不搬数据，`cute::copy`才会最终调用底层CopyOperation。
4. `cp.async`是一线程一次4/8/16-byte的GMEM到SMEM异步copy；`L2::128B`只是prefetch hint。
5. `ldmatrix`是warp级collective load，source地址提供方式和destination fragment分发方式不同，所以Copy Atom必须区分SrcLayout和DstLayout。
6. `make_tiled_copy_A/B`和`retile_D`把LDSM Copy映射与已有MMA fragment衔接起来，但不会制造额外的数据重排。

下一篇将以一个完整的SM80 HGEMM为例，把Tensor、Layout Algebra、MMA Atom和本文的Copy Atom串起来，逐层追踪数据如何经过GMEM、SMEM和RMEM，最终进入Tensor Core。

# Reference

https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html

https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix

https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async
