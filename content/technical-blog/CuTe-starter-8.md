+++
title = 'CuTe学习8-SM90 WGMMA'
date = '2026-08-27T10:29:37+08:00'
draft = true
description = '基于CuTe封装讲解SM90 WGMMA'
readingTimeText = '阅读此文大概需要 分钟'
tags = []
categories = ['Technical Blog']
+++


# WGMMA
WGMMA（Warp-Group Matrix Multiply-Accumulate，线程束组矩阵乘加是NVIDIA在Hopper架构中引入的第四代Tensor Core核心PTX指令。其新特性为更宽的指令形状，Shared Memory操作数和异步计算。

{{< image src="wgmma.png" alt="wgmma" maxWidth="500px" caption="SM90_64x64x16_F16F16F16_SS">}}

我们观察wgmma PTX的新形态和其在CuTe中的命名。
```CPP
SM90_64x64x16_F16F16F16_SS
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
```
- 一条wgmma指令可以处理`64x64x16`的tile
- `SS`代表矩阵A和B从SMEM直接读取
- `mma_async`表示这是一条异步指令，指令在发出之后立即退休，不阻塞后续指令

下面我们将分别介绍这三个wgmma新特性。

## Warp Group and Larger Tile
前代Ampere通常使用以一个Warp32个线程为单位的 mma.sync 指令。而Hopper引入了Warp Group概念，由4个Warp（共128个线程）共同issue指令并执行MMA。PTX中的`sync.aligned`标志要求该128个线程必须到达相同的代码执行点，并以相同的参数同时发射该指令，这个同步是由硬件进行同步的，开销极小。

由128个线程共同协作带来了诸多好处。
首先是显著增大的MMA tile，Ampere架构上FP16最宽指令只有16x8x16，即使是四个MMA沿M方向排列，也只有16x32x16,远远小于Hopper上的最大尺寸64x256x16。更大的tile带来了诸多好处。

首先是减少了SMEM的读取次数。如下图所示，计算C1和C3时，需要重复读取B1。但如果使用更大的MMA，就只需要读取一次。
{{< image src="reuse.png" alt="reuse" maxWidth="500px">}}

其次是减少了指令数量。SM内的指令发射器（Dispatch Unit）的配额是非常稀缺的，Ampere时代的MMA tile小，周期数少，所以就需要不断发射MMA以让Tensor Core保持运算。但是kernel内部计算地址搬运内存的指令同样需要发射，所以在Ampere上需要更精致的软件流水线(e.g. ping-pong buffer)以达到更好的性能。但wgmma一条指令可以计算数十倍于Ampere的MMA tile，这就减少了很多指令发射。同时wgmma一条指令会占用数十甚至上百个cycle，这使保持Tensor Core计算变得相对容易。

最后是减小了MMA间同步开销。Ampere时代的mma必须显式syncthreads以保证所有线程都不再需要SMEM里面的数据。wgmma得益于其指令宽度，只需要等待自身退休，不再需要和其他MMA进行同步。

当然，如此大的Tile也带来了巨大的寄存器压力，单条 wgmma 的 D accumulator 寄存器占用 = M × N × 4 / (4 × 128) (F32, 4 byte/elem, 128
thread/warpgroup)。

| Shape | D regs/thread | 占 thread 寄存器上限 (255) 比例 |
| --- | --- | --- |
| m64n64k16 | 32 | 13% |
| m64n128k16 | 64 | 25% |
| m64n192k16 | 96 | 38% |
| m64n256k16 | 128 | 50% |

我们可以发现，选用m64n256k16时仅累加寄存器就要占用一半的寄存器上限，所以选用MMA形状时必须要考虑寄存器压力。在工程上，如FA3的attention算子会更常用m64n128k16等小tile来为中间状态（P/logits/lse）留足足够的寄存器。GEMM因为其除了累加器之外不需要中间变量，所以会采用更激进的大tile来达到更高的吞吐。

### Warp Specialize and Warp Group
我们在介绍TMA的时候也简单提到了Warp Specialize这一新编程范式，现在详细讲一下一个CTA中角色到底是如何分配的。ProducerWarp专门负责数据搬运与地址计算。ConsumerWarp专门负责计算。

TMA仅需要一个线程发射指令，wgmma需要4个warp同时发射指令。同时我们还需要为ProducerWarp分配大量的累加寄存器。ConsumerWarp和ProducerWarp需要的硬件资源明显是不对等的。虽然 160 线程（1 个 TMA Warp + 1 个 WGMMA WG）可行，但在真实的深度学习算子中，最推荐的BlockSize依然是 128、256、384、512等128的整数倍。SM内部的4个Sub-SM是对称划分的，按128线程（Warp Group）对齐能让 SM 在调度、SMEM 分配上达到最佳硬件效率。

在CUTLASS SM90 warpspecialized gemm中，有如下的定义。
```CPP
enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
};
enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3
};
```
可以看到，为ProducerWarp分配了一整个WarpGroup，ProducerWarp内部又进行了细分，其中Mainloop负责发射TMA指令。


## Shared Memory Operand
为了减轻寄存器压力，wgmma可以从SMEM直接取操作数，我们传入PTX指令的是一个64-bit描述符，用于描述操作数的地址和形状。

| Bits | 字段 | 说明 |
| --- | --- | --- |
| 13:0 (14b) | SMEM start address | shared memory 起始地址 >> 4 (16-byte 对齐) |
| 29:16 (14b) | Leading-dim byte offset | K 方向相邻 group 之间的字节距离 >> 4 |
| 45:32 (14b) | Stride byte offset | M (或 N) 方向 group 跨度 >> 4 |
| 48:46 (3b) | Matrix base offset | 对齐微调 (一般 0) |
| 51:49 (3b) | Swizzle mode | 0 = none, 1 = 32B swizzle, 2 = 64B, 3 = 128B |
| 其余 | reserved | 未使用 / 必须为 0 |

*Source from RedNote [sylar]()*

其中swizzle由硬件自动处理，这和TMA的自动swizzle是一致的。

## Async pipeline
WGMMA是一条异步MMA指令，异步语义意味着指令退休不代表D寄存器可读。和Ampere类似，Hopper wgmma也有commit_group和wait_group两条同步指令，也仅对当前WarpGroup可见。在ConsumerWarp内部，为了掩盖Tensor Core的计算延迟，通常会连续发射2-4个WGMMA指令填充pipeline，这些wgmma之间的同步就要依靠前文两条同步指令。

在 GPU 硬件层面，不同的硬件模块访问内存的物理通道和硬件缓存是独立的。这就导致了Hopper需要维护两个硬件队列。Generic Proxy指常规CUDA线程执行单元（ALU/Load-Store Unit）的同步队列。比如LDG/LDS/STS/CUDA Core ALU操作。Async Proxy指SM内部专门的硬件异步加速引擎。在 Hopper 中，Async Proxy包括TMA，WGMMA和mbarrier。

内存模型分离导致不同操作之间缺乏隐式一致性。CUDA Programming Guide §3.2.2.3.1 明确指出wgmma.mma_async.*在 async proxy 中执行, 而普通内存操作或者MMA在generic proxy 中执行。两个 proxy各自维护内存顺序, 跨proxy的同地址访问不保证顺序。
