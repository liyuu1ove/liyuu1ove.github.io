+++
title = '集合通讯学习0-分布式计算与硬件网络'
date = '2026-06-11T16:26:55+08:00'
draft = false
description = '简单介绍分布式计算的概念和现代数据中心硬件网络'
readingTimeText = '阅读此文大概需要24分钟'
tags = ['Distributed System','RoCE','IB','NVLink']
categories = ['Technical Blog']
+++

# 写在前面

在讨论集合通讯之前，我们先把视角从单张GPU拉远一点。

分布式计算并不是机器学习时代才出现的概念。在更早的互联网服务和传统分布式系统中，机器之间的通信更多是围绕“请求-响应”展开的。比如一个Web服务调用用户服务，一个订单服务调用库存服务，或者一个计算节点向存储节点请求一段数据。这类通信通常可以抽象成RPC（Remote Procedure Call）。

``` plain text
Client -> RPC Request  -> Server
Client <- RPC Response <- Server
```

RPC的核心目标是把远端调用包装得像本地函数调用一样。开发者关心的是接口、序列化、超时、重试、负载均衡和服务治理。它的通信模式通常是点对点的：一个客户端请求一个服务端，或者少量服务之间互相调用。即使底层系统很复杂，单次通信的语义仍然比较像“我问你答”。

而机器学习训练中的通信有些不同。一个训练任务不是几台服务之间偶尔传几条业务消息，而是大量计算设备在每一个step里反复交换张量。尤其是进入大模型时代之后，Scaling Law让大家逐渐形成了一个朴素但昂贵的共识：在数据、模型和计算量合理匹配的前提下，更大的模型、更大的数据集和更多的训练算力通常能带来更强的模型能力。

这件事直接改变了训练系统的形态。早期很多模型可以在单张GPU上训练，后来变成单机多卡，再后来变成多机多卡，最后演变成上千甚至上万张GPU协同训练。模型参数量从几千万、几亿增长到数百亿、数千亿甚至更大时，单张GPU不再只是“训练慢一点”，而是根本放不下完整模型、优化器状态和中间激活。

![Parameters of LLMs](estimated-capability-density-for-open-source-base-LLMs.png)

这张图片上的参数量还停留在几B级别，deepseek-v4-pro已经有1.6T参数了。

但是当模型参数、训练数据、batch size或者吞吐需求超过单张GPU能够承载的范围时，这个过程就会被拆到多张GPU、多个节点甚至整个机群上执行。此时问题就从“如何让一个kernel跑得快”，变成了“如何让一群计算设备像一个整体一样工作”。训练系统里的通信也从传统RPC式的请求响应，变成了高频、规则、带宽敏感的张量搬运。

集合通讯（Collective Communication）就是这个问题中最核心的一块。它关心的不是某一个rank向另一个rank发送一条消息，而是一组rank如何协同完成广播、归约、聚合、切分等通信模式。比如数据并行训练中最常见的梯度同步，本质上就是一次`AllReduce`。

这也是为什么近几年大模型公司越来越重视训练infra。DeepSeek被称为宇宙最强infra厂，并不是因为它发布了一个强模型，而是把模型结构、低精度训练和通信系统放在一起做了很强的协同优化。以DeepSeek-V3为例，它在H800集群上训练大规模MoE模型，同时使用FP8混合精度训练、面向MoE的通信优化、计算通信重叠以及更贴合硬件约束的网络组织方式，把硬件纳入模型结构设计。这类工作说明，大模型训练已经不是单纯堆GPU的问题，而是模型、算法、编译、通信库和数据中心网络共同参与的系统工程。

这篇文章作为集合通讯的第一篇，先介绍集合通讯背后的分布式计算语境和硬件网络。因为集合通讯的性能很少只由“调用了哪个函数”决定，它通常同时受算法拓扑、链路带宽、网络延迟、PCIe/NVLink/IB/RoCE等硬件路径影响。

## Strong-Scaling and Weak-Scaling

讨论分布式性能时，经常会看到两个概念：Strong Scaling和Weak Scaling。

**Strong Scaling**指固定问题规模，增加计算资源，观察总耗时是否下降。

``` plain text
固定训练数据量 / 固定模型 / 固定global batch size

1 GPU  -> 100 min
2 GPU  ->  55 min
4 GPU  ->  32 min
8 GPU  ->  25 min
```

理想情况下，GPU数量翻倍，训练时间减半。但现实中很难做到。因为每张GPU承担的计算量变少后，通信、同步、调度等开销占比会越来越高。当GPU数量继续增加时，新增GPU带来的计算收益可能被通信开销吃掉。

**Weak Scaling**指每个计算设备处理的问题规模不变，随着设备数量增加，总问题规模也一起增加。

``` plain text
1 GPU  -> local batch = 32, global batch = 32
2 GPU  -> local batch = 32, global batch = 64
4 GPU  -> local batch = 32, global batch = 128
8 GPU  -> local batch = 32, global batch = 256
```

Weak Scaling更关心系统是否能维持单位设备的效率。它在大规模训练里非常常见，因为我们增加GPU数量时，往往也希望增加global batch size或者训练更大的模型。

这两个概念对应的瓶颈并不完全一样。Strong Scaling中，每张GPU的计算量变小，通信延迟和同步开销更容易成为主导。Weak Scaling中，单卡计算量不变，但是总通信规模和网络拥塞会随着集群变大而变复杂。

## Scale-Up and Scale-Out

从硬件组织方式看，扩展计算资源有两条路线：Scale-Up和Scale-Out。

**Scale-Up**通常指在一个节点内部堆更多、更强、更紧密连接的设备。例如一台服务器内放置多张GPU，并用NVLink、NVSwitch或PCIe把它们连接起来。它的特点是节点内带宽高、延迟低、拓扑相对可控。比如说每个做机器学习的人的梦中情机NVL72，就是如此的一个超节点。


**Scale-Out**则是把多个节点通过网络连接起来，形成更大的集群。节点之间通常依赖InfiniBand或RoCE网卡通信。它的特点是容量扩展能力强，但是通信路径更长，网络层级更多，也更容易遇到拥塞、路由、交换机带宽收敛等问题。

![scaling](scale-up-vs-scale-out-011675.png)

在真实训练系统里，Scale-Up和Scale-Out通常同时存在。比如一个节点内有8张GPU，节点内走NVLink/NVSwitch；多个节点之间通过IB或RoCE连接。集合通讯库要做的事情，就是在这种分层硬件上选择合适的通信算法，让数据尽可能沿着高带宽、低延迟的路径流动。

# 大规模网络

理解完通信在分布式计算为什么如此重要之后，我们再看通信本身如何组织。

这里可以分成两层：逻辑拓扑和硬件连接。

逻辑拓扑描述算法层面上rank之间如何传数据，比如Ring和Tree。硬件连接描述真实机器里数据实际会经过哪些链路，比如PCIe、NVLink、RoCE和InfiniBand。高性能集合通讯的核心，就是让逻辑拓扑尽量贴合硬件连接。

## 逻辑拓扑

逻辑拓扑并不一定等于物理拓扑。即使机器之间实际是全连接（Full Mesh），集合通讯算法也可以在rank之间构造一个Ring或者Tree。

![Topology](An-overview-of-basic-types-of-network-topologies-including-the-A-chain-or-line-B.webp)

### Ring

Ring是集合通讯里最经典的逻辑拓扑之一。所有rank被组织成一个环，每个rank只和前驱、后继通信。


Ring的优点是每个rank的通信模式非常均匀。以`AllReduce`为例，Ring算法通常可以拆成`ReduceScatter + AllGather`两个阶段。在每一步中，每个rank向下一个rank发送一个数据块，同时从上一个rank接收一个数据块。

假设有`p`个rank，总数据大小为`N`，每个rank每一步只发送约`N / p`的数据块。经过多轮传递之后，所有数据完成归约并重新分发。Ring的一个直观优点是可以较好地利用全双工链路带宽，尤其适合大消息。

但是Ring也有代价。它的通信轮数和rank数量相关，rank越多，需要的step越多。当消息很小或者集群非常大时，`steps * alpha`中的延迟开销会变得明显。

有关Ring AllReduce的数值计算将会在后面的其他文章详细说明。

### Tree

Tree是另一类常见拓扑。rank被组织成一棵树，数据沿着树边向上汇聚或向下广播。

以Reduce为例，叶子节点先把数据发送给父节点，父节点完成局部归约后继续向上发送，最终根节点得到完整结果。Broadcast则反过来，根节点把数据逐层向下分发。

Tree的优势是通信轮数通常是`O(log p)`，比Ring的线性轮数更少。因此在小消息、延迟敏感场景中，Tree往往更有吸引力。

Tree的问题是流量不一定均匀。越靠近根节点，承担的聚合压力越大。如果树的构造没有贴合硬件拓扑，根附近链路可能成为瓶颈。为了解决这个问题，实际系统中还会使用双树、多树、分层树等变体，把流量摊到更多链路上。

Ring和Tree没有绝对好坏。更准确地说，它们分别偏向不同的性能目标。集合通讯库通常会根据消息大小、rank数量、硬件拓扑和实际测量结果选择不同算法。

现代数据中心节点间连接一般使用Tree，节点内GPU连接一般是Full Mesh(NVLink)/2D Lotus(Google Fabric for TPU)，但是在通信组织上可能会使用Ring。

## 硬件连接

逻辑拓扑最终要落在硬件链路上执行。对于GPU集群来说，我们可以先按节点边界分成两类：Intra-Node和Inter-Node。

### Intra-Node（节点内）

节点内通信发生在同一台服务器的多张GPU之间。它通常比跨节点通信更快，但并不意味着没有拓扑差异。

一台多GPU服务器里，GPU之间可能通过PCIe Switch连接，也可能通过NVLink/NVSwitch连接。CPU、NUMA、网卡的位置也会影响数据路径。比如某张GPU离网卡更近，另一张GPU则需要跨PCIe Switch或跨CPU socket才能访问网卡。

``` plain text
GPU -> PCIe Switch -> CPU
GPU -> NVLink      -> GPU
GPU -> PCIe Switch -> NIC -> Network
```

集合通讯库通常会先探测系统拓扑，再决定rank如何排序、环如何构造、树如何分层，以及哪些GPU负责跨节点收发。

#### PCIe

PCIe（Peripheral Component Interconnect Express）是通用的高速外设总线。GPU、网卡、NVMe等设备通常都通过PCIe接入CPU或PCIe Switch。

PCIe的优势是通用、成熟、生态广。它的问题是，PCIe本质上是围绕CPU和外设构建的互连方式，并不是专门为GPU之间的密集通信设计的。多GPU之间如果只通过PCIe通信，数据路径可能会经过PCIe Switch、CPU Root Complex，甚至跨NUMA节点。

``` plain text
GPU0
  |
PCIe Switch --- CPU
  |
GPU1
```

在这种情况下，GPU0和GPU1虽然在同一台机器里，但它们之间的通信带宽和延迟仍然受PCIe拓扑限制。如果多张GPU同时通信，还可能共享同一个PCIe Switch的上行带宽。

不过PCIe仍然非常重要。即使在配备NVLink的系统中，GPU访问网卡、CPU控制GPU、主机内存和设备内存之间的数据搬运，通常都绕不开PCIe路径。因此理解PCIe拓扑对于理解跨节点通信也很关键。

#### NVLink

NVLink是NVIDIA为GPU之间高速互连设计的链路。相比PCIe，NVLink更贴近GPU通信需求，通常提供更高带宽和更低延迟。多条NVLink可以组成更复杂的节点内拓扑，而NVSwitch则进一步提供类似交换网络的GPU互连能力。

在没有NVSwitch的系统中，GPU之间可能是部分互联。某些GPU对之间有直接NVLink，另一些GPU对之间则需要经过中间GPU转发或退回PCIe。此时rank顺序和通信环构造会显著影响性能。

``` plain text
GPU0 === GPU1
 ||      ||
GPU2 === GPU3
```

在带NVSwitch的系统中，多张GPU可以通过交换芯片获得更均匀的互联。对于集合通讯来说，这意味着节点内AllReduce、AllGather等操作更容易获得稳定带宽，也更适合构造多条并行通信路径。

NVLink并不替代跨节点网络。它主要解决节点内GPU之间的高性能通信；一旦数据需要离开服务器，仍然需要通过网卡进入RoCE或InfiniBand网络。

### Inter-Node（跨节点）

跨节点通信发生在不同服务器之间。此时数据通常需要从GPU显存经过网卡发送到网络，再到另一台机器的网卡和GPU。

如果没有GPU Direct RDMA，数据路径可能需要CPU和主机内存参与：

``` plain text
GPU memory -> CPU memory -> NIC -> Network -> NIC -> CPU memory -> GPU memory
```

GPU Direct RDMA允许网卡直接读写GPU显存，减少CPU参与和额外拷贝：

``` plain text
GPU memory -> NIC -> Network -> NIC -> GPU memory
```

这对大规模训练非常关键。因为梯度、参数分片、激活值等数据量很大，如果每次跨节点通信都要绕一圈CPU内存，延迟和带宽都会受到明显影响。

#### RoCE

RoCE（RDMA over Converged Ethernet）是在以太网上承载RDMA语义的技术。它的目标是在保留以太网生态和交换设备优势的同时，提供低延迟、低CPU开销的数据传输能力。

RDMA的关键点是“远端直接内存访问”。简单来说，一台机器的网卡可以在授权范围内直接读写另一台机器的内存区域，而不需要远端CPU频繁参与数据拷贝。

``` plain text
Sender GPU memory -> NIC == Ethernet Fabric == NIC -> Receiver GPU memory
```

RoCE的优势是可以利用以太网生态，部署和管理方式更接近传统数据中心网络。但它对网络配置比较敏感。为了让RDMA在以太网上稳定低延迟运行，通常需要处理PFC、ECN、拥塞控制、交换机buffer等问题。配置不当时，性能可能会因为丢包、拥塞或队头阻塞出现明显抖动。

因此，RoCE不是“普通以太网换个API就能变成高性能训练网络”。它更像是在以太网之上构建一个需要精细调优的低延迟传输层。

#### InfiniBand

InfiniBand是一套面向高性能计算的网络技术，从设计上就服务于低延迟、高带宽和RDMA通信。它在HPC和大规模AI训练集群中非常常见。

相比RoCE，InfiniBand的优势在于网络语义、拥塞控制、子网管理等能力更加专用，整体软硬件栈也更围绕高性能通信设计。对于追求稳定通信性能的大规模训练集群，InfiniBand通常是非常强的选择。

``` plain text
GPU -> NIC(HCA) -> InfiniBand Switch Fabric -> NIC(HCA) -> GPU
```

这里的网卡在IB语境中常被称为HCA（Host Channel Adapter）。在使用GPU Direct RDMA时，HCA可以直接和GPU显存交互，从而减少CPU内存中转。

InfiniBand的代价也很直观。它是一套专用网络，需要相应的网卡、交换机、线缆、管理工具和运维经验。对于已经有成熟以太网基础设施的数据中心，RoCE可能更容易融入现有体系，对于从一开始就以AI/HPC训练为目标建设的集群，InfiniBand则常常能提供更稳定的性能边界。

# 写在最后

本文主要介绍了集合通讯出现的背景：为什么单卡计算会扩展到分布式计算，为什么扩展之后通信会成为瓶颈，以及现代GPU集群中常见的逻辑拓扑和硬件连接。

这里最重要的直觉是：集合通讯性能不是单一API的结果，而是算法和硬件共同决定的结果。Ring、Tree这些逻辑拓扑要映射到PCIe、NVLink、RoCE、InfiniBand这些真实链路上，最终表现才会落到我们看到的训练吞吐和GPU利用率上。

下一篇文章将进入NCCL和集合通讯原语，具体介绍常用的集合通讯库NCCL，然后介绍Broadcast、Reduce、AllReduce、ReduceScatter、AllGather等集合通讯操作。

# Reference

https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html

https://docs.nvidia.com/cuda/gpudirect-rdma/

https://network.nvidia.com/products/infiniband/

https://arxiv.org/abs/2412.19437
