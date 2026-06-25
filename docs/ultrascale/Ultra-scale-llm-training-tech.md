# Paradigms of Parallelism 并行范式

## Introduction

随着深度学习的发展，对并行训练的需求日益增长。模型和数据集规模持续扩大，单 GPU 训练的时间成本已难以接受。本节将简要概述现有的并行训练方法。

## Data Parallel

数据并行因其简单性成为最常见的并行形式。在数据并行训练中，数据集被分割为若干分片，每个分片分配给一个设备。这相当于沿批次维度并行化训练过程。每个设备持有完整的模型副本，并在分配的数据分片上进行训练。反向传播后，模型梯度通过全归约（All-Reduce）操作进行聚合，确保不同设备上的模型参数保持同步。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/WSAensMqjwHdOlR.png" alt="img" style="zoom:33%;" />
</div>

> 数据并行示意图

数据并行通过并行处理数据显著减少训练时间，可扩展性取决于可用 GPU 数量。但同步各 GPU 计算结果可能带来额外的通信开销。

## Model Parallel

数据并行训练中，每个 GPU 都持有整个模型权重的副本，这带来了冗余问题。模型并行是另一种并行范式，其核心思想是将模型拆分并分布到多个设备上。

模型并行主要包含两种类型：张量并行和流水线并行。张量并行在矩阵乘法等运算内部实现并行计算；流水线并行在层间实现并行计算。换言之，张量并行可视为层内并行，流水线并行可视为层间并行。

模型并行是训练超出单个 GPU 内存容量的大型模型的有效策略。但由于同一时间仅有一个 GPU 处于活跃状态，会导致 GPU 利用率不均衡。此外，GPU 间的结果传递会引入通信开销，可能成为性能瓶颈。

### Tensor Parallel

张量并行训练将张量沿特定维度分割为 $N$ 个分块，每个设备仅持有整个张量的 $1/N$ 部分，同时不影响计算图的正确性。这需要额外的通信来确保结果正确。

以通用矩阵乘法 $C = AB$ 为例。假设将 $B$ 沿列维度分割为 $[B_0, B_1, B_2, \ldots, B_n]$，每个设备持有一列。然后在各设备上将 $A$ 与 $B$ 的对应列相乘，得到 $[AB_0, AB_1, AB_2, \ldots, AB_n]$。此时每个设备仅持有部分结果，例如设备 0 持有 $AB_0$。为确保结果正确，需要通过全收集（All-Gather）操作汇总这些部分结果，并沿列维度拼接张量。通过这种方式，可在设备间分布张量的同时保证计算流程的正确性。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/2ZwyPDvXANW4tMG.png" alt="img" style="zoom:33%;" />
</div>

> 张量并行示意图

### Pipeline Parallel

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/at3eDv7kKBusxbd.png" alt="img" style="zoom:33%;" />
</div>

> 流水线并行示意图

流水线并行的核心思想是将模型按层分割为若干块，每块分配给不同设备。前向传播时，每个设备将中间激活值传递给下一阶段；反向传播时，各设备将输入张量的梯度回传给前一流水线阶段。这种机制使设备能够并行计算，从而提升训练吞吐量。

流水线并行的缺点在于会产生气泡时间（Bubble Time）——部分设备处于计算等待状态，导致计算资源浪费。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/sDNq51PS3Gxbw7F.png" alt="img" />
</div>

> Source: [GPipe](https://arxiv.org/abs/1811.06965)

## Sequence Parallelism

序列并行是一种沿序列维度进行划分的并行策略，适用于训练长文本序列场景。成熟的序列并行方法包括 Megatron 序列并行、DeepSpeed-Ulysses 序列并行以及 Ring Attention 序列并行。

### Megatron SP

Megatron 序列并行构建于张量并行基础之上。在模型并行的每个 GPU 上，样本数据保持独立且被复制。对于无法应用张量并行的部分（如 `LayerNorm` 等非线性运算），可沿序列维度将样本数据分割为多个片段，由各 GPU 分别计算。对于注意力机制和 MLP 等需要聚合激活值的线性运算部分，则采用张量并行处理。

这种方案在模型分区时能进一步降低激活值内存占用。需注意，此序列并行方法必须与张量并行配合使用。

### DeepSpeed-Ulysses

DeepSpeed-Ulysses 将样本沿序列维度进行分割，并采用全交换通信操作（All-to-All），使每个 GPU 能获取完整序列但仅计算注意力头的非重叠子集，从而实现序列并行。该方法支持完全通用的注意力机制，可同时处理稠密与稀疏注意力。

All-to-All 是一种完整的数据交换操作，类似于分布式转置运算。在注意力计算前，样本沿序列维度分割，每个设备仅持有 $N/P$ 长度的序列片段。经过 All-to-All 操作后，QKV 子部分的形状转变为 $[N, d/P]$，确保注意力计算时能考量整体序列信息。

### Ring Attention

环形注意力（Ring Attention）在概念上与 Flash Attention 类似。每个 GPU 仅计算局部注意力，最终通过归约运算汇总各注意力块以得到全局注意力。

该算法沿序列维度将输入切分为多个分块，由不同 GPU 分别处理。其核心采用"环形通信"策略：通过点对点通信在 GPU 间传递键值子块进行迭代计算，从而支持超长文本的多 GPU 训练。各处理器仅与前驱和后继节点交换信息，形成环形网络拓扑。这种设计无需全局同步即可高效传递中间计算结果，显著降低了通信开销。

## Optimizer-Level Parallel

另一种范式在优化器层面发挥作用。当前该范式最著名的方法是 ZeRO（[零冗余优化器](https://arxiv.org/abs/1910.02054)，Zero Redundancy Optimizer）。ZeRO 将参数、梯度和优化器状态分区到不同数据并行进程，显著提升内存使用效率。它包含三个优化阶段：

- 阶段 1：对优化器状态进行分区
- 阶段 2：对优化器状态及梯度进行分区
- 阶段 3：对优化器状态、梯度和参数全部分区

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png" alt="img" />
</div>

## Parallelism on Heterogeneous System

上述方法通常需要大量 GPU 来训练大型模型。然而 CPU 内存容量远大于 GPU——在典型服务器上，CPU 可轻松配备数百 GB 内存，而单个 GPU 通常仅有 16 或 32 GB 内存。这一现状促使业界思考：为何不利用 CPU 内存进行分布式训练？

近期进展依赖 CPU 甚至 NVMe 磁盘来训练大型模型。核心思路是在张量未被使用时将其卸载回 CPU 内存或 NVMe 磁盘。通过采用异构系统架构，可以在单台机器上容纳超大规模模型。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/qLHD5lk97hXQdbv.png" alt="img" />
</div>

> 异构系统示意图

### Hybrid Parallelism 混合并行

可以组合多种并行方法来实现更大的内存节省，并更高效地训练具有数十亿参数的模型。
