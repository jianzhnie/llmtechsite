# MindSpeed

##  MindSpeed 介绍

训练加速库，提供丰富加速算法和模型，使能训练全流程加速。基于昇腾硬件的分布式训练加速套件，提供多维度加速算法，支持主流大语言、多模态大模型。面向大模型全流程加速，降低开发门槛，获得最优开箱性能。

MindSpeed 是一款专为昇腾平台打造的高性能加速库，涵盖了 MindSpeed Core 亲和加速模块、MindSpeed LLM 套件、MindSpeed MM 套件以及 MindSpeed RL 套件这四个重要组成部分。

MindSpeed 凭借其卓越的性能表现与深度优化的算法架构，为客户在 AI 领域实现大模型训练提供了强有力的支持。借助 MindSpeed，用户能够充分挖掘并利用昇腾设备的高性能计算能力，加速大模型训练过程。

### 总体架构

**图1** MindSpeed整体架构

![img](https://www.hiascend.com/doc_center/source/zh/MindSpeed/220/productoverview/figure/zh-cn_image_0000002501882927.png)

表1 组件介绍

展开

| 组件名称                   | 说明                                                         |
| -------------------------- | ------------------------------------------------------------ |
| MindSpeed Core 亲和加速模块 | 基于昇腾设备的大模型加速模块，提供计算、内存、通信、并行四个维度的优化，支持长序列、MoE 等场景加速特性。 |
| MindSpeed LLM 套件          | 基于昇腾生态的大语言模型套件。旨在提供端到端的大语言模型训练方案，包含分布式预训练、分布式指令微调以及对应的开发工具链，如：多模态数据预处理、权重转换、在线推理、基线评估，覆盖业内主流大语言模型。 |
| MindSpeed MM 套件           | 面向大规模分布式训练的昇腾多模态大模型套件，聚焦多模态生成、多模态理解，提供多模态大模型端到端训练流程，包含多模态数据预处理、训练微调、在线推理以及效果评估等能力，覆盖业内主流多模态大模型。 |
| MindSpeed RL 套件           | 提供超大昇腾集群训推共卡、异步流水调度、训推异构切分通信等核心加速能力。 |

## MindSpeed Core

针对昇腾设备推出的亲和加速模块。提供包括并行优化、内存优化、通信优化及计算优化的多维度加速算法，极大提高了模型训练速度，并提升了支持的模型规模。作为套件核心加速模块，为用户提供强劲的性能支持。

- 并行算法优化：支持模型并行、优化器并行、专家并行、长序列并行等多维并行策略，针对昇腾软硬件架构进行亲和优化，显著提升了集群训练的性能和效率。
- 内存资源优化：提供内存压缩、复用、内存交换，以及差异化的重计算技术，最大限度地利用内存资源，有效缓解内存瓶颈，提升训练效率。
- 通信性能优化：采用通算融合、通算掩盖等策略，配合高效的算网协同机制，大幅提高算力利用率，减少通信延迟，优化整体训练性能。
- 计算性能优化：集成高性能融合算子库，结合昇腾亲和的计算优化，充分释放昇腾算力，显著提升计算效率。
- 差异化能力支持：在长序列、权重保存、并行策略自动搜索等场景提供差异化能力。

## MindSpeed LLM

针对昇腾硬件推出的大语言模型套件。提供了开箱即用的主流开源大模型，支持[预训练](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/docs/pytorch/solutions/pretrain/pretrain.md)、[全参微调](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/docs/pytorch/solutions/finetune/instruction_finetune.md)、[低参微调](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/docs/pytorch/solutions/finetune/lora_finetune.md)等算法，完备的数据预处理、[权重转换](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/pytorch/solutions/checkpoint/checkpoint_convert.md)、[多样评估方式](https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/docs/pytorch/solutions/evaluation/evaluation_guide.md)的工具链支持。提升用户在昇腾硬件上高效、便捷地开发与使用大型语言模型的能力。

- 主流LLM大语言模型：支持Qwen3/DeepSeek/Mamba2系列等100+主流LLM模型，涵盖Dense/MoE/SSM等LLM架构，提供针对昇腾架构的高性能训练脚本，开箱即用。
- 分布式预训练：支持分布式预训练，提供数据预处理方案与包含TP/PP/DP/CP/EP在内的多维并行策略。
- 分布式指令微调：支持业界主流的全参微调/LoRA/QLoRA微调训练算法，并提供微调性能/显存优化手段。
- 模型权重转换：支持Megatron/HuggingFace格式的权重转换和LoRA微调权重的独立/合并转换。
- 在线推理与评估：支持模型分布式在线推理与公版数据的在线评估。

## MindSpeed MM

针对昇腾硬件推出的多模态模型套件。支持多模态模型开发全流程，整合先进的[数据工程能力](https://gitcode.com/ascend/MindSpeed-MM/blob/master/docs/features/bucket_reordering.md)，预置开箱即用的主流多模态大模型，涵盖图文理解、图像生成、视频生成等[多种训练策略](https://gitcode.com/ascend/MindSpeed-MM/blob/master/docs/features/lora_finetune.md) ，支持不同模态和分布式训练的分离训练和[长序列训练能力](https://gitcode.com/ascend/MindSpeed-MM/blob/master/docs/features/dit_ring_attention.md)，并配备完善的[评测能力](https://gitcode.com/ascend/MindSpeed-MM/blob/master/docs/features/vbench-evaluate.md)保证模型效果。

- 主流多模态模型：支持LLaVA/InternVL/QwenVL系列等主流多模态理解模型；支持OpenSoraPlan/CogVideoX/HunyuanVideo/Wan2.1系列等主流视频生成模型；支持SDXL/FLUX/SANA系列等主流文生图模型。提供针对昇腾架构的高性能训练脚本，开箱即用。
- 分布式训练：支持分布式全参微调，提供数据预处理方案与包含异构PP/TP/SP等多维并行策略；通过细粒度选择性重计算和Async-offload异步卸载技术，充分利用显存、H2D、D2H等异构资源，达成了超长序列性能优化。支持LoRA微调和DPO训练。
- 模型权重转换：支持Megatron/HuggingFace格式的权重转换和LoRA微调权重的独立/合并转换。
- 在线推理与评估：支持模型分布式在线推理与公版数据的在线评估。

## MindSpeed RL

针对昇腾硬件推出的强化学习开发套件。提供端到端的RL后训练解决方案，支持超大昇腾集群[训推共卡](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/features/integrated_worker.md)/分离部署模式，通过[多模型异步流水调度](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/features/data_module_design.md)机制实现算力资源高效利用，依托[训推异构切分通信](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/features/resharding.md)架构降低数据交互延迟，同时[序列合并](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/features/remove_padding.md)与[长序列并行](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/features/context_parallel.md)处理技术，解决序列计算瓶颈问题，全面提升分布式训推效率。

- 内存资源优化：支持训推共卡，训推最优并行的共卡切换，以及精细化的训推内存管理技术。
- 计算流编排优化：支持异步Replay Buffer，解耦数据依赖，使能异步训练与任务流水掩盖，大幅提升端到端吞吐。
- 负载均衡优化：支持变长序列的数据负载均衡，显著提升算力利用率和端到端的吞吐。
- 大规模RL优化：支持千亿MoE长序列的高性能训练。

# MindSpeed LLM 训练：支持多场景下的加速特性与内存优化特性

| 场景 | 加速特性名称 | 说明 |
|------|--------------|------|
| **长序列** | Ring Attention 长序列并行 | Megatron-Ring算法实现 |
| | Ulysses 长序列并行 | DeepSpeed-Ulysses算法实现 |
| | 混合长序列并行 | 结合Ring和Ulysses的长序列算法 |
| | Adaptive-CP | 序列重映射寻优 & 调度寻优 |
| **显存与计算均衡** | Noop Layer | 通过插入空操作层，均衡PP stage计算不均，有效减少流水并行空泡 |
| | Dynamic PP | 手动指定PP stage层数，均衡内存&计算 |
| **MOE** | All gather Dispatcher优化 | --moe-permutation-async-comm<br>--moe-token-dispatcher-type allgather |
| | All2All Dispatcher优化 | --moe-permutation-async-comm<br>--moe-token-dispatcher-type alltoall |
| | All gather Dispatcher EP通信隐藏优化 | 在async-comm基础上进一步掩盖EP通信 |
| | All2A Dispatcher EP通信隐藏优化 | 在async-comm基础上进一步掩盖EP通信 |
| **通算掩盖** | MC2 | TP SP场景，对线性层做通算掩盖 |
| | 梯度reduce通算掩盖 | 梯度更新与Reduce通算掩盖 |
| | 权重all-gather通算掩盖 | 前向计算与Param Gather通算掩盖 |
| **通信优化** | 高维张量并行 | 切分传统 1D-TP 通信域，提升 TP通信效率 |
| | Recompute in advance | 解除重计算与后一个stage反向计算的依赖关系，降低bubble |

# MindSpeed LLM 训练：支持多场景下的加速特性与内存优化特性

## 显存优化特性

| 显存优化特性名称 | 说明 |
|------------------|------|
| 参数副本复用 | 显存时分复用，计算参数&参数副本 |
| 分布式优化器 | 将优化器状态在 DP 域上拆分 |
| Swap Attention | 利用设备内存来存放激活值 |
| 激活函数重计算 | 对激活函数做重计算，节省内存 |
| O2 BF16 Optimizer | 使用半精度优化器参数&梯度 |

## 融合算子

| 融合算子名称 | 应用场景 |
|--------------|----------|
| Flash attention | All |
| Fused Ring Attention Update | 长序列<br>CP-Ring 算法 |
| Fused Token Permute/Unpermute | Moe All2All<br>Dispatcher Dropless |
| GMM | Moe-小专家 |
| Matmul Add | 稠密模型梯度累积 |
| Fused swiglu | All |
| Fused rotary position embedding | All |
| Fused rmsnorm | All |



# 基于短序列 8K 研发长序列

------

## 概述

在进行**长序列训练优化**之前，应首先确保在**短序列场景下（如 8K）的计算效率已达到较为理想的状态**。
通过优化短序列性能，可以为后续长序列训练打下良好的基础，避免因底层算力未充分释放而导致的性能瓶颈。

> ✅ **核心思想**：先优化“小问题”，再扩展到“大问题”。

------

## 实际案例

- **项目路径**：`examples/mcore/chatglm3/pretrain_chatglm3_6B_32K.sh`
- **目标模型**：ChatGLM3-6B
- **训练场景**：预训练任务，序列长度为 32K（长序列）

------

## 性能对比表

| 项目                 | 吞吐量 Token/s/p | MFU  |
| -------------------- | ---------------- | ---- |
| GPU A100 Megatron-LM | 4543.22357       | /    |
| NPU A2 MindSpeed-LLM | 4854.518519      | 65%  |

> 🔍 分析：
>
> - NPU A2 在吞吐量上优于 GPU A100
> - MFU 达到 65%，表明硬件利用率较高
> - 可作为长序列调优的基础参考

------

## 关键配置项及其意义

| 配置项                        | 值   | 意义                                                       |
| ----------------------------- | ---- | ---------------------------------------------------------- |
| **TP**                        | 1    | TP 在 GQA=2 的场景下使用受限，开启后会导致较低的算力利用率 |
| **PP**                        | 2    | 在内存充足的情况下用 PP 代替 TP，减少通信开销              |
| **num-query-groups**          | 2    | 社区原生设定，导致 TP 无法超过 2（影响并行策略选择）       |
| **use-fused-swiglu**          | True | 融合 SwiGLU 过程，降低激活值显存占用                       |
| **use-fused-rmsnorm**         | True | 融合 RmsNorm 过程，降低激活值显存占用                      |
| **use-distributed-optimizer** | True | 使用分布式并行优化器，降低显存消耗                         |
| **overlap-grad-reduce**       | True | 梯度更新与 Reduce 通算掩盖，提升通信效率                   |

------

## 核心优化策略总结

### ✅ 降低显存占用

- `use-fused-swiglu` + `use-fused-rmsnorm`：融合计算过程，减少中间变量存储
- `use-distributed-optimizer`：将优化器状态分布到多个设备，缓解单卡显存压力

### ✅ 提升计算效率

- `overlap-grad-reduce`：重叠梯度同步与参数更新，隐藏通信延迟
- 使用 PP 替代 TP：在内存允许时优先选择流水并行，避免张量并行带来的通信瓶颈

### ✅ 适应特定架构限制

- `num-query-groups=2`：受社区默认设置限制，需在设计时考虑其对 TP 的约束
- TP=1：GQA=2 场景下 TP 效果不佳，因此不启用

## 结论与建议

1. **优先保障短序列性能**：确保在 8K 序列长度下吞吐量和 MFU 达到预期水平。
2. **合理选择并行策略**：
   - 内存充裕时优先使用 PP 而非 TP
   - 避免在 GQA 场景下盲目启用 TP
3. **启用融合算子与分布式优化**：
   - 显著降低显存占用
   - 提高整体训练效率
4. **为长序列扩展做准备**：
   - 当短序列性能稳定后，再逐步增加序列长度至 32K 或更长

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)

# 长序列并行算法一览

------

## 概述

在训练长序列大模型（如上下文长度 > 32K）时，传统的并行策略难以应对 **Attention 计算带来的高显存占用问题**。为此，业界提出了多种 **Context Parallel（上下文并行）** 算法，旨在降低 Attention 阶段的动态显存消耗。

本节总结三种主流的 Context Parallel 算法：**RingAttention、Ulysses 和 Hybrid**，并对比其优缺点。

------

## 一、Context Parallel 总体特点

### ✅ 优点：

- 降低 Attention 计算阶段的动态显存占用
  （通过切分序列或注意力头维度）

### ❌ 缺点：

- 引入额外的通信耗时
  （需在设备间同步数据）

> 📌 核心目标：**在不牺牲太多计算效率的前提下，突破显存瓶颈**

## 二、具体算法对比

### 1. RingAttention

#### ✅ 优点：

- 切分 **S 轴**（序列长度轴），将长序列分片到不同设备
- 显著降低每个设备上的 KV Cache 和 Attention Matrix 的显存占用

#### ❌ 缺点：

- 需要频繁的 `SendRecv` 通信操作
- 当序列不够长时，通信开销占比大，性能下降明显

> 💡 适用场景：**超长序列（如 64K+）且通信带宽充足**

### 2. Ulysses

#### ✅ 优点：

- 切分 **Dim 轴**（特征维度轴），即对 Q/K/V 的 hidden dimension 进行切分
- 降低 Attention 阶段的动态显存占用，且性能更稳定

#### ❌ 缺点：

- 对 GQA（Grouped Query Attention）或 Head 数量有要求
- 若无法满足切分条件，则需额外复制数据，增加冗余

> 💡 适用场景：**支持 Dim 切分的模型结构，如标准 Transformer**

### 3. Hybrid（混合模式）

#### ✅ 优点：

- 兼具 RingAttention 和 Ulysses 的优势
- 可根据实际场景灵活切换切分方式
- 综合能力强，适应性广

#### ❌ 缺点：

- 在 Ring 或 Ulysses 的优势场景下，性能不如单一方案
- 实现复杂度更高，调试成本上升

> 💡 适用场景：**多变的训练任务、混合负载环境**

## 三、总结对比表

| 算法              | 切分维度           | 显存优化 | 通信开销 | 性能稳定性 | 适用场景             |
| ----------------- | ------------------ | -------- | -------- | ---------- | -------------------- |
| **RingAttention** | S 轴（序列长度）   | ⭐⭐⭐⭐☆    | ⭐⭐⭐⭐☆    | ⭐⭐⭐☆☆      | 超长序列、高带宽     |
| **Ulysses**       | Dim 轴（隐藏维度） | ⭐⭐⭐⭐☆    | ⭐⭐⭐☆☆    | ⭐⭐⭐⭐☆      | 标准模型、稳定训练   |
| **Hybrid**        | S + Dim 双轴       | ⭐⭐⭐⭐⭐    | ⭐⭐⭐⭐☆    | ⭐⭐⭐⭐☆      | 多样化任务、综合需求 |

------

## 四、选择建议

- **优先使用 Ulysses**：适用于大多数通用场景，性能稳定且实现简单
- **使用 RingAttention**：当序列极长（如 64K~128K）且通信资源丰富时
- **考虑 Hybrid**：在需要兼顾灵活性与效率的复杂系统中

> 🔁 实际部署中可结合 **动态调度策略**，根据序列长度自动选择最优算法。

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)

# Ulysses：DeepSpeed 序列并行算法详解

------

## 概述

**Ulysses** 是 DeepSpeed 提出的一种 **上下文并行（Context Parallelism）** 算法，旨在解决长序列训练中因 Attention 计算导致的 **显存瓶颈问题**。其核心思想是通过 **all-to-all 通信操作**，将 Q/K/V 在序列维度上的切分转换为在注意力头维度上的切分，从而实现每个设备上独立完成 Attention 计算。

> ✅ 目标：降低动态显存占用，提升长序列训练效率

------

## 一、基本原理

### 核心机制：

- **输入数据切分**：原始输入 $ x \in [N, d] $ 被切分为多个部分（如 $[N/P, d]$），分配到不同设备
- **Q/K/V 投影后仍保持序列维度切分**：即 $ Q, K, V \in [N, d] $ → 各设备持有 $[N/P, d]$
- **关键步骤**：通过 **all-to-all 通信**，将 $ Q, K, V $ 的序列维度切分（$ N/P $）转换为 **Attention head 维度切分**（$ d/P $）
- **最终效果**：每个设备仅处理 $[N, d/P]$ 的 Attention head，避免全量 KV Cache 存储

------

## 二、详细流程图解析

### 参数定义：

- $ N $: sequence length（序列长度）
- $ d $: hidden size（隐藏维度）
- $ hc $: head count（注意力头数）
- $ P $: total processor (GPU) count（设备总数）
- 假设 $ P = hc = 4 $

------

### 步骤分解：

#### 1. 输入与投影阶段（本地计算）

- 输入 $ x \in [N, d] $ 被切分为 $[N/P, d]$ 分布到各设备
- 使用权重矩阵 $ W_Q, W_K, W_V $ 进行线性变换：
  - $ Q = xW_Q \in [N, d] $
  - $ K = xW_K \in [N, d] $
  - $ V = xW_V \in [N, d] $
- 各设备仅持有局部数据：$[N/P, d]$

#### 2. all-to-all 通信（关键步骤）

- 对 $ Q $ 和 $ K^T $ 执行 **all-to-all** 操作：
  - 将 $ Q $ 从序列维度切分 $[N/P, d]$ 转换为 head 维度切分 $[N, d/P]$
  - 类似地处理 $ K^T $
- 结果：
  - 每个设备获得完整的 $ Q_h \in [N, d/P] $ 和 $ K_h^T \in [d/P, N] $
  - 实现了“跨设备”的 head 切分

> 🔹 注意：此步骤确保后续 Attention 可在本地完成，无需全局存储 KV

#### 3. Attention 计算（本地执行）

- 在每个设备上独立计算：
  - $ S_h = \text{softmax}(Q_h K_h^T / \sqrt{d/P}) $
  - $ F_h = S_h V_h $
- 其中 $ V_h \in [N, d/P] $ 是通过类似方式切分后的输出

#### 4. 输出还原（all-to-all）

- 对 $ F_h \in [N, d/P] $ 再次执行 **all-to-all** 操作
- 将 head 维度切分恢复为序列维度切分 $[N/P, d]$
- 最终通过 $ W_O $ 得到输出 $ O \in [N, d] $

------

## 三、优势分析

### ✅ 优点：

| 优势                 | 说明                                                        |
| -------------------- | ----------------------------------------------------------- |
| **显著降低显存占用** | 每个设备只需存储 $[N, d/P]$ 的 KV 缓存，而非完整的 $[N, d]$ |
| **支持长序列训练**   | 解决了传统方法中 KV Cache 显存爆炸的问题                    |
| **计算高效**         | Attention 操作可在本地独立进行，减少通信频率                |

### ❌ 缺点：

| 缺点                     | 说明                                                       |
| ------------------------ | ---------------------------------------------------------- |
| **依赖 all-to-all 通信** | 需要高性能网络支持，通信开销较大                           |
| **对模型结构有要求**     | GQA 或 Head 数量需足够多才能有效切分，否则可能需要复制数据 |

------

## 四、总结

> 🎯 **Ulysses 的核心思想**：
> 通过 **all-to-all 通信** 将序列维度切分转化为 head 维度切分，使每个设备只处理部分 Attention head，从而实现 **低显存、高并发** 的长序列训练。

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)



# Ulysses：上下文并行算法详解（补充）

------

## 概述

本节补充说明 **Ulysses** 上下文并行算法的**关键步骤、开启方式与使用限制**，帮助开发者在实际训练中正确配置和应用该技术。

------

## 一、核心流程补充

### 最后一步：all-to-all 转置还原

- 在完成 Attention 计算后，需要将结果从 **head 维度切分** 转换回 **序列维度切分**
- 通过一个 **all-to-all 操作** 将 $[N, d/P]$ 的输出转置为 $[N/P, d]$
- 从而恢复原始的序列结构，便于后续层处理或输出

> ✅ 这一步确保了模型整体计算的一致性，是 Ulysses 算法完整闭环的关键。

------

## 二、开启方式（命令行参数）

要启用 Ulysses 并行策略，需在训练脚本中设置以下参数：

```bash
--context-parallel-size 8 \
--context-parallel-algo ulysses_cp_algo \
```

### 参数说明：

| 参数                      | 含义                                               |
| ------------------------- | -------------------------------------------------- |
| `--context-parallel-size` | 设置上下文并行的设备数量（如 8）                   |
| `--context-parallel-algo` | 指定使用的上下文并行算法，此处为 `ulysses_cp_algo` |

> 💡 示例：
>
> ```bash
> deepspeed train.py \
>   --context-parallel-size 8 \
>   --context-parallel-algo ulysses_cp_algo \
>   ...
> ```

------

## 三、使用限制

Ulysses 对模型结构有一定要求，主要体现在 **Grouped Query Attention (GQA)** 配置上：

### 1. 必须启用 GQA

```bash
--group-query-attention \
```

- GQA 是 Ulysses 实现 head 切分的基础
- 允许多个 query 头共享 key/value 头，降低 KV Cache 显存占用

### 2. 设置合理的查询组数

```bash
--num-query-groups 8 \
```

- `num-query-groups` 应等于或大于 `context-parallel-size`
- 若不满足，则无法有效切分，可能触发数据复制或性能下降

> ⚠️ 注意：
>
> - 当 `num-query-groups < context-parallel-size` 时，系统会自动进行冗余复制，导致显存浪费
> - 建议保持两者一致以获得最佳效果

------

## 四、总结

| 项目         | 内容                                                         |
| ------------ | ------------------------------------------------------------ |
| **最后操作** | 通过 all-to-all 将输出转置回序列维度                         |
| **开启方式** | `--context-parallel-size N`, `--context-parallel-algo ulysses_cp_algo` |
| **必要条件** | 启用 `--group-query-attention`                               |
| **关键配置** | `--num-query-groups >= context-parallel-size`                |

> ✅ 推荐组合：
>
> ```
> --context-parallel-size 8 \
> --context-parallel-algo ulysses_cp_algo \
> --group-query-attention \
> --num-query-groups 8
> ```

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)



# Ulysses 应用案例：ChatGLM3-6B 32K 长序列训练

------

## 概述

本节以 **ChatGLM3-6B 模型在 32K 长序列上的预训练任务** 为例，展示如何通过 **Ulysses 上下文并行算法** 实现高效长序列训练。该案例来自 Ascend 官方示例脚本：

> 📁 `examples/mcore/chatglm3/pretrain_chatglm3_6B_32K.sh`

目标是解决长序列带来的 **KV Cache 显存瓶颈**，同时保持高吞吐与算力利用率。

------

## 一、性能对比表

| 项目                 | 吞吐量 Token/s/p |
| -------------------- | ---------------- |
| GPU A100 Megatron-LM | 2887.84          |
| NPU A2 MindSpeed-LLM | 3006.24          |

> 🔍 分析：
>
> - NPU A2 在相同配置下表现更优
> - 表明 Ulysses 算法在 Ascend 架构上优化良好，支持高效率长序列训练

------

## 二、关键配置项详解

| 配置项                        | 值                | 意义                                                         |
| ----------------------------- | ----------------- | ------------------------------------------------------------ |
| **TP**                        | 1                 | TP 开启后会因 GQA=2 场景限制导致较低算力利用率，故关闭       |
| **PP**                        | 1                 | PP 引入通信开销较大，在此场景下不启用                        |
| **CP**                        | 8                 | 使用上下文并行（Context Parallel），切分序列维度，降低激活值显存占用 |
| **context-parallel-algo**     | `ulysses_cp_algo` | 选择 Ulysses 算法实现 CP，适用于 GQA 场景下的高效 KV 切分    |
| **num-query-groups**          | 2                 | 社区原生设定，导致 TP/CP 最大为 2；但此处 CP=8 是通过其他方式实现的（可能为扩展支持） |
| **use-fused-swiglu**          | True              | 融合 SwiGLU 过程，减少中间变量存储，降低显存占用             |
| **use-fused-rmsnorm**         | True              | 融合 RmsNorm 计算，提升计算效率，节省显存                    |
| **use-distributed-optimizer** | True              | 使用分布式优化器，将 optimizer state 分布到多设备，缓解单卡显存压力 |
| **overlap-grad-reduce**       | True              | 梯度更新与 Reduce 通信重叠，隐藏通信延迟，提升整体效率       |

------

## 三、核心优化策略总结

### ✅ 为何选择 Ulysses？

- **序列长度达 32K**，传统方法中 KV Cache 显存爆炸
- **GQA 场景下无法有效使用 RingAttention**
- **Ulysses 支持 KV 复制 + Head 切分**，可在不牺牲精度的前提下实现低显存运行

### ✅ 为什么不用 TP 或 PP？

| 并行方式 | 不选用原因                                                   |
| -------- | ------------------------------------------------------------ |
| **TP**   | GQA=2 时开启 TP 会导致算力利用率下降（如参数复制、通信开销） |
| **PP**   | 引入额外通信耗时，且对短序列效率不高，不适合此场景           |

> 💡 因此，**优先采用 CP（Ulysses）** 来突破显存瓶颈。

## 四、实际应用建议

1. **长序列训练优先考虑 Context Parallel**
   - 特别是在序列 > 16K 的场景下
   - 推荐使用 Ulysses 或 RingAttention
2. **合理设置 num-query-groups**
   - 若模型支持 GQA，应确保 `num-query-groups >= context-parallel-size`
   - 否则需权衡是否启用 CP
3. **融合操作不可忽视**
   - `use-fused-swiglu` 和 `use-fused-rmsnorm` 是提升吞吐的关键
   - 可显著降低激活值显存占用
4. **通信优化至关重要**
   - `overlap-grad-reduce` 能有效掩盖通信时间，提升 MFU

------

## 五、结论

> ✅ **Ulysses 成功应用于 ChatGLM3-6B 的 32K 长序列训练**，实现了：

- **低显存占用**：避免 KV Cache 爆炸
- **高吞吐量**：NPU 达到 3006.24 Token/s/p
- **高算力利用率**：结合融合算子和通信优化，最大化硬件性能

> 🚀 该方案可推广至其他基于 GQA 的大模型长序列训练任务。

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)



# RingAttention：长序列并行计算的核心技术

------

## 概述

**RingAttention** 是一种用于高效处理**长序列 Transformer 模型**的上下文并行（Context Parallelism）算法，旨在解决传统 Attention 计算中因序列长度增长导致的 **KV Cache 显存爆炸问题**。该技术由尤洋团队提出，最初称为 **RingSelfAttention (RSA)**，后经优化演变为更高效的 **Blockwise Parallel Transformer (BPT)** 方案。

> ✅ 核心思想：在序列维度上进行切分，并通过设备间通信实现分布式 Attention 计算。

------

## 一、发展历程

### 1. RingSelfAttention (RSA)

- **提出者**：尤洋团队
- **核心机制**：
  - 将输入序列按块切分，分配到不同设备
  - 各设备持有局部 Query、Key、Value
  - 通过 **环形通信（Ring Communication）** 在设备间传递 Key 和 Value
- **优点**：
  - 实现了序列维度的并行化
  - 降低单设备显存占用
- **缺点**：
  - 通算不可掩盖（Compute and Communication 不重叠）
  - 性能受限于通信延迟

### 2. Blockwise Parallel Transformer (BPT)

- **改进目标**：优化 RSA 的性能瓶颈
- **关键优化**：
  - 引入 **分块独立 Attention 计算**
  - 实现 **通信与计算重叠（Overlap）**
  - 支持 `local Q` 的本地化处理
- **优势**：
  - 解决了“通算不可掩盖”的问题
  - 提升整体吞吐和效率

------

## 二、工作流程详解

### 图 (a)：Key Embeddings 的传输（计算注意力得分）

| 步骤                                                         | 描述 |
| ------------------------------------------------------------ | ---- |
| 1. 每个设备持有自己的 Query、Key、Value 块                   |      |
| 2. 当前设备需要计算 Attention Score 时，必须获取其他设备的 Key 向量 |      |
| 3. 通过 **SendRecv** 或 **Ring Communication** 将 Key 从一个设备传送到下一个设备 |      |
| 4. 所有设备依次接收并使用 Key 来计算 $ QK^T $                |      |

> 🔹 示例：
>
> - Device 1 发送 Key 到 Device 2
> - Device 2 接收 Key 并计算其 Query 与所有 Key 的得分
> - 然后将 Key 传给 Device 3，依此类推

------

### 图 (b)：Value Embeddings 的传输（生成输出）

| 步骤                                                         | 描述 |
| ------------------------------------------------------------ | ---- |
| 1. 在完成 Softmax 得到 attention weights 后，需要计算最终输出：$\text{output} = \text{attention score} \cdot V $ |      |
| 2. 每个设备需从其他设备获取对应的 Value 向量                 |      |
| 3. 使用类似环形方式传输 Value，确保每个设备都能访问完整 Value 集合 |      |
| 4. 最终输出在本地聚合                                        |      |

------

## 三、BPT 架构图解析（右侧图）

### 外层循环：Query Outer Loop

- 按照 Query 分组进行迭代
- 每次处理一组 Query（如 Query1, Query2...）

### 内层循环：Key and Value Inner Loop

- 对每一个 Query，执行以下操作：
  1. **Blockwise Attention**：
     - 使用本地 Query 与来自前一个设备的 Key 计算 attention score
     - 将结果发送给下一个设备
  2. **Blockwise FeedForward**：
     - 完成 Attention 后，进行 FFN 计算
     - 输出可直接传递或本地存储

> 📌 特点：
>
> - **通信与计算交替进行**
> - **支持流水线式处理**
> - **减少全局同步开销**

------

## 四、优缺点分析

### ✅ 优点：

| 优势                   | 说明                             |
| ---------------------- | -------------------------------- |
| **显著降低显存占用**   | 每个设备仅存储部分序列的 KV 缓存 |
| **支持超长序列训练**   | 可扩展至数十万 token             |
| **结构清晰，易于实现** | 基于环形通信，逻辑简单           |

### ❌ 缺点：

| 缺点                   | 说明                                       |
| ---------------------- | ------------------------------------------ |
| **通信开销大**         | 每次都需要 SendRecv，对网络带宽要求高      |
| **序列不够长时效率低** | 若序列较短，通信开销占比过大，性能下降明显 |
| **通算难以完全掩盖**   | 虽然 BPT 有所改善，但仍存在通信瓶颈        |

------

## 五、适用场景

| 场景                         | 是否推荐                   |
| ---------------------------- | -------------------------- |
| **超长序列（> 32K）**        | ✅ 强烈推荐                 |
| **高带宽网络环境**           | ✅ 推荐                     |
| **内存受限但通信能力强**     | ✅ 推荐                     |
| **短序列（< 8K）**           | ❌ 不推荐，通信开销主导     |
| **GQA 或 Head 数量少的模型** | ⚠️ 效果较差，建议用 Ulysses |

------

## 六、总结对比（与 Ulysses）

| 特性         | RingAttention      | Ulysses            |
| ------------ | ------------------ | ------------------ |
| **切分维度** | 序列轴（S）        | 注意力头轴（Dim）  |
| **通信模式** | Ring / SendRecv    | All-to-all         |
| **显存优化** | 高（仅存局部 KV）  | 高（头切分）       |
| **通信开销** | 较大（逐设备传递） | 中等（全连接通信） |
| **适合场景** | 超长序列、高带宽   | GQA 模型、稳定训练 |

> 💡 建议：
>
> - 长序列 + 高带宽 → 优先考虑 **RingAttention**
> - GQA 模型 + 中等序列 → 推荐 **Ulysses**

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)

# RingAttention：MindSpeed 长序列方案演进

------

## 概述

本节介绍 **RingAttention** 在 MindSpeed-LLM 框架中的进一步优化演进，基于 **Blockwise Parallel Transformer (BPT)** 的思想，通过引入 **ring p2p（点对点环形通信）** 实现了对两层循环计算的并行化处理，从而提升了长序列训练的效率与可扩展性。

------

## 一、技术背景

### BPT（Blockwise Parallel Transformer）

- 是对早期 RingSelfAttention (RSA) 的重要改进
- 核心思想：
  - 将 Attention 计算分解为两个嵌套循环：
    1. 外层：按 Query 分块处理（Query Outer Loop）
    2. 内层：按 Key/Value 块进行通信和计算（Key and Value Inner Loop）
- 实现了分块独立的 Attention 计算，缓解了显存压力

> 🔹 缺点：两层循环仍存在串行依赖，难以完全并行化

------

## 二、RingAttention 的演进：Liu 等人的贡献

### 关键创新：

> ✅ **将 BPT 的两层循环通过 ring p2p 形式进行并行切分**

- Liu 等人提出了一种新的并行策略：
  - 利用 **环形点对点通信（ring p2p）** 结构
  - 对 BPT 中的 **Query 和 Key/Value 的双层循环** 进行空间上的并行划分
  - 使得原本串行执行的计算过程可以在多个设备上同时推进

### 效果：

- **实现真正的并行化**：避免了单个设备成为瓶颈
- **提升通信效率**：利用环形拓扑减少通信延迟
- **支持更大规模模型**：适用于超长序列（如 32K~128K）训练场景

------

## 三、形成当前 MindSpeed 的 RingAttention 方案

### 最终形态：

- 当前 **MindSpeed-LLM** 所采用的 **RingAttention** 正是基于上述优化而来
- 具备以下特性：
  - 支持 **高效长序列训练**
  - 降低 **KV Cache 显存占用**
  - 提升 **通信与计算重叠能力**
  - 适配 Ascend NPU 架构的高带宽互联特性

> 🚀 成为 MindSpeed 在长序列大模型训练中的核心技术之一

------

## 四、总结

| 技术                        | 贡献                                   |
| --------------------------- | -------------------------------------- |
| **RSA**                     | 首次提出序列维度切分 + 环形通信        |
| **BPT**                     | 优化 RSA，实现分块独立计算             |
| **Liu 等人改进**            | 将 BPT 的双层循环并行化，引入 ring p2p |
| **MindSpeed RingAttention** | 综合以上成果，形成工业级长序列解决方案 |

> 💡 演进路径：
>
> ```
> RSA → BPT → ring p2p 并行化 → MindSpeed RingAttention
> ```

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)

# RingAttention：使用方式与通信开销分析

------

## 概述

本节介绍 **RingAttention** 的实际使用方法及其关键的 **通信开销（communication overhead）分析**，帮助开发者理解如何配置并优化该算法在长序列训练中的性能表现。

------

## 一、使用方式

要启用 RingAttention 并行策略，需在训练脚本中设置以下参数：

```bash
--context-parallel-algo megatron_cp_algo \
--context-parallel-size 8 \
```

### 参数说明：

| 参数                      | 含义                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `--context-parallel-algo` | 指定上下文并行算法，此处为 `megatron_cp_algo`（即 RingAttention 实现） |
| `--context-parallel-size` | 设置参与上下文并行的设备数量，例如 8                         |

> ✅ 示例：
>
> ```bash
> deepspeed train.py \
>   --context-parallel-algo megatron_cp_algo \
>   --context-parallel-size 8 \
>   ...
> ```

------

## 二、通信开销分析

### 假设条件：

- 设备间通信带宽为 $ B $ （单位：byte/s）
- 单个设备算力为 $ F $ （单位：flop/s）
- 每个 block 分配的序列长度为 $ c $
- 模型隐藏维度为 $ h $
- batch size 为 $ b $

------

### 核心目标：实现计算掩盖通信

为了使 **计算操作能有效掩盖通信延迟**，需要满足以下不等式：

$$
\frac{4h c^2}{F} \geq \frac{4c h}{B}
$$

#### 推导过程：

1. 左边：每个 block 的计算量（FLOPs）除以算力 → 计算耗时
   - 注意力计算中，$ QK^T $ 和 $ SV $ 等操作约为 $ O(hc^2) $
   - 总计算量 ≈ $ 4h c^2 $ flops
   - 计算时间 = $ \frac{4h c^2}{F} $
2. 右边：通信数据量除以带宽 → 通信耗时
   - 每个 block 需要传输的数据量 ≈ $ 4c h $ bytes（Key/Value 等）
   - 通信时间 = $ \frac{4c h}{B} $
3. 要求：计算时间 ≥ 通信时间
   $$
   \frac{4h c^2}{F} \geq \frac{4c h}{B}
   $$
4. 化简得：
   $$
   c \geq \frac{F}{B}
   $$

------

### 结论：关键阈值

> 📌 **每一个 block 分配的序列长度 $ c $ 必须大于等于算力与通信带宽之比 $ F/B $**。

#### 意义：

- 若 $ c < F/B $：通信耗时 > 计算耗时 → 通信成为瓶颈
- 若 $ c \geq F/B $：计算可掩盖通信 → 整体效率提升

> 💡 实际建议：
>
> - 在高算力、低带宽环境下，应增加 block 大小（即增大每块序列长度）
> - 在高带宽网络中（如 NVLink、InfiniBand），可以容忍更小的 block

------

## 三、总结

| 项目             | 内容                                                         |
| ---------------- | ------------------------------------------------------------ |
| **开启方式**     | `--context-parallel-algo megatron_cp_algo`, `--context-parallel-size N` |
| **通信效率关键** | block 序列长度 $ c \geq F/B $                                |
| **优化建议**     | 提升通信带宽或增大 block 大小以掩盖通信延迟                  |

> ✅ 推荐实践：
>
> - 使用高速互联网络（如 Ascend NPU 的高速互联）
> - 合理划分 block 大小，避免过短导致通信主导

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)



# 混合长序列并行：Hybrid Context Parallel

------

## 概述

**Hybrid Context Parallel** 是一种融合 **Ulysses** 和 **RingAttention** 两种上下文并行算法的混合策略，旨在结合两者优势，实现更高效、更具适应性的长序列训练方案。

> ✅ 核心思想：
> 在 Ulysses 的基础上，将原本本地执行的 Feed-Forward Attention（FA）替换为 **RingAttention**，并引入维度细分机制，以优化通信效率与显存占用。

------

## 一、技术原理

### 1. 替换本地 FA 为 RingAttention

- 原始 Ulysses 算法中：
  - Q/K/V 在序列维度上切分
  - 经过 all-to-all 转换后，在每个设备上独立完成 Attention 计算（即“本地 FA”）
- Hybrid 改进：
  - 将该“本地 FA”部分替换为 **RingAttention**
  - 利用 RingAttention 的环形通信机制处理注意力计算

> 📌 目标：提升在超长序列下的通信效率和可扩展性

------

### 2. CP 维度细分为 Ulysses 和 Ring 两个子维度

- 在并行初始化阶段，将整体上下文并行（CP）维度划分为两个子组：
  - `ulysses_pg`：用于 all-to-all 通信（Ulysses 部分）
  - `ring_pg`：用于 ring p2p 通信（RingAttention 部分）

> ⚠️ 注意：由于 all-to-all 对带宽要求更高，因此：

- **Ulysses 维度设置得更低**（如 4 或 2）
- Ring 维度则可以更大（如 8）

> 💡 例如：总 CP=8 → Ulysses=2, Ring=4

------

## 二、统一单进程注意力实现算法（Algorithm 1）

```python
function UNIFIED_SP_ATTENTION(ulysses_pg, ring_pg, Q, K, V, scatter_idx, gather_idx)
    1: Q ← AllToAll4D(Q, scatter_idx, gather_idx, group = ulysses_pg)
    2: K ← AllToAll4D(K, scatter_idx, gather_idx, group = ulysses_pg)
    3: V ← AllToAll4D(V, scatter_idx, gather_idx, group = ulysses_pg)
    4: O ← LoadBalance-RingAttention(Q, K, V, group = ring_pg)
    5: O ← AllToAll4D(ulysses_pg, O, gather_idx, scatter_idx, group = ulysses_pg)
    6: return O
end function
```

### 步骤解析：

| 步骤 | 功能                                                         |
| ---- | ------------------------------------------------------------ |
| 1~3  | 使用 **all-to-all** 将 Q/K/V 从序列维度切分转换为 head 维度切分（Ulysses 阶段） |
| 4    | 调用 **LoadBalance-RingAttention** 进行注意力计算，利用 ring p2p 实现高效通信 |
| 5    | 再次使用 all-to-all 将输出还原为原始序列维度结构             |
| 6    | 返回最终结果                                                 |

> ✅ 整体流程实现了 **Ulysses + RingAttention 的无缝集成**

------

## 三、优势分析

| 优点             | 说明                                             |
| ---------------- | ------------------------------------------------ |
| **兼具二者优点** | Ulysses 降低显存占用，RingAttention 提升通信效率 |
| **综合能力强**   | 可适应不同长度序列和网络环境                     |
| **灵活配置**     | 可根据硬件条件动态调整 Ulysses/Ring 维度比例     |

------

## 四、适用场景

| 场景                        | 是否推荐                                   |
| --------------------------- | ------------------------------------------ |
| **超长序列（> 64K）**       | ✅ 推荐，充分发挥 RingAttention 优势        |
| **高带宽网络环境**          | ✅ 推荐，支持更大的 Ring 维度               |
| **GQA 模型**                | ✅ 推荐，Ulysses 支持良好                   |
| **中等序列长度（16K~32K）** | ⚠️ 可选，若通信带宽不足则可能不如纯 Ulysses |
| **低带宽网络**              | ❌ 不推荐，all-to-all 成本过高              |

------

## 五、总结

> 🌟 **Hybrid Context Parallel 是一种先进的混合并行策略**，通过以下方式实现性能突破：

- 将 Ulysses 的 all-to-all 与 RingAttention 的 ring p2p 结合
- 细分 CP 维度，平衡通信开销与显存占用
- 实现统一的单进程注意力接口，便于部署和维护

> 💡 最佳实践建议：
>
> - 总 CP 数量 ≥ 4
> - 设置 `ulysses_dim < ring_dim`
> - 在高带宽系统中优先启用 Hybrid 方案

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)



# 混合长序列并行：Ulysses、Ring 与 Hybrid 方案对比分析

------

## 概述

本节从 **优劣势角度** 对比 Ulysses、RingAttention 和 Hybrid 混合方案，深入剖析其在长序列训练中的适用场景与性能权衡，为实际部署提供决策依据。

------

## 一、Ulysses 优劣势分析

### ✅ 优点：

- **充分利用 all-to-all 通信带宽**：
  - 支持全连接式数据交换，通信效率高
  - 在高带宽网络（如 NVLink、InfiniBand）中表现优异
- **Attention 计算矩阵较大，充分挖掘计算带宽**：
  - 每个设备处理完整的 $ Q_h K_h^T $ 矩阵（$[N, d/P]$），计算密集度高
  - 更好地利用 GPU/NPU 的算力资源

### ❌ 缺点：

- **CP 并行维度难扩展**：
  - 要求 `head_size` 能被 `TP * CP` 整除
  - 若不满足，则需复制数据，导致显存浪费或无法启用
- **部分模型结构不支持**：
  - 如 GQA（Grouped Query Attention）、MQA（Multi-Query Attention）等因 head 数量少，难以进行有效切分
  - 可能完全无法使用 Ulysses
- **跨机 all-to-all 带宽受限**：
  - 多节点环境下，跨机通信延迟和带宽瓶颈显著
  - 影响整体吞吐与可扩展性

> 💡 小结：Ulysses 适合 **头数多、GQA 不用、高带宽内网环境** 的场景。

------

## 二、RingAttention 优劣势分析

### ✅ 优点：

- **通信模式简单，易于实现**：
  - 基于 ring p2p，逻辑清晰，适合大规模集群

### ❌ 缺点：

- **通信与计算难掩盖**：
  - 需要满足条件：每个 block 分配的序列长度 $ c \geq F/B $
    - 其中 $ F $：设备算力（flop/s）
    - $ B $：通信带宽（byte/s）
  - 否则通信成为瓶颈，引入额外开销
- **对序列长度要求严格**：
  - 实际场景中还需考虑算力利用率因素
  - 通常需要 **更长的序列** 才能保证效率

> 💡 小结：RingAttention 适合 **超长序列（>32K）、高带宽网络** 场景，但对短序列效率较低。

------

## 三、Hybrid 混合方案的优势与价值

### ✅ 核心优势：**弥补两种方案的缺陷**

#### 1. 解决 head_size 过小问题

- 在 `head_size` 较小的场景下（如 GQA/MQA）：
  - 可设置较大的 **ring 维度**，从而增大总 CP 规模
  - 避免因 head 切分失败而限制并行度

#### 2. 提升 RingAttention 的通算掩盖能力

- 在执行 RingAttention 前，先通过 **all-to-all** 进行数据预处理
- 可以获取更大的局部序列长度（即“block”长度）
- 使得后续 RingAttention 中的 $ c \geq F/B $ 更容易满足
- 极大提升通信与计算的重叠效率

> 🎯 总结：Hybrid 是一种**灵活且高效的折中策略**，能够在不同硬件和模型条件下动态平衡显存、通信与计算资源。

------

## 四、总结对比表

| 方案              | 优点                                   | 缺点                                     | 推荐场景                                |
| ----------------- | -------------------------------------- | ---------------------------------------- | --------------------------------------- |
| **Ulysses**       | 充分利用 all-to-all 带宽，计算密集度高 | head_size 必须整除 TP×CP；不支持 GQA/MQA | 头数多、支持 full attention、高带宽内网 |
| **RingAttention** | 通信简单，适合超长序列                 | 通信难掩盖，需足够长序列                 | 超长序列（>32K）、高带宽网络            |
| **Hybrid**        | 兼具两者优点，可灵活配置               | 实现复杂，调试成本较高                   | 头数少、GQA 模型、混合负载环境          |

------

## 五、实践建议

1. **优先尝试 Ulysses**：若模型支持且 head 数较多，首选 Ulysses
2. **长序列 + 低 head 数 → 使用 Hybrid**：结合 Ulysses 的 all-to-all 和 Ring 的高效通信
3. **极端长序列（>64K）→ 考虑纯 Ring 或 Hybrid**：确保通信开销可控
4. **跨机训练 → 注意 all-to-all 带宽瓶颈**：尽量避免跨机 all-to-all，或优化拓扑结构

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)

# 混合长序列并行：最佳实践与配置指南

------

## 概述

本节介绍 **Hybrid Context Parallel** 的**最佳实践策略**和**具体使用方式**，旨在通过合理划分 Ulysses 与 RingAttention 的通信维度，最大化训练效率，避免跨节点通信瓶颈。

------

## 一、最佳实践原则

> ✅ **核心目标**：在保证计算性能的前提下，最小化通信开销，尤其是避免跨节点的 all-to-all 操作。

### 推荐策略：

- **所有 Ulysses rank 设置到节点内**
  - 即：`TP * ulysses_size <= 8`
  - 建议将 Ulysses 维度限制在单个计算节点内部（如 1 台服务器）
  - 利用节点内的高速互联（如 NVLink、PCIe）实现高效 all-to-all 通信
- **Ulysses_size 尽量小**
  - 因为 all-to-all 对带宽要求高，应尽量减少其规模
  - 例如：设置 `ulysses_size = 2` 或 `4`
- **跨节点仅执行 p2p 通信的 RingAttention**
  - 跨节点部分只进行点对点（p2p）通信，不涉及 all-to-all
  - 避免跨机 all-to-all 导致的高延迟和低吞吐

> 🎯 结果：此时系统效率最优，通信成本最低，算力利用率最高。

------

## 二、使用方式（命令行参数）

要启用 Hybrid 混合上下文并行方案，需配置以下参数：

```bash
--context-parallel-size 8 \
--context-parallel-algo hybrid_cp_algo \
--ulysses-degree-in-cp 2 \
```

### 参数说明：

| 参数                      | 含义                                            |
| ------------------------- | ----------------------------------------------- |
| `--context-parallel-size` | 总的上下文并行设备数，此处为 8                  |
| `--context-parallel-algo` | 指定使用混合算法：`hybrid_cp_algo`              |
| `--ulysses-degree-in-cp`  | 在总 CP 中分配给 Ulysses 的维度大小，此处设为 2 |

> 🔍 解读：
>
> - 总 CP = 8
> - Ulysses 维度 = 2 → 用于 all-to-all 通信（建议在节点内完成）
> - Ring 维度 = 8 / 2 = 4 → 用于 ring p2p 通信（可跨节点）

> 💡 示例拓扑：
>
> ```
> Node 1: [GPU0, GPU1] → Ulysses group (size=2)
> Node 2: [GPU2, GPU3] → Ulysses group (size=2)
> ...
> ```

------

## 三、总结

| 项目         | 内容                                                     |
| ------------ | -------------------------------------------------------- |
| **最佳实践** | Ulysses rank 放在节点内，尺寸尽量小；Ring 用于跨节点 p2p |
| **关键参数** | `--ulysses-degree-in-cp` 控制 Ulysses 维度               |
| **通信优化** | 避免跨节点 all-to-all，提升整体效率                      |

> ✅ 推荐组合：
>
> ```bash
> --context-parallel-size 8 \
> --context-parallel-algo hybrid_cp_algo \
> --ulysses-degree-in-cp 2
> ```

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)

# 长序列64K性能优化：Hybrid方案实战案例

------

## 概述

本节以 **ChatGLM3-6B 模型在 64K 长序列上的预训练任务** 为例，展示如何通过 **Hybrid Context Parallel（混合上下文并行）** 方案实现高效训练。该案例来自 Ascend 官方示例脚本：

> 📁 `examples/mcore/chatglm3/pretrain_chatglm3_6B_64K.sh`

目标是解决超长序列带来的 **KV Cache 显存瓶颈** 和 **通信开销问题**，同时保持高吞吐与算力利用率。

------

## 一、性能指标

| 项目                 | 吞吐量 Token/s/p |
| -------------------- | ---------------- |
| NPU A2 MindSpeed-LLM | 1724.6           |

> 🔍 分析：
>
> - 在 64K 超长序列下仍能维持较高吞吐
> - 表明 Hybrid 方案有效提升了长序列训练效率

------

## 二、关键配置项详解

| 配置项                        | 值               | 意义                                                         |
| ----------------------------- | ---------------- | ------------------------------------------------------------ |
| **TP**                        | 1                | TP 开启会导致算力利用率下降（尤其在 GQA 场景），故关闭       |
| **PP**                        | 1                | PP 引入的通信耗时较大，避免使用                              |
| **CP**                        | 16               | 使用上下文并行（Context Parallel），切分序列维度，降低激活值显存占用 |
| **context-parallel-algo**     | `hybrid_cp_algo` | 采用混合方案，结合 Ulysses 和 RingAttention 的优势           |
| **ulysses-degree-in-cp**      | 8                | 在总 CP=16 中分配 8 给 Ulysses，用于 all-to-all 通信 → 卡均序列长度为 32K，满足通算掩盖条件 |
| **use-fused-swiglu**          | True             | 融合 SwiGLU 过程，减少中间变量存储，降低显存占用             |
| **use-distributed-optimizer** | True             | 使用分布式优化器，将 optimizer state 分布到多设备，缓解单卡显存压力 |
| **overlap-grad-reduce**       | True             | 梯度 reduce 与反向计算重叠，隐藏通信延迟，提升整体效率       |

------

## 三、核心优化策略分析

### ✅ 为何选择 Hybrid 方案？

- **序列长度达 64K**，传统方法中 KV Cache 显存爆炸
- **GQA 场景限制了纯 Ulysses 的使用**
- **RingAttention 在短 block 下效率低**，难以完全掩盖通信

> 💡 因此，采用 **Hybrid 混合方案** 是最优解：
>
> - 利用 Ulysses 实现 head 切分和数据预处理
> - 利用 RingAttention 实现跨节点 p2p 通信
> - 兼顾显存、通信与计算效率

------

### ✅ 关键参数解读：`ulysses-degree-in-cp = 8`

- 总 CP = 16 → 每个 GPU 处理 $ \frac{64K}{16} = 4K $ token
- 但通过 Ulysses 将局部序列扩展为 32K（即每张卡在 Ulysses 内部处理 32K）
- 此时每个 block 的序列长度 $ c = 32K $
- 算力通信比 $ F/B $ 通常在 10~100 之间
- $ c > F/B $ → 满足通算掩盖条件

> 🎯 结果：RingAttention 的 SendRecv 通信可被计算完全掩盖，效率最大化

------

## 四、实际应用建议

1. **超长序列训练优先考虑 Hybrid 方案**
   - 特别适用于 32K~128K 的场景
   - 支持 GQA/MQA 模型
2. **合理设置 `ulysses-degree-in-cp`**
   - 建议设置为总 CP 的 1/2 ~ 1/4
   - 确保卡均序列长度足够大（> F/B）
3. **融合操作不可忽视**
   - `use-fused-swiglu` 和 `use-fused-rmsnorm` 是降低显存的关键
   - 可显著提升内存效率
4. **通信优化至关重要**
   - `overlap-grad-reduce` 能有效掩盖梯度同步延迟
   - 提升整体 MFU（Memory-Floating Point Utilization）

------

## 五、总结

> ✅ **Hybrid 方案成功应用于 ChatGLM3-6B 的 64K 长序列训练**，实现了：

- **低显存占用**：避免 KV Cache 爆炸
- **高吞吐量**：达到 1724.6 Token/s/p
- **高算力利用率**：通过通算掩盖和融合操作最大化硬件性能

> 🚀 该方案可推广至其他基于 GQA 的大模型超长序列训练任务。

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)



# 长序列算法实操展示：32K 序列训练对比分析

------

## 概述

本节通过 **实际训练场景** 对比不同并行算法在 **32K 长序列** 下的性能表现，揭示为何需要使用专门的长序列优化算法（如 Ring Attention 和 Ulysses），并明确其适用场景与优势。

> 📌 场景设定：常见 32K 序列长度训练任务
> 🖥️ 硬件平台：NPU A2 + MindSpeed-LLM 框架

------

## 一、问题引导

### 1. 使用常见 TP/PP 并行 —— 为什么要使用长序列算法？

- 传统 Tensor Parallel (TP) + Pipeline Parallel (PP) 在处理长序列时：
  - KV Cache 显存占用随序列长度平方增长（$ O(S^2) $）
  - 单卡显存迅速耗尽
  - 导致 **OOM（Out of Memory）**

> ❌ 结论：常规并行方式无法支撑 32K 长序列训练

------

### 2. 使用 Ring Attention 算法 —— Ring Attention 的使用场景是什么？

- Ring Attention 通过 **环形通信切分序列维度**
- 每个设备仅存储部分 KV 缓存
- 支持超长序列训练，避免显存爆炸

> ✅ 实测吞吐量：**2674.93 Token/s/p**

> 💡 适用场景：
>
> - 序列长度 > 16K
> - 高带宽网络环境（如 NPU 互联）
> - 不支持 GQA/MQA 的模型

------

### 3. 使用 Ulysses 算法 —— 成功执行训练并达成较高性能

- Ulysses 利用 **all-to-all 通信 + head 维度切分**
- 实现高效上下文并行，降低显存占用
- 支持 GQA 场景下的 KV 复制机制

> ✅ 实测吞吐量：**3006.24 Token/s/p**

> 💡 优势：
>
> - 显存效率高
> - 计算密集度大，充分利用算力
> - 适合 GQA、MQA 等头数较少的模型

------

## 二、实战效果对比表

| 软硬件框架           | 算法                       | 吞吐量 Token/s/p |
| -------------------- | -------------------------- | ---------------- |
| NPU A2 MindSpeed-LLM | Tensor / Pipeline Parallel | OOM              |
| NPU A2 MindSpeed-LLM | Ring Attention             | 2674.93          |
| NPU A2 MindSpeed-LLM | Ulysses                    | 3006.24          |

------

## 三、关键结论

| 对比项         | 分析                                                        |
| -------------- | ----------------------------------------------------------- |
| **是否能运行** | TP/PP → ❌ OOM；Ring & Ulysses → ✅ 成功运行                  |
| **性能差异**   | Ulysses > Ring Attention（+12.4% 吞吐）                     |
| **适用性**     | Ulysses 更适合 GQA 模型；Ring 更适合纯 attention 模型       |
| **通信要求**   | Ulysses 需要高 all-to-all 带宽；Ring 需要足够长序列掩盖通信 |

------

## 四、总结建议

> ✅ **在 32K 长序列训练中，必须采用专用长序列算法**：

1. **若模型支持 GQA/MQA → 推荐使用 Ulysses**
   - 性能更高，显存更优
2. **若模型为 full attention 或 head 数较多 → 推荐使用 Ring Attention**
   - 通信简单，扩展性强
3. **若序列极长（>64K）→ 可考虑 Hybrid 方案**
   - 兼顾 Ulysses 和 Ring 的优势

> 🚀 最终目标：**在不牺牲性能的前提下，突破显存瓶颈，实现高效长序列训练**

------

> 来源：Ascend 官方文档 [www.hiascend.com](http://www.hiascend.com/)



## 学习资料

- [昇腾后训练强化学习最佳实践](https://www.hiascend.com/zh/developer/techArticles/20251107-1)
