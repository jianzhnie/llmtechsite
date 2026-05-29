## LLM 性能优化方法论（基于 MindSpeed LLM）

> **核心目标**：在不同业务场景下，识别性能瓶颈（Device Bound / Host Bound），并应用针对性的加速与显存优化策略。

### 一、性能瓶颈分类与典型场景

| 业务阶段             | 模型类型                         | 主要瓶颈                                      | 典型场景                          |
| -------------------- | -------------------------------- | --------------------------------------------- | --------------------------------- |
| **预训练**           | 稠密模型（如 Llama, GPT-Qwen）   | Device Bound                                  | 计算密集，受限于芯片算力          |
|                      | MoE 模型（如 Mixtral, Deepseek） | Device Bound（Dropless）或 Host Bound（Drop） | MoE 路由/通信开销大               |
| **全参微调**         | 静态/动态序列                    | Device Bound                                  | 显存压力大，计算密集              |
| **LoRA 微调**        | 静态/动态序列                    | Host Bound                                    | 小参数更新，通信或 CPU 成瓶颈     |
| **偏好对齐（RLHF）** | Actor/Critic/RM                  | Device Bound                                  | 多模型协同，显存与计算压力        |
| **在线推理 & 搜索**  | —                                | Host Bound                                    | 请求调度、KV Cache 管理、I/O 延迟 |

> ✅ **关键洞察**：
>
> - **稠密模型**通常为 **Device Bound**（算力/显存受限）；
> - **MoE 模型**需关注 **Token Dispatch 开销**（AllGather vs All2All）；
> - **轻量微调（如 LoRA）** 和 **推理服务**易受 **Host（CPU/通信）限制**。

### 二、性能优化方法论三大维度

#### 1. **计算优化（提升算力利用率）**

- **算子融合**：
  - `FlashAttention`、`Fused RotaryEmbedding`、`Fused RMSNorm`、`Fused SwiGLU`
  - MoE 专用：`Fused Token Permute/Unpermute`
- **算法优化**：
  - 使用高效 GEMM/BMM 实现
  - 非连续内存 → 连续内存（提升访存效率）
  - Vector 指令优化、AICPU 算子卸载
- **流水线优化**：
  - 多流水并行下发
  - FA（FlashAttention）推理引擎 + 融合下发

#### 2. **显存优化（降低峰值显存，提升容量）**

- **重计算（Recomputation）**：
  - 激活函数重计算（节省激活值存储）
  - 全重计算（Full Recompute）
  - **Recompute in Advance**：提前重计算，掩盖通信延迟，减少 Pipeline Bubble
- **显存复用与交换**：
  - 参数副本复用（时分复用）
  - Swap Attention（用主机内存换显存）
- **精度优化**：
  - O2 BF16 Optimizer（半精度优化器状态）

#### 3. **通信优化（通算掩盖 & 带宽提升）**

- **通算掩盖（Overlap Computation & Communication）**：
  - MC2：在线性层做通信掩盖（TP/SP 场景）
  - 梯度 Reduce 与参数更新重叠
  - 权重 All-Gather 与前向计算重叠
- **MoE 通信优化**：
  - `--moe-permutation-async-comm`（异步通信）
  - Dispatcher 选择：`allgather`（适合小专家） vs `alltoall`（适合大专家）
  - EP（Expert Parallel）通信隐藏
- **并行策略优化**：
  - 高维张量并行（切分 1D-TP 通信域，提升 Cube 利用率）
  - 最小 DP 组优化、多机通信优化

### 三、并行策略对比分析

| 并行方式                            | 优点                          | 缺点                             | 适用场景             |
| ----------------------------------- | ----------------------------- | -------------------------------- | -------------------- |
| **Tensor Parallel (TP)**            | 降低单卡显存                  | TP 域内通信开销大，Cube 效率下降 | 稠密模型，大 MatMul  |
| **Sequence Parallel (SP)**          | 切分 LayerNorm/Dropout 显存   | 需配合 TP，通信复杂              | 长序列训练           |
| **Pipeline Parallel (PP)**          | 降低静态显存                  | Bubble（空泡）、计算/显存不均衡  | 模型层数多           |
| **Virtual PP**                      | 减少 Bubble                   | Send/Recv 次数增加，动态显存上升 | 深层模型             |
| **Expert Parallel (EP)**            | 切分 MoE 专家，降显存         | 专家负载不均 → 显存瓶颈          | MoE 模型             |
| **Distributed Optimizer (ZeRO-DP)** | 降低优化器状态 & 参数副本内存 | 通信量增加                       | 所有场景（尤其微调） |

> 💡 **组合建议**：
>
> - MoE 模型常用 **TP + EP + DP** 混合并行；
> - 长序列场景引入 **Ring/Ulysses SP**；
> - 使用 **Noop Layer** 或 **Dynamic PP** 手动均衡 PP Stage。

### 四、MindSpeed LLM 支持的关键特性（Ascend 平台）

#### 🔹 长序列优化

- Ring Attention（Megatron-Ring）
- Ulysses SP（DeepSpeed-Ulysses）
- 混合长序列并行（Ring + Ulysses）
- Adaptive-CP（序列重映射 & 调度寻优）

#### 🔹 MoE 优化

- AllGather / All2All Dispatcher 选择
- 异步通信（`--moe-permutation-async-comm`）
- EP 通信隐藏

#### 🔹 通用加速

- Noop Layer（插入空操作层，均衡 PP 计算）
- Dynamic PP（手动指定每 Stage 层数）
- Recompute in Advance（解耦反向依赖，降 Bubble）

#### 🔹 显存优化

- 参数副本复用、Swap Attention、激活重计算、BF16 优化器

#### 🔹 融合算子（Fused Kernels）

- FlashAttention、Fused Ring Attention Update（长序列）
- Fused Token Permute/Unpermute（MoE）
- GMM（Grouped MatMul，用于小专家 MoE）
- Matmul Add（梯度累积）
- Fused SwiGLU / Rotary / RMSNorm（通用）

### 五、总结：性能优化“暗线”

> **“识别瓶颈 → 选择策略 → 应用特性”**
>
> - 先判断是 **Device Bound** 还是 **Host Bound**；
> - 再根据 **模型结构**（稠密/MoE）、**任务类型**（预训练/微调/推理）、**序列长度** 选择并行策略；
> - 最后启用 MindSpeed 对应的 **加速特性** 与 **融合算子**，实现端到端优化。
