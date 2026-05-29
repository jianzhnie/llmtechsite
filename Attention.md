# Transformer 注意力机制：MHA、MQA 与 GQA 的深度对比

## 背景

Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) 架构的提出彻底改变了自然语言处理（NLP）领域。该架构最初基于编码器-解码器（Encoder-Decoder）结构，随后演化出一系列变体：如仅包含编码器的 BERT ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805))，以及仅包含解码器的 GPT ([Radford et al., 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)) 系列。目前主流的大型语言模型（LLM），如 LLaMA ([Touvron et al., 2023](https://arxiv.org/abs/2302.13971)) 和 GPT-4，大多延续了**仅解码器（Decoder-only）**的架构。

## 符号定义

| **符号**                                   | **含义**                                                     |
| ------------------------------------------ | ------------------------------------------------------------ |
| $B$                                        | 批量大小（Batch Size）                                       |
| $S$                                        | 序列长度（Sequence Length）                                  |
| $d$                                        | 隐藏层维度 / 模型维度（Model Dimension）                     |
| $H$                                        | 注意力头数量（Number of Attention Heads）                    |
| $G$                                        | 分组数量（Group Number），用于 GQA                           |
| $d_{\text{head}}$                          | 每个注意力头的维度，通常 $d_{\text{head}} = d / H$           |
| $\mathbf{X}$                               | 输入张量，$\mathbf{X} \in \mathbb{R}^{B \times S \times d}$  |
| $\mathbf{Q}, \mathbf{K}, \mathbf{V}$       | 经过线性变换后的查询（Query）、键（Key）、值（Value）矩阵    |
| $W_Q, W_K, W_V$                            | 映射矩阵，$W \in \mathbb{R}^{d \times d}$，用于生成 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ |
| $W_O$                                      | 输出映射矩阵，$W_O \in \mathbb{R}^{d \times d}$              |
| $\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h$ | 第 $h$ 个注意力头对应的子矩阵                                |
| $\mathbf{K}^*, \mathbf{V}^*$               | MQA 中所有头共享的键和值矩阵                                 |

## Transformer 中的注意力机制

Transformer 的核心在于**自注意力机制（Self-Attention）**，它赋予了模型动态捕捉序列内部长程依赖的能力。

对于输入序列 $\mathbf{X} \in \mathbb{R}^{B \times S \times d}$，模型通过可训练的权重矩阵 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ 将其投影到三个子空间：
$$
\mathbf{Q} = \mathbf{X} W_Q, \quad \mathbf{K} = \mathbf{X} W_K, \quad \mathbf{V} = \mathbf{X} W_V
$$
Transformer 采用的是**缩放点积注意力（Scaled Dot-Product Attention）**。其基本思想是计算查询与键之间的相关性，并将其作为权重对值进行加权求和：
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_{\text{head}}}}\right)\mathbf{V}
$$

### 1. 多头注意力（Multi-Head Attention, MHA）

MHA 通过将 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 划分为 $H$ 个独立的头，允许模型在不同的表示子空间中并行学习信息：
$$
\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
$$
其中每个头的计算公式为：
$$
\text{head}_h = \text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_{\text{head}}}}\right)\mathbf{V}_h
$$
**MHA 的优势：**

- **多维度特征捕捉**：不同头可以关注序列中不同的语法或语义特征。
- **增强表达能力**：通过子空间集成，提升了模型对复杂依赖关系的建模精度。
- **计算并行性**：各头的计算逻辑相互独立，适合 GPU/TPU 硬件加速。

#### 缩放因子 $\sqrt{d_{\text{head}}}$ 的必要性

引入缩放因子的主要目的是**维持数值稳定性**，防止 Softmax 函数进入梯度饱和区：

1. **防止梯度消失**：若点积结果过大，Softmax 的输出会集中在极小或极大的区域，导致导数接近于 0。

2. **数学推导**：假设 $\mathbf{q}$ 和 $\mathbf{k}$ 的各分量是独立同分布的随机变量，且满足均值为 0、方差为 1。则其点积 $\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_{\text{head}}} q_i k_i$ 的方差为 $d_{\text{head}}$。

   - 通过除以 $\sqrt{d_{\text{head}}}$，可以使缩放后点积的方差恢复为 1：
     $$
     \text{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_{\text{head}}}}\right) = \frac{1}{d_{\text{head}}} \text{Var}(\mathbf{q} \cdot \mathbf{k}) = \frac{d_{\text{head}}}{d_{\text{head}}} = 1
     $$

### 2. 多查询注意力（Multi-Query Attention, MQA）

MQA ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) 是一种旨在提升推理效率的变体。在 MQA 中，所有的查询头共享同一组键（Key）和值（Value）。

其核心逻辑如下：
$$
\mathbf{K}^* = \text{Linear}_{shared}(\mathbf{X}), \quad \mathbf{V}^* = \text{Linear}_{shared}(\mathbf{X})
$$

$$
\text{MQA}(\mathbf{Q}, \mathbf{K}^*, \mathbf{V}^*) = \text{Concat}(\text{Attention}(\mathbf{Q}_1, \mathbf{K}^*, \mathbf{V}^*), \dots, \text{Attention}(\mathbf{Q}_H, \mathbf{K}^*, \mathbf{V}^*)) W_O
$$

**核心价值**：显著减少了推理阶段 **KV Cache** 的显存占用和访存开销（Memory Bandwidth），这对长文本生成尤为重要。

### 3. 分组查询注意力（Grouped-Query Attention, GQA）

GQA ([Ainslie, 2023](https://arxiv.org/pdf/2305.13245)) 是 MHA 与 MQA 的折中方案，它在保持推理效率的同时，尽可能保留多头机制的表达能力。

GQA 将查询头分为 $G$ 组，每组内的查询头共享一对 KV 头：

- 若 $G=1$，则等同于 MQA。
- 若 $G=H$，则等同于 MHA。

### 三者对比总结

- **MHA**：$H$ 个 Query 头，$H$ 个 KV 头。性能最优，但推理时 KV Cache 显存压力大。
- **MQA**：$H$ 个 Query 头，$1$ 个 KV 头。推理速度最快，显存占用最低，但可能损失一定的模型容量。
- **GQA**：$H$ 个 Query 头，$G$ 个 KV 头（$1 < G < H$）。在速度与性能之间取得最佳平衡，是目前主流大模型（如 Llama 3）的首选。

## 复杂度分析

### 1. 时间复杂度（Time Complexity）

无论是 MHA、MQA 还是 GQA，对于**完整序列的一次性前向传播**，其计算复杂度量级是相同的。

- **矩阵乘法 $\mathbf{Q}\mathbf{K}^\top$**：复杂度为 $\mathcal{O}(B \times S^2 \times d)$。
- **加权求和（与 $\mathbf{V}$ 相乘）**：复杂度同样为 $\mathcal{O}(B \times S^2 \times d)$。
- **总体量级**：$\mathcal{O}(B \times S^2 \times d)$。注意，注意力机制的计算开销随序列长度 $S$ 呈二次方增长。

增量解码（Incremental Decoding）场景：

在 LLM 推理时，利用 KV Cache 缓存历史信息。每生成一个新 Token，只需计算当前 Query 与历史 KV 的关联：

- **单步复杂度**：$\mathcal{O}(B \times H \times S_{\text{past}} \times d_{\text{head}}) = \mathcal{O}(B \times S_{\text{past}} \times d)$。

### 2. 空间复杂度（Space Complexity）

空间复杂度主要由**参数量**和**中间激活值（KV Cache）**组成。

- **参数量**：$W_Q, W_K, W_V, W_O$ 四个矩阵的参数量均为 $\mathcal{O}(d^2)$，总参数量约为 $4d^2$。MQA/GQA 虽然减少了 KV 头的数量，但由于投影矩阵的维度变化，其参数量微减，通常仍视为 $\mathcal{O}(d^2)$。
- **KV Cache 显存占用**：这是 MQA/GQA 优化的核心。
  - **MHA**：每个 Token 需要存储 $2 \times H \times d_{\text{head}} = 2d$ 个数值。
  - **MQA**：每个 Token 仅需存储 $2 \times 1 \times d_{\text{head}} = 2d/H$ 个数值。显存占用降低为原来的 $1/H$。
  - **GQA**：每个 Token 需要存储 $2 \times G \times d_{\text{head}}$ 个数值。显存占用介于两者之间。

## 结论

在 LLM 时代，显存带宽往往是推理性能的瓶颈（Memory-Bound）。**MQA** 通过极致的共享策略解决了访存效率问题，但可能影响复杂任务的表现；**GQA** 则通过灵活的分组机制，在推理延迟、显存占用与模型效果之间找到了黄金平衡点，已成为当前工业界的事实标准。
