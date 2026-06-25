
> 关键要点

> 在人工智能快速发展的今天，高效利用大型语言模型变得越来越重要。本文将介绍如何以计算高效的方式，使用低秩适应（LoRA）对 LLM 进行微调。

## 为什么要微调？

预训练的大型语言模型通常被称为基础模型，原因在于它们在各类任务上表现良好，可作为目标任务微调的起点。正如上一篇文章（[了解大型语言模型的参数高效微调：从 Prefix Tuning 到 LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)）所述，微调能使模型适应目标领域和目标任务。然而，模型越大，参数更新的计算成本越高。

作为全层更新的替代方案，研究者已开发了多种参数高效方法，如 Prefix Tuning 和 Adapters。目前，一种更为流行的参数高效微调技术是 [Low-Rank Adaptation（LoRA），Hu et al.](https://arxiv.org/abs/2106.09685)。什么是 LoRA？它如何工作？与其他微调方法相比表现如何？本文将逐一解答。

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-1.jpg" alt="PCA 变换" style="zoom:50%;" />

## 使权重更新更高效

基于上述思想，[LoRA（Hu et al.）](https://arxiv.org/abs/2106.09685)提出将权重变化 $\Delta W$ 分解为低秩表示。（严格来说，LoRA 并非直接分解矩阵，而是通过反向传播学习分解后的矩阵——这一细节后文会详细说明。）

在深入 LoRA 之前，先简要回顾常规微调的训练过程。什么是权重变化 $\Delta W$？设 $W$ 为某神经网络层的权重矩阵，通过常规反向传播可得到权重更新 $\Delta W$，通常计算为损失的负梯度乘以学习率：

$$
\Delta W = \alpha (-\nabla_W L)
$$

得到 $\Delta W$ 后，按如下方式更新原始权重：$W' = W + \Delta W$。下图对此进行了说明（为简化，省略偏置向量）：

或者，我们可以将权重更新矩阵分离，按如下方式计算输出：$h = Wx + \Delta Wx$

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-2.png" alt="常规反向传播" style="zoom: 33%;" />

其中 $x$ 代表输入，如下图所示。

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-3.png" alt="img" style="zoom: 50%;" />

为什么要这样做？目前这种替代公式有助于说明 LoRA 的原理，后文将进一步解释。

在神经网络中训练全连接（即"密集"）层时，权重矩阵通常具有满秩——即矩阵没有线性相关（"冗余"）的行或列。相反，低秩意味着矩阵存在冗余行或列。

虽然预训练模型的权重在预训练任务上具有满秩，但 LoRA 作者引用 [Aghajanyan](https://arxiv.org/abs/2012.13255)（2020）的研究指出：预训练大型语言模型在适应新任务时具有较低的"内在维度"。

低内在维度意味着数据可通过低维空间有效表示或近似，同时保留大部分关键信息和结构。换言之，我们可以将适应任务的新权重矩阵分解为低维（更小）矩阵，而不会丢失太多重要信息。

例如，假设 $\Delta W$ 是 $A \times B$ 维权重矩阵的权重更新。我们可以将其分解为两个更小的矩阵：$\Delta W = W_A W_B$，其中 $W_A$ 是 $A \times r$ 维矩阵，$W_B$ 是 $r \times B$ 维矩阵。这里保持原始权重 $W$ 不变，只训练新矩阵 $W_A$ 和 $W_B$。这就是 LoRA 方法的核心思想，如下图所示。

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-4.png" alt="img" style="zoom:50%;" />



### 选择秩

上图中的 $r$ 是一个超参数，用于指定低秩矩阵的秩。较小的 $r$ 意味着更简单的低秩矩阵，适应过程中需学习的参数更少，可加快训练速度并降低计算需求。然而，$r$ 越小，低秩矩阵捕获任务特定信息的能力越弱，可能导致适应质量下降。

总之，在 LoRA 中选择 $r$ 值需要在模型复杂度、适应能力和过拟合/欠拟合风险之间权衡。建议用不同的 $r$ 值进行实验，以找到最佳平衡点。

### 实现 LoRA

LoRA 的实现相对简单，可将其视为 LLM 中全连接层的改进前向传播。伪代码如下：

```python
input_dim = 768  # 例如，预训练模型的隐藏尺寸
output_dim = 768  # 例如，层的输出尺寸
rank = 8  # 低秩适应的秩'r'
W = ... # 来自具有形状input_dim x output_dim的预训练网络

W_A = nn.Parameter(torch.empty(input_dim, rank)) # LoRA权重A
W_B = nn.Parameter(torch.empty(rank, output_dim)) # LoRA权重B

# 初始化LoRA权重
nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
nn.init.zeros_(W_B)

def regular_forward_matmul(x, W):
    h = x @ W
    return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # 常规矩阵乘法
    h += x @ (W_A @ W_B)alpha # 使用缩放的LoRA权重
    return h
```

在上述伪代码中，`alpha` 是一个缩放因子，用于调整组合结果（原始模型输出加低秩适应）的幅度。它平衡了预训练模型的知识和新的任务特定适应——默认情况下 `alpha` 通常设为 1。另请注意，$W_A$ 初始化为小随机权重，而 $W_B$ 初始化为 0，因此训练开始时 $\Delta W = W_A W_B = 0$，即从原始权重开始训练。

### 参数效率

如果引入了新的权重矩阵，为何还能实现参数高效？关键在于新矩阵 $W_A$ 和 $W_B$ 可以非常小。例如，假设 $A=100$，$B=500$，则 $\Delta W$ 大小为 $100 \times 500 = 50,000$。将其分解为一个 $100 \times 5$ 的矩阵 $W_A$ 和一个 $5 \times 500$ 的矩阵 $W_B$，两者总共仅有 $5 \times 100 + 5 \times 500 = 3,000$ 个参数。

### 减少推理开销

在实践中，如果训练后保持原始权重 $W$ 与矩阵 $W_A$、$W_B$ 分离，推理时会产生额外计算步骤，造成轻微效率损失。为此，可在训练后通过 $W' = W + W_A W_B$ 合并权重，类似于前文提到的 $W' = W + \Delta W$。

然而，保持 $W_A$ 和 $W_B$ 分离也有实际优势。例如，假设我们以预训练模型为基础，为不同客户创建各自的微调 LLM。此时无需为每位客户存储完整的 $W'$（对于拥有数十亿到数万亿参数的 LLM 而言，存储开销巨大），只需保留原始模型 $W$，并为每位客户存储轻量级的 $W_A$ 和 $W_B$ 即可。

以具体数字说明：一个完整的 7B LLaMA checkpoint 需要 23GB 存储空间，而选择 $r=8$ 时，LoRA 权重可小至 8MB。

### 实践效果

LoRA 在实践中表现如何？与全量微调及其他参数高效方法相比如何？根据 [LoRA 论文](https://arxiv.org/abs/2106.09685)，在多个任务特定基准测试中，LoRA 的建模性能略优于 [Adapters](https://arxiv.org/abs/2110.07280)、[Prompt Tuning](https://arxiv.org/abs/2104.08691) 和 [Prefix Tuning](https://arxiv.org/abs/2101.00190)。通常，LoRA 的性能甚至优于全层微调，如下表所示（ROUGE 是评估语言生成质量的指标）。

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-5.png" alt="img" style="zoom:50%;" />

值得注意的是，LoRA 与其他微调方法是正交的，即它可以与 Prefix Tuning 和 Adapters 组合使用。
