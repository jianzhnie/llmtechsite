# 连续批处理 (Continuous Batching) 的原理与应用优化

**摘要**：本文旨在深入探讨大型语言模型 (LLM) 推理中的**连续批处理 (Continuous Batching)** 技术。我们将从注意力机制 (Attention Mechanism) 和 KV 缓存 (KV Caching) 的基础出发，推导出连续批处理如何通过优化吞吐量来显著提升LLM的服务性能。

在使用像 Qwen、Claude 或其他AI聊天机器人时，用户通常会观察到这样的现象：模型生成**第一个** Token 需要一段短暂的时间，随后 Token 会以稳定且快速的节奏逐个显示在屏幕上。这种现象源于 LLM 的本质：它们是复杂的**下一个 Token 预测器**。LLM 首先需要处理完整的输入提示词 (prompt) 来生成第一个新的 Token，随后进入一个**自回归 (autoregressive)** 循环，逐个生成后续 Token，直到模型决定终止生成。

这个生成过程的计算成本极高。每生成一个 Token，都需要将输入数据流经包含数十亿参数的模型网络。为了使这些模型在实际应用中具有实用性，尤其是在同时服务大量并发用户时，研究人员和工程师开发了一系列高效的推理优化技术。其中，最具影响力的优化之一就是**连续批处理 (Continuous Batching)**。该技术通过并行处理多个对话，并在任一对话完成时立即用新的等待请求替换它，从而最大化GPU的利用率和整体吞吐量。

为了理解连续批处理的工作原理及其在高负载服务场景中的卓越效率，我们首先需要回顾LLM处理Token的基础知识。



## 一、注意力机制 (Attention Mechanism)

注意力机制是LLM能够理解并生成连贯文本的核心。语言模型首先将输入文本切分为称为 **Token (词元)** 的片段（概念上可类比于“单词”，但一个词可能由多个Token构成）。对于每一个Token序列，网络都会计算并预测最有可能的下一个Token。

网络中的许多操作是**逐 Token (token-wise)** 的，例如层归一化 (Layer Normalization) 或矩阵乘法，它们的输出仅依赖于当前Token的内容。然而，为了在序列中的Token之间建立语义联系，模型需要引入Token之间能够相互影响的操作——这就是注意力机制的作用。**注意力层是序列中不同Token之间进行信息交互的唯一场所。** 理解注意力机制，即是理解网络如何将序列中的Token连接起来。

让我们以单个输入提示词为例，观察这一过程是如何实际运作的。

### 1. 预填充阶段 (Prefill Stage)

考虑初始输入提示词 "I am sure this project"，它被分词为 7 个 Token：`[<bos>, I, am, sure, this, pro, ject]`。这里的 `<bos>`（Beginning of Sequence，序列开始）是一个特殊的Token，用于标记新的对话的起点。

在网络内部，每个Token被表示为一个长度为 $d$（隐藏维度，Hidden Dimension）的向量。因此，这 $n=7$ 个输入Token构成了一个形状为 $(1, n, d)$ 的输入张量 $X$（其中 $1$ 为批大小）。

输入张量 $X$ 随后通过三个投影矩阵 $W_Q$、$W_K$ 和 $W_V$ 进行投影，分别生成**查询状态 (Query State)** $Q$、**键状态 (Key State)** $K$ 和**值状态 (Value State)** $V$。它们的形状均为 $(1, n, A)$，其中 $A$ 是注意力头 (Attention Head) 的维度。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/proj_and_mul.png" alt="proj_and_mul.png" style="zoom:33%;" />



注意力计算的核心步骤如下：

1. **相似度计算：** $Q$ 和 $K$ 的转置相乘 ($QK^T$)，用于测量序列中所有Token对之间的相似度（即关联性），生成一个形状为 $(1, n, n)$ 的得分张量。
   - **复杂度分析：** 这一步是注意力机制计算成本的主要来源。计算 $QK^T$ 需要 $O(n^2d)$ 次操作，其时间复杂度与序列长度 $n$ 的**平方**成正比。
2. **注意力掩码应用：** 随后，对 $QK^T$ 应用一个**布尔注意力掩码 (Boolean Attention Mask)**，以控制哪些Token可以交互。
   - 在文本生成任务中，我们通常使用**因果掩码 (Causal Mask)**，它强制要求每个Token只能关注（或影响）它自身及它之前的Token。位置 $(i, j)$ 的掩码值为 $\text{True}$ 表示 Token $j$ 可以影响 Token $i$。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/masking_and_softmax.png" alt="masking_and_softmax.png" style="zoom:33%;" />

3. **最终输出：** 在应用注意力掩码和逐行 Softmax 归一化后，将结果与值投影 $V$ 相乘，得到注意力层的输出。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/attention.png" alt="attention.png" style="zoom:33%;" />

这个过程——将整个输入序列送入模型并计算下一个Token的预测分数——被称为**预填充 (Prefill)**。由于LLM生成是自回归的，在这个阶段计算得到的许多中间结果可以被缓存和重用，从而为后续的Token生成（称为**解码 (Decoding)** 阶段）显著提速。

## 二、KV 缓存 (KV Caching)

为了生成下一个Token，模型仍然需要依赖于先前所有Token的键 ($K$) 和值 ($V$) 投影结果。如果每次都从头计算，就会浪费大量的计算资源。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/simple_attention.png" alt="simple_attention.png" style="zoom:50%;" />

**KV 缓存 (KV-cache)** 机制正是为了解决这个问题而设计的：它是一个存储了生成过程中所有先前Token的 $K$ 和 $V$ 状态的列表。通过重用这些缓存的状态，生成Token $n+1$ 的计算成本可以从预填充阶段的 $O(n^2)$ 降低到解码阶段的 $O(n)$，代价是 $O(n)$ 的内存开销。

### 1. 缓存大小计算

对于一个拥有 $L$ 个注意力层、$H$ 个注意力头且头维度为 $A$ 的模型，存储一个Token所需的总缓存大小为：

$$
\text{CacheSize}_{\text{token}} = 2 \times L \times A \times H \times \text{DtypeSize}
$$
其中，系数 $2$ 代表同时存储 $K$ 和 $V$。例如，对于 Llama-2-7B 模型 ($L=32, H=32, A=128$)，在 $\text{float16}$ 精度下 ($\text{DtypeSize}=2 \text{ Bytes}$)，每层每个Token需要 $2 \times 32 \times 128 = 8,192$ 个值，总共占用 $16 \text{ KB}$ 的内存。

### 2. 分块预填充 (Chunked Prefill)

在实际应用中，初始提示词的长度 $n$ 可能会非常大（例如，将代码库作为上下文输入）。如果存储 $n$ 个Token的激活值所需的显存超过了GPU的可用显存，我们就无法在一次前向传播中完成预填充。此时，我们必须将预填充操作拆分为多个**块 (chunks)**，这被称为**分块预填充 (Chunked Prefill)**。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/chunked_prefill.png" alt="chunked prefill.png" style="zoom:33%;" />

假设可用内存限制每次前向传播只能处理 $m$ 个Token。对于一个 $n=7$ 个Token的提示词，如果 $m=4$，我们就需要 $\lceil n/m \rceil = 2$ 个块。

利用 KV 缓存，我们可以增量地处理提示词：

1. 在第一个块的预填充过程中，计算并存储 KV 状态。
2. 在处理第二个块时，将已缓存的 KV 状态**前置 (prepend)** 到新的 KV 状态中，并相应地调整注意力掩码。

**核心洞察**：KV 缓存使得我们能够灵活地将预填充拆分成任意大小的块，以适应不同的内存限制，而不会丢失任何信息。

## 三、连续批处理 (Continuous Batching) 的优化

在模型服务场景中，我们的目标是同时为多个用户的请求生成Token，即最大化**吞吐量**（每秒生成的Token数）。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/padding.png" alt="padding.png" style="zoom:33%;" />

### 1. 朴素批处理的局限性

将多个提示词批处理在一起的**朴素方法 (Naïve Method)** 是在输入张量中增加一个**批次维度 (Batch Dimension)**。然而，这要求批次中的所有提示词必须具有相同的序列长度，因为张量必须是矩形的。为了满足这一要求，短序列通常需要通过**填充 (padding)** 到最长序列的长度。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/batched_generation.png" alt="batched_generation.png" style="zoom:33%;" />

这种**批处理生成 (Batched Generation)** 在所有序列长度相同时是高效的，但在长度变化时会造成极大的浪费：

- **填充浪费：** 当一个请求生成了特殊的 `<eos>` (End Of Sequence) Token 完成其生成时，它所占用的空间在批次中会持续浪费，直到批次中最长的请求完成。

虽然可以通过**动态调度 (Dynamic Scheduling)** 来解决第一个问题（将已完成的请求替换为新的等待请求），但这又引入了新的填充问题：新插入的请求需要进行预填充，而批次中的其他请求正在进行解码。新请求的全部长度几乎全部转化为填充Token，造成巨大的计算浪费。

![dynamic_batching.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/dynamic_batching.png)

**问题总结：** 填充成本随着批大小 $B$ 和提示词长度 $n$ 呈**二次方**增长。此外，像 CUDA graphs 等优化技术需要静态张量形状，这迫使我们将所有提示词填充到固定的最大长度，进一步加剧了浪费。

### 2. 参差批处理 (Ragged Batching)



连续批处理的关键在于**彻底消除批次维度**及其带来的填充问题。如果移除批次维度，将提示词批处理在一起的唯一方法就是将它们的Token序列**拼接 (concatenate)** 起来，形成一个超长的单序列。



<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/concatenate.png" alt="concatenate.png" style="zoom:33%;" />



这种将长度不均匀的序列拼接在一起的方式被称为**参差批处理 (Ragged Batching)**。它的核心优势在于**彻底消除了填充Token**，显著提高了吞吐量。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/ragged_batching.png" alt="ragged_batching.png" style="zoom:33%;" />

然而，拼接序列带来了一个问题：我们必须确保“提示词 0”的Token不会与“提示词 1”的Token进行交互。幸运的是，我们已经有工具来控制这种交互——**注意力掩码**。通过精心设计的注意力掩码，我们可以将所有请求的Token合并在一个批次中进行处理，同时确保因果关系和跨请求的隔离。

### 3. 连续批处理的完整算法

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/continuous_batching/continuous_batching.png" alt="continuous_batching.png" style="zoom:33%;" />

**连续批处理 (Continuous Batching)** 是参差批处理和动态调度的结合，旨在最大化 GPU 的内存和计算资源利用率。其调度算法的核心逻辑如下：

1. **确定内存预算 $m$：** 始终以达到每批次总Token数 $m$ 的内存预算为目标。
2. **优先解码序列：** 首先将所有处于**解码阶段**（每个只占用 1 个Token）的请求添加到当前批次中。
3. **填充预填充序列：** 利用**分块预填充**的灵活性，用处于**预填充阶段**的请求来填充批次的剩余空间。如果一个预填充请求的长度过长，则根据剩余容量将其拆分为合适的块。
4. **动态调度：** 一旦批次中的任一请求完成生成 (生成 `<eos>`)，立即将其从批次中移除，并用一个新的、等待处理的请求（以其第一个预填充块的形式）替换它。

通过在同一批次中高效地混合预填充和解码阶段，并结合动态调度和无填充的参差批处理，连续批处理显著提高了LLM服务系统的效率。

## 结论

**连续批处理**是现代高性能 LLM 服务系统的核心技术，它巧妙地结合了以下三种关键技术以最大化吞吐量：

- **KV 缓存 (KV Caching)**：避免在自回归生成过程中重复计算过去的Token表示。
- **分块预填充 (Chunked Prefill)**：允许模型在内存受限的情况下，灵活、增量地处理可变且超长的提示词。
- **参差批处理与动态调度 (Ragged Batching & Dynamic Scheduling)**：通过消除批次维度和填充，以及动态替换已完成请求，确保GPU始终处于高负载运行状态。

这种技术上的突破允许服务（如 ChatGPT）能够以高效率和低延迟同时处理成千上万的并发用户请求。
