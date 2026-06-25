# Multi-Head Latent Attention (MLA)

> **约定**：所有计算使用行向量，即 $y = xW$。

## Q 的计算

$$
\begin{align}
\mathbf{c}_{t}^{Q} &= \mathbf{h}_{t} W^{DQ}, \\
[\mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} &= \mathbf{c}_{t}^{Q} W^{UQ}, \\
[\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \operatorname{RoPE}(\mathbf{c}_{t}^{Q} {W^{QR}}), \\
\mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}],
\end{align}
$$

其中：
- $\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_c^{\prime}}$ 是查询的压缩隐向量；
- $d_c^{\prime} (\ll d_h n_h)$ 表示查询压缩维度；
- $W^{DQ} \in \mathbb{R}^{d \times d_c^{\prime}}$、$W^{UQ} \in \mathbb{R}^{d_c^{\prime} \times d_h n_h}$ 分别是查询的下投影和上投影矩阵；
- $W^{QR} \in \mathbb{R}^{d_c^{\prime} \times d_h^R n_h}$ 用于生成携带 RoPE 的解耦查询。

## KV 的计算

$$
\begin{align}
\boxed{\color{blue} \mathbf{c}_{t}^{KV}} &= \mathbf{h}_{t} W^{DKV}, \\
[\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} &= \mathbf{c}_{t}^{KV} W^{UK}, \\
\boxed{\color{blue}\mathbf{k}_{t}^{R}} &= \operatorname{RoPE}(\mathbf{h}_{t} {W^{KR}}), \\
\mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\
[\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} &= \mathbf{c}_{t}^{KV} W^{UV},
\end{align}
$$

其中：
- $\mathbf{c}_{t}^{KV} \in \mathbb{R}^{d_c}$ 是键值的压缩隐向量；
- $d_c (\ll d_h n_h)$ 表示 KV 压缩维度；
- $W^{DKV} \in \mathbb{R}^{d \times d_c}$ 是下投影矩阵；
- $W^{UK}, W^{UV} \in \mathbb{R}^{d_c \times d_h n_h}$ 是键和值的上投影矩阵；
- $W^{KR} \in \mathbb{R}^{d \times d_h^R}$ 用于生成携带 RoPE 的解耦键；
- $\operatorname{RoPE}(\cdot)$ 表示应用旋转位置编码的操作。

> **注意**：对于 MLA，仅需缓存蓝色框中的向量（$\color{blue} \mathbf{c}_{t}^{KV}$ 和 $\color{blue}\mathbf{k}_{t}^{R}$），从而显著减少 KV 缓存大小，同时保持与标准多头注意力（MHA）相当的性能。

最终，注意力查询（$\mathbf{q}_{t, i}$）、键（$\mathbf{k}_{j, i}$）和值（$\mathbf{v}_{j, i}^{C}$）组合得到最终输出 $\mathbf{u}_{t}$：

$$
\begin{align}
\mathbf{o}_{t, i} &= \sum_{j=1}^{t} \operatorname{Softmax}_j\left(\frac{\mathbf{q}_{t, i} \mathbf{k}^T_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}\right) \mathbf{v}_{j, i}^{C}, \\
\mathbf{u}_{t} &= [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}] W^{O},
\end{align}
$$

其中 $W^{O} \in \mathbb{R}^{d_h n_h \times d}$ 是输出投影矩阵。

## 实际参数配置

- $d = \text{hidden\_size} = 7168$
- $d_c = \text{kv\_lora\_rank} = 512$
- $d_c^{\prime} = \text{q\_lora\_rank} = 1536$
- $n_h = \text{num\_heads} = 128$
- $d_h = \text{qk\_nope\_head\_dim} = 128$
- $d_h^R = \text{qk\_rope\_head\_dim} = 64$

- $W^{UQ}$ 和 $W^{QR}$ 可合并，$q\_head\_dim = qk\_nope\_head\_dim + qk\_rope\_head\_dim = 192$。
- $W^{DKV}$ 和 $W^{KR}$ 可合并，$kv\_lora\_rank + qk\_rope\_head\_dim = 576$。

## 矩阵吸收（Absorb）

考虑如下计算：

$$
Y = X A B, \quad C = A B
$$

其中：
- $X \in \mathbb{R}^{m \times d}$ 是输入隐状态（hidden states），
- $A \in \mathbb{R}^{d \times d_c}$、$B \in \mathbb{R}^{d_c \times n}$ 是权重矩阵，
- $C \in \mathbb{R}^{d \times n}$ 是 absorb 后的等效权重矩阵。

直接计算的 FLOPs 为：
$$
2 m d d_c + 2 m n d_c = 2 m d_c (d + n)
$$

合并权重后计算的 FLOPs 为：$2 m d n$

当 $d_c$ 较小时，通常有：
$$
\boxed{d n > d_c (d + n)}
$$
因此**不一定需要合并两个权重矩阵**！

不考虑 RoPE 部分，仅从 $\mathbf{c}^Q$ 和 $\mathbf{c}^{KV}$ 计算 $\mathbf{q}_i \mathbf{k}_i^T$（第 $i$ 个 head）：

$$
\begin{align*}
q_i k_i^T &= \boxed{\mathbf{c}^{Q} W^{UQ}_i} \; \boxed{(\mathbf{c}^{KV} W^{UK}_i)^T}, \\
          &= \boxed{\mathbf{c}^{Q} W^{UQ}_i (W^{UK}_i)^T} (\mathbf{c}^{KV})^T, \\
          &= \boxed{q_i (W^{UK}_i)^T} (\mathbf{c}^{KV})^T, & \text{(Absorb)} \\
          &= q_i \boxed{(\mathbf{c}^{KV} W^{UK}_i)^T}, & \text{(Normal)} \\
\end{align*}
$$

> ⚠️ **警告**：此处 “Absorb” 的真实含义是**利用矩阵乘法结合律**，优先将 $\mathbf{q}$ 与 $W^{UK}$ 结合，并缓存压缩隐向量 $\mathbf{c}^{KV}$。它**并非合并权重矩阵**，“Absorb” 这一命名具有一定误导性！

### 为什么计算时不把 $W^{UQ}_i (W^{UK}_i)^T$ 合并？

对单个 token、单个 head，FLOPs 分别为：
- 分开计算：$2 d_h (d_c^{\prime} + d_c) = 524288$
- 合并计算：$2 d_c^{\prime} d_c = 1572864 = 3 \times 524288$

**合并后计算量反而是原来的 3 倍！**

### 为什么 Prefill 阶段显式计算 k 和 v，而 Decode 阶段不需要？

假设输入 shape 如下：
- $\mathbf{q} : (b, n_h, s_q, d_h)$
- $\mathbf{c}^{KV} : (b, 1, s_{kv}, d_c)$
- $W^{UK} : (d_c, n_h d_h)$

#### **Prefill 阶段**（$s_q = s_{kv} = s$）

FLOPs 对比：
$$
\begin{align*}
T_{\text{Normal}} &= 2 b n_h d_h s (d_c + s), \\
T_{\text{Absorb}} &= 2 b n_h d_c s (d_h + s), \\
\frac{T_{\text{Normal}}}{T_{\text{Absorb}}} &= \frac{d_h (d_c + s)}{d_c (d_h + s)} = \frac{s + 512}{4s + 512} \in \left( \frac{1}{4}, 1 \right)
\end{align*}
$$

→ **Prefill 阶段 Normal 更快**，且此阶段是**计算瓶颈**，故显式计算 $\mathbf{q}$ 和 $\mathbf{k}$。

#### **Decode 阶段**（$s_q = 1, s_{kv} = s$）

FLOPs 对比：
$$
\begin{align*}
T_{\text{Normal}}^{K} &= 2 b n_h d_h (d_c + s), & \text{(缓存 k)} \\
T_{\text{Normal}}^{L} &= 2 b n_h d_h (d_c s + s), & \text{(缓存 latent)} \\
T_{\text{Absorb}} &= 2 b n_h d_c (d_h + s), \\
\frac{T_{\text{Normal}}^{K}}{T_{\text{Absorb}}} &= \frac{d_h (d_c + s)}{d_c (d_h + s)} = \frac{s + 512}{4s + 512} \in (0.25, 1) \\
\frac{T_{\text{Normal}}^{L}}{T_{\text{Absorb}}} &= \frac{513 s}{4s + 512} \in (0.99, 128.25)
\end{align*}
$$

虽然缓存 k 的计算量最小（极限为 Absorb 的 1/4），但 **Decode 阶段瓶颈是显存带宽**。

##### 内存读取量对比（bfloat16 精度）：

- **MLA (Absorb)**：$(b, n_h, 1, d_c) \times (b, 1, s, d_c)$
  $$
  M_{\text{MLA}} = 2 b d_c (n_h + s)
  $$

- **标准 MHA**：$(b, n_h, 1, d_h) \times (b, n_h, s, d_h)$
  $$
  M_{\text{MHA}} = 2 b d_h n_h (1 + s)
  $$

内存读取比例：
$$
\frac{M_{\text{MLA}}}{M_{\text{MHA}}} = \frac{d_c (n_h + s)}{d_h n_h (1 + s)} = \frac{128 + s}{32 (1 + s)}
$$

- 当 $s = 20$，比值 ≈ 0.22；
- 极限情况（$s \to \infty$）：比值 → $1/32$。

✅ 因此，**Decode 阶段采用 Absorb 方式**，可大幅降低显存带宽压力，并**复用 MQA（Multi-Query Attention）实现**。

## 矩阵吸收问题总结

“矩阵吸收”的本质是**如何应用矩阵乘法结合律**：

$$
\begin{align*}
Y &= (X A) B = X (A B), \\
Z &= (X W) Y = X (W Y),
\end{align*}
$$

其中 $A, B, W$ 均为权重矩阵。

**决策依据**应综合权衡：
- 计算量（FLOPs）
- 显存读写量（Memory Traffic）
- 当前阶段瓶颈（计算 or 带宽）

可借助 **Roofline Model** 进行系统性分析。
