# 旋转式位置编码（RoPE）

旋转式位置编码（RoPE）最早由论文 `[1]` 提出，是一种将相对位置信息依赖集成到 self-attention 中并提升 Transformer 架构性能的位置编码方式。目前广受关注的 LLaMA 模型也采用了该位置编码方式。

## **基本概念**

首先论文中定义一个长度为 $N$ 的输入序列为：
$$
S_{N}=\{ {token}_{i} \}_{i=1}^{N} \\
$$
其中 ${token}_i$ 表示输入序列中第 $i$ 个 token，而输入序列 $S_N$ 对应的 embedding 表示为：
$$
E_{N}=\{ x_i \}_{i=1}^N\\
$$
其中 $x_i$ 表示第 $i$ 个 token $w_i$ 对应的 $d$ 维词嵌入向量。

接着在执行 self-attention 之前，会用词嵌入向量计算 $\mathbf{q}, \mathbf{k}, \mathbf{v}$ 向量并同时加入位置信息，函数公式表达如下：
$$
q_m=f_q(x_m,m) \\ k_n=f_k(x_n,n) \\ v_n=f_v(x_n,n) \\
$$
其中 $q_m$ 表示第 $m$ 个 token 对应的词向量 $x_m$ 集成位置信息 $m$ 之后的 query 向量。而 $k_n$ 和 $v_n$ 则表示第 $n$ 个 token 对应的词向量 $x_n$ 集成位置信息 $n$ 之后的 key 和 value 向量。

基于 Transformer 的位置编码方法都着重于构造合适的 $f_{\{q,k,v\}}$ 函数形式。计算第 $m$ 个词嵌入向量 $x_m$ 对应的 self-attention 输出结果时，$q_m$ 与所有 $k_n$ 计算 attention score，然后将 attention score 乘以对应的 $v_n$ 再求和，得到输出向量 $o_m$：

$$
a_{m,n}=\frac{exp(\frac{q_m^Tk_n}{\sqrt{d}})}{\sum_{j=1}^Nexp(\frac{q_m^Tk_j}{\sqrt{d}})} \\ o_m=\sum_{n=1}^Na_{m,n}v_n \\
$$

## **绝对位置编码**

对于位置编码，常规做法是在计算 query、key 和 value 向量之前，先计算一个位置编码向量 $p_i$ 加到词嵌入 $x_i$ 上。位置编码向量 $p_i$ 同样是 $d$ 维向量，然后再乘以对应的变换矩阵 $W_{\{q,k,v\}}$：
$$
f_{\{q,k,v\}}(x_i,i)=W_{\{q,k,v\}}(x_i+p_i) \\
$$
而经典的位置编码向量 $p_i$ 的计算方式是：
$$
p_{i,2t}=sin(\frac{i}{10000^{\frac{2t}{d}}}) \\ p_{i,2t+1}=cos(\frac{i}{10000^{\frac{2t}{d}}})\\
$$
其中 $p_{i,2t}$ 表示 $d$ 维位置向量 $p_i$ 中第 $2t$ 个分量（偶数索引位置）的计算公式，而 $p_{i,2t+1}$ 对应第 $2t+1$ 个分量（奇数索引位置）的计算公式。

## **旋转式位置编码**

接下来介绍 Rotary Transformer（RoFormer）模型。它的主要改动是引入"旋转式位置编码（Rotary Position Embedding，RoPE）"，这是一种配合 Attention 机制能达到"以绝对位置编码的方式实现相对位置编码"的设计。正因如此，它也是目前唯一一种可用于线性 Attention 的相对位置编码。

### 基本思路

在 RoPE 中，出发点是"通过绝对位置编码的方式实现相对位置编码"。这一设计既有理论上的优雅之处，也有实践上的实用价值，例如它可以扩展到线性 Attention 中。

在机器学习中，我们通常只关注实数，但对于旋转嵌入来说，使用复数作为空间的基域在数学上更为方便。先考虑二维情形，然后借助复数来求解。将 query 向量和 key 向量的元素视为单个复数，我们使用 $\mathbb{C}^{d/2}$ 而非通常的 $\mathbb{R}^{d}$ 空间来表示。具体而言，不再将 $\mathbf{q}=(q_1,q_2,q_3,q_4,\ldots,q_{d})$ 视为 $d$ 维实数向量，而是将其视为 $\mathbf{q}=(q_1+iq_2, q_3+iq_4,\ldots q_{d-1} + iq_{d})\in\mathbb{C}^{d/2}$。若 $d$ 为奇数，可用零虚部填充以确保对齐。

$\mathbf{q}$ 和 $\mathbf{k}$ 分别为 query 向量和 key 向量，$m$ 和 $n$ 分别为相应 token 的绝对位置。假设 $f(\mathbf{x},\ell)$ 是一个函数，它接收位于位置 $\ell$ 的嵌入 $\mathbf{x}$，并输出一个包含相对位置信息的新嵌入。我们假设通过下述运算来给 $\mathbf{q}, \mathbf{k}$ 添加绝对位置信息：
$$
\tilde{\boldsymbol{q}}_{m}=\boldsymbol{f}(\boldsymbol{q}, m), \quad \tilde{\boldsymbol{k}}_{n}=\boldsymbol{f}(\boldsymbol{k}, n)
$$

也就是说，分别为 $\boldsymbol{q}, \boldsymbol{k}$ 设计操作 $f(\cdot, m), \boldsymbol{f}(\cdot, n)$，使得经过该操作后，$\tilde{\boldsymbol{q}}_{m}, \tilde{\boldsymbol{k}}_{n}$ 就带有了位置 $m, n$ 的绝对位置信息。Attention 的核心运算是内积，因此我们希望内积的结果带有相对位置信息，假设存在恒等关系：

$$
\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle=g(\boldsymbol{q}, \boldsymbol{k}, m-n)
$$

因此需要给出该恒等式的一个（尽可能简单的）解。求解过程还需要初始条件，显然可以合理地设 $f(\boldsymbol{q}, 0)=\boldsymbol{q}$ 和 $f(\boldsymbol{k}, 0)=\boldsymbol{k}$。

### 求解过程

在复数中有 $\langle\boldsymbol{q}, \boldsymbol{k}\rangle=\operatorname{Re}\left[\boldsymbol{q} \boldsymbol{k}^{*}\right]$，$\operatorname{Re}[]$ 代表复数的实部，所以有：
$$
\operatorname{Re}\left[\boldsymbol{f}(\boldsymbol{q}, m) \boldsymbol{f}^{*}(\boldsymbol{k}, n)\right]=g(\boldsymbol{q}, \boldsymbol{k}, m-n)
$$

简单起见，假设存在复数 $\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n)$，使得 $f(\boldsymbol{q}, m) \boldsymbol{f}^{*}(\boldsymbol{k}, n)=\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n)$。然后用复数的指数形式，设：

$$
\begin{aligned}
\boldsymbol{f}(\boldsymbol{q}, m) & =R_{f}(\boldsymbol{q}, m) e^{\mathrm{i} \Theta_{f}(\boldsymbol{q}, m)} \\
\boldsymbol{f}(\boldsymbol{k}, n) & =R_{f}(\boldsymbol{k}, n) e^{\mathrm{i} \Theta_{f}(\boldsymbol{k}, n)} \\
\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) & =R_{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) e^{\mathrm{i} \Theta_{g}(\boldsymbol{q}, \boldsymbol{k}, m-n)}
\end{aligned}
$$

$$
\begin{aligned}
\boldsymbol{R}_{f}(\boldsymbol{q}, m) R_{f}(\boldsymbol{k}, n) & =R_{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) \\
\Theta_{f}(\boldsymbol{q}, m)-\Theta_{f}(\boldsymbol{k}, n) & =\Theta_{g}(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{aligned}
$$

对于第一个方程，代入 $m=n$ 得到：

$$
R_{f}(\boldsymbol{q}, m) R_{f}(\boldsymbol{k}, m)=R_{g}(\boldsymbol{q}, \boldsymbol{k}, 0)=R_{f}(\boldsymbol{q}, 0) R_{f}(\boldsymbol{k}, 0)=\|\boldsymbol{q}\|\|\boldsymbol{k}\|
$$

最后一个等号源于初始条件 $f(\boldsymbol{q}, 0)=\boldsymbol{q}$ 和 $f(\boldsymbol{k}, 0)=\boldsymbol{k}$。因此可以直接设 $R_{f}(\boldsymbol{q}, m)=\|\boldsymbol{q}\|, R_{f}(\boldsymbol{k}, m)=\|\boldsymbol{k}\|$，即它不依赖于 $m$。至于第二个方程，同样代入 $m=n$ 得到：

$$
\Theta_{f}(\boldsymbol{q}, m)-\Theta_{f}(\boldsymbol{k}, m)=\Theta_{g}(\boldsymbol{q}, \boldsymbol{k}, 0)=\Theta_{f}(\boldsymbol{q}, 0)-\Theta_{f}(\boldsymbol{k}, 0)=\Theta(\boldsymbol{q})-\Theta(\boldsymbol{k})
$$

这里的 $\Theta(\boldsymbol{q}), \Theta(\boldsymbol{k})$ 是 $\boldsymbol{q}, \boldsymbol{k}$ 本身的幅角，最后一个等号同样源于初始条件。根据上式可得：

$\Theta_{f}(\boldsymbol{q}, m)-\Theta(\boldsymbol{q})=\Theta_{f}(\boldsymbol{k}, m)-\Theta(\boldsymbol{k})$，所以 $\Theta_{f}(\boldsymbol{q}, m)-\Theta(\boldsymbol{q})$ 应是一个只与 $m$ 相关、与 $\boldsymbol{q}$ 无关的函数，记为 $\varphi(m)$，即 $\Theta_{f}(\boldsymbol{q}, m)=\Theta(\boldsymbol{q})+\varphi(m)$。接着代入 $n=m-1$，整理得到：

$$
\varphi(m)-\varphi(m-1)=\Theta_{g}(\boldsymbol{q}, \boldsymbol{k}, 1)+\Theta(\boldsymbol{k})-\Theta(\boldsymbol{q})
$$

即 $\{\varphi(m)\}$ 是等差数列，代入初始值 $\varphi(0) = 0,  \varphi(1) = \theta$，解得 $\varphi(m)=m \theta$。

将前面所有的公式推导汇总，即可得到 Rotary Position Embedding 的最终表达式：
$$
f(\mathbf{q}, m) = R_f(\mathbf{q}, m)e^{i\Theta_f(\mathbf{q}, m)}=\mathbf{q}e^{i(\Theta(\mathbf{q})+m\mathbf{\theta})} = \sum_{j=1}^{d/2} q_je^{im\theta_j} \vec{e_j}
$$
因此，对于任意的 $0 < \varepsilon \leq \frac \pi {2N}$，其中 $N$ 是最大序列长度。当按元素计算 $\mathbf{q}$ 和 $\mathbf{k}$ 时，以 $j$ 作为元素索引，RoPE 可以表示如下：
$$
\begin{align}
\mathrm{RoPE}(x, m) &= xe^{mi\varepsilon} \\
\langle \mathrm{RoPE}(q_j, m), \mathrm{RoPE}(k_j, n)\rangle &= \langle q_j e^{mi\varepsilon}, k_j e^{ni\varepsilon} \rangle \\
&= q_j k_j e^{mi\varepsilon} \overline{e^{ni\varepsilon}} \\
&= q_j k_j e^{(m - n)i\varepsilon} \\
&= \mathrm{RoPE}(q_j k_j, m - n)
\end{align}
$$
由于与复数相比，计算机更喜欢实数和矩阵，因此将此表达式转换为矩阵方程很方便。
$$
f(\mathbf{q}, m) =
\begin{pmatrix}
M_1 & & & \\
& M_2 & & \\
& & \ddots & \\
& & & M_{d/2}
\end{pmatrix}
\begin{pmatrix}
q_1\\
q_2\\
\vdots\\
q_d
\end{pmatrix} = \mathbf{\Theta_m Q_m} = \mathbf{\Theta_m W_q X_m}
$$
其中，$M_j=\begin{pmatrix}\cos m\theta_j & -\sin m\theta_j \\\sin m\theta_j & \cos m\theta_j\end{pmatrix}$，$\mathbf{\Theta_m}$ 为块对角矩阵，$\mathbf{W_q}$ 为可学习的 query 权重，$\mathbf{X_m}$ 为位置 $m$ 处的嵌入。

### 编码形式

综上，我们得到二维情况下用复数表示的 RoPE：

$$
\boldsymbol{f}(\boldsymbol{q}, m)=R_{f}(\boldsymbol{q}, m) e^{\mathrm{i} \Theta_{f}(\boldsymbol{q}, m)}=\|q\| e^{\mathrm{i}(\Theta(\boldsymbol{q})+m \theta)}=\boldsymbol{q} e^{\mathrm{i} m \theta}
$$

根据复数乘法的几何意义，该变换实际上对应着向量的旋转，因此称之为"旋转式位置编码"。它还可以写成矩阵形式：

$$
\boldsymbol{f}(\boldsymbol{q}, m)=\left(\begin{array}{cc}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{array}\right)\left(\begin{array}{l}
q_{0} \\
q_{1}
\end{array}\right)
$$

由于内积满足线性叠加性，任意偶数维的 RoPE 都可以表示为二维情形的拼接，即：

$$
\underbrace{\left(\begin{array}{ccccccc}
\cos m \theta_{0} & -\sin m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_{0} & \cos m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_{1} & -\sin m \theta_{1} & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_{1} & \cos m \theta_{1} & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d / 2-1} & -\sin m \theta_{d / 2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d / 2-1} & \cos m \theta_{d / 2-1}
\end{array}\right)}\left(\begin{array}{c}
q_{0} \\
q_{1} \\
q_{2} \\
q_{3} \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right)
$$

也就是说，给位置为 $m$ 的向量 $\mathbf{q}$ 乘上矩阵 $\boldsymbol{R}_{m}$、位置为 $n$ 的向量 $\mathbf{k}$ 乘上矩阵 $\boldsymbol{R}_{n}$，用变换后的 $Q, K$ 序列做 Attention，则 Attention 就自动包含相对位置信息，因为成立恒等式：

$$
\left(\boldsymbol{\mathcal { R }}_{m} \boldsymbol{q}\right)^{\top}\left(\boldsymbol{\mathcal { R }}_{n} \boldsymbol{k}\right)=\boldsymbol{q}^{\top} \boldsymbol{\mathcal { R }}_{m}^{\top} \boldsymbol{\mathcal { R }}_{n} \boldsymbol{k}=\boldsymbol{q}^{\top} \boldsymbol{\mathcal { R }}_{n-m} \boldsymbol{k}
$$

值得指出的是，$\boldsymbol{R}_{m}$ 是一个正交矩阵，它不会改变向量的模长，因此通常不会影响原模型的稳定性。
由于 $\boldsymbol{R}_{m}$ 的稀疏性，直接用矩阵乘法来实现会浪费算力，推荐通过下述方式来实现 RoPE：

$$
\left(\begin{array}{c}
q_{0} \\
q_{1} \\
q_{2} \\
q_{3} \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\cos m \theta_{0} \\
\cos m \theta_{0} \\
\cos m \theta_{1} \\
\cos m \theta_{1} \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_{d / 2-1}
\end{array}\right)+\left(\begin{array}{c}
-q_{1} \\
q_{0} \\
-q_{3} \\
q_{2} \\
\vdots \\
-q_{d-1} \\
q_{d-2}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin m \theta_{0} \\
\sin m \theta_{0} \\
\sin m \theta_{1} \\
\sin m \theta_{1} \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_{d / 2-1}
\end{array}\right)
$$

其中 $\otimes$ 是逐元素相乘，即 Numpy、Tensorflow 等计算框架中的 `*` 运算。从这个实现也可以看到，RoPE 可以视为三角函数式位置编码的变体。

### LLaMA 模型中的 RoPE

LLaMA 模型使用了 Rotary Position Embedding。对于 $\mathbf{Q}$ 的第 $m$ 个位置向量 $\mathbf{q}$，通过以下方式注入位置编码。

#### Step1:初始化 $\theta$ 矩阵

$$
\left(\begin{array}{c}
\theta_{0}  & \theta_{1}   &  \cdots & \theta_{d/2-1}   &
\theta_{0}  & \theta_{1}   &  \cdots & \theta_{d/2-1}   \\

\theta_{0}  & \theta_{1}   &  \cdots & \theta_{d/2-1}    &
\theta_{0}  & \theta_{1}   &  \cdots & \theta_{d/2-1}     \\

2\theta_{0} & 2\theta_{1}  &  \cdots & 2\theta_{d/2-1}   &
2\theta_{0} & 2\theta_{1}  &  \cdots & 2\theta_{d/2-1}   \\

 \vdots     & \vdots       &  \ddots &  \vdots            &
 \vdots     & \vdots       &  \ddots &  \vdots            \\

m\theta_{0} & m\theta_{1}  &  \cdots & m\theta_{d/2-1}   &
m\theta_{0} & m\theta_{1}  &  \cdots & m\theta_{d/2-1}   \\
\end{array}\right)
$$

#### Step2:计算 $cos$ 矩阵和 $sin$ 矩阵

$$
\left(\begin{array}{c}

\cos\theta_{0}  & \cos\theta_{1}   & \cdots & \cos\theta_{d/2-1}   &
\cos\theta_{0}  & \cos\theta_{1}   &  \cdots & \cos\theta_{d/2-1}   \\

\cos\theta_{0}  & \cos\theta_{1}   &  \cdots & \cos\theta_{d/2-1}   &
\cos\theta_{0}  & \cos\theta_{1}   &  \cdots & \cos\theta_{d/2-1}   \\

\cos2\theta_{0} & \cos2\theta_{1}  &  \cdots & \cos2\theta_{d/2-1}  &
\cos2\theta_{0} & \cos2\theta_{1}  &  \cdots & \cos2\theta_{d/2-1}  \\

 \vdots     & \vdots       &  \ddots &  \vdots     &  \vdots
 \vdots     & \vdots       &  \ddots &  \vdots                 \\

\cos m\theta_{0} & \cos  m\theta_{1}  &  \cdots &\cos  m\theta_{d/2-1}  &
\cos m\theta_{0} & \cos  m\theta_{1}  &  \cdots & \cos  m\theta_{d/2-1}   \\

\end{array}\right)
$$

#### Step3：计算 Query 向量

```python
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

$$
\left(\begin{array}{c}
q_{0} \\
q_{1} \\
\vdots \\
q_{d / 2-1} \\
q_{d /2} \\
\vdots \\
q_{d-2}\\
q_{d-1}
\end{array}\right) \otimes

\left(\begin{array}{c}
\cos m \theta_{0} \\
\cos m \theta_{1} \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\cos m \theta_{0} \\
\cos m \theta_{1} \\
\vdots \\
\cos m \theta_{d / 2-1} \\
\end{array}\right)

+\left(\begin{array}{c}
-q_{d/2} \\
-q_{d/2+1} \\
\vdots \\
-q_{d-1} \\
q_{0} \\
q_{1} \\
\vdots \\
q_{d/2-1}
\end{array}\right) \otimes\left(\begin{array}{c}
\sin m \theta_{0} \\
\sin m \theta_{1} \\
\vdots \\
\sin m \theta_{d / 2-1} \\
\sin m \theta_{0} \\
\sin m \theta_{1} \\
\vdots \\
\sin m \theta_{d / 2-1}
\end{array}\right)
$$



## RoPE 证明过程

#### 简单证明

简单起见，先假设 $\boldsymbol{q}_{m}, \boldsymbol{k}_{n}$ 是所在位置分别为 $m, n$ 的二维行向量。既然是二维，可以将其当作复数来运算。Attention 的关键之处在于向量的内积，用复数表示为：

$$
\left\langle\boldsymbol{q}_{m}, \boldsymbol{k}_{n}\right\rangle=\operatorname{Re}\left[\boldsymbol{q}_{m} \boldsymbol{k}_{n}^{*}\right]
$$

其中 \* 是共轭复数，右端的乘法是普通的复数乘法，$\operatorname{Re}[]$ 表示取结果的实部。上式意味着：如果将 $\boldsymbol{q}_{m}, \boldsymbol{k}_{n}$ 分别乘以 $e^{\mathrm{i} m\theta}, e^{\mathrm{i} n \theta}$ 变成 $\boldsymbol{q}_{m} e^{\mathrm{i} m\theta}, \boldsymbol{k}_{n} e^{\mathrm{i} n \theta}$，那么就相当于给它们配上了绝对位置编码（因为显式地依赖绝对位置 $m, n$）。然后代入内积，有：

$$
\left\langle\boldsymbol{q}_{m} e^{\mathrm{i} i \theta}, \boldsymbol{k}_{n} e^{\mathrm{i} n \theta}\right\rangle=\operatorname{Re}\left[\left(\boldsymbol{q}_{m} e^{\mathrm{i} m \theta}\right)\left(\boldsymbol{k}_{n} e^{\mathrm{i} n \theta}\right)^{*}\right]=\operatorname{Re}\left[\boldsymbol{q}_{m} \boldsymbol{k}_{n}^{*} e^{\mathrm{i}(m-n) \theta}\right]
$$

值得注意的是，内积只依赖于相对位置 $m-n$！这就巧妙地将绝对位置与相对位置融合在了一起。

由上述结果可知，对于位置为 $n$ 的二维实数向量 $[x, y]$，将其当作复数运算并乘以 $e^{\mathrm{i} n \theta}$，得到恒等式：
$$
(x+y \mathrm{i}) e^{\mathrm{i} n \theta}=(x \cos n \theta-y \sin n \theta)+\mathrm{i}(x \sin n \theta+y \cos n \theta)
$$

这意味着，通过

$$
\left(\begin{array}{l}
x \\
y
\end{array}\right) \rightarrow\left(\begin{array}{c}
x \cos n \theta-y \sin n \theta \\
x \sin n \theta+y \cos n \theta
\end{array}\right)=\left(\begin{array}{l}
x \\
y
\end{array}\right) \cos n \theta+\left(\begin{array}{c}
-y \\
x
\end{array}\right) \sin n \theta
$$

来赋予 $[x, y]$ 绝对位置信息，那么在 Attention 运算时就等价于相对位置编码。如果是多于二维的向量，可以每两维为一组执行同样的运算，每组的 $\theta$ 可以不同。

这样一来，我们得到了一种融合绝对位置与相对位置的位置编码方案。从形式上看它类似乘性的绝对位置编码：通过在 $\mathbf{q}, \mathbf{k}$ 中施加该位置编码，效果等价于相对位置编码。如果还需要显式的绝对位置信息，则可以同时在 $\mathbf{v}$ 上施加该编码。

#### 完整证明

假定 query 向量 $\mathbf{q}_m$ 和 key 向量 $\mathbf{k}_n$ 之间的内积操作可以用函数 $g$ 表示，该函数的输入是词嵌入向量 $x_m$、$x_n$ 和它们之间的相对位置 $m-n$：
$$
<f_q(x_m,m),f_k(x_n,n)>=g(x_m,x_n,m-n) \\
$$
我们的目标是找到一个等价的位置编码方式，使得上述关系成立，即构造出函数 $f$ 和 $g$，使得上述等式成立。

假定词嵌入向量的维度为二维 $d=2$，这样就可以利用二维平面上向量的几何性质。论文中提出了满足上述关系的 $f$ 和 $g$ 的形式如下：
$$
f_q(x_m,m)=(W_qx_m)e^{im\theta} \\ f_k(x_n,n)=(W_kx_n)e^{in\theta} \\ g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)^{*}e^{i(m-n)\theta}] \\
$$
这里 Re 表示复数的实部。

首先看到上述 $f$ 和 $g$ 公式中有个指数函数：$$e^{ix}$$

这是欧拉公式 `[2]`，其中 $x$ 表示任意实数，$e$ 是自然对数的底数，$i$ 是复数中的虚数单位。根据欧拉公式有：
$$
e^{ix} = \cos x + i\sin x \\
$$
即上述指数函数可以表示为实部为 $\cos x$、虚部为 $\sin x$ 的复数。欧拉公式 `[2]` 建立了指数函数、三角函数和复数之间的桥梁。

则上述 $f$ 和 $g$ 公式中：
$$
e^{im\theta}=\cos (m\theta) + i\sin (m\theta) \\ e^{in\theta}=\cos (n\theta) + i\sin (n\theta) \\ e^{i(m-n)\theta}=\cos ((m-n)\theta) + i\sin ((m-n)\theta) \\
$$
然后看回公式：
$$
f_q(x_m,m)=(W_qx_m)e^{im\theta} \\
$$
其中 $W_q$ 是个二维矩阵，$x_m$ 是个二维向量，相乘结果也是一个二维向量，用 $q_m$ 表示：
$$
q_m= \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix} = W_qx_m =\begin{pmatrix} W_q^{(11)} & W_q^{(12)} \\ W_q^{(21)} & W_q^{(22)} \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix} \\
$$
首先将 $q_m$ 表示成复数形式：
$$
q_m = [q_m^{(1)}, q_m^{(2)}] = [q_m^{(1)} + iq_m^{(2)}] \\
$$
接着
$$
f_q(x_m,m)=(W_qx_m)e^{im\theta}=q_me^{im\theta} \\
$$
其实就是两个复数相乘：
$$
(a+ib) \cdot (c+id) = ac + ibc + iad + i^2bd=(ac-bd)+i(bc+ad) \\
$$
复数乘法使用分配律，并利用 $i^2=-1$ 的性质。代入可得：
$$
q_me^{im\theta}=(q_m^{(1)} + iq_m^{(2)}) * (\cos (m\theta) + i\sin (m\theta)) \\
$$
复习一下复数乘法的性质：

$$
q_me^{im\theta}=(q_m^{(1)} + iq_m^{(2)}) * (\cos (m\theta) + i\sin (m\theta)) \\ =(q_m^{(1)}cos (m\theta) - q_m^{(2)} \sin (m\theta) ) + i(q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)) \\
$$
将结果重新表达成实数向量形式就是：
$$
q_me^{im\theta}=[q_m^{(1)} \cos (m\theta) - q_m^{(2)} \sin (m\theta), q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)] \\
$$

$$
f_q(x_m,m)=(W_qx_m)e^{im\theta}=q_me^{im\theta}\\ =[q_m^{(1)} \cos (m\theta) - q_m^{(2)} \sin (m\theta), q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)] \\ = \begin{pmatrix} \cos (m\theta) & -\sin (m\theta) \\ \sin (m\theta) & \cos (m\theta) \end{pmatrix} \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix} \\
$$

看到这里会发现，这就是 query 向量乘以了一个旋转矩阵。这就是"旋转位置编码"名称的由来。

同理，$f_k$ 可以表示成下面的式子：
$$
\begin{align} f_k\left( {x}_m,m \right) &= \begin{pmatrix} \cos m\theta & -\sin m\theta) \\ \sin m \theta & \cos m \theta \end{pmatrix} \begin{pmatrix} W^{(1,1)}_{k} & W^{(1,2)}_{k} \\ W^{(2,1)}_{k} & W^{(2,2)}_{k} \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix} \\ &= \begin{pmatrix} \cos m\theta & -\sin m\theta) \\ \sin m \theta & \cos m \theta \end{pmatrix}\begin{pmatrix} k_m^{(1)} \\ k_m^{(2)} \end{pmatrix} \end{align}
$$
同理可得 key 向量 $k_n$ ：
$$
f_k(x_n,n)=(W_kx_n)e^{in\theta}=k_ne^{in\theta}\\ =[k_n^{(1)} \cos (n\theta) - k_n^{(2)} \sin (n\theta), k_n^{(2)}\cos (n\theta) + k_n^{(1)}\sin (n\theta)] \\ = \begin{pmatrix} \cos (n\theta) & -\sin (n\theta) \\ \sin (n\theta) & \cos (n\theta) \end{pmatrix} \begin{pmatrix} k_n^{(1)} \\ k_n^{(2)} \end{pmatrix} \\
$$
最后还有个函数 $g$：
$$
g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)^{*}e^{i(m-n)\theta}] \\
$$
其中 `Re[x]` 表示复数 $x$ 的实部，而 $(W_kx_n)^{*}$ 表示复数 $W_kx_n$ 的共轭。

复习一下共轭复数的定义：
$$
z=a+ib\\ z^*=a-ib \\
$$
所以可得：
$$
W_qx_m = q_m = q_m^{(1)} + iq_m^{(2)} \\ W_kx_n=k_n= k_n^{(1)} + ik_n^{(2)} \\ (W_kx_n)^*=k_n^*= k_n^{(1)} - ik_n^{(2)} \\ e^{i(m-n)\theta}=\cos((m-n)\theta) + i \sin((m-n)\theta) \\
$$
继续可得：
$$
g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)^{*}e^{i(m-n)\theta}] \\ = Re[(q_m^{(1)} + iq_m^{(2)})(k_n^{(1)} - ik_n^{(2)})(\cos((m-n)\theta) + i \sin((m-n)\theta))] \\ = Re[((q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)}) + i(q_m^{(2)}k_n^{(1)} - q_m^{(1)}k_n^{(2)}))(\cos((m-n)\theta) + i \sin((m-n)\theta))] \\ = (q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)})\cos((m-n)\theta) - (q_m^{(2)}k_n^{(1)} - q_m^{(1)}k_n^{(2)})\sin((m-n)\theta) \\
$$
接下来我们就要证明函数 $g$ 的计算公式是成立的。

首先回顾一下 attention 操作， 位置 m 的 query 和位置 n 的 key 会做一个内积操作：
$$
f_q(x_m,m)=[q_m^{(1)} \cos (m\theta) - q_m^{(2)} \sin (m\theta), q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta)] \\ f_k(x_n,n) =[k_n^{(1)} \cos (n\theta) - k_n^{(2)} \sin (n\theta), k_n^{(2)}\cos (n\theta) + k_n^{(1)}\sin (n\theta)] \\ <f_q(x_m,m),f_k(x_n,n)> = \\ (q_m^{(1)} \cos (m\theta) - q_m^{(2)} \sin (m\theta))(k_n^{(1)} \cos (n\theta) - k_n^{(2)} \sin (n\theta)) \\+ (q_m^{(2)}\cos (m\theta) + q_m^{(1)}\sin (m\theta))(k_n^{(2)}\cos (n\theta) + k_n^{(1)}\sin (n\theta))\\ =q_m^{(1)} \cos (m\theta) k_n^{(1)} \cos (n\theta) - q_m^{(1)} \cos (m\theta)k_n^{(2)} \sin (n\theta)\\ - q_m^{(2)} \sin (m\theta)k_n^{(1)} \cos (n\theta) + q_m^{(2)} \sin (m\theta)k_n^{(2)} \sin (n\theta) \\ + q_m^{(2)}\cos (m\theta)k_n^{(2)}\cos (n\theta) + q_m^{(2)}\cos (m\theta)k_n^{(1)}\sin (n\theta) \\ + q_m^{(1)}\sin (m\theta)k_n^{(2)}\cos (n\theta) + q_m^{(1)}\sin (m\theta)k_n^{(1)}\sin (n\theta) \\
$$
接着继续之前先复习一下三角函数的和差公式 `[3]`：
$$
\sin(a+b) = \sin a \cos b + \cos a \sin b \\ \sin(a-b) = \sin a \cos b - \cos a \sin b \\ \cos(a+b) = \cos a \cos b - \sin a \sin b \\ \cos(a-b) = \cos a \cos b + \sin a \sin b \\
$$
回到上面的式子，整理得到：
$$
<f_q(x_m,m),f_k(x_n,n)> = \\ q_m^{(1)}k_n^{(1)}(\cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) ) \\ + q_m^{(1)}k_n^{(2)}(-\cos(m\theta)\sin(n\theta) + \sin(m\theta)\cos(n\theta) ) \\ + q_m^{(2)}k_n^{(1)}(-\sin(m\theta)\cos(n\theta) + \cos(m\theta)\sin(n\theta) ) \\ + q_m^{(2)}k_n^{(2)}(\sin(m\theta)\sin(n\theta) + \cos(m\theta)\cos(n\theta) ) \\ = q_m^{(1)}k_n^{(1)}\cos((m-n)\theta) \\ + q_m^{(1)}k_n^{(2)}\sin((m-n)\theta) \\ - q_m^{(2)}k_n^{(1)}\sin((m-n)\theta) \\ + q_m^{(2)}k_n^{(2)}\cos((m-n)\theta) \\ = (q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)})\cos((m-n)\theta) + (q_m^{(1)}k_n^{(2)}- q_m^{(2)}k_n^{(1)})\sin((m-n)\theta) \\ = (q_m^{(1)}k_n^{(1)} + q_m^{(2)}k_n^{(2)})\cos((m-n)\theta) - (q_m^{(2)}k_n^{(1)} - q_m^{(1)}k_n^{(2)})\sin((m-n)\theta) \\ =g(x_m,x_n,m-n) \\
$$
这就证明了上述关系成立：位置 $m$ 的 query 和位置 $n$ 的 key 的内积即为函数 $g$。

把上面的式子用矩阵向量乘的形式来表达就是：
$$
<f_q(x_m,m),f_k(x_n,n)> \\ =\begin{pmatrix} \begin{pmatrix} \cos (m\theta) & -\sin (m\theta) \\ \sin (m\theta) & \cos (m\theta) \end{pmatrix} \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix} \end{pmatrix}^T \begin{pmatrix} \begin{pmatrix} \cos (n\theta) & -\sin (n\theta) \\ \sin (n\theta) & \cos (n\theta) \end{pmatrix} \begin{pmatrix} k_n^{(1)} \\ k_n^{(2)} \end{pmatrix} \end{pmatrix} \\ = \begin{pmatrix} q_m^{(1)} & q_m^{(2)} \\ \end{pmatrix} \begin{pmatrix} \cos (m\theta) & \sin (m\theta) \\ -\sin (m\theta) & \cos (m\theta) \end{pmatrix} \begin{pmatrix} \cos (n\theta) & -\sin (n\theta) \\ \sin (n\theta) & \cos (n\theta) \end{pmatrix} \begin{pmatrix} k_n^{(1)} \\ k_n^{(2)} \end{pmatrix} \\ = \begin{pmatrix} q_m^{(1)} & q_m^{(2)} \\ \end{pmatrix} \begin{pmatrix} \cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) & -\cos(m\theta)\sin(n\theta) + \sin(m\theta)\cos(n\theta) \\ -\sin(m\theta)\cos(n\theta) + \cos(m\theta)\sin(n\theta) & \sin(m\theta)\sin(n\theta) + \cos(m\theta)\cos(n\theta) \end{pmatrix} \begin{pmatrix} k_n^{(1)} \\ k_n^{(2)} \end{pmatrix} \\ =\begin{pmatrix} q_m^{(1)} & q_m^{(2)} \\ \end{pmatrix} \begin{pmatrix} \cos((m-n)\theta) & -\sin((m-n)\theta) \\ \sin((m-n)\theta) & \cos((m-n)\theta) \end{pmatrix} \begin{pmatrix} k_n^{(1)} \\ k_n^{(2)} \end{pmatrix} \\
$$
上面的推导假定词嵌入维度为 2 维向量。对于 $d \geq 2$ 的通用情况，将词嵌入向量元素按两两一组分组，每组应用同样的旋转操作，且每组的旋转角度计算方式如下：
$$
\theta_j=10000^{-2(j-1)/d}, j \in [1,2,...,d/2] \\
$$
综上，RoPE 的 self-attention 操作流程为：对 token 序列中的每个词嵌入向量，首先计算对应的 query 和 key 向量，然后对每个 token 位置计算对应的旋转位置编码，接着对 query 和 key 向量的元素按两两一组应用旋转变换，最后计算 query 和 key 之间的内积得到 self-attention 的计算结果。

## RoPE 的性质

### 远程衰减

可以看到，RoPE 形式上和 Sinusoidal 位置编码有一定相似性，只不过 Sinusoidal 位置编码是加性的，而 RoPE 可视为乘性的。在 $\theta_{i}$ 的选择上，同样沿用了 Sinusoidal 位置编码的方案，即 $\theta_{i}=10000^{-2 i / d}$，它可以带来一定的远程衰减性。

具体证明如下：将 $\mathbf{q}, \mathbf{k}$ 两两分组后，加上 RoPE 后的内积可以用复数乘法表示为：

$$
\left(\boldsymbol{\mathcal { R }}_{m} \boldsymbol{q}\right)^{\top}\left(\boldsymbol{\mathcal { R }}_{n} \boldsymbol{k}\right)=\operatorname{Re}\left[\sum_{i=0}^{d / 2-1} \boldsymbol{q}_{[2 i: 2 i+1]} \boldsymbol{k}_{[2 i: 2 i+1]}^{*} e^{\mathrm{i}(m-n) \theta_{i}}\right]
$$

记 $h_{i}=\boldsymbol{q}_{[2 i: 2 i+1]} \boldsymbol{k}_{[2 i: 2 i+1]}^{*}, S_{j}=\sum_{i=0}^{j-1} e^{\mathrm{i}(m-n) \theta_{i}}$，并约定 $h_{d / 2}=0, S_{0}=0$，由 Abel 变换（分部求和法）可以得到：

$$
\sum_{i=0}^{d / 2-1} \boldsymbol{q}_{[2 i: 2 i+1]} \boldsymbol{k}_{[2 i: 2 i+1]}^{*} e^{\mathrm{i}(m-n) \theta_{i}}=\sum_{i=0}^{d / 2-1} h_{i}\left(S_{i+1}-S_{i}\right)=-\sum_{i=0}^{d / 2-1} S_{i+1}\left(h_{i+1}-h_{i}\right)
$$

所以

$$
\begin{aligned}
\left|\sum_{i=0}^{d / 2-1} \boldsymbol{q}_{[2 i: 2 i+1]} \boldsymbol{k}_{[2 i: 2 i+1]}^{*} e^{\mathrm{i}(m-n) \theta_{i}}\right| & =\left|\sum_{i=0}^{d / 2-1} S_{i+1}\left(h_{i+1}-h_{i}\right)\right| \\
& \leq \sum_{i=0}^{d / 2-1}\left|S_{i+1}\right|\left|h_{i+1}-h_{i}\right| \\
& \leq\left(\max _{i}\left|h_{i+1}-h_{i}\right|\right) \sum_{i=0}^{d / 2-1}\left|S_{i+1}\right|
\end{aligned}
$$

因此可以考察 $\frac{1}{d / 2} \sum_{i=1}^{d / 2}\left|S_{i}\right|$ 随相对距离的变化情况来体现衰减性。可以观察到随着相对距离增大，内积结果呈现衰减趋势。因此，选择 $\theta_{i}=10000^{-2 i / d}$ 确实能带来一定的远程衰减性。

### 线性场景

最后指出，RoPE 是目前唯一一种可用于线性 Attention 的相对位置编码。这是因为其他相对位置编码直接基于 Attention 矩阵进行操作，而线性 Attention 并不事先计算 Attention 矩阵，因此无法应用。RoPE 以绝对位置编码的方式实现相对位置编码，不需要操作 Attention 矩阵，因而具备应用到线性 Attention 的可能性。

线性 Attention 的常见形式是：

$$
\operatorname{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})_{i}=\frac{\sum_{j=1}^{n} \operatorname{sim}\left(\boldsymbol{q}_{i}, \boldsymbol{k}_{j}\right) \boldsymbol{v}_{j}}{\sum_{j=1}^{n} \operatorname{sim}\left(\boldsymbol{q}_{i}, \boldsymbol{k}_{j}\right)}=\frac{\sum_{j=1}^{n} \phi\left(\boldsymbol{q}_{i}\right)^{\top} \varphi\left(\boldsymbol{k}_{j}\right) \boldsymbol{v}_{j}}{\sum_{j=1}^{n} \phi\left(\boldsymbol{q}_{i}\right)^{\top} \varphi\left(\boldsymbol{k}_{j}\right)}
$$

其中 $\phi, \varphi$ 是值域非负的激活函数。可以看到，线性 Attention 也是基于内积的，因此很自然的想法是将 RoPE 插入到内积中：

$$
\frac{\sum_{j=1}^{n}\left[\boldsymbol{\mathcal { R }}_{i} \phi\left(\boldsymbol{q}_{i}\right)\right]^{\top}\left[\boldsymbol{\mathcal { R }}_{j} \varphi\left(\boldsymbol{k}_{j}\right)\right] \boldsymbol{v}_{j}}{\sum_{j=1}^{n}\left[\boldsymbol{\mathcal { R }}_{i} \phi\left(\boldsymbol{q}_{i}\right)\right]^{\top}\left[\boldsymbol{\mathcal { R }}_{j} \varphi\left(\boldsymbol{k}_{j}\right)\right]}
$$

但这样存在的问题是，内积 $\left[\boldsymbol{R}_{i} \phi\left(\boldsymbol{q}_{i}\right)\right]^{\top}\left[\boldsymbol{R}_{j} \varphi\left(\boldsymbol{k}_{j}\right)\right]$ 可能为负数，因此不再是常规的概率注意力，且分母有为零的风险，可能带来优化上的不稳定。考虑到 $\boldsymbol{R}_{i}, \boldsymbol{R}_{j}$ 都是正交矩阵，不改变向量的模长，因此可以抛弃常规的概率归一化要求，使用如下运算作为一种新的线性 Attention：

$$
\frac{\sum_{j=1}^{n}\left[\boldsymbol{\mathcal { R }}_{i} \phi\left(\boldsymbol{q}_{i}\right)\right]^{\top}\left[\boldsymbol{\mathcal { R }}_{j} \varphi\left(\boldsymbol{k}_{j}\right)\right] \boldsymbol{v}_{j}}{\sum_{j=1}^{n} \phi\left(\boldsymbol{q}_{i}\right)^{\top} \varphi\left(\boldsymbol{k}_{j}\right)}
$$

也就是说，RoPE 只插入分子中，分母保持不变。这样的注意力不再是基于概率的（注意力矩阵不再满足非负归一性），但某种意义上也是一种归一化方案。目前也没有证据表明非概率式的注意力效果更差（例如 Nyströmformer 也未严格依据概率分布构建注意力）。因此将其作为候选方案之一进行实验，初步实验结果显示这样的线性 Attention 也是有效的。

### RoPE 的长度扩展

在 LLM 的应用中，有一个非常重要的参数——上下文长度（max context length）。更长的上下文长度允许进行更多轮次的对话、对更长的文本进行总结分析，也允许生成更长的文章。然而在训练 LLM 时，训练语料大部分不够长，许多 LLM 训练时设计的最大文本长度仅为 2k（即最长 2048 个 token）。那么，能否在训练时使用较短的文本，而在推理时扩展到长文本上呢？

这是可行的，可以对 RoPE 进行长度扩展。下面介绍三种扩展方案。

- 第一种是直接外推

直接外推即继续沿用现有位置编码公式，不做任何修改。在扩展长度不太大时（例如由 2k 扩展到 2.5k），此方法对性能的影响不大。旋转位置编码只与相对位置 $m-n$ 的大小有关，通常具有远程衰减性，即相对距离越大的两个 token 相关性越弱。

因此，如果模型已从训练数据中学习到 token 之间在 0-2k 范围内合适的衰减规律，将其应用到 0-2.5k 通常也没有问题。但若扩展到更长的长度（例如从 2k 扩展到 32k），直接外推通常会严重影响性能。因为学习到的衰减规律可能在 5k 处就完全衰减为零，导致无法捕捉超过 5k 相对距离的 token 之间的相互作用。

总结：直接外推对衰减规律在长距离情况下的使用容易出现问题。为减少性能影响，可以让训练好的模型在更长的上下文上做少量步骤的微调。

- 第二种是线性内插

线性内插需要改变位置编码公式，等效于将位置序号等比例缩小。

例如从 2k 扩展到 32k 时，等效于将位置序号缩小为原来的 1/16。线性内插未改变模型学习到的衰减规律的应用范围，不做微调时其效果一般优于直接外推方案。但当扩展倍数非常大时（如从 2k 到 32k），性能也会明显受影响。原因在于短距离情况下的使用受到较大影响：本来距离为 1 的两个 token，扩展后相当于距离为 1/16，而衰减规律在短距离时可能变化率极大，对相关性的评估可能偏离合理值。

应用线性内插时，在长文本上做少量步骤的微调也能明显改善性能。

- 第三种是NTK扩展方式

这种方式综合了外推和内插的优点，做长度扩展后即使不微调也能保持较好的性能。

前面的分析表明：直接外推对衰减规律在长距离情况下的使用容易出问题，在短距离下不受影响；线性内插对衰减规律在短距离下的使用容易出问题，在长距离下影响较小。那么能否将两者综合——在短距离情况下具有外推特性（与扩展前基本一致），在长距离情况下具有内插特性（缩放到扩展前的范围）？

观察 RoPE 位置编码的元素计算公式，可以发现 $i$ 越大，三角函数对应的角频率系数越小（即越低频），三角函数变化越慢。由此可得到直观结论：短距离之间的差异主要体现在高频分量（$i$ 较小）上；长距离之间的差异主要体现在低频分量（$i$ 较大）上。

为了在短距离情况下具有外推特性、长距离情况下具有内插特性，可以设计一个与频率相关的位置序号缩放因子：在最高频时取值为 1（与扩展前一致），在最低频时恰好为缩放倍数的倒数（缩放到扩展前的范围）。一种有效的选择方案是对 base 做指数缩放。NTK 扩展方式的要点是**高频外推、低频内插**，实现方法是直接对底数 base 进行缩放，类似进制编码转换。采用 NTK 扩展到长文本，即使不做微调，性能也仅略有下降。

## 代码实现

旋转位置嵌入的简单实现使用前面所示的块对角矩阵形式。在实践中，这种实现方式效率较低，更优化的形式很容易获得。RoPE 的原始实现可在 [roformer](https://github.com/ZhuiyiTechnology/roformer) 和 [bert4keras](https://github.com/bojone/bert4keras) 中找到。

此外，在 [x-transformers](https://github.com/lucidrains/x-transformers)、[GPT-Neo](https://github.com/EleutherAI/gpt-neo)、[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) 和 [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax) 中也实现了旋转位置嵌入。以下是从这些代码库中提取的 PyTorch 实现。

```python
import torch

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```



## 总结

从理论上看，RoPE 与 Sinusoidal 位置编码有相通之处，但 RoPE 不依赖泰勒展开，更具严谨性与可解释性。从预训练模型 RoFormer 的结果来看，RoPE 具有良好的外推性，应用到 Transformer 中体现出较好的处理长文本的能力。此外，RoPE 是目前唯一一种可用于线性 Attention 的相对位置编码。

## Reference

[1] [https://arxiv.org/pdf/2104.09864.pdf](https://arxiv.org/pdf/2104.09864.pdf)

[2] [https://en.wikipedia.org/wiki/Euler's_formula](https://en.wikipedia.org/wiki/Euler's_formula)

[3] [https://en.wikipedia.org/wiki/List_of_trigonometric_identities](https://en.wikipedia.org/wiki/List_of_trigonometric_identities)

[4] [https://github.com/facebookresearch/llama/tree/main](https://github.com/facebookresearch/llama/tree/main)

[5] [https://zh.wikipedia.org/wiki/旋转矩阵](https://zh.wikipedia.org/wiki/%E6%97%8B%E8%BD%AC%E7%9F%A9%E9%98%B5)

[6] Jianlin Su. 让研究人员绞尽脑汁的 Transformer 位置编码. https://kexue.fm/archives/8130, 2021. [Online; accessed 18-April-2021].

[7] Jianlin Su. Transformer 升级之路：2、博采众长的旋转式位置编码. https://kexue.fm/archives/8265, 2021. [Online; accessed 18-April-2021].

[8] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv preprint [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)*, 2021.
