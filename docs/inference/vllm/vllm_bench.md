# vLLM 性能基准测试指南：全面评估大模型推理性能

## 引言

在大语言模型（Large Language Models, LLMs）的生产部署中，推理性能是决定用户体验和系统成本的关键因素。[vLLM](https://github.com/vllm-project/vllm) 作为业界领先的高效推理引擎，不仅提供了强大的功能支持（如 PagedAttention、LoRA、多模态等），还内置了一套**全面且灵活的基准测试工具集**，帮助开发者量化评估模型在不同场景下的表现。

本文将系统介绍 vLLM 的基准测试体系，涵盖其 CLI 工具、支持的数据集、典型使用场景以及高级功能，助您精准衡量推理性能。

## 一、vLLM 基准测试体系概览

vLLM 的基准测试主要分为三大类：

1. **Benchmark CLI 工具**
   提供命令行接口（`vllm bench`），用于交互式或脚本化地执行在线/离线性能测试。
2. **自动化 CI 性能基准**
   在每次代码提交时自动运行，确保新功能不会引入性能退化。
3. **横向对比基准（Nightly Benchmarks）**
   定期将 vLLM 与竞品（如 TGI、TensorRT-LLM、LMDeploy）进行横向对比，结果公开发布于 [vLLM 性能仪表盘](https://docs.vllm.ai/en/stable/contributing/benchmarks.html#nightly-benchmarks)。

本文重点聚焦于第一类——开发者可直接使用的 **Benchmark CLI**。

### 基准测试命令行接口 (Benchmark CLI)

所有基准测试的入口点都是统一的命令行工具，通常通过 `vllm bench` 命令调用。

这一命令是您执行所有性能评估的起点，不同的子命令和参数将引导您进入特定的测试场景。例如：

- 运行延迟测试：`vllm bench latency [...]`
- 运行服务吞吐量测试：`vllm bench serve [...]`
- 运行离线吞吐量测试：`vllm bench throughput [...]`


## 二、核心性能基准测试套件和测试场景

vLLM 的性能评估被精细地划分为两大核心类别，分别对应在线服务和离线批处理场景，这是任何 LLM 推理引擎都必须优化的关键指标。

### 1. 在线基准测试 (Online Benchmark)

**核心目标：** 模拟真实的用户请求流量，评估系统的**延迟**和**服务能力**，直接关系到终端用户体验和 SLA（服务等级协议）。

- **延迟测试 (Latency)**：衡量请求的端到端时间，包括首个 Token 的延迟（Time To First Token, TTFT）和后续 Token 的平均生成延迟（Time Per Output Token, TPOT）。这是评估调度器和 KV Cache 访问效率的关键。
- **服务性能测试 (Serve)**：在多并发、高负载环境下运行，评估系统的最大并发请求处理能力和在特定延迟限制下的稳定吞吐量。

### 2. 离线吞吐量基准测试 (Offline Throughput Benchmark)

**核心目标：** 评估系统在资源饱和状态下的**最大处理能力**，通常以每秒生成的 Token 数（Tokens/s）来衡量。

此测试适用于评估底层 CUDA/Triton 内核、张量并行（Tensor Parallelism）和 KV Cache 管理等组件的纯粹计算效率。它关注的是在不考虑实时延迟限制下的极限性能。

### 3. 特性与场景特定的基准测试

vLLM 的基准测试套件还包含一系列针对特定高级特性和复杂工作负载设计的测试，以确保关键创新不会引入性能退化。

| 基准测试名称       | 核心评估目标                                                       | 对应场景                  | 关键技术深度                                                     |
| ------------------ | ------------------------------------------------------------------ | ------------------------- | ---------------------------------------------------------------- |
| 结构化输出基准测试 | 评估在特定格式（如JSON Schema、Regex）约束下生成内容的效率和正确性 | RAG、函数调用、Agent 任务 | 约束解码、Logits Processor 的性能开销                            |
| 长文档问答基准测试 | 评估处理超长上下文窗口时的性能和内存管理效率                       | 长上下文模型、文档处理    | 长上下文 KV Cache 的内存开销、Paged Attention 在大规模块上的效率 |
| 前缀缓存基准测试   | 评估**自动前缀缓存 (Automatic Prefix Caching)** 机制的实际加速效果 | 多用户或重复提示词场景    | KV Cache 的命中率、Prefill 阶段的计算优化                        |
| 请求优先级基准测试 | 评估调度器处理具有不同优先级的并发请求时的响应速度和SLA保障能力    | 多租户、SLA 保证场景      | 调度器逻辑、Request Prioritization 机制的有效性                  |
| 多模态基准测试     | 评估集成图像、音频等输入后，模型的推理性能和延迟                   | 多模态输入处理            | 多模态输入预处理、数据并行与计算资源的协调                       |
| Embedding 基准测试 | 评估 vLLM 在执行向量嵌入任务时的吞吐量和延迟                       | 检索增强、向量数据库集成  | Pooling Model 的效率、批处理能力                                 |
| Reranker 基准测试  | 评估重排序模型（Reranker Model）的性能                             | 检索增强、向量数据库集成  | 专用于评估非生成类（Encoder-only）模型的推理能力                 |

## 三、支持的数据集

vLLM 支持多种真实世界和合成数据集，覆盖文本、图像、视频等多种模态：

| 数据集名称                | 在线测试 | 离线测试 | 数据来源                 |
| ------------------------- | -------- | -------- | ------------------------ |
| ShareGPT                  | ✅        | ✅        | Hugging Face             |
| ShareGPT4V (图像)         | ✅        | ✅        | Hugging Face + COCO 图像 |
| ShareGPT4Video (视频)     | ✅        | ✅        | Hugging Face             |
| BurstGPT                  | ✅        | ✅        | GitHub Release           |
| Random                    | ✅        | ✅        | synthetic                |
| Prefix Repetition         | ✅        | ✅        | synthetic                |
| HuggingFace-VisionArena   | ✅        | ✅        | Hugging Face             |
| HuggingFace-MMVU          | ✅        | ✅        | Hugging Face             |
| HuggingFace-InstructCoder | ✅        | ✅        | Hugging Face             |
| HuggingFace-AIMO          | ✅        | ✅        | Hugging Face             |
| MT-Bench / Blazedit       | ✅        | ✅        | Hugging Face             |
| Spec-Bench                | ✅        | ✅        | GitHub                   |
| 自定义数据集              | ✅        | ✅        | 本地 `.jsonl` 文件       |

> **提示**：对于 Hugging Face 数据集，请使用 `--dataset-name hf` 并指定 `--hf-name`；对于本地文件，使用 `--dataset-name custom`。

为了确保测试的公平性和代表性，vLLM 使用了一系列标准化的数据集和提示词模板，贡献者应关注：

1. **数据集选择:** 了解不同基准测试使用的数据集，确保自己的性能改进在相关数据集上有效。
2. **示例参考:** 官方文档提供了运行各个基准测试的完整示例命令，是运行测试的首选参考。



## 四、核心基准测试场景与示例

### 1. 在线服务基准测试（Online Serving Benchmark）

模拟真实 API 调用，测量端到端延迟和吞吐量。

```bash
# 启动模型服务
vllm serve NousResearch/Hermes-3-Llama-3.1-8B

# 执行基准测试
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

**输出指标包括**：
- 请求成功率、总耗时
- 输入/输出 token 数量
- **请求吞吐量（req/s）**
- **Token 吞吐量（tok/s）**
- **首 Token 延迟（TTFT）**
- **后续 Token 生成延迟（TPOT / ITL）**

### 2. 离线吞吐量基准测试（Offline Throughput Benchmark）

适用于批量推理场景，不依赖服务进程，直接调用引擎。

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000
```

> 📌 对于多模态模型，输入 Token 数会包含图像 Token。

### 3. 结构化输出性能测试

评估 JSON Schema、正则表达式、语法约束等结构化输出的生成开销。

```bash
# 需先启动服务
vllm serve NousResearch/Hermes-3-Llama-3.1-8B

# 运行结构化输出基准
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset json \
  --structured-output-ratio 1.0 \
  --request-rate 10 \
  --num-prompts 1000
```

支持类型：`json`、`grammar`、`regex`、`choice`、`xgrammar_bench`。

### 4. 长文档 QA 与前缀缓存（Prefix Caching）

测试在重复长上下文场景下，前缀缓存对性能的提升效果。

```bash
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 16 \
  --document-length 2000 \
  --output-len 50 \
  --repeat-count 5
```

支持三种重复模式：`random`（随机）、`tile`（整体循环）、`interleave`（逐个重复）。

### 5. 请求优先级调度测试

验证高优先级请求是否能更快获得响应。

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority
```

### 6. 多模态（图像/视频）推理基准

以 Qwen-VL 为例：

```bash
# 启动多模态服务
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --limit-mm-per-prompt '{"image": 1}' \
  --allowed-local-media-path /path/to/images

# 发送带图请求
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /path/to/sharegpt4v.json \
  --num-prompts 100
```

还支持**合成多模态数据**（`--dataset-name random-mm`），用于压力测试。

## 五、高级功能与技巧

### 动态请求速率调节（Ramp-up）

模拟流量增长，寻找系统瓶颈：

```bash
vllm bench serve \
  --ramp-up-strategy linear \
  --ramp-up-start-rps 1 \
  --ramp-up-end-rps 50 \
  --duration 60
```

支持 `linear` 和 `exponential` 两种策略。

### Speculative Decoding（推测解码）测试

结合 N-gram 或小模型进行加速：

```bash
VLLM_USE_V1=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5}'

vllm bench throughput \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5}'
```

实测可显著提升吞吐量（示例：**104.77 req/s**）。

## 六、结果保存与分析

使用以下参数保存详细结果：

```bash
--save-result \
--save-detailed \
--result-dir "./log/"
```

结果包含：
- 汇总统计（Markdown/JSON）
- 每个请求的详细延迟数据（CSV）
- 可用于 Grafana/Prometheus 监控集成

## 七、结语

vLLM 的基准测试工具不仅是性能验证的利器，更是优化推理配置、对比模型版本、评估新功能影响的**标准流程**。无论你是部署线上服务，还是参与 vLLM 开发，掌握这套工具都将极大提升你的工作效率。

> 📚 **延伸阅读**
> - [vLLM 官方文档 - Benchmarking](https://docs.vllm.ai/en/stable/contributing/benchmarks.html)
> - [vLLM 性能仪表盘](https://vllm.ai/performance)
> - [Spec-Bench: 结构化推理基准](https://github.com/hemingkx/Spec-Bench)
