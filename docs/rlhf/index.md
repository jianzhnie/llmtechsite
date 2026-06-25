## LLM RLHF Framework

- [大模型 RL 框架的演进与发展趋势](/rlhf/infra/RL-Infra_overview)
- [面向 LLM 的开源强化学习库](/rlhf/infra/Open-source-rl-library)
- [RLHF 训练框架 Slime](/rlhf/infra/Slime)
- [RLHF 训练框架 ROLL](/rlhf/infra/ROLL)
- [RLHF 中的 PPO 代码拆解](/rlhf/infra/RLHF中的PPO代码拆解)
- [RLHF 训练框架 NeMo-Aligner](/rlhf/infra/NeMo-Aligner)
- [RLHF 训练框架 DeepSpeedChat](/rlhf/infra/DeepSpeedChat)
- [RLHF 训练框架 OpenR](/rlhf/infra/OpenR)
- [RLHF 训练框架 AReaL](/rlhf/infra/AReaL)
- [RLHF 训练框架 ARealLite](/rlhf/infra/ARealLite)
- [RLHF 训练框架 AsyncFlow](/rlhf/infra/AsyncFlow)
- [RLHF 训练框架 OpenRLHF](/rlhf/infra/OpenRLHF)
- [RLHF 训练框架 OpenRLHF 源码解读](/rlhf/infra/OpenRLHF源码解读)
- [RLHF 训练框架 VeRL](/rlhf/infra/Verl)
- [RLHF 训练框架 VeRL 源码解读](/rlhf/infra/Verl源码解读)
- [RLHF 训练框架 VeRL 参数配置指南](/rlhf/infra/Verl参数配置)
- [OpenRLHF & veRL 参数转换指南](/rlhf/infra/OpenRLHF&Verl参数转换指南)
- [从 Ray 角度分析 OpenRLHF 和 veRL 的工程设计](/rlhf/infra/Ray_OpenRLHF_Verl)
- [Ray 与 LLM 强化学习框架设计](/rlhf/infra/Ray与LLM强化学习框架设计)


## veRL 源码分析

### 核心实现
- [核心算法实现](/rlhf/infra/verl/core_algos)
- [veRL 单控制器设计详解](/rlhf/infra/verl/verl.single_controller设计详解)
- [veRL 源码解析与 Hybrid Flow 编程范式](/rlhf/infra/verl/verl_design)
- [veRL 中 PPO 示例架构详解](/rlhf/infra/verl/verl_ppo)

### Actor 实现
- [FSDP Actor 实现](/rlhf/infra/verl/fsdp_actor)
- [FSDP Actor Worker](/rlhf/infra/verl/fsdp_actor_worker)
- [Megatron Actor 实现](/rlhf/infra/verl/megatron_actor)
- [FSDP Backend](/rlhf/infra/verl/fsdp_backend)
- [Megatron Backend](/rlhf/infra/verl/megatron_backend)

### Critic 实现
- [FSDP Critic 实现](/rlhf/infra/verl/fsdp_critic)
- [FSDP Critic Worker](/rlhf/infra/verl/fsdp_critic_worker)
- [Megatron Critic 实现](/rlhf/infra/verl/megatron_critic)

### Rollout 相关
- [Hugging Face Rollout](/rlhf/infra/verl/hf_rollout)
- [vLLM Rollout](/rlhf/infra/verl/vllm_rollout)
- [Rollout Schemas](/rlhf/infra/verl/rollout_schemas)

### vLLM 集成
- [FSDP vLLM 集成](/rlhf/infra/verl/fsdp_vllm)
- [Megatron vLLM 集成](/rlhf/infra/verl/megatron_vllm)
- [vLLM Server](/rlhf/infra/verl/vllm_server)

### 奖励管理
- [朴素奖励管理器](/rlhf/infra/verl/naive_reward_manager)



## LLM RLHF Intro

- [理解 RLHF](/rlhf/intro/rlhf_advance)
- [Chip Huyen 对 RLHF 的分析](/rlhf/intro/rlhf_chiphuyen)
- [RLHF 相关知识整理](/rlhf/intro/rlhf_overview)
- [RLHF 中 KL 散度的近似计算](/rlhf/intro/KL散度的近似计算方法)
- [RLHF 中的 Policy Gradient Algorithms](/rlhf/intro/rlhf_policy_gradient)
- [浅谈 GRPO 的系列改进（From GRPO to DAPO and GSPO）](/rlhf/intro/grpo-to-dapo-and-gspo)
- [重新思考 PPO-Clip — GRPO 时代下的各种变体](/rlhf/intro/ppo_clip)
- [截断重要性采样（TIS）](/rlhf/intro/truncated_importance_sampling)
- [动态微调（Dynamic Fine-Tuning）](/rlhf/intro/Dynamic-Fine-Tuning)


## LLM RLHF Algorithm and Paper


- [直接偏好优化 (DPO)](/rlhf/paper/rlhf_dpo)
- [直接偏好优化 (DPO) 推导](/rlhf/paper/rlhf_dpo_notes)
- [Kahneman-Tversky-Optimization (KTO)](/rlhf/paper/rlhf_kto)
- [RLOO](/rlhf/paper/RLOO)
- [DeepSeek-R1：通过强化学习激励 LLMs 的推理能力](/rlhf/paper/DeepSeek-R1)
- [Kimi k1.5：使用 LLM 扩展强化学习](/rlhf/paper/KimiK1.5)
- [DAPO: 一个开源的大规模 LLM 强化学习系统](/rlhf/paper/DAPO)
- [深入理解 R1-Zero 类训练：一个批判性视角](/rlhf/paper/DR.GRPO)
- [DeepScaleR：通过扩展强化学习超越 o1](/rlhf/paper/deepscaler)
- [REINFORCE++：一种简单高效的大型语言模型对齐方法](/rlhf/paper/REINFORCE++)
- [ChatGPT O1 Reasoning](/rlhf/paper/chatgpt_O1)
- [过程奖励模型（Process Reward Model）](/rlhf/paper/PRM)
- [数学推理中过程奖励模型的开发经验](/rlhf/paper/PRM_Reasoning)
- [ReFT: 通过强化微调提升推理能力](/rlhf/paper/ReFT)
- [拒绝采样（Reject Sampling）在 RLHF 中的应用](/rlhf/paper/RejectSampling)
- [ReST-MCTS：通过过程奖励引导的树搜索实现 LLM 自训练](/rlhf/paper/ReST-MCTS)
- [rStar-Math：小型语言模型通过自我进化的深度思考掌握数学推理](/rlhf/paper/rStar-Math)
- [GRPO-λ (动态长度惩罚)](/rlhf/paper/GRPO-lambda)
