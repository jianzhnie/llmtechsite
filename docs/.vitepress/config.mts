import { defineConfig } from 'vitepress'

// https://vitepress.dev/zh/reference/site-config
export default defineConfig({
  base: '/llmtech/',
  lang: 'zh-CN',
  title: "Robin's AI Lab",
  description: '探索人工智能的无限可能 — 大语言模型、强化学习、深度学习等前沿AI技术的研究与实践',

  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,

  head: [
    ['link', { rel: 'icon', href: '/llmtech/favicon.ico' }],
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css' }],
    ['script', { defer: '', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js' }],
    ['script', { defer: '', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' }],
    ['script', {}, `
      function renderKaTeX() {
        if (typeof renderMathInElement === 'undefined') {
          setTimeout(renderKaTeX, 100);
          return;
        }
        renderMathInElement(document.body, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
          ],
          throwOnError: false
        });
      }
      document.addEventListener('DOMContentLoaded', renderKaTeX);
      document.addEventListener('vitepress:afterRouteChange', renderKaTeX);
    `]
  ],

  markdown: {
    lineNumbers: true,
    container: {
      tipLabel: '提示',
      warningLabel: '警告',
      dangerLabel: '危险',
      infoLabel: '信息',
      detailsLabel: '详情'
    }
  },

  themeConfig: {
    // https://vitepress.dev/zh/reference/default-theme-config
    search: {
      provider: 'local'
    },

    nav: [
      { text: '首页', link: '/' },
      {
        text: 'AI 技术',
        items: [
          { text: '🦙 大语言模型', link: '/aigc/' },
          { text: '🤗 强化微调', link: '/rlhf/' },
          { text: '✌️ 推理部署', link: '/inference/' },
          { text: '🏗️ 模型架构', link: '/architecture/' },
          { text: '🚀 大规模并行', link: '/ultrascale/' },
          { text: '🍭 扩散模型', link: '/diffusion/' },
          { text: '🍔 多模态', link: '/multimodal/' },
          { text: '🍒 强化学习', link: '/rlwiki/' }
        ]
      },
      { text: '🇨🇳 Ascend生态', link: '/ascend/' },
      {
        text: '更多',
        items: [
          { text: '⚒️ 效率工具', link: '/toolbox/' },
          { text: '📖 深度学习百科', link: '/dlwiki/' },
          { text: '🤖 机器学习百科', link: '/mlwiki/' },
          { text: '🔗 外部链接', link: '/links' }
        ]
      }
    ],

    sidebar: {
      '/aigc/': [
        {
          text: '大语言模型',
          collapsed: false,
          items: [
            { text: '概述', link: '/aigc/' },
            { text: 'ChatGPT', link: '/aigc/chatgpt/chatGPT' },
            { text: 'Langchain', link: '/aigc/chatgpt/langchain' },
            { text: 'LLM Challenges', link: '/aigc/chatgpt/llm_challenges' },
            { text: 'Useful Chatgpt Prompt', link: '/aigc/chatgpt/useful_prompt' },
            { text: 'T5 模型', link: '/aigc/llm_pretrain/T5model' },
            { text: 'NLP 任务分类', link: '/aigc/llm_pretrain/nlptasks' },
            { text: '位置编码', link: '/aigc/llm_pretrain/pe' },
            { text: '旋转位置编码 (RoPE)', link: '/aigc/llm_pretrain/rope' },
            { text: 'LLM 微调方法', link: '/aigc/llm_finetune/finetune_llm' },
            { text: 'LoRA', link: '/aigc/llm_finetune/lora_llm' },
            { text: 'Instruction 数据集', link: '/aigc/llm_dataset/instruction_dataset' },
            { text: 'Preference 数据集', link: '/aigc/llm_dataset/preference_dataset' },
            { text: '行业数据集', link: '/aigc/llm_dataset/prompt_dataset' },
            { text: 'Autonomous AI Agents', link: '/aigc/llm_agent/llm_agent0' },
            { text: 'LLM 赋能 Agent', link: '/aigc/llm_agent/llm_agent1' },
            { text: 'AutoGPT 与 LLM Agent', link: '/aigc/llm_agent/llm_agent2' },
            { text: 'Agent: MarkAgent', link: '/aigc/llm_agent/market_agent' },
            { text: 'Agent: SightPro', link: '/aigc/llm_agent/sightpro' },
            { text: 'LLM StarCraftII (1-3)', link: '/aigc/llm_agent/llm_starcraft_1' },
            { text: 'LLM 评测体系', link: '/aigc/llm_eval/LLM评测体系' },
            { text: 'LLM 评估工具', link: '/aigc/llm_eval/LLM评估工具' },
            { text: 'LLM 评估指标', link: '/aigc/llm_eval/LLM评估指标' },
            { text: '8-bit 优化器', link: '/aigc/quantization/int8_opt' },
            { text: 'LLM.int8()', link: '/aigc/quantization/llm_int8' },
            { text: 'Bitsandbytes 集成', link: '/aigc/quantization/hf-bitsandbytes-integration' },
            { text: 'QLora', link: '/aigc/quantization/qlora' },
            { text: 'QLora 源码分析', link: '/aigc/quantization/qlora_usage' },
            { text: 'AI News and Blogs', link: '/aigc/ai-news' },
            { text: 'HuggingFace 模型下载', link: '/aigc/hf_download' }
          ]
        }
      ],

      '/rlhf/': [
        {
          text: '强化微调',
          collapsed: false,
          items: [
            { text: '概述', link: '/rlhf/' },
            { text: 'RL框架演进', link: '/rlhf/infra/RL-Infra_overview' },
            { text: '开源强化学习库', link: '/rlhf/infra/Open-source-rl-library' },
            { text: 'Slime', link: '/rlhf/infra/Slime' },
            { text: 'ROLL', link: '/rlhf/infra/ROLL' },
            { text: 'PPO 代码拆解', link: '/rlhf/infra/RLHF中的PPO代码拆解' },
            { text: 'NeMo-Aligner', link: '/rlhf/infra/NeMo-Aligner' },
            { text: 'DeepSpeedChat', link: '/rlhf/infra/DeepSpeedChat' },
            { text: 'OpenR', link: '/rlhf/infra/OpenR' },
            { text: 'AReaL', link: '/rlhf/infra/AReaL' },
            { text: 'AsyncFlow', link: '/rlhf/infra/AsyncFlow' },
            { text: 'OpenRLHF', link: '/rlhf/infra/OpenRLHF' },
            { text: 'OpenRLHF 源码解读', link: '/rlhf/infra/OpenRLHF源码解读' },
            { text: 'VeRL', link: '/rlhf/infra/Verl' },
            { text: 'Verl 源码解读', link: '/rlhf/infra/Verl源码解读' },
            { text: 'Verl 参数配置', link: '/rlhf/infra/Verl参数配置' },
            { text: 'OpenRLHF & Verl 参数转换', link: '/rlhf/infra/OpenRLHF&Verl参数转换指南' },
            { text: 'Ray 角度分析', link: '/rlhf/infra/Ray_OpenRLHF_Verl' },
            { text: 'Ray与LLM强化学习框架设计', link: '/rlhf/infra/Ray与LLM强化学习框架设计' },
            { text: 'Verl 核心算法', link: '/rlhf/infra/verl/core_algos' },
            { text: 'Verl 单控制器设计', link: '/rlhf/infra/verl/verl.single_controller设计详解' },
            { text: 'Verl Hybrid Flow', link: '/rlhf/infra/verl/verl_design' },
            { text: 'Verl PPO 架构', link: '/rlhf/infra/verl/verl_ppo' },
            { text: 'FSDP Actor', link: '/rlhf/infra/verl/fsdp_actor' },
            { text: 'FSDP Actor Worker', link: '/rlhf/infra/verl/fsdp_actor_worker' },
            { text: 'Megatron Actor', link: '/rlhf/infra/verl/megatron_actor' },
            { text: 'FSDP Backend', link: '/rlhf/infra/verl/fsdp_backend' },
            { text: 'Megatron Backend', link: '/rlhf/infra/verl/megatron_backend' },
            { text: 'FSDP Critic', link: '/rlhf/infra/verl/fsdp_critic' },
            { text: 'Megatron Critic', link: '/rlhf/infra/verl/megatron_critic' },
            { text: 'HuggingFace Rollout', link: '/rlhf/infra/verl/hf_rollout' },
            { text: 'VLLM Rollout', link: '/rlhf/infra/verl/vllm_rollout' },
            { text: 'Rollout Schemas', link: '/rlhf/infra/verl/rollout_schemas' },
            { text: 'FSDP VLLM 集成', link: '/rlhf/infra/verl/fsdp_vllm' },
            { text: 'Megatron VLLM 集成', link: '/rlhf/infra/verl/megatron_vllm' },
            { text: 'VLLM Server', link: '/rlhf/infra/verl/vllm_server' },
            { text: '朴素奖励管理器', link: '/rlhf/infra/verl/naive_reward_manager' },
            { text: '理解 RLHF', link: '/rlhf/intro/rlhf_advance' },
            { text: 'Chip Huyen: RLHF 分析', link: '/rlhf/intro/rlhf_chiphuyen' },
            { text: 'RLHF 知识整理', link: '/rlhf/intro/rlhf_overview' },
            { text: 'KL 散度近似计算', link: '/rlhf/intro/KL散度的近似计算方法' },
            { text: 'Policy Gradient Algorithms', link: '/rlhf/intro/rlhf_policy_gradient' },
            { text: 'GRPO 改进 (DR.GRPO→DAPO→GSPO)', link: '/rlhf/intro/grpo-to-dapo-and-gspo' },
            { text: '重新思考 PPO-Clip', link: '/rlhf/intro/ppo_clip' },
            { text: '截断重要性采样 (TIS)', link: '/rlhf/intro/truncated_importance_sampling' },
            { text: '动态微调', link: '/rlhf/intro/Dynamic-Fine-Tuning' },
            { text: 'DPO', link: '/rlhf/paper/rlhf_dpo' },
            { text: 'DPO 推导', link: '/rlhf/paper/rlhf_dpo_notes' },
            { text: 'KTO', link: '/rlhf/paper/rlhf_kto' },
            { text: 'RLOO', link: '/rlhf/paper/RLOO' },
            { text: 'DeepSeek-R1', link: '/rlhf/paper/DeepSeek-R1' },
            { text: 'Kimi k1.5', link: '/rlhf/paper/KimiK1.5' },
            { text: 'DAPO', link: '/rlhf/paper/DAPO' },
            { text: 'DR.GRPO', link: '/rlhf/paper/DR.GRPO' },
            { text: 'DeepScaleR', link: '/rlhf/paper/deepscaler' },
            { text: 'REINFORCE++', link: '/rlhf/paper/REINFORCE++' },
            { text: 'ChatGPT O1 Reasoning', link: '/rlhf/paper/chatgpt_O1' },
            { text: '过程奖励模型 (PRM)', link: '/rlhf/paper/PRM' },
            { text: 'PRM 开发经验', link: '/rlhf/paper/PRM_Reasoning' },
            { text: 'ReFT', link: '/rlhf/paper/ReFT' },
            { text: '拒绝采样', link: '/rlhf/paper/RejectSampling' },
            { text: 'ReST-MCTS', link: '/rlhf/paper/ReST-MCTS' },
            { text: 'rStar-Math', link: '/rlhf/paper/rStar-Math' },
            { text: 'GRPO-λ', link: '/rlhf/paper/GRPO-lambda' }
          ]
        }
      ],

      '/inference/': [
        {
          text: '推理部署',
          collapsed: false,
          items: [
            { text: '概述', link: '/inference/' },
            { text: '解码采样参数解析', link: '/inference/GenerateConfig' },
            { text: '解码策略基础', link: '/inference/解码策略基础' },
            { text: '解码策略高级方法', link: '/inference/解码策略高级方法' },
            { text: 'KVCaching 机制详解', link: '/inference/KVCaching机制详解' },
            { text: 'Continuous-Batching', link: '/inference/Continuous-Batching' },
            { text: 'Prefill-decode-disaggregation', link: '/inference/Prefill-decode-disaggregation' },
            { text: 'PagedAttention', link: '/inference/vllm/PageAttention' },
            { text: 'vLLM 设计文档', link: '/inference/vllm/vllm设计文档' },
            { text: 'Auto Prefix Caching', link: '/inference/vllm/vllm_auto_prefix_cache' },
            { text: 'vLLM 性能调优', link: '/inference/vllm/vllm_tuning' },
            { text: 'vLLM 性能基准测试', link: '/inference/vllm/vllm_bench' },
            { text: '图解 vLLM 系统', link: '/inference/vllm/vllm_design' },
            { text: 'vLLM DP 部署', link: '/inference/vllm/vllm_dp_deploy' },
            { text: 'SGLang 介绍', link: '/inference/sglang/SGLang' },
            { text: 'SGLang 性能调优', link: '/inference/sglang/sglang_tuning' },
            { text: 'SGLang Router', link: '/inference/sglang/sglang_router' },
            { text: 'SGLang Bench', link: '/inference/sglang/sglang_bench_serving' },
            { text: 'SGLang PD 分离', link: '/inference/sglang/sglang_pd_disaggregation' }
          ]
        }
      ],

      '/architecture/': [
        {
          text: '模型架构',
          collapsed: false,
          items: [
            { text: '概述', link: '/architecture/' },
            { text: 'MOE 图解指南', link: '/architecture/MoE图解指南' }
          ]
        }
      ],

      '/ultrascale/': [
        {
          text: '大规模并行',
          collapsed: false,
          items: [
            { text: '概述', link: '/ultrascale/' },
            { text: '大模型训练技术概论', link: '/ultrascale/Ultra-scale-llm-training-tech' },
            { text: 'DeepSpeed 教程', link: '/ultrascale/deepspeed/DeepSpeed教程' },
            { text: 'ZeRO 技术原理 (上)', link: '/ultrascale/deepspeed/zero-optimizer-1' },
            { text: 'ZeRO 技术原理 (下)', link: '/ultrascale/deepspeed/zero-optimizer-2' },
            { text: 'Pytorch 分布式教程', link: '/ultrascale/torch/torch_distributed' },
            { text: 'Pytorch FSDP', link: '/ultrascale/torch/torch_fsdp' },
            { text: 'TorchTitan', link: '/ultrascale/torch/torch_titan' },
            { text: 'Ray 核心概念', link: '/ultrascale/Ray核心概念' },
            { text: 'MMEngine', link: '/ultrascale/mmengine/engine' },
            { text: 'Runner', link: '/ultrascale/mmengine/runner' },
            { text: 'Model', link: '/ultrascale/mmengine/model' },
            { text: 'Hooks', link: '/ultrascale/mmengine/hooks' },
            { text: 'Logger', link: '/ultrascale/mmengine/logger' }
          ]
        }
      ],

      '/ascend/': [
        {
          text: 'Ascend 生态',
          collapsed: false,
          items: [
            { text: '概述', link: '/ascend/' },
            { text: 'Ascend Pytorch 适配方案', link: '/ascend/AscendPytorch适配方案介绍' },
            { text: 'LLamaFactory NPU Docker', link: '/ascend/llamafactory_docker' },
            { text: 'MindSpeed-RL 使用指南', link: '/ascend/MindSpeed-RL使用指南' },
            { text: 'MindSpeed-LLM 使用指南', link: '/ascend/MindSpeed-LLM使用指南' },
            { text: 'MindSpeed 并行特性', link: '/ascend/mindspeed/' },
            { text: '脑海 2B 模型强化微调', link: '/ascend/脑海2B模型强化微调' },
            { text: 'WandB Tables 记录文本', link: '/ascend/使用WandBTables记录生成的文本数据' }
          ]
        }
      ],

      '/diffusion/': [
        {
          text: '扩散模型',
          collapsed: false,
          items: [
            { text: '概述', link: '/diffusion/' },
            { text: '扩散模型理论第一课', link: '/diffusion/theory' },
            { text: '什么是扩散模型', link: '/diffusion/summary' },
            { text: '扩散模型数学原理', link: '/diffusion/math101' },
            { text: '深入理解扩散模型', link: '/diffusion/deepdive' },
            { text: '重参数化技巧', link: '/diffusion/reparameterization' },
            { text: '基于分数的生成模型', link: '/diffusion/score_model' },
            { text: 'Autoencoder to Beta-VAE', link: '/diffusion/vae_model' }
          ]
        }
      ],

      '/multimodal/': [
        {
          text: '多模态',
          collapsed: false,
          items: [
            { text: '概述', link: '/multimodal/' },
            { text: '多模态学习概览', link: '/multimodal/overview' },
            { text: '多模态模型', link: '/multimodal/lmm' },
            { text: '声音生成文本', link: '/multimodal/video2text' },
            { text: '文生视频', link: '/multimodal/text2video' },
            { text: 'ALBEF', link: '/multimodal/albef' },
            { text: 'BLIP', link: '/multimodal/blip' },
            { text: 'BLIP-2', link: '/multimodal/blip2' },
            { text: 'CoCa', link: '/multimodal/coca' },
            { text: 'Flamingo', link: '/multimodal/flamingo' }
          ]
        }
      ],

      '/rlwiki/': [
        {
          text: '强化学习',
          collapsed: false,
          items: [
            { text: '概述', link: '/rlwiki/' },
            { text: '入门教程', link: '/rlwiki/hfrlclass/' },
            { text: '第一章: 深度强化学习简介', link: '/rlwiki/hfrlclass/ch1_introduction' },
            { text: '第二章: Q-Learning', link: '/rlwiki/hfrlclass/ch2_q-learning' },
            { text: '第三章: Deep Q-Learning', link: '/rlwiki/hfrlclass/ch3_dqn' },
            { text: '第四章: Policy Gradient', link: '/rlwiki/hfrlclass/ch4_pg' },
            { text: '第五章: Actor-Critic', link: '/rlwiki/hfrlclass/ch5_a2c' },
            { text: '第六章: PPO', link: '/rlwiki/hfrlclass/ch6_ppo' },
            { text: '第七章: Decision Transformer', link: '/rlwiki/hfrlclass/ch7_decision-transformer' },
            { text: '第八章: Multi-Agent RL', link: '/rlwiki/hfrlclass/ch8_marl' },
            { text: '第九章: RL 前沿主题', link: '/rlwiki/hfrlclass/ch9_advanced' },
            { text: '第十章: RLHF', link: '/rlwiki/papers/RLHF' },
            { text: '进阶教程', link: '/rlwiki/algorithms/' },
            { text: 'Policy Gradient 证明', link: '/rlwiki/algorithms/ch1_supp_pg' },
            { text: 'A2C Baseline 方差减小', link: '/rlwiki/algorithms/ch1_supp_a2c' },
            { text: '深入 TRPO', link: '/rlwiki/algorithms/ch1_supp_trpo' },
            { text: 'HyAR', link: '/rlwiki/algorithms/ch2_supp_hyar' },
            { text: 'PPO vs DDPG', link: '/rlwiki/algorithms/ch2_supp_ppovsddpg' },
            { text: '重参数化与RL', link: '/rlwiki/algorithms/ch2_supp_reparameterization' },
            { text: 'Awesome RL Envs', link: '/rlwiki/envs/awesome_envs' },
            { text: 'OpenAI Gym', link: '/rlwiki/envs/gym' },
            { text: 'Mujoco', link: '/rlwiki/envs/mujoco' },
            { text: 'SMAC', link: '/rlwiki/envs/smac' },
            { text: 'MARL Envs', link: '/rlwiki/envs/marl_env' },
            { text: 'PettingZoo', link: '/rlwiki/envs/pettingzoo' },
            { text: 'Cyber Env Summary', link: '/rlwiki/envs/cyber_env_summary' },
            { text: 'CyberWheel', link: '/rlwiki/envs/cyberwheel_env' },
            { text: 'YawningTitan', link: '/rlwiki/envs/cyber_yawningtitan_env' },
            { text: 'RL 代表人物/机构', link: '/rlwiki/rltools/awesome_rl' },
            { text: 'EnvPool', link: '/rlwiki/rltools/envpool' },
            { text: 'MARL 代码汇总', link: '/rlwiki/rltools/marltool' },
            { text: '具身智能', link: '/rlwiki/robots/embodyai' },
            { text: 'LeRobot', link: '/rlwiki/robots/lerobot' },
            { text: 'MCTS 入门指南', link: '/rlwiki/muzero/mcts_guide' },
            { text: 'MCTS 详解', link: '/rlwiki/muzero/mcts' },
            { text: 'AlphaGoZero', link: '/rlwiki/muzero/alphazero' },
            { text: 'MuZero 介绍', link: '/rlwiki/muzero/muzero_intro' },
            { text: 'MuZero 伪代码', link: '/rlwiki/muzero/muzero_pseudocode' },
            { text: 'MARL Overview', link: '/rlwiki/papers/Overview' },
            { text: 'DRQN', link: '/rlwiki/papers/DRQN' },
            { text: 'IQL', link: '/rlwiki/papers/IQL' },
            { text: 'COMA', link: '/rlwiki/papers/COMA' },
            { text: 'VDN', link: '/rlwiki/papers/VDN' },
            { text: 'QTRAN', link: '/rlwiki/papers/QTRAN' },
            { text: 'QMIX', link: '/rlwiki/papers/QMIX' },
            { text: 'MADDPG', link: '/rlwiki/papers/MADDPG' },
            { text: 'MAT', link: '/rlwiki/papers/MAT' },
            { text: '零和博弈', link: '/rlwiki/papers/league' },
            { text: 'Self-Play', link: '/rlwiki/papers/self-play' },
            { text: 'Douzero', link: '/rlwiki/papers/Douzero' }
          ]
        }
      ],

      '/toolbox/': [
        {
          text: '效率工具',
          collapsed: false,
          items: [
            { text: '概述', link: '/toolbox/' },
            { text: 'Git 手册', link: '/toolbox/technical/git-manual' },
            { text: 'Linux 手册', link: '/toolbox/technical/linux-manual' },
            { text: 'Homebrew 手册', link: '/toolbox/technical/homebrew' },
            { text: 'zsh 配置指南', link: '/toolbox/technical/install_zsh' },
            { text: '其他常用指令', link: '/toolbox/technical/tools' },
            { text: 'Docker 教程', link: '/toolbox/technical/docker_tourial' },
            { text: '端口转发', link: '/toolbox/technical/端口转发' },
            { text: '深度学习环境配置', link: '/toolbox/technical/ubuntu-nvidia-cuda-install' },
            { text: '计算机科学书单', link: '/toolbox/study/cs_books' },
            { text: '计算机科学课程', link: '/toolbox/study/cs_class' },
            { text: '计算机科学技术栈', link: '/toolbox/study/cs_techself' },
            { text: 'AutoCut 视频剪辑', link: '/toolbox/study/autocut' },
            { text: 'Kindle 电子书下载', link: '/toolbox/study/kindle' }
          ]
        }
      ],

      '/dlwiki/': [
        {
          text: '深度学习百科',
          collapsed: false,
          items: [
            { text: '概述', link: '/dlwiki/' },
            { text: 'DeepLearning4j', link: '/dlwiki/deeplearning4j/' },
            { text: 'Arbiter', link: '/dlwiki/deeplearning4j/Arbiter' },
            { text: '分布式训练基础', link: '/dlwiki/distributed-training/' },
            { text: 'Horovodrun 背后', link: '/dlwiki/distributed-training/ch3' },
            { text: '网络基础 & Driver', link: '/dlwiki/distributed-training/ch4' },
            { text: '融合框架', link: '/dlwiki/distributed-training/ch5' },
            { text: 'DistributedOptimizer', link: '/dlwiki/distributed-training/ch7' },
            { text: 'On Spark', link: '/dlwiki/distributed-training/ch8' },
            { text: '弹性训练架构', link: '/dlwiki/distributed-training/ch12' },
            { text: '弹性训练 Driver', link: '/dlwiki/distributed-training/ch13' },
            { text: '弹性训练容错', link: '/dlwiki/distributed-training/ch17' },
            { text: 'MPI', link: '/dlwiki/distributed-training/MPI' },
            { text: 'Model 转换', link: '/dlwiki/model-convert/' },
            { text: '目标检测', link: '/dlwiki/object-detection/' },
            { text: 'FocalLoss', link: '/dlwiki/object-detection/FocalLoss' },
            { text: 'NLP', link: '/dlwiki/nlp/' },
            { text: 'ChatGPT', link: '/dlwiki/nlp/chaptGPT' },
            { text: 'Transformer', link: '/dlwiki/nlp/Transformer' },
            { text: 'Word2Vec', link: '/dlwiki/nlp/Word2Vec_1' },
            { text: 'ViT', link: '/dlwiki/transformers/' },
            { text: 'SwinTransformer', link: '/dlwiki/transformers/SwinTransformer' },
            { text: 'Vision Transformer', link: '/dlwiki/transformers/vision_transformer' },
            { text: '推荐系统', link: '/dlwiki/recommender/' },
            { text: 'Wide&Deep', link: '/dlwiki/recommender/wide&deep' }
          ]
        }
      ],

      '/mlwiki/': [
        {
          text: '机器学习百科',
          collapsed: false,
          items: [
            { text: '概述', link: '/mlwiki/' },
            { text: 'AutoML 简述', link: '/mlwiki/AutoML/AutoML简述' },
            { text: 'auto-sklearn', link: '/mlwiki/AutoML/auto-sklearn' },
            { text: 'AutoFeaturetools', link: '/mlwiki/AutoML/AutoFeaturetools' },
            { text: 'DeepTables', link: '/mlwiki/AutoML/DeepTables' },
            { text: 'Hyperopt', link: '/mlwiki/AutoML/hyperopt_1' },
            { text: 'Optuna', link: '/mlwiki/AutoML/optuna' },
            { text: 'Ray.tune', link: '/mlwiki/AutoML/Ray.tune' },
            { text: 'TPOT', link: '/mlwiki/AutoML/TPOT' },
            { text: 'Hadoop', link: '/mlwiki/Spark/Hadoop' },
            { text: 'Scala 编程', link: '/mlwiki/Spark/Scala编程' },
            { text: 'Spark', link: '/mlwiki/Spark/Spark' },
            { text: 'SparkSQL', link: '/mlwiki/Spark/SparkSQL' },
            { text: 'Python 多进程', link: '/mlwiki/python/multiprocess' },
            { text: 'Python Hook', link: '/mlwiki/python/How-to-use-hook-in-python' },
            { text: 'Lambda 函数', link: '/mlwiki/python/Lambda函数' },
            { text: '行为树', link: '/mlwiki/python/behavior_tree' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/jianzhnie' }
    ],

    footer: {
      message: 'Maintained by Robin',
      copyright: '© 2017 — 2025. Powered by VitePress'
    },

    outline: {
      level: [2, 3],
      label: '文章目录'
    },

    editLink: {
      pattern: 'https://github.com/jianzhnie/llmtech/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'short', timeStyle: 'short' }
    },

    notFound: {
      title: '页面未找到',
      quote: '您访问的页面不存在',
      linkLabel: '返回首页',
      linkText: '返回首页'
    },

    darkModeSwitchLabel: '主题切换',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部'
  }
})
