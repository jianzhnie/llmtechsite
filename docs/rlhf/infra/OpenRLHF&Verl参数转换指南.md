

## OpenRLHF & veRL 参数转换

### 模型参数

| 介绍           | OpenRLHF        | veRL                         |
| -------------- | --------------- | ---------------------------- |
| Actor 模型路径  | pretrain        | actor_rollout_ref.model.path |
| Reward 模型路径 | reward_pretrain | reward_model.model.path      |
| Critic 模型路径 | critic_pretrain | critic.model.path            |

### 优化参数

| 介绍             | OpenRLHF             | veRL                             |
| ---------------- | -------------------- | -------------------------------- |
| Actor 模型学习率  | actor_learning_rate  | actor_rollout_ref.actor.optim.lr |
| Critic 模型学习率 | critic_learning_rate | critic.optim.lr                  |

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 默认使用的是 Warmup-Decay 的学习率调度器，所以也支持设置 warmup 步数等其他的相关参数，具体的可以在 [train_ppo_ray.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo_ray.py) 中找到相关的参数。

veRL 中对于学习率的调度方式可以选择为 `constant/cosine`，在 yaml 文件中依然可以找对应的[位置](https://github.com/volcengine/verl/blob/f8acd9017b4db4eead1f34beb39fce9c39143194/verl/trainer/config/ppo_trainer.yaml%23L45)。

### 数据参数

| 介绍                             | OpenRLHF                 | Verl                                                         |
| -------------------------------- | ------------------------ | ------------------------------------------------------------ |
| 训练阶段单卡分配的experience数量 | micro_train_batch_size   | actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu         |
| 训练时全局的experience数量       | train_batch_size         | actor_rollout_ref.actor.ppo_mini_batch_size*actor_rollout_ref.rollout.n(本节末有解释) |
| 探索阶段单卡分配的experience数量 | micro_rollout_batch_size | \(自动计算)                                                  |
| 探索阶段的prompt数量             | rollout_batch_size       | data.train_batch_size                                        |
| 实际使用的最大prompt数量         | max_samples              | \                                                            |
| 单个prompt采样次数               | n_samples_per_prompt     | actor_rollout_ref.rollout.n                                  |
| 训练阶段experience的学习次数     | max_epochs               | actor_rollout_ref.actor.ppo_epochs                           |
| 数据集迭代次数                   | num_episodes             | trainer.total_epochs                                         |

#### OpenRLHF 流程：

(1) 首先，当给定一个数据集之后，框架会从中选择至多 max_samples 个 prompt。假设我们的数据集仅有 1024 个 prompt，并且 1024 小于 max_samples，则 1024 个 prompt 全部保留。

(2) 之后进入探索阶段，由于一次探索完 1024 个 prompt 的时间太长，所以选择一次只对 rollout_batch_size 个 prompt 进行探索。我们假设 rollout_batch_size 为 32，则一共需要探索 1024÷32=32 步。这个 32 步就是我们在 wandb 或者 tensorboard 上面看到的步骤，我们称之为 explore step。我们会用 vLLM 对每个 prompt 进行采样 n_samples_per_prompt 次，得到所有的 samples。我们假设 n_samples_per_prompt 为 8，则得到了 32×8=256 个样本，即每个样本都是一个问答对，一共有 32 个问题，并且相同的问题回答了 8 次。

(3) 之后需要生成 experience，这时就需要切换到训练引擎，即在 1 步内单卡负责生成 micro_rollout_batch_size 个经验，我们假设 micro_rollout_batch_size 为 4，我们有 8 张卡，则 1 步一共可以生成 32 个 experience。由于我们一共有 256 个样本，所以需要一共需要 256÷32=8 步可以得到全部的 experience。在 make experience 阶段我们主要利用 Reward 模型得到每个答案的奖励分数、用 Critic 模型给出每个答案每一步的 Value 值（如果有 Critic 模型），以及用 Reference 模型和 Actor 模型给出每个答案每一步的预测概率并且计算出对应的 KL 惩罚值（如果有 Reference 模型）

(4) 在结束探索阶段后，就进入了训练阶段。刚才一共得到了 256 个 experience，但是我们的显卡不足以一次性在所有的样本上进行训练，因此我们设置 train_batch_size 为 128，即每次只更新 128 个 experience，则需要 256÷128=2 步，也就是说在训练阶段模型更新了 2 次，我们称之为 update step。假设 micro_train_batch_size 为 4，我们有 8 张卡，则 1 步一共可以训练 32 个 experience，那么我们需要 4 步梯度累计，然后才进行反向传播。当 update step 大于 1，也就是所有的 experience 不能一次更新完的时候，就称之为 off policy，反之如果 update step=1，也就是模型探索一步就更新一步，则称之为 on policy。并且需要注意，如果 max_epochs＞1，此时这一组经验被训练了多次，即对 256 个 experience 进行了多次优化，那么此时的策略一定是 off policy，所以一般情况下我们默认这个参数为 1 即可，因为我们希望尽可能地确保我们的优化是 on policy 的。

(5) 我们再强调一遍，我们的数据集有 1024 个 prompt，每次探索和训练其中的 32 个，则经过以上流程循环 1024÷32=32 步，我们已经探索并且训练完了整个数据集，之后我们再训练 num_episodes 次，则完成了整个训练流程。

#### veRL 流程

veRL 流程上，从算法逻辑上与 OpenRLHF 的是相同的，但是其中需要关注的是 `actor_rollout_ref.actor.ppo_mini_batch_size` 参数是一个与采样数 `n` 无关的参数，从上述的 1024 个 prompt 的数据集来看，如果 `ppo_mini_batch_size` 的设置为 32 的话，则一定进行 1024÷32=32 次的参数更新，无论 `sample_n` 的设置值为多少。

但要注意，veRL 中的 `ppo_micro_batch_size_per_gpu` 是已经考虑了 `sample_n`，因此又重新对 OpenRLHF 的参数含义相同。

### 批处理参数部分

上文提到的actor的训练bs设置不再赘述。

| 介绍                                     | OpenRLHF               | veRL                                                        |
| ---------------------------------------- | ---------------------- | ----------------------------------------------------------- |
| rollout 计算 log_prob 时每个 GPU 的微批量大小 | micro_train_batch_size | actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu |
| ref 计算 log_prob 时每个 GPU 的微批量大小     | micro_train_batch_size | actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu     |
| Critic 模型单 GPU 计算的批量大小            | micro_train_batch_size | critic.forward_micro_batch_size_per_gpu                     |
| Reward 模型单 GPU 计算的批量大小            | micro_train_batch_size | reward_model.micro_batch_size_per_gpu                       |

在 OpenRLHF 中，通常由 `micro_train_batch_size` 来管理所有的 bs 信息，对于不同的模型角色，实际上使用的是一个参数进行管理。但在 veRL 中，这个部分的参数则是可以指定的。一般来说如果脚本中不特地指定数值，则会保持与 `actor_rollout_ref.actor` 中设置一致。并且在 veRL 中，落实到 `per_gpu` 的参数设置，一般都是已经考虑了 sample_n 次的结果。

### 生成参数

| 介绍               | OpenRLHF         | veRL                     |
| ------------------ | ---------------- | ------------------------ |
| 采样阶段的温度系数 | temperature      | rollout.temperature      |
| prompt 最大长度     | prompt_max_len   | data.max_prompt_length   |
| 生成回答的最大长度 | generate_max_len | data.max_response_length |
