# O2-Searcher 训练流程完整梳理

> 本文档基于仓库代码 + 多次问答迭代整理而成。覆盖 SFT（冷启动）、GRPO（强化学习）两个训练阶段，以及数据准备、搜索引擎子系统、奖励函数设计等全部关键模块。

---

## 目录

1. [项目概览](#1-项目概览)
2. [数据准备](#2-数据准备)
   - 2.1 [SFT 数据（coldstart）](#21-sft-数据coldstart)
   - 2.2 [GRPO 数据（hybrid）](#22-grpo-数据hybrid)
3. [搜索引擎子系统](#3-搜索引擎子系统)
   - 3.1 [闭卷：Wiki Search](#31-闭卷wiki-search)
   - 3.2 [开放题：Web Search](#32-开放题web-search)
4. [SFT 训练（冷启动）](#4-sft-训练冷启动)
5. [GRPO 训练（强化学习）](#5-grpo-训练强化学习)
   - 5.1 [Rollout 阶段](#51-rollout-阶段)
   - 5.2 [Reward 计算](#52-reward-计算)
   - 5.3 [优势计算](#53-优势计算)
   - 5.4 [策略更新](#54-策略更新)
6. [奖励函数详解](#6-奖励函数详解)
   - 6.1 [F1 Reward（闭卷）](#61-f1-reward闭卷)
   - 6.2 [LFS Reward（开放题）](#62-lfs-reward开放题)
   - 6.3 [Format Reward](#63-format-reward)
   - 6.4 [DV Reward（查询多样性）](#64-dv-reward查询多样性)
7. [关键技术细节](#7-关键技术细节)
   - 7.1 [匈牙利算法配对](#71-匈牙利算法配对)
   - 7.2 [1对2 配对的损失](#72-1对2-配对的损失)
   - 7.3 [GRPO 组大小 n_agent](#73-grpo-组大小-n_agent)
8. [完整时序图](#8-完整时序图)

---

## 1. 项目概览

**O2-Searcher** 是一个 3B 参数的搜索增强 LLM agent，基于 Qwen2.5-3B-Instruct 训练。

训练分两阶段：
- **阶段 1：SFT 冷启动** — 用专家示范的多轮搜索轨迹微调基座模型
- **阶段 2：GRPO 强化学习** — 用搜索 + LLM-judge 的 reward 优化策略

**核心创新**：
- 同时支持**闭卷**（NQ/HotpotQA）和**开放题**（O²-QA）的搜索增强生成
- 训练用真实搜索引擎，开放题走 meilisearch + LLM 压缩 pipeline
- 奖励函数不只评"答案对不对"，还评"搜索 query 多样性"和"输出格式"

---

## 2. 数据准备

### 2.1 SFT 数据（coldstart）

**文件**：`o2searcher/data/coldstart/train.json` / `test.json`

**来源**：人工标注 + 专家 agent 跑出来的搜索轨迹

**结构**（messages 列表，多轮对话）：

```json
{
  "id": "819_1",
  "messages": [
    {"role": "system", "content": "...搜索 prompt..."},
    {"role": "user", "content": "Initial query: who was the singing voice of elsa in frozen?"},
    {"role": "assistant", "content": "<think>...</think>\n<search><query>...</query></search>"},
    {"role": "user", "content": "Search learnings: <learnings>Doc 1(Title: \"Elsa (Frozen)\")...真实搜索结果...</learnings>"},
    {"role": "assistant", "content": "<think>...Idina Menzel...</think>\n<answer>Idina Menzel</answer>"}
  ]
}
```

**统计**：
- 总样本数：**2453**（train）
- 平均 assistant 轮数：3.44（min 2, max 6）
- 平均 `<search>` 次数：2.45（min 1, max 5）
- ≥ 2 次搜索的样本：1617（66%）
- ≥ 3 次搜索的样本：1069（44%）

**关键设计**：
- 搜索结果是**预先离线抓取**的，写在 user 消息里
- SFT 阶段模型学习的是"看到 search 标签 → 假装拿到结果 → 继续推理"的格式模式
- 真实搜索在 GRPO rollout 阶段才发生

### 2.2 GRPO 数据（hybrid）

**生成脚本**：`scripts/data/o2searcher_dataset.py`

**处理逻辑**：

```python
# NQ/HotpotQA：从 parquet 读，sample 1200 条
# openended：从 JSON 读，重复 4 次
if split == 'train':
    datanames = ['nq_hotpotqa'] + ['openended']*4
else:
    datanames = ['nq_hotpotqa']  # test 只用闭卷
```

**最终数据量**：

| 来源 | 数量 |
|------|------|
| NQ/HotpotQA | 1200（sample n=1200, random_state=42）|
| openended × 4 | 240 × 4 = 960 |
| **总计** | **2160**（NQ:open ≈ 1.25:1）|

**为什么 `*4` 不是 `*5`**：历史遗留，README 写"300 manually curated"，原本可能是 300 训练 + 0 测试，后来切出 60 条做 O²-QA benchmark，训练剩 240 条，但 `*4` 没跟着改成 `*5` 凑完美 1:1。

**测试集设计**：
- `hybrid/test.parquet`：**只有 NQ**，240 条（从原始 NQ 测试分出）
- `openended/test_gt.json`：60 条开放题，**完全独立**的评测脚本 `scripts/eval/test_llm.py` 消费

**240 这个数字在代码里找不到** —— 它是数据准备阶段离线定的，训练脚本对"文件里有几条"完全无感。

---

## 3. 搜索引擎子系统

GRPO 训练时必须**先启动 3 个外部服务**（端口配置见各 `run_*.sh`）：

| 服务 | 端口 | 用途 | 代码 |
|------|------|------|------|
| Wiki dense retriever | 8000 | 闭卷向量检索 | 外部（如 Search-R1 提供）|
| wiki_search wrapper | 10001 | 转发到 8000 | `o2searcher/searcher/run_closedended.py` |
| meilisearch | 10000 | 开放题真实搜索 | 外部 |
| openended_search wrapper | 10102 | 包装搜索 + 抓网页 + LLM 压缩 | `o2searcher/searcher/run_openended.py` |
| Metric server | 11000 | 算 F1 / DV / LFS reward | `o2searcher/rewards/metrics/server.py` |

### 3.1 闭卷：Wiki Search

**端点**：`POST http://127.0.0.1:10001/wiki_search`

**输入**：
```json
{"queries": [["q1", "q2"], ["q3"]], "topk": 3}
```

**Pipeline**（[run_closedended.py](o2searcher/searcher/run_closedended.py)）：
1. 拉平 queries → 调用 dense retriever → 拿到 topk 文档
2. 每个 query 取 topk 文档的 contents
3. 拼接成字符串 → 返回

**特点**：
- **没有 LLM 压缩**——直接返回 raw retriever 命中内容
- 检索语料是预构建的本地 wiki dump
- 速度快、成本低

### 3.2 开放题：Web Search

**端点**：`POST http://127.0.0.1:10102/search`

**Pipeline**（[run_openended.py](o2searcher/searcher/run_openended.py)）：

```
query (英文)
  │
  ▼
[可选] 翻译为中文（如果 SEARCH_CN=True）
  │
  ▼
meilisearch 搜 → 拿到 URL + 摘要列表
  │
  ▼
[并发] 对每个 URL：
  ├─ aiohttp 抓 HTML
  ├─ BeautifulSoup 清洗（去 script/style）
  ├─ 截取前 32000 字符
  └─ LLM-1: compress_prompts → 压缩到 ≤2K token
  │
  ▼
把所有压缩后的内容拼起来
  │
  ▼
LLM-2: learnings_prompts → 提取 N 条 key learnings
  │
  ▼
[可选] 写入 memory bank（json 文件缓存）
  │
  ▼
返回 learnings 字符串（多 query 用 \n 拼接）
```

**两个 LLM 调用的 prompt 模板**：

`compress_prompts.py`：
```
You are an expert researcher.
Given raw webpage contents: <contents>{contents}</contents>, 
compress to a maximum 2K-token contents...
- Preserve critical information
- Prioritize content relevance to the research query
```

`learnings_prompts.py`：
```
Given the research query <query>{query}</query>, 
your task is to extract a list of key learnings from the provided contents.
Return a maximum of {num_learnings} distinct learnings.
...
```

**特点**：
- **真的搜互联网**（meilisearch + tavily 等）
- **两次 LLM 调用**做内容压缩
- 有 memory bank 缓存（`memory.json`），重复 query 不再搜
- 慢、贵、但能搜到最新信息

---

## 4. SFT 训练（冷启动）

**脚本**：`scripts/train/run_o2searcher_sft.sh`

**核心参数**：

```bash
model_path="Qwen2.5-3B-Instruct"
train_files="./o2searcher/data/coldstart/train.json"  # 2453 条
val_files="./o2searcher/data/coldstart/test.json"

data.train_batch_size=16
data.micro_batch_size=4
data.max_length=10240
optim.lr=1e-5
trainer.total_epochs=2
```

**训练器**：`verl.trainer.fsdp_sft_trainer`

**目标**：让基座模型学会
1. 看到 user 问题就生成 `<think>...</think>` 推理
2. 推理完打 `<search><query>...</query></search>` 标签
3. 看到 `<learnings>` 标签就把它当搜索结果处理
4. 信息够用就打 `<answer>...</answer>` 收尾

**注意点**：
- 这是普通 SFT（行为克隆），**没有 reward**，纯模仿专家轨迹
- 输入/输出长度 10240 是个比较宽的窗口，足够塞多轮搜索
- 跑 2 个 epoch，模型路径输出到 `./checkpoints/o2searcher/coldstart/global_step_306`

---

## 5. GRPO 训练（强化学习）

**脚本**：`scripts/train/run_o2searcher_grpo.sh`

**核心参数**：

```bash
algorithm.adv_estimator=grpo                # GRPO 而非 PPO
algorithm.kl_ctrl.kl_coef=0.001             # KL 惩罚系数
data.train_batch_size=32                    # 每次 step 32 条 prompt
data.max_prompt_length=8096
data.max_response_length=2048
data.max_start_length=2048
data.max_obs_length=2048
actor_rollout_ref.actor.lr=1e-6
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.rollout.n_agent=8         # GRPO 组大小（关键）
actor_rollout_ref.rollout.temperature=1     # 探索用
agent.max_turns=4                           # 最多 4 轮 search
```

**启动前必须**：
```bash
# 三个 server
python o2searcher/searcher/run_closedended.py --local_url http://127.0.0.1:8000/retrieve
python o2searcher/searcher/run_openended.py --local_url http://127.0.0.1:10000/search
python o2searcher/rewards/metrics/server.py --port 11000
```

### 5.1 Rollout 阶段

**核心类**：`LLMGenerationManager`（[o2searcher/generation.py](o2searcher/generation.py)）

**单次 step 流程**：

```
1. DataLoader 取 32 条 prompts
2. batch.repeat(n_agent=8, interleave=True) → 256 条 rollout
3. 每条 rollout 走多轮（最多 4 轮）：
   ┌────────────────────────────────────────────────┐
   │ a. vLLM 生成 response                          │
   │ b. _postprocess_responses: 截断到第一个 </search>│
   │    或 </answer>（避免一次生成太长）            │
   │ c. label_check: 校验有没有正确打标签          │
   │    - 没 <think> 但有其他标签 → penalty 0.5    │
   │    - 多标签 → penalty 0.1*(tag_count-1)       │
   │ d. postprocess_predictions: 解析 action        │
   │    - <search>...</search> → action='search'   │
   │    - <answer>...</answer> → action='answer'   │
   │ e. execute_predictions:                       │
   │    - 如果 action='search':                   │
   │      · batch_search 调对应 searcher           │
   │      · 用 <learnings>{result}</learnings>      │
   │        包成 next_obs 注入下一轮               │
   │    - 如果 action='answer': done=1            │
   │    - 格式错: 返回 error_prompt                │
   │ f. _update_rolling_state: 把 response+obs     │
   │    拼到 rolling state 里                      │
   │ g. 检查 active_mask，全部 done 才结束         │
   └────────────────────────────────────────────────┘
4. 收集所有 rollout 的完整轨迹
```

**关键代码**（[generation.py:373-421](o2searcher/generation.py#L373)）：

```python
def execute_predictions(self, predictions, pad_token, active_mask, abilities, do_search):
    cur_actions, contents, penalties = self.postprocess_predictions(predictions)
    next_obs, dones, valid_action, is_search = [], [], [], []
    
    # 批量调搜索引擎
    search_queries = [c for a, c in zip(cur_actions, contents) if a == 'search']
    search_abilities = [ab for a, ab in zip(cur_actions, abilities) if a == 'search']
    if do_search:
        search_results, all_queries = self.batch_search(search_queries, search_abilities)
    
    for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
        if not active:
            next_obs.append(''); dones.append(1)
        elif action == 'answer':
            next_obs.append(''); dones.append(1)
        elif action == 'search':
            # 把 search 结果包成 <learnings>...</learnings>
            next_obs.append(extra_prompt.format(learning_str=search_results.pop(0).strip()))
            dones.append(0)
```

**`batch_search` 关键代码**（[generation.py:479-497](o2searcher/generation.py#L479)）：

```python
def batch_search(self, queries, abilities):
    all_queries = []
    for query in queries:
        # 抽 <query>...</query> 里的内容
        sep = re.findall(r'<query>(.*?)</query>', query, re.DOTALL)
        sep = [p.strip() for p in sep] if sep else []
        all_queries.append(sep)
    results = self._batch_search(all_queries, abilities)
    return results, all_queries
```

**`label_check` 关键代码**（[generation.py:424-443](o2searcher/generation.py#L424)）：

```python
def label_check(self, prediction):
    think_matches = re.findall(r'<think>(.*?)</think>', prediction)
    search_matches = re.findall(r'<search>(.*?)</search>', prediction)
    answer_matches = re.findall(r'<answer>(.*?)</answer>', prediction)
    tag_count = len(search_matches) + len(answer_matches)
    
    if not think_matches:
        if tag_count > 0:
            return 0.5
        else:
            return 0
    penalty = (tag_count - 1) * 0.1
    return penalty
```

### 5.2 Reward 计算

**在 rollout 完成后立刻算**，不是训练完再算。

每条 rollout 走完（即出现 `<answer>` 或 `max_turns=4`）后：
1. 抽 `<answer>` 内容
2. 按 `ability` 字段分流：
   - `closedended` → F1 reward（[f1_reward.py](o2searcher/rewards/metrics/f1_reward.py)）
   - `openended` → LFS reward（[LFS.py](o2searcher/rewards/metrics/LFS.py)）
3. 算 format reward（[rewards_format.py](o2searcher/rewards/rewards_format.py)）
4. 把 rollout 期间的所有 search query 收集起来算 DV reward

### 5.3 优势计算

**核心代码**（[core_algos.py:111-148](verl/verl/trainer/ppo/core_algos.py#L111)）：

```python
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)  # 每条样本的 scalar reward
    
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])  # 按 index 分组
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    
    # advantage = (score - group_mean) / group_std
    advantages = scores - ...
```

**关键点**：
- `index` 来自 `extra_info['index']`，作为分组键（= `uid`）
- 同 `index` 的 8 条 rollout 形成一个 GRPO 组
- advantage = (这条的 reward - 组内均值) / 组内标准差
- **目标**：让 reward 高的样本概率上升，reward 低的样本概率下降

### 5.4 策略更新

标准的 PPO-clip 目标 + KL 惩罚：

```
loss = -E[advantage * clip(ratio, 1-ε, 1+ε)] + β * KL(π || π_ref)
```

- `actor_rollout_ref.actor.kl_loss_coef=0.001`（β）
- `actor_rollout_ref.actor.kl_loss_type=low_var_kl`
- `use_kl_loss=True`（把 KL 作为 loss 项而非 reward 项）
- `ppo_epochs=1`（每批数据只更新 1 次）

---

## 6. 奖励函数详解

### 6.1 F1 Reward（闭卷）

**端点**：`POST http://127.0.0.1:11000/calculate_finding_scores`

**Pipeline**（[f1_reward.py](o2searcher/rewards/metrics/f1_reward.py)）：

```
1. 抽 <answer> 里的 - bullet points
2. 过滤空内容、过滤纯 "and" 等无效内容
3. 调 LLM 翻译中文 → 英文（用 doubao-32k）
4. 用 all-MiniLM-L6-v2 编码所有 points
5. 算 N×M 余弦相似度矩阵
6. 软去重：每个 ref 最多保留相似度最高的 1 个 gen
7. 匈牙利算法做最优 1对1 配对
8. 数相似度 ≥ 0.85 的匹配对
9. precision = 匹配数 / num_gen
   recall    = 匹配数 / num_ref
   f1        = 2*P*R/(P+R)
```

**关键参数**：
- 句向量模型：`all-MiniLM-L6-v2`（轻量、本地）
- 翻译模型：`doubao-32k`（云端 LLM）
- 相似度阈值：**0.85**
- 匈牙利算法：`scipy.optimize.linear_sum_assignment`

**示例**：

```
模型输出（1 条 bullet）: "边缘计算既能处理近源数据降低延迟，又支持本地化决策"
参考答案（2 条）:
  ref_1: "处理近源数据降低延迟"
  ref_2: "本地化决策减少云端依赖"

相似度: 0.866 / 0.914
→ 匈牙利配对: 只能配 1 对
→ precision = 1/1 = 1.0
→ recall    = 1/2 = 0.5
→ F1 = 0.667（损失 1 个 ref 的 credit）
```

### 6.2 LFS Reward（开放题）

**端点**：`POST http://127.0.0.1:11000/calculate_finding_scores_llm`

**Pipeline**（[LFS.py](o2searcher/rewards/metrics/LFS.py)）：

```
1. 抽 <answer> 里的 - bullet points
2. 构造 LLM prompt：
   "Your task is to determine the semantic similarity between 
    input findings and target findings.
    Input: ...
    Target: ...
    Output JSON: [[\"input_1\", \"matched_target_1\"], ...]"
3. 调 LLM（默认 deepseek-v3）返回匹配对
4. precision = unique(input) 匹配数 / num_input
   recall    = unique(target) 匹配数 / num_target
   f1        = 2*P*R/(P+R)
```

**关键差异**：
- 不用句向量，**直接用 LLM 做语义匹配**
- 同样是 1对1 配对（LLM 自己保证）
- LLM judge 不"知道"正确答案，只是给参考答案做语义匹配

### 6.3 Format Reward

**代码**：[rewards_format.py](o2searcher/rewards/rewards_format.py)

```python
def calculate_format_reward(model_answer):
    lines = [line.strip() for line in model_answer.split('\n') if line.strip()]
    
    valid_bullets = 0
    content_list = []
    format_errors = 0
    for line in lines:
        if line.startswith('- '):
            content = line[2:].strip()
            if content:
                valid_bullets += 1
                content_list.append(content)
            else:
                format_errors += 1
        else:
            format_errors += 1
    
    format_reward     = 1 - (format_errors / len(lines))
    completeness      = min(valid_bullets / 10, 1)  # 10 条得满分
    diversity_reward  = calculate_diversity_reward(content_list)  # TF-IDF
    duplicate_penalty = 1 - len(set(content_list)) / max(1, len(content_list))
    
    weights = [0.5, 0.3, 0.5]
    reward = (weights[0]*format_reward + weights[1]*completeness + weights[2]*diversity_reward) / sum(weights) - 3*duplicate_penalty
    return FormatOutput(reward=max(0, min(1, reward)), ...)
```

**4 个分量**：
- `format_reward`：`- ` 开头才计数
- `completeness`：凑到 10 条得满分
- `diversity_reward`：TF-IDF 算 bullet 之间的相似度，相似度低得分高
- `duplicate_penalty`：3 倍惩罚重复 bullet

**关键设计**：
- 强制 ` - ` 分隔格式（避免整段塞一个 bullet）
- 鼓励拆细、拆多
- 这间接缓解了 F1 的 1对2 配对损失

### 6.4 DV Reward（查询多样性）

**端点**：`POST http://127.0.0.1:11000/calculate_query_independence`

**Pipeline**（[dv_reward.py](o2searcher/rewards/metrics/dv_reward.py)）：

```
1. 收集一轮 rollout 里的所有 <query>
2. 翻译为英文
3. 用 all-MiniLM-L6-v2 编码
4. 算 N×N 余弦相似度矩阵
5. 计算 overall_independence_score（基于 1 - 相似度）
6. 考虑 query 数量权重（3-10 个加分，>15 略降）
7. 应用 token 长度惩罚（< 8 token 的 query 扣分）
```

**目的**：惩罚模型反复搜相似 query，鼓励从不同角度探索

**示例**：
```
query 列表: ["AI in healthcare", "AI medical applications", "AI drug discovery"]
→ 3 个 query 高度相似
→ DV score 极低
→ reward 惩罚
```

---

## 7. 关键技术细节

### 7.1 匈牙利算法配对

**作用**：把"模型生成的 N 个要点"和"参考答案的 N 个要点"做最优 1对1 配对

**为什么需要**：
- 模型输出的 bullet 顺序可以错乱
- 匈牙利不要求顺序一致，按内容语义重新配对
- 比贪心稳定（贪心会被局部最优坑）

**完整流程**（[f1_reward.py:118-145](o2searcher/rewards/metrics/f1_reward.py#L118)）：

```python
# Step 1: 软去重（避免一个 ref 被多个 gen 抢）
for j in range(similarity_matrix.shape[1]):
    col = similarity_matrix[:, j]
    matches = col >= threshold
    if np.sum(matches) > 1:
        best_idx = np.argmax(col)
        matches = np.zeros_like(matches)
        matches[best_idx] = True
        similarity_matrix[:, j] = matches * col

# Step 2: 匈牙利
cost_matrix = 1 - similarity_matrix
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Step 3: 数有效匹配
valid_matches = [(i,j) for i,j in zip(row_ind, col_ind) 
                 if similarity_matrix[i,j] >= 0.85]
```

**复杂度**：O(N³)，scipy 工业级实现

### 7.2 1对2 配对的损失

**问题**：一条 bullet 塞两个信息，匈牙利只能配 1 对，**recall 砍半**

**实测**：

| 写法 | 实际 F1 |
|------|---------|
| 1 条 bullet 含 2 信息 | 0.667（recall 砍半）|
| 拆成 2 条 bullet | 1.000（完美）|

**缓解手段**：
- format_reward 强制 `- ` 分隔
- completeness_reward 鼓励 10 条
- 不在 reward 层做 1对多（数学上不允许）

### 7.3 GRPO 组大小 n_agent

**设置**：`actor_rollout_ref.rollout.n_agent=8`

**含义**：每条 prompt 复制 8 份组成 1 个 GRPO 组，组内算 advantage

**数据流**（[ray_trainer.py:679-756](verl/verl/trainer/ppo/ray_trainer.py#L679)）：

```python
# 取 batch
for batch_dict in self.train_dataloader:
    batch = DataProto.from_single_dict(batch_dict)
    
    # 复制 8 份
    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, 
                        interleave=True)
    
    # rollout ...
    
    # 关键：uid = index（不是重新生成）
    batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
    
    # 算 advantage
    batch = compute_advantage(batch, adv_estimator='grpo', ...)
```

**注意隐藏 bug**：
- `extra_info['index']` 直接当 `uid`
- openended 的 `index` 在 4 次重复里都从 0 开始（[o2searcher_dataset.py:87](scripts/data/o2searcher_dataset.py#L87)）
- NQ 的 `index` 来自原始 parquet，可能与 openended 撞车
- 撞车后不同 prompt 会被分到同一 GRPO 组，advantage 计算会乱

**修复方法**：用全局 counter：

```python
global_idx = 0
for data_source in datanames:
    for idx, (query, gt) in enumerate(data.items()):
        processed_example = process_fn(query, gt, global_idx)
        global_idx += 1
```

---

## 8. 完整时序图

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 0：数据准备                                              │
├─────────────────────────────────────────────────────────────┤
│  NQ parquet  ─┐                                              │
│                ├─► hybrid/train.parquet (2160 条)            │
│  openended ×4 ─┘                                              │
│  coldstart JSON (2453 条 SFT)                                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1：SFT 冷启动                                          │
├─────────────────────────────────────────────────────────────┤
│  for 2 epochs:                                               │
│    取 16 条 SFT 样本                                         │
│    拼成 messages，跑普通 SFT loss                            │
│    反向传播                                                  │
│  输出：./checkpoints/o2searcher/coldstart/global_step_306    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 2：GRPO 训练（启动 3 个 server）                        │
├─────────────────────────────────────────────────────────────┤
│  # 前置                                                       │
│  启动 wiki dense retriever (8000) + wrapper (10001)         │
│  启动 meilisearch (10000) + wrapper (10102)                  │
│  启动 metric server (11000)                                  │
│                                                               │
│  # 主循环 (151 steps × 3 epochs)                             │
│  for step in [1, 151]:                                       │
│    ① 取 32 条 prompt                                         │
│    ② batch.repeat(8) → 256 条 rollout                        │
│    ③ vLLM 并行 rollout（最多 4 轮 search）:                  │
│       for turn in [0, 4):                                    │
│         模型生成 response                                     │
│         截断到 </search> 或 </answer>                         │
│         label_check (本地 penalty 计算)                      │
│         parse action: search / answer / error                │
│         if search:                                           │
│           按 ability 分流：                                    │
│             - closedended → POST :10001/wiki_search          │
│               → 拿 topk 文档 → 返回 raw 内容                  │
│             - openended  → POST :10102/search                │
│               → meilisearch 搜 URL → aiohttp 抓 HTML        │
│               → LLM-1 compress → LLM-2 learnings            │
│               → 返回 learnings 字符串                         │
│           包成 <learnings>...</learnings> 注入 next_obs       │
│         if answer: done=1                                    │
│                                                               │
│    ④ Reward 计算（rollout 完立刻算）:                         │
│       F1/LFS reward: 抽 <answer> 调 :11000 metric server     │
│       format reward: 检查 - 开头、10 条、TF-IDF 多样性        │
│       DV reward: 这一轮所有 <query> 的相似度                  │
│       → scalar reward per rollout                            │
│                                                               │
│    ⑤ 优势计算（[core_algos.py](verl/verl/trainer/ppo/core_algos.py)）:                  │
│       按 uid (= index) 分 8 条一组                            │
│       advantage_i = (r_i - mean) / std                       │
│                                                               │
│    ⑥ 策略更新:                                                │
│       loss = -E[adv * clip(ratio, 1-ε, 1+ε)] + β*KL         │
│       FSDP 分布式反向传播                                     │
│                                                               │
│  输出：O²-Searcher-Qwen2.5-3B-GRPO 权重                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 阶段 3：评测                                                  │
├─────────────────────────────────────────────────────────────┤
│  # 训练中（val）                                             │
│  hybrid/test.parquet (240 NQ) → F1 reward                    │
│  test_freq=150，每 150 step 跑一次                             │
│                                                               │
│  # 训练后（O²-QA benchmark）                                 │
│  openended/test_gt.json (60 条) → 调独立脚本                  │
│  scripts/eval/test_llm.py                                    │
│  → LFS 评分（f1 或 lfs）                                     │
│  → 输出 ./outputs/                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录 A：关键文件索引

| 文件 | 作用 |
|------|------|
| `scripts/data/o2searcher_dataset.py` | 数据预处理 |
| `scripts/train/run_o2searcher_sft.sh` | SFT 训练脚本 |
| `scripts/train/run_o2searcher_grpo.sh` | GRPO 训练脚本 |
| `scripts/eval/test_llm.py` | 训练后 benchmark |
| `o2searcher/generation.py` | Rollout 主循环 |
| `o2searcher/searcher/run_closedended.py` | Wiki search server |
| `o2searcher/searcher/run_openended.py` | Web search + LLM 压缩 server |
| `o2searcher/searcher/generator.py` | LLM 客户端 |
| `o2searcher/searcher/prompts/compress_prompts.py` | 网页压缩 prompt |
| `o2searcher/searcher/prompts/learnings_prompts.py` | 关键要点提取 prompt |
| `o2searcher/rewards/metrics/server.py` | Metric server（FastAPI）|
| `o2searcher/rewards/metrics/f1_reward.py` | 匈牙利 + 句向量 F1 |
| `o2searcher/rewards/metrics/LFS.py` | LLM judge F1 |
| `o2searcher/rewards/metrics/dv_reward.py` | 查询多样性 |
| `o2searcher/rewards/metrics/utils.py` | 通用 LLM 客户端 + 文本解析 |
| `o2searcher/rewards/rewards_format.py` | 格式 reward |
| `o2searcher/rewards/rewards_score.py` | HTTP 调用入口 |
| `o2searcher/rewards/config.py` | metric server URL 配置 |
| `verl/verl/trainer/ppo/ray_trainer.py` | GRPO 训练主循环（verl 框架）|
| `verl/verl/trainer/ppo/core_algos.py` | GRPO 优势计算 |
| `verl/verl/trainer/config/ppo_trainer.yaml` | PPO/GRPO 配置模板 |

---

## 附录 B：常见疑问

**Q1: 240 这个数字哪来的？**
A: 数据准备阶段离线定的，代码里完全找不到。最可能是从 300 拆 240+60 时留下的历史。

**Q2: 为什么是 `*4` 不是 `*5`？**
A: 原本可能是 300 train × 4 = 1200（完美对齐 NQ），后来切了 60 条做 test，剩 240，没改 *4。

**Q3: LLM 怎么知道答案对不对？**
A: 不知道。LFS reward 把参考答案给 LLM，让 LLM 做语义匹配判断，**不是事实判断**。

**Q4: 1对2 配对会不会丢分？**
A: 会。F1 = 0.667 而非 1.0。format reward 鼓励拆细以缓解。

**Q5: reward 是训练完再算的吗？**
A: 不是。每条 rollout 完成后立刻算，advantage 立刻算，参数立刻更新——这是 RL 的核心。

**Q6: 闭卷和开放题为什么用不同搜索引擎？**
A: 闭卷用本地 wiki 向量检索（快、便宜、答案权威），开放题必须用真实互联网搜索（最新、覆盖广）。
