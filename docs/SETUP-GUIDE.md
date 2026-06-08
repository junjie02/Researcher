# O²-Searcher 环境搭建 + 数据下载 + 训练启动 完整指南

按本文档**从上到下顺序执行**即可。所有命令都基于项目根目录 `f:/CODE/SuperAssist/O2-Searcher`。

---

## 0. 前置条件检查

| 项 | 最低要求 | 推荐 |
|---|---|---|
| 磁盘空间（稳态） | 100 GB | 150 GB（留 buffer + checkpoint） |
| 磁盘空间（峰值，cat 时） | 145 GB | 同上 |
| GPU | 1 × 24GB（如 RTX 3090/4090） | 4 × A100-80GB（论文配置） |
| 系统内存 | 32 GB | 64 GB |
| Python | 3.10 | 3.10 |
| CUDA | 12.1 | 12.1 |

**磁盘明细**（实测）：
- `wiki-18.jsonl` 解压后 ~15 GB
- `e5_Flat.index` **64.5 GB**（part_aa: 42.9G + part_ab: 21.6G，fp32 全精度）
- `e5-base-v2` 模型 ~440 MB
- `Web_data.json` ~1-3 GB
- `nq_hotpotqa` parquet ~100 MB
- 训练 checkpoint ~10 GB / step

⚠️ **`cat part_* > e5_Flat.index` 这步会临时占用 2× 空间（约 129 GB）**，磁盘紧时见下面"流式拼接"方案。

---

## 1. 环境安装

### 1.1 创建主训练环境（用于 SFT / GRPO）

```bash
cd f:/CODE/SuperAssist/O2-Searcher

conda create -n o2searcher python=3.10 -y
conda activate o2searcher

pip install -e ./verl
pip install -e .
pip install wandb uvicorn bs4 tavily-python "huggingface_hub[cli]"
```

### 1.2 创建检索后端环境（用于 wiki_server / web_search）

```bash
conda create -n searcher python=3.10 -y
conda activate searcher

# 用 conda 装 torch 才能配套 faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install transformers datasets pyserini

# faiss-gpu 必须用 conda 装
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y

pip install uvicorn fastapi meilisearch httpx pydantic
```

### 1.3 （可选）国内 HuggingFace 镜像

```bash
# 加到 ~/.bashrc 或当前 shell
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 2. 创建必需目录

```bash
cd f:/CODE/SuperAssist/O2-Searcher

mkdir -p o2searcher/data/nq_hotpotqa
mkdir -p o2searcher/data/hybrid
mkdir -p o2searcher/searcher/search_env/wiki_search/data
mkdir -p o2searcher/searcher/search_env/wiki_search/model
mkdir -p o2searcher/searcher/search_env/web_search/data
mkdir -p checkpoints/o2searcher
```

---

## 3. 下载所有数据

> 仓库自带、**不用下**的：
> - `o2searcher/data/coldstart/train.json` 和 `test.json`（SFT 冷启动数据）
> - `o2searcher/data/openended/train_gt.json` / `test_gt.json` / `difficulty_labels.json`

### 3.1 NQ + HotpotQA 训练问答（封闭式 RL 训练用）

```bash
cd f:/CODE/SuperAssist/O2-Searcher

huggingface-cli download PeterJinGo/nq_hotpotqa_train \
    --repo-type dataset \
    --local-dir ./o2searcher/data/nq_hotpotqa
```

验证：`./o2searcher/data/nq_hotpotqa/` 应该有 `train.parquet` 和 `test.parquet`。

### 3.2 Wiki-18 语料 + 已建好的 Faiss 索引（封闭式检索用）

```bash
cd f:/CODE/SuperAssist/O2-Searcher/o2searcher/searcher/search_env/wiki_search

# ⚠️ wiki_download.py 有 bug，只下 corpus 不下 index，必须分两步

# 第一步：下 corpus
python wiki_download.py --save_path ./data

# 第二步：手动下 index（wiki_download.py 漏的，**64.5 GB**）
huggingface-cli download PeterJinGo/wiki-18-e5-index \
    --repo-type dataset \
    --local-dir ./data

# 把分片拼回完整索引
# ⚠️ 磁盘充裕（>130GB 可用）时直接：
cat ./data/part_* > ./data/e5_Flat.index && rm ./data/part_*

# ⚠️ 磁盘紧时用流式拼接（边拼边删，省 43GB 峰值）：
# cat ./data/part_aa > ./data/e5_Flat.index && rm ./data/part_aa
# cat ./data/part_ab >> ./data/e5_Flat.index && rm ./data/part_ab

# 解压 corpus
gzip -d ./data/wiki-18.jsonl.gz
```

验证：`./data/` 应该有：
- `wiki-18.jsonl`（~15 GB）
- `e5_Flat.index`（**~64.5 GB**）

### 3.3 E5 编码模型

```bash
# 仍然在 wiki_search/ 目录
huggingface-cli download intfloat/e5-base-v2 \
    --local-dir ./model/e5-base-v2
```

验证：`./model/e5-base-v2/` 应该有 `config.json`、`pytorch_model.bin` / `model.safetensors`、`tokenizer.json` 等。

### 3.4 Web Knowledge Corpus（开放式检索用）

```bash
cd f:/CODE/SuperAssist/O2-Searcher/o2searcher/searcher/search_env/web_search

huggingface-cli download TaoTao0216/O2-QA-Web_data \
    --repo-type dataset \
    --local-dir ./data
```

验证：`./data/Web_data.json` 存在。

---

## 4. 配置 LLM API（开放式服务必需）

编辑 [o2searcher/config.json](../o2searcher/config.json)，**至少填一个模型**的完整配置：

```json
{
    "qwen2.5": {
        "model_name": "Qwen2.5-72B-Instruct",
        "api_key_var": "sk-你的key",
        "base_url": "https://你的-llm-endpoint.com/v1",
        "proxy": false
    }
}
```

> 也可以用 deepseek-v3、doubao、gpt-4o 等，启动 `run_openended.py` 时加 `--model_name xxx` 切换。

---

## 5. 生成训练用的 hybrid parquet

```bash
cd f:/CODE/SuperAssist/O2-Searcher
conda activate o2searcher

python ./scripts/data/o2searcher_dataset.py --split train
python ./scripts/data/o2searcher_dataset.py --split test
```

验证：`./o2searcher/data/hybrid/` 应该有 `train.parquet` 和 `test.parquet`。
- train：~1200（封闭式）+ 4 × N（开放式）行
- test：仅封闭式

---

## 6. 下载基础模型（SFT 训练用）

[scripts/train/run_o2searcher_sft.sh:7](../scripts/train/run_o2searcher_sft.sh#L7) 默认用 `Qwen2.5-3B-Instruct`：

```bash
cd f:/CODE/SuperAssist/O2-Searcher

huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir ./models/Qwen2.5-3B-Instruct
```

然后修改 [scripts/train/run_o2searcher_sft.sh:7](../scripts/train/run_o2searcher_sft.sh#L7)：

```bash
model_path="./models/Qwen2.5-3B-Instruct"   # 改成本地路径
```

---

## 7. 启动 4 个后端服务（GRPO 训练前必须全部启动）

**建议每个服务开一个 tmux/screen 窗口**，方便观察日志。

### 窗口 1：Meilisearch 引擎（端口 7700）

```bash
cd f:/CODE/SuperAssist/O2-Searcher/o2searcher/searcher/search_env/web_search
conda activate searcher

# 第一次需要安装二进制
curl -L https://install.meilisearch.com | sh

# 启动（前台）
./meilisearch --master-key="Web_Knowledge_Corpus"
```

### 窗口 2：导入 Web_data + 启动 web_search wrapper（端口 10000）

```bash
cd f:/CODE/SuperAssist/O2-Searcher/o2searcher/searcher/search_env/web_search
conda activate searcher

# 一次性：把 Web_data.json 灌入 Meilisearch（约 10 分钟）
python web_data_upload.py

# ⚠️ 等 Meilisearch 后台索引彻底跑完再启动下一步
# 可以打开 http://localhost:7700 看 task 状态

# 启动 web_search FastAPI 服务
python web_search.py
```

### 窗口 3：Wiki dense 检索（端口 8000，需 GPU）

```bash
cd f:/CODE/SuperAssist/O2-Searcher/o2searcher/searcher/search_env/wiki_search
conda activate searcher

python wiki_server.py \
    --index_path ./data/e5_Flat.index \
    --corpus_path ./data/wiki-18.jsonl \
    --retriever_model ./model/e5-base-v2 \
    --topk 3
```

> 启动需要 1-2 分钟（加载 30GB 索引到 GPU + 21M 行 corpus 用 mmap 加载）。

### 窗口 4：openended wrapper（端口 10102）

```bash
cd f:/CODE/SuperAssist/O2-Searcher
conda activate o2searcher

python o2searcher/searcher/run_openended.py \
    --local_url http://127.0.0.1:10000/search \
    --model_name qwen2.5 \
    --port 10102
```

### 窗口 5：closedended wrapper（端口 10001）

```bash
cd f:/CODE/SuperAssist/O2-Searcher
conda activate o2searcher

python o2searcher/searcher/run_closedended.py \
    --local_url http://127.0.0.1:8000/retrieve \
    --port 10001
```

---

## 8. Smoke test：验证 4 个服务都通

```bash
cd f:/CODE/SuperAssist/O2-Searcher
conda activate o2searcher

# 默认会测 closedended (:10001)
python o2searcher/searcher/test_api.py
```

应该看到 wiki 段落字符串返回。

测开放式（手动）：

```bash
curl -X POST http://127.0.0.1:10102/search \
    -H "Content-Type: application/json" \
    -d '{"queries": [["What is edge computing?"]], "topk": 3}'
```

应该看到 `<learnings>...</learnings>` 风格的文本返回（耗时 5-15s，因为后端要调 LLM 抽 learnings）。

---

## 9. 启动训练

### 9.1 SFT 冷启动（先跑这个）

```bash
cd f:/CODE/SuperAssist/O2-Searcher
conda activate o2searcher

# 参数 4 = GPU 数量
./scripts/train/run_o2searcher_sft.sh 4
```

输出 checkpoint 到 `./checkpoints/o2searcher/coldstart/global_step_<n>/`。论文里用的是 `global_step_306`。

### 9.2 GRPO 训练（依赖 step 7 的 4 个服务全部在跑）

确认 [scripts/train/run_o2searcher_grpo.sh:12](../scripts/train/run_o2searcher_grpo.sh#L12) 里的 MODEL_PATH 指向你 SFT 完的 checkpoint：

```bash
MODEL_PATH="./checkpoints/o2searcher/coldstart/global_step_306"
```

启动：

```bash
./scripts/train/run_o2searcher_grpo.sh
```

---

## 10. 常见踩坑

| 问题 | 解决 |
|---|---|
| `huggingface-cli` 国内下载慢 / 中断 | `export HF_ENDPOINT=https://hf-mirror.com` 然后重跑 |
| `wiki_download.py` 没下 e5 index | 用本文档 3.2 的第二步 `huggingface-cli download PeterJinGo/wiki-18-e5-index` |
| `web_search.py` 返回空 hits | Meilisearch 后台索引没建完，等 5-10 分钟再试 |
| `wiki_server.py` OOM | 减小 GPU 数量或把 `faiss_gpu=False`（CPU 模式，会慢） |
| `run_openended.py` 报 401 / API key error | 检查 `o2searcher/config.json` 里 api_key / base_url 填了没 |
| GRPO 训练卡在搜索请求 | 检查 4 个后端服务都在跑：`curl http://127.0.0.1:10001 http://127.0.0.1:10102 http://127.0.0.1:10000 http://127.0.0.1:8000` |
| wandb 上传失败 | 在两个训练脚本第 1 行附近替换 WANDB_API_KEY，或注释掉 wandb logger |

---

## 11. 一键 smoke test 脚本（可选）

把下面存为 `scripts/check_setup.sh`，跑一遍能验证所有服务是否就绪：

```bash
#!/bin/bash
set -e
echo "=== 检查必需文件 ==="
test -f ./o2searcher/data/nq_hotpotqa/train.parquet && echo "✅ nq_hotpotqa train" || echo "❌ 缺 nq_hotpotqa train"
test -f ./o2searcher/data/hybrid/train.parquet && echo "✅ hybrid train" || echo "❌ 缺 hybrid（跑 o2searcher_dataset.py）"
test -f ./o2searcher/searcher/search_env/wiki_search/data/wiki-18.jsonl && echo "✅ wiki corpus" || echo "❌ 缺 wiki-18.jsonl"
test -f ./o2searcher/searcher/search_env/wiki_search/data/e5_Flat.index && echo "✅ e5 index" || echo "❌ 缺 e5_Flat.index"
test -d ./o2searcher/searcher/search_env/wiki_search/model/e5-base-v2 && echo "✅ e5 model" || echo "❌ 缺 e5-base-v2"
test -f ./o2searcher/searcher/search_env/web_search/data/Web_data.json && echo "✅ web corpus" || echo "❌ 缺 Web_data.json"

echo ""
echo "=== 检查 4 个服务 ==="
for port in 7700 8000 10000 10001 10102; do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$port | grep -qE "200|404|405"; then
        echo "✅ port $port 在跑"
    else
        echo "❌ port $port 不可达"
    fi
done
```

跑：

```bash
chmod +x scripts/check_setup.sh
./scripts/check_setup.sh
```

---

## 12. 端口/服务总表（速查）

| 端口 | 服务 | 启动命令 | 启动顺序 |
|---|---|---|---|
| 7700 | Meilisearch | `./meilisearch --master-key="Web_Knowledge_Corpus"` | 1 |
| 10000 | web_search wrapper | `python web_search.py` | 2（Meilisearch 索引就绪后） |
| 8000 | Wiki dense 检索 | `python wiki_server.py` | 3 |
| 10001 | closedended wrapper | `python run_closedended.py` | 4 |
| 10102 | openended wrapper | `python run_openended.py` | 5 |

---

## 13. 训练前最终 checklist

- [ ] 5 个后端服务全部启动且互通（`check_setup.sh` 全绿）
- [ ] `o2searcher/data/hybrid/{train,test}.parquet` 存在
- [ ] `o2searcher/config.json` 的 LLM API key 填好且可用
- [ ] SFT 冷启动 checkpoint 已生成（`./checkpoints/o2searcher/coldstart/global_step_*/`）
- [ ] `run_o2searcher_grpo.sh` 里 `MODEL_PATH` 指向上一步的 checkpoint
- [ ] `WANDB_API_KEY` 替换为自己的（或注释掉 wandb logger）
- [ ] 至少有 1 张可用 GPU（GRPO 默认要 4 张，可改 `trainer.n_gpus_per_node`）

全部勾上即可 `./scripts/train/run_o2searcher_grpo.sh` 开训。