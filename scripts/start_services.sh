#!/usr/bin/env bash
# ============================================================================
# O2-Searcher 一键启动：服务 + 数据 + 索引 + 训练数据
# 
# 用法：
#   chmod +x scripts/start_all.sh
#   ./scripts/start_all.sh
#
# 启动完成后会显示「可以跑 GRPO」的提示，照着敲即可
# ============================================================================

set -uo pipefail

# ============================================================================
# 配置（可通过环境变量覆盖）
# ============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/autodl-tmp/Researcher/o2searcher}"
SEARCH_ENV="${SEARCH_ENV:-researcher}"       # meilisearch / web_search / run_openended
TRAIN_ENV="${TRAIN_ENV:-researcher}"         # wiki_server / closedended / metrics / GRPO
LLM_ALIAS="${LLM_ALIAS:-deepseek-v4}"            # openended wrapper 用的模型别名
WANDB_KEY="${WANDB_KEY:-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}"                   # 留空则不配置 wandb

# 路径
WEB_DIR="$PROJECT_ROOT/searcher/search_env/web_search"
WIKI_DIR="$PROJECT_ROOT/searcher/search_env/wiki_search"

# ============================================================================
# 工具函数
# ============================================================================
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR ]${NC} $*"; }
step() { echo -e "\n${CYAN}== $* ==${NC}"; }

# 等待端口就绪
wait_port() {
    local port=$1
    local name=$2
    local timeout=$3
    for ((i=0; i<timeout; i+=2)); do
        sleep 2
        local code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$port" 2>/dev/null || echo "000")
        if [ "$code" != "000" ]; then
            info "  ✅ $name 就绪 (HTTP $code, ${i}s)"
            return 0
        fi
    done
    err "  ❌ $name 启动超时（${timeout}s）"
    return 1
}

# 检测进程是否在跑
is_running() {
    pgrep -f "$1" > /dev/null
}

# ============================================================================
# 0. 预检
# ============================================================================
step "0/7 预检：项目目录、conda 环境、数据文件"

if [ ! -d "$PROJECT_ROOT" ]; then
    err "找不到项目目录: $PROJECT_ROOT"
    err "请通过 PROJECT_ROOT 环境变量指定，例如："
    err "  PROJECT_ROOT=/path/to/o2searcher $0"
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p logs

info "  PROJECT_ROOT: $PROJECT_ROOT"
info "  SEARCH_ENV:   $SEARCH_ENV"
info "  TRAIN_ENV:    $TRAIN_ENV"

command -v conda > /dev/null || { err "找不到 conda"; exit 1; }

for env in "$SEARCH_ENV" "$TRAIN_ENV"; do
    if ! conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$env"; then
        err "conda 环境不存在: $env"
        err "请先 conda create -n $env python=3.10"
        exit 1
    fi
done

# 数据文件检查
MISSING=0
for f in \
    "$WIKI_DIR/data/e5_Flat.index" \
    "$WIKI_DIR/data/wiki-18.jsonl" \
    "$WIKI_DIR/model/e5-base-v2" \
    "$WEB_DIR/data/Web_data.json"
do
    if [ ! -e "$f" ]; then
        err "  缺：$f"
        MISSING=1
    fi
done
[ $MISSING -eq 1 ] && { err "数据文件不齐全，请按 docs/SETUP-GUIDE.md 下载"; exit 1; }
info "  ✅ 数据文件齐全"

# config.json 检查
if grep -q '"api_key_var": ""' o2searcher/config.json 2>/dev/null; then
    warn "  ⚠️  config.json 里 api_key_var 留空"
    warn "  ⚠️  开放式检索会失败，但 SFT / 闭集 GRPO 不受影响"
fi

# ============================================================================
# 1. Meilisearch
# ============================================================================
step "1/8 启动 Meilisearch (port 7700)"

cd "$WEB_DIR"
if [ ! -x ./meilisearch ]; then
    warn "  meilisearch 二进制不存在，正在下载..."
    curl -L https://install.meilisearch.com | sh
    chmod +x ./meilisearch
fi

if is_running meilisearch; then
    info "  Meilisearch 已在跑"
else
    nohup ./meilisearch --master-key='Web_Knowledge_Corpus' > "$PROJECT_ROOT/logs/meili.log" 2>&1 &
    wait_port 7700 "Meilisearch" 30 || { tail -30 "$PROJECT_ROOT/logs/meili.log"; exit 1; }
fi

# ============================================================================
# 2. 上传 Web_data 到 Meilisearch（幂等：有数据就跳过）
# ============================================================================
step "2/7 检查/上传 Web_data 到 Meilisearch"

# 给索引 5 秒就绪时间
sleep 3

# 拿当前 Meilisearch 里的文档数
EXISTING=$(curl -s -X GET 'http://localhost:7700/indexes/Web_Corpus/stats' \
    -H "Authorization: Bearer Web_Knowledge_Corpus" 2>/dev/null | \
    python -c "import json,sys; print(json.load(sys.stdin).get('numberOfDocuments', 0))" 2>/dev/null || echo 0)

# 拿 Web_data.json 里的文档数
EXPECTED=$(python -c "import json; print(len(json.load(open('./data/Web_data.json', encoding='utf-8'))))" 2>/dev/null || echo 0)

# ⭐ 关键：只要有数据就跳过，不管数量对不对
if [ "$EXISTING" -gt 0 ]; then
    if [ "$EXPECTED" -gt 0 ] && [ "$EXISTING" -lt "$EXPECTED" ]; then
        warn "  ⚠️  现有 $EXISTING 条 < 预期 $EXPECTED 条，数据可能不完整"
        warn "  ⚠️  如果想重新灌：先停 Meilisearch、删掉 ./data/.ms_data/、再起 Meilisearch、再跑本脚本"
    fi
    info "  ✅ Meilisearch 已有 $EXISTING 条文档，跳过上传"
else
    info "  Meilisearch 是空的，开始上传 Web_data.json（这一步可能要 10 分钟）..."
    conda run -n "$SEARCH_ENV" python web_data_upload.py
    
    info "  等待 Meilisearch 后台索引完成..."
    for ((i=0; i<60; i++)); do
        sleep 10
        DONE=$(curl -s -X GET 'http://localhost:7700/indexes/Web_Corpus/stats' \
            -H "Authorization: Bearer Web_Knowledge_Corpus" 2>/dev/null | \
            python -c "import json,sys; print(json.load(sys.stdin).get('numberOfDocuments', 0))" 2>/dev/null || echo 0)
        if [ "$DONE" -gt 0 ]; then
            info "  ✅ 索引就绪（$DONE 条）"
            break
        fi
        printf "    [%2d/60] 等待 Meilisearch 索引...\r" $((i+1))
    done
    echo ""
fi


# ============================================================================
# 3. wiki_server (慢，要 1-2 分钟)
# ============================================================================
step "3/7 启动 wiki_server (port 8000)"

cd "$PROJECT_ROOT"
if is_running wiki_server.py; then
    info "  wiki_server 已在跑"
else
    nohup conda run -n "$TRAIN_ENV" python "$WIKI_DIR/wiki_server.py" \
        --index_path "$WIKI_DIR/data/e5_Flat.index" \
        --corpus_path "$WIKI_DIR/data/wiki-18.jsonl" \
        --retriever_model "$WIKI_DIR/model/e5-base-v2" \
        --topk 3 \
        > logs/wiki.log 2>&1 &
    
    info "  等待 wiki_server 加载 30 GB 索引到 GPU（1-2 分钟）..."
    wait_port 8000 "wiki_server" 180 || { tail -50 logs/wiki.log; exit 1; }
fi

# ============================================================================
# 4. web_search wrapper
# ============================================================================
step "4/7 启动 web_search (port 10000)"

cd "$WEB_DIR"
if is_running 'python web_search.py'; then
    info "  web_search 已在跑"
else
    nohup conda run -n "$SEARCH_ENV" python web_search.py > "$PROJECT_ROOT/logs/web.log" 2>&1 &
    wait_port 10000 "web_search" 15
fi

# ============================================================================
# 5. openended + closedended + metrics
# ============================================================================
step "5/7 启动 openended (10102) / closedended (10001) / metrics (11000)"

cd "$PROJECT_ROOT"

# openended
if is_running 'run_openended.py'; then
    info "  run_openended 已在跑"
else
    nohup conda run -n "$SEARCH_ENV" python o2searcher/searcher/run_openended.py \
        --local_url http://127.0.0.1:10000/search \
        --model_name "$LLM_ALIAS" --port 10102 \
        > logs/open.log 2>&1 &
    wait_port 10102 "run_openended" 15
fi

# closedended
if is_running 'run_closedended.py'; then
    info "  run_closedended 已在跑"
else
    nohup conda run -n "$TRAIN_ENV" python o2searcher/searcher/run_closedended.py \
        --local_url http://127.0.0.1:8000/retrieve --port 10001 \
        > logs/closed.log 2>&1 &
    wait_port 10001 "run_closedended" 15
fi

# metrics
if is_running 'o2searcher.rewards.metrics.server'; then
    info "  metrics server 已在跑"
else
    nohup conda run -n "$TRAIN_ENV" python -m o2searcher.rewards.metrics.server --port 11000 \
        > logs/metrics.log 2>&1 &
    wait_port 11000 "metrics server" 15
fi


# ============================================================================
# 6. SFT 冷启动 checkpoint
# ============================================================================
step "6/8 检查 SFT 冷启动 checkpoint"

if [ -d "checkpoints/o2searcher/coldstart" ]; then
    CKPT=$(ls -d checkpoints/o2searcher/coldstart/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CKPT" ] && [ -d "$CKPT/actor" ]; then
        info "  ✅ SFT checkpoint: $CKPT"
    else
        warn "  ⚠️  coldstart 目录存在但没有完整 checkpoint"
        warn "  需要跑："
        echo "      conda activate $TRAIN_ENV"
        echo "      ./scripts/train/run_o2searcher_sft.sh 4    # 4 = GPU 数"
    fi
else
    warn "  ⚠️  还没有 SFT checkpoint"
    warn "  GRPO 需要从 SFT 后的模型接着训，请运行："
    echo "      conda activate $TRAIN_ENV"
    echo "      ./scripts/train/run_o2searcher_sft.sh 4    # 4 = GPU 数"
    echo "  论文用的是 global_step_306。跑完后把 run_o2searcher_grpo.sh 里的 MODEL_PATH 改对"
fi

# ============================================================================
# 7. 终检
# ============================================================================
step "7/7 终检"

ALL_OK=1
echo ""
echo "  ┌────────────────┬─────────┬──────────┐"
echo "  │ 服务           │ 端口     │ 状态      │"
echo "  ├────────────────┼─────────┼──────────┤"
for entry in "7700:Meilisearch" "8000:wiki_server" "10000:web_search" "10001:closedended" "10102:openended" "11000:metrics"; do
    port="${entry%%:*}"
    name="${entry##*:}"
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$port" 2>/dev/null || echo "000")
    if [ "$code" != "000" ]; then
        printf "  │ %-14s │ %-7s │ ✅ HTTP %-3s │\n" "$name" "$port" "$code"
    else
        printf "  │ %-14s │ %-7s │ ❌ DOWN   │\n" "$name" "$port"
        ALL_OK=0
    fi
done
echo "  └────────────────┴─────────┴──────────┘"

echo ""
echo "  数据:"
for f in \
    "data/hybrid/train.parquet" \
    "data/hybrid/test.parquet" \
    "data/nq_hotpotqa/train.parquet" \
    "data/nq_hotpotqa/test.parquet"
do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        printf "    ✅ %-50s %s\n" "$f" "$size"
    else
        printf "    ❌ %-50s MISSING\n" "$f"
        ALL_OK=0
    fi
done

echo ""
echo "=========================================="
if [ $ALL_OK -eq 1 ]; then
    info "🎉 全部就绪！可以开 GRPO："
    echo ""
    echo "    conda activate $TRAIN_ENV"
    echo "    ./scripts/train/run_o2searcher_grpo.sh"
    echo ""
else
    warn "⚠️  有服务/数据缺失，请看上面的状态和 logs/*.log"
fi
echo "=========================================="
