#!/usr/bin/env bash
# ============================================================================
# O2-Searcher 一键启动：终极修复版（彻底解决 wiki_server 000000 探测误报）
# 
# 用法：
#    chmod +x scripts/start_all.sh
#    ./scripts/start_all.sh
# ============================================================================

set -uo pipefail

# ============================================================================
# 配置（自适应你的 AutoDL 绝对路径）
# ============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/autodl-tmp/Researcher}"
CONDA_BASE_PATH="${CONDA_EXE%/bin/conda}" # 自动获取 conda 根路径

# 缓存重定向：HF / ModelScope / pip / torch 全部下到数据盘
# 后续每条 nohup bash -c "..." 也会重复 source，保证子 shell 一定有
source "$PROJECT_ROOT/scripts/env.sh"
ENV_SH="$PROJECT_ROOT/scripts/env.sh"

# 内部子模块路径自适应
if [ -d "$PROJECT_ROOT/o2searcher" ]; then
    WIKI_DIR="$PROJECT_ROOT/o2searcher/searcher/search_env/wiki_search"
    WEB_DIR="$PROJECT_ROOT/o2searcher/searcher/search_env/web_search"
    SRC_PREFIX="o2searcher/"
else
    WIKI_DIR="$PROJECT_ROOT/searcher/search_env/wiki_search"
    WEB_DIR="$PROJECT_ROOT/searcher/search_env/web_search"
    SRC_PREFIX=""
fi

# ============================================================================
# 工具函数
# ============================================================================
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR ]${NC} $*"; }
step() { echo -e "\n${CYAN}== $* ==${NC}"; }

# 修复核心：针对不同网关和底座，使用特定的协议和路由进行健康检查
#
# 注意一个隐蔽 bug：curl 连不上时，%{http_code} 仍会输出 "000" 到 stdout，
# 再叠加 "|| echo 000" 后 code 会变成 6 位的 "000000"。原来的判断
# `[ "$code" != "000" ] && [ "$code" != "00" ]` 对 "000000" 恒真，导致
# 服务挂了脚本反而报 ✅ ONLINE。下面用 case 严格只认 200/404/405。
wait_port() {
    local port=$1
    local name=$2
    local timeout=$3
    local code=""
    local curl_rc=0

    for ((i=0; i<timeout; i+=2)); do
        sleep 2
        code="000"
        curl_rc=0

        # 针对 8000 端口的维基服务，使用 POST 请求探测其核心路由 /retrieve
        # 注意：body 必须匹配 QueryRequest(queries, topk, return_scores) 的 Pydantic schema；
        #       旧版 "text" 字段会被 422 拒掉，且 return_scores=true 才能避开 endpoint 里
        #       `results, scores = batch_search(...)` 在 return_score=False 时只返回 1 个值的 bug
        if [ "$port" == "8000" ]; then
            code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:8000/retrieve" \
                  -H "Content-Type: application/json" -d '{"queries": ["test"], "topk": 1, "return_scores": true}' --max-time 5 2>/dev/null)
        else
            # 其他标准 FastAPI 网关使用标准的 GET 探测
            code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$port" 2>/dev/null)
        fi
        curl_rc=$?

        # 真正"在线"必须满足两个条件：
        #   1) curl 退出码 0（成功发出了请求并收到响应）
        #   2) HTTP 状态码是 200 / 404 / 405 之一（404 路由未匹配、405 方法不允许，都说明服务进程在）
        if [ "$curl_rc" -eq 0 ]; then
            case "$code" in
                200|404|405)
                    info "   ✅ $name 状态正常 (HTTP $code, 耗时 ${i}s)"
                    return 0
                    ;;
            esac
        fi
    done
    err "   ❌ $name 响应超时（最近一次 code=$code, curl_rc=$curl_rc），请检查 logs/ 下对应的日志文件。"
    return 1
}

is_running() {
    pgrep -f "$1" > /dev/null
}

# ============================================================================
# 0. 环境与资产检查
# ============================================================================
step "0/7 预检：项目目录、资产文件状态"

if [ ! -d "$PROJECT_ROOT" ]; then
    err "项目目录缺失: $PROJECT_ROOT"
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p logs

# 验证核心大文件完整性
MISSING=0
for f in \
    "$WIKI_DIR/data/e5_Flat.index" \
    "$WIKI_DIR/data/wiki-18.jsonl" \
    "$WIKI_DIR/model/e5-base-v2" \
    "$WEB_DIR/data/Web_data.json"
do
    if [ ! -e "$f" ]; then
        err "   缺失资产: $f"
        MISSING=1
    fi
done
[ $MISSING -eq 1 ] && { err "核心资产不全，请检查路径。"; exit 1; }
info "   ✅ 稠密索引与本地知识库资产校验成功"

# ============================================================================
# 1. 启动 Meilisearch
# ============================================================================
step "1/7 启动 7700 端口 Meilisearch 搜索引擎"

cd "$WEB_DIR"
if [ ! -x ./meilisearch ]; then
    curl -L https://install.meilisearch.com | sh
    chmod +x ./meilisearch
fi

if is_running meilisearch; then
    info "   Meilisearch 已经在跑，跳过拉起"
else
    nohup ./meilisearch --master-key='Web_Knowledge_Corpus' > "$PROJECT_ROOT/logs/meili.log" 2>&1 &
    wait_port 7700 "Meilisearch" 30 || { tail -30 "$PROJECT_ROOT/logs/meili.log"; exit 1; }
fi

# ============================================================================
# 2. 检查并同步索引数据
# ============================================================================
step "2/7 检查并同步开放域 Web 数据"
sleep 1

EXISTING=$(curl -s -X GET 'http://localhost:7700/indexes/Web_Corpus/stats' \
    -H "Authorization: Bearer Web_Knowledge_Corpus" 2>/dev/null | \
    python -c "import json,sys; print(json.load(sys.stdin).get('numberOfDocuments', 0))" 2>/dev/null || echo 0)

if [ "$EXISTING" -gt 0 ]; then
    info "   ✅ 索引库中已有 $EXISTING 条记录，无需重复灌数"
else
    info "   正在向 Meilisearch 灌入本地 Web 知识库..."
    # 修复：显式加载 conda 环境变量环境，避免后台静默死锁
    source "$CONDA_BASE_PATH/etc/profile.d/conda.sh" && conda activate researcher
    python web_data_upload.py
    sleep 3
fi

# ============================================================================
# 3. 启动维基底座（修复 000000 的核心部分）
# ============================================================================
step "3/7 启动 8000 端口 维基百科底座服务"

cd "$PROJECT_ROOT"
if is_running wiki_server.py; then
    info "   wiki_server 已经在后台运行中"
else
    # 修复：放弃 conda run，改用最硬核稳定的后台启动
    nohup bash -c "source $ENV_SH && source $CONDA_BASE_PATH/etc/profile.d/conda.sh && conda activate researcher && python $WIKI_DIR/wiki_server.py --index_path $WIKI_DIR/data/e5_Flat.index --corpus_path $WIKI_DIR/data/wiki-18.jsonl --retriever_model $WIKI_DIR/model/e5-base-v2 --topk 3" > logs/wiki.log 2>&1 &
    
    info "   正在将 60GB 索引 + 14GB 语料搬运至多卡 GPU + HuggingFace datasets 缓存（约需 2-5 分钟）..."
    wait_port 8000 "wiki_server底座" 360 || { tail -80 logs/wiki.log; exit 1; }
fi

# ============================================================================
# 4. 启动网页抽象包装层
# ============================================================================
step "4/7 启动 10000 端口 网页检索包装器"

cd "$WEB_DIR"
if is_running 'python web_search.py'; then
    info "   web_search 已经在运行中"
else
    nohup bash -c "source $ENV_SH && source $CONDA_BASE_PATH/etc/profile.d/conda.sh && conda activate researcher && python web_search.py" > "$PROJECT_ROOT/logs/web.log" 2>&1 &
    wait_port 10000 "web_search包装器" 20
fi

# ============================================================================
# 5. 级联启动大模型网关：Closed(10001) / Open(10102) / Metrics(11000)
# ============================================================================
step "5/7 级联对齐大模型网关端口映射"

cd "$PROJECT_ROOT"

# 开放题网关 (10102)
if is_running 'run_openended.py'; then
    info "   run_openended 已经在运行中"
else
    nohup bash -c "source $ENV_SH && source $CONDA_BASE_PATH/etc/profile.d/conda.sh && conda activate researcher && python ${SRC_PREFIX}searcher/run_openended.py --local_url http://127.0.0.1:10000/search --model_name qwen-turbo --port 10102" > logs/open.log 2>&1 &
    wait_port 10102 "OpenEnded网关" 20
fi

# 闭合题网关 (10001)
if is_running 'run_closedended.py'; then
    info "   run_closedended 已经在运行中"
else
    nohup bash -c "source $ENV_SH && source $CONDA_BASE_PATH/etc/profile.d/conda.sh && conda activate researcher && python ${SRC_PREFIX}searcher/run_closedended.py --local_url http://127.0.0.1:8000/retrieve --port 10001" > logs/closed.log 2>&1 &
    wait_port 10001 "ClosedEnded网关" 20
fi

# 强化学习指标监控端 (11000)
if is_running 'o2searcher.rewards.metrics.server'; then
    info "   metrics server 已经在运行中"
else
    nohup bash -c "source $ENV_SH && source $CONDA_BASE_PATH/etc/profile.d/conda.sh && conda activate researcher && export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/o2searcher && export HF_ENDPOINT=https://hf-mirror.com && python -m o2searcher.rewards.metrics.server --port 11000" > logs/metrics.log 2>&1 &
    # server.py 模块顶层就实例化 QueryIndependenceTransformer + FindingSentenceEvaluator，
    # uvicorn.run 之前要等这两个 evaluator 初始化完，端口 11000 才会被绑定，20s 完全不够
    wait_port 11000 "MetricServer" 180
fi

# ============================================================================
# 6. 大盘点与终检
# ============================================================================
step "6/7 全生命周期健康大盘点"

ALL_OK=1
echo ""
echo "   ┌────────────────┬─────────┬──────────┐"
echo "   │ 服务网关名称   │ 监听端口│ 当前状态 │"
echo "   ├────────────────┼─────────┼──────────┤"
for entry in "7700:Meilisearch" "8000:WikiBase" "10000:WebSearch" "10001:ClosedGateway" "10102:OpenGateway" "11000:MetricServer"; do
    port="${entry%%:*}"
    name="${entry##*:}"

    code="000"
    curl_rc=0

    if [ "$port" == "8000" ]; then
        code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:8000/retrieve" -H "Content-Type: application/json" -d '{"queries": ["t"], "topk": 1, "return_scores": true}' --max-time 5 2>/dev/null)
    else
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$port" 2>/dev/null)
    fi
    curl_rc=$?

    # 同样的假阳性修复：必须 curl 成功退出且 code 是 200/404/405 才算在线
    if [ "$curl_rc" -eq 0 ] && { [ "$code" = "200" ] || [ "$code" = "404" ] || [ "$code" = "405" ]; }; then
        printf "   │ %-14s │ %-7s │ ✅ ONLINE │\n" "$name" "$port"
    else
        printf "   │ %-14s │ %-7s │ ❌ DOWN   │\n" "$name" "$port"
        ALL_OK=0
    fi
done
echo "   └────────────────┴─────────┴──────────┘"

echo ""
echo "===================================================================="
if [ $ALL_OK -eq 1 ]; then
    info "🎉 完美！所有后台进程全部健康挂载，探测机制完美切合服务路径！"
    info "现在直接去你的训练窗口拉起 GRPO 即可，绝不会再发生进程打不开的误报："
    echo ""
    echo "    conda activate researcher"
    echo "    ./scripts/train/run_o2searcher_grpo.sh"
    echo ""
else
    warn "⚠️ 仍有网关返回空响应，请执行 'cat logs/文件名.log' 捞取底层信息。"
fi
echo "===================================================================="