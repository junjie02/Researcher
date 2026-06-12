#!/usr/bin/env bash
# ============================================================================
# 全局缓存路径声明 —— 让 HuggingFace / ModelScope / pip / torch 等所有
# 自动下载行为，统统落到数据盘 /root/autodl-tmp/.cache，不再撑系统盘
#
# 用法：在任何会触发下载的脚本（启动脚本、训练脚本、推理脚本）顶部加一行
#   source "$(dirname "$0")/../env.sh"
# 或者
#   source /root/autodl-tmp/Researcher/scripts/env.sh
# ============================================================================

export AUTODL_CACHE_ROOT="/root/autodl-tmp/.cache"

# 目录懒创建
mkdir -p "$AUTODL_CACHE_ROOT/huggingface/hub" \
         "$AUTODL_CACHE_ROOT/huggingface/datasets" \
         "$AUTODL_CACHE_ROOT/modelscope" \
         "$AUTODL_CACHE_ROOT/pip" \
         "$AUTODL_CACHE_ROOT/torch" 2>/dev/null

# HuggingFace 体系
export HF_HOME="$AUTODL_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_METRICS_CACHE="$HF_HOME/metrics"

# ModelScope
export MODELSCOPE_CACHE="$AUTODL_CACHE_ROOT/modelscope"

# pip
export PIP_CACHE_DIR="$AUTODL_CACHE_ROOT/pip"

# torch / torch hub
export TORCH_HOME="$AUTODL_CACHE_ROOT/torch"

# XDG 兜底（其他库的通用缓存习惯）
export XDG_CACHE_HOME="$AUTODL_CACHE_ROOT"
