#!/usr/bin/env bash
# 一键关停所有 O2-Searcher 服务
set -uo pipefail

echo "停止所有 O2-Searcher 服务..."

pkill -f meilisearch               && echo "  ✓ Meilisearch"
pkill -f 'python web_search.py'    && echo "  ✓ web_search"
pkill -f 'run_openended.py'        && echo "  ✓ run_openended"
pkill -f 'wiki_server.py'          && echo "  ✓ wiki_server"
pkill -f 'run_closedended.py'      && echo "  ✓ run_closedended"
pkill -f 'o2searcher.rewards.metrics.server' && echo "  ✓ metrics server"

# 保险起见
sleep 1
pkill -9 -f meilisearch 2>/dev/null
pkill -9 -f 'o2searcher' 2>/dev/null

echo "完成"
