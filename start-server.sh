#!/bin/bash
echo "=========================================="
echo " Xiaohao Cai 论文精读可视化系统"
echo "=========================================="
echo ""
echo "正在启动本地服务器..."
echo "访问地址:"
echo "  - 精读Dashboard: http://localhost:8080/docs/"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

cd "$(dirname "$0")"
python3 -m http.server 8080
