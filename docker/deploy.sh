#!/bin/bash
# deploy.sh

# 构建镜像
docker build -t xcai:latest .

# 运行容器
docker run -d -p 8888:8888 -v $(pwd)/data:/app/data xcai:latest

# 或者使用docker-compose
docker-compose up -d
