# 数字孪生港口协同优化 — Docker 运行环境
# 基于 Python 3.12 slim，安装所有依赖后自动运行验证

FROM python:3.12-slim

LABEL maintainer="author@example.com"
LABEL description="Digital Twin Port Collaborative Optimization - Reproducible Package"

WORKDIR /replication

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY 02_code/ 02_code/
COPY 03_results/ 03_results/
COPY output/ output/
COPY 05_appendix/ 05_appendix/
COPY 01_data/codebook/ 01_data/codebook/
COPY experiments/chapter4/results/ experiments/chapter4/results/
COPY Data_Availability_Statement.md .
COPY LICENSE .
COPY README.md .
COPY verify_replication.py .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 入口：运行验证脚本
CMD ["python", "verify_replication.py", "--quick"]
