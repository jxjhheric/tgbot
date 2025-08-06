FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y gcc libffi-dev libxml2-dev libxslt1-dev && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 拷贝代码
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 启动命令
CMD ["python", "tgbot.py"]
