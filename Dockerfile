# 使用官方 Python 运行时作为父镜像
FROM python:3.10-slim

# 在容器中设置工作目录
WORKDIR /app

# 将依赖文件复制到容器中
COPY ./requirements.txt /app/

# 安装 requirements.txt 中指定的任何所需包
RUN pip install --no-cache-dir -r requirements.txt

# 将应用程序的其余代码复制到容器中
COPY ./src /app/src
COPY ./key.json /app/key.json
COPY ./db.db /app/db.db

# 让容器的 8000 端口可供外部访问
EXPOSE 8000

# 定义环境变量
ENV PYTHONPATH=/app

# 容器启动时运行 uvicorn
# --reload 标志用于在代码更改时自动重载服务
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
