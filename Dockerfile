FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制应用文件
COPY . .

# 安装 Python 依赖
RUN pip install -r requirements.txt

# 暴露端口
EXPOSE 8094

# 启动应用
CMD ["python", "app.py"]