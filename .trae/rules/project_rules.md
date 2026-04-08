# Project Rules

## 项目名称
sam3-video-labeling

## 技术栈
- Python 3.13
- Flask Web UI
- SAM3 (Segment Anything Model 3) 图像分割
- Hugging Face Transformers
- OpenCV 视频处理

## 核心文件
- app.py: 主应用文件，包含 SAM3 模型加载和 mask 生成逻辑

## 端口配置
- Flask 服务端口: 8095

## 模型配置
- 模型路径: models/sam3/facebook/sam3
- 目标帧率: 16 fps
- 默认阈值: 0.4 (降低阈值以适应架构不匹配)