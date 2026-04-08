# SAM3 Video Labeling Tool

基于 Meta Segment Anything Model 3 (SAM3) 的视频标注工具。

## 功能

- 📹 上传视频文件
- ✏️ 输入提示词描述
- 🎯 自动生成视频帧标注
- 💾 导出标注结果

## 环境要求

- Python 3.10+
- CUDA (推荐)
- 至少 8GB GPU 显存

## 安装

```bash
pip install -r requirements.txt
```

## 模型下载

SAM3 模型会自动下载，或手动下载：
- https://github.com/facebookresearch/sam3#checkpoints

## 运行

```bash
python app.py
```

访问 http://localhost:8094

## Docker 部署

```bash
docker build -t sam3-video-labeling .
docker run -d -p 8094:8094 --gpus all sam3-video-labeling
```

## 技术栈

- Flask - Web 框架
- SAM3 - 图像分割模型
- OpenCV - 视频处理
- PyTorch - 深度学习