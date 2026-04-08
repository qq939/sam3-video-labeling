#!/usr/bin/env python3
"""
SAM3 Video Labeling Tool
基于 SAM3 的视频标注 Web UI
"""

import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import json
from transformers import pipeline
from PIL import Image

# ==========================================
# 全局参数配置 (Global Parameters)
# ==========================================
# 允许上传的视频扩展名
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'} # 使用位置: app.py L150
# 模型名称
SAM3_MODEL_NAME = "facebook/sam3-hiera-large" # 使用位置: app.py L35
# 提取视频帧的最大数量
MAX_FRAMES = 30 # 使用位置: app.py L75
# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 使用位置: app.py L36

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/sam3_uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/sam3_outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局模型实例
sam_generator = None

def init_sam3():
    """初始化 SAM3 模型，使用 Hugging Face transformers pipeline"""
    global sam_generator
    try:
        print(f"正在加载 SAM3 模型: {SAM3_MODEL_NAME} ...")
        # 使用 transformers pipeline 进行 mask-generation
        sam_generator = pipeline(
            "mask-generation", 
            model=SAM3_MODEL_NAME, 
            device=DEVICE
        )
        print("SAM3 模型加载成功 (Hugging Face Pipeline)")
        return True
    except Exception as e:
        print(f"SAM3 模型加载失败: {e}")
        # 备选方案：尝试加载 SAM2
        try:
            from transformers import pipeline as hf_pipeline
            sam_generator = hf_pipeline(
                "mask-generation", 
                model="facebook/sam2-hiera-large", 
                device=DEVICE
            )
            print("使用 SAM2 模型作为备选 (Hugging Face Pipeline)")
            return True
        except Exception as e2:
            print(f"SAM2 也加载失败: {e2}")
            return False

def extract_frames(video_path, max_frames=MAX_FRAMES):
    """从视频中提取帧"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # BGR to RGB for PIL/Transformers
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        frame_idx += 1
    
    cap.release()
    return frames

def generate_masks(image_np):
    """生成图像分割 masks"""
    if sam_generator is None:
        return []
    
    # 转换为 PIL Image 供 pipeline 使用
    image_pil = Image.fromarray(image_np)
    
    try:
        # pipeline 返回格式通常是 {"masks": [...], "scores": [...]} 或列表
        outputs = sam_generator(image_pil, points=None) # 自动模式
        
        # 处理不同版本的返回格式
        if isinstance(outputs, dict) and "masks" in outputs:
            masks_list = outputs["masks"]
        elif isinstance(outputs, list):
            masks_list = outputs
        else:
            masks_list = []
            
        return masks_list[:10]  # 限制返回数量
    except Exception as e:
        print(f"生成 mask 失败: {e}")
        return []

def process_video(video_path, prompt):
    """处理视频并生成标注"""
    frames_rgb = extract_frames(video_path)
    results = []
    
    for idx, frame_rgb in enumerate(frames_rgb):
        # 生成 masks
        masks_data = generate_masks(frame_rgb)
        
        # 保存标注后的图像 (转换回 BGR 用于 OpenCV 保存)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        result_frame = frame_bgr.copy()
        
        for i, m_data in enumerate(masks_data):
            # 获取 mask，可能是 numpy 数组或 PIL Image
            mask = m_data.get("mask") if isinstance(m_data, dict) else m_data
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            
            color = tuple(np.random.randint(0, 255, 3).tolist())
            mask_bool = mask.astype(bool)
            result_frame[mask_bool] = result_frame[mask_bool] * 0.7 + np.array(color) * 0.3
        
        output_filename = f'frame_{idx}.jpg'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, result_frame)
        results.append(output_filename)
        
    return results

@app.route('/')
def index():
    return "SAM3 Video Labeling Tool is Running"

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 异步处理或直接处理
    results = process_video(filepath, "")
    
    return jsonify({"message": "Success", "frames": results})

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    # 初始化模型
    init_sam3()
    # 启动服务
    app.run(host='0.0.0.0', port=8094)
