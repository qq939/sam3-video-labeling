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
import threading
import time
from transformers import AutoProcessor, AutoModelForMaskGeneration
from PIL import Image

# ==========================================
# 全局参数配置 (Global Parameters)
# ==========================================
# 允许上传的视频扩展名
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'} # 使用位置: app.py L150
# 模型名称
SAM3_MODEL_NAME = "facebook/sam3-hiera-large" # 使用位置: app.py L43
# 提取视频帧的最大数量
MAX_FRAMES = 60 # 使用位置: app.py L110
# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 使用位置: app.py L45
# 项目根目录下的存储路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 使用位置: app.py L35
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads') # 使用位置: app.py L36
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'results') # 使用位置: app.py L37

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局变量
sam_processor = None # 使用位置: app.py L42
sam_model = None # 使用位置: app.py L44
tasks = {} # 使用位置: app.py L165 (任务状态存储)

def init_sam3():
    """初始化 SAM3 模型与处理器"""
    global sam_processor, sam_model
    try:
        print(f"正在加载 SAM3 模型: {SAM3_MODEL_NAME} ...")
        sam_processor = AutoProcessor.from_pretrained(SAM3_MODEL_NAME)
        sam_model = AutoModelForMaskGeneration.from_pretrained(SAM3_MODEL_NAME).to(DEVICE)
        print("SAM3 模型加载成功")
        return True
    except Exception as e:
        print(f"SAM3 模型加载失败: {e}")
        return False

def extract_frames(video_path, max_frames=MAX_FRAMES):
    """从视频中提取帧并返回帧率和尺寸"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    interval = max(1, total_video_frames // max_frames)
    
    count = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        count += 1
    
    cap.release()
    return frames, fps, (width, height)

def generate_masks(image_np, prompt=None):
    """根据提示词生成 masks 并进行可视化处理"""
    if sam_model is None or sam_processor is None:
        return []
    
    image_pil = Image.fromarray(image_np)
    
    try:
        # 如果有提示词，使用文本提示
        if prompt and prompt.strip():
            inputs = sam_processor(text=prompt, images=image_pil, return_tensors="pt").to(DEVICE)
        else:
            # 否则进行自动分割
            inputs = sam_processor(images=image_pil, return_tensors="pt").to(DEVICE)
            
        with torch.no_grad():
            outputs = sam_model(**inputs)
            
        # 提取 mask
        if hasattr(outputs, 'pred_masks'):
            masks = outputs.pred_masks
        elif hasattr(outputs, 'masks'):
            masks = outputs.masks
        else:
            # 备选：尝试从字典中获取
            masks = outputs.get('pred_masks') or outputs.get('masks')

        if masks is None:
            return []

        # 处理维度 [batch, num_masks, H, W] -> [num_masks, H, W]
        if masks.ndim == 4:
            masks = masks[0]
            
        # 二值化并转为 numpy
        # 某些模型输出是 logits，需要 sigmoid
        masks_np = (torch.sigmoid(masks) > 0.0).cpu().numpy()
        
        # 过滤掉几乎为空的 mask
        valid_masks = []
        for m in masks_np:
            if m.sum() > 10: # 至少有 10 个像素
                valid_masks.append(m)
                
        return valid_masks[:5]
    except Exception as e:
        print(f"生成 mask 失败: {e}")
        return []

def combine_frames_to_video(frames_bgr, output_path, fps, size):
    """将帧合成视频，使用 H.264 编码以确保浏览器兼容性"""
    # 优先尝试使用 avc1 (H.264)，如果失败则回退到 mp4v
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        if not out.isOpened():
            raise Exception("avc1 编码器不可用")
    except:
        print("回退到 mp4v 编码器")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
    for frame in frames_bgr:
        out.write(frame)
    out.release()

def background_process_video(task_id, video_path, prompt):
    """后台处理视频任务"""
    try:
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['current_action'] = '正在读取视频帧...'
        
        frames_rgb, fps, size = extract_frames(video_path)
        tasks[task_id]['total_frames'] = len(frames_rgb)
        processed_frames_bgr = []
        
        for idx, frame_rgb in enumerate(frames_rgb):
            tasks[task_id]['current_action'] = f'正在处理第 {idx+1}/{len(frames_rgb)} 帧...'
            tasks[task_id]['processed_frames'] = idx
            tasks[task_id]['progress'] = (idx / len(frames_rgb)) * 80 # 前80%是处理帧
            tasks[task_id]['log'] = f'正在对帧 {idx} 进行 {prompt if prompt else "自动"} 分割...'
            
            masks = generate_masks(frame_rgb, prompt)
            
            # 绘制结果
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            for m in masks:
                # 获取随机颜色并确保是 list of ints
                color = [int(c) for c in np.random.randint(0, 255, 3)]
                # 确保 mask 是 2D (H, W)
                mask_bool = m.astype(bool)
                if mask_bool.ndim == 3: 
                    mask_bool = mask_bool[0]
                
                # 调整 mask 尺寸以匹配原图 (如果模型输出了缩小的 mask)
                if mask_bool.shape != (size[1], size[0]):
                    mask_bool = cv2.resize(mask_bool.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)

                # 应用半透明颜色叠加
                mask_overlay = frame_bgr.copy()
                mask_overlay[mask_bool] = color
                cv2.addWeighted(mask_overlay, 0.4, frame_bgr, 0.6, 0, frame_bgr)
            
            processed_frames_bgr.append(frame_bgr)

        tasks[task_id]['current_action'] = '正在合成视频文件...'
        tasks[task_id]['progress'] = 90
        
        output_filename = f'result_{task_id}.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        combine_frames_to_video(processed_frames_bgr, output_path, fps, size)
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['video_url'] = f'/outputs/{output_filename}'
        tasks[task_id]['current_action'] = '处理完成'
        
    except Exception as e:
        print(f"任务 {task_id} 出错: {e}")
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "未选择视频文件"}), 400
    
    file = request.files['video']
    prompt = request.form.get('prompt', '')
    
    if file.filename == '':
        return jsonify({"error": "文件名不能为空"}), 400
    
    task_id = str(uuid.uuid4())
    filename = secure_filename(f"{task_id}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 初始化任务状态
    tasks[task_id] = {
        'status': 'pending',
        'progress': 0,
        'processed_frames': 0,
        'total_frames': 0,
        'current_action': '排队中...',
        'log': '任务已创建'
    }
    
    # 启动后台线程
    thread = threading.Thread(target=background_process_video, args=(task_id, filepath, prompt))
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(tasks[task_id])

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    init_sam3()
    app.run(host='0.0.0.0', port=8094)
