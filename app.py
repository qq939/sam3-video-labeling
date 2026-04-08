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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/sam3_uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/sam3_outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局变量
sam_model = None

def init_sam3():
    """初始化 SAM3 模型"""
    global sam_model
    try:
        # 尝试导入 SAM3
        from sam3 import SamAutomaticMaskGenerator
        sam_model = SamAutomaticMaskGenerator(
            model=torch.hub.load('facebookresearch/sam3', 'vit_h')
        )
        print("SAM3 模型加载成功")
        return True
    except Exception as e:
        print(f"SAM3 模型加载失败: {e}")
        # 使用 SAM2 作为备选
        try:
            from sam2 import SAM2AutomaticMaskGenerator
            sam_model = SAM2AutomaticMaskGenerator.from_pretrained(
                "facebook/sam2-hiera-large",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("使用 SAM2 模型作为备选")
            return True
        except Exception as e2:
            print(f"SAM2 也加载失败: {e2}")
            return False

def extract_frames(video_path, max_frames=30):
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
            frames.append(frame)
        frame_idx += 1
    
    cap.release()
    return frames

def generate_masks(image):
    """生成图像分割 masks"""
    if sam_model is None:
        return []
    
    with torch.no_grad():
        masks = sam_model.generate(image)
    
    return masks[:10]  # 限制返回数量

def process_video(video_path, prompt):
    """处理视频并生成标注"""
    frames = extract_frames(video_path)
    results = []
    
    for idx, frame in enumerate(frames):
        # 生成 masks
        masks = generate_masks(frame)
        
        # 保存标注后的图像
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'frame_{idx}.jpg')
        
        # 绘制 masks
        result_frame = frame.copy()
        for i, mask in enumerate(masks):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            mask_bool = mask['segmentation']
            result_frame[mask_bool] = result_frame[mask_bool] * 0.7 + np.array(color) * 0.3
        
        cv2.imwrite(output_path, result_frame)
        
        results.append({
            'frame_idx': idx,
            'mask_count': len(masks),
            'image_url': f'/outputs/frame_{idx}.jpg'
        })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': sam_model is not None,
        'model_name': 'SAM3' if sam_model else 'Not loaded'
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    video = request.files['video']
    prompt = request.form.get('prompt', '')
    
    if video.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # 保存视频
    task_id = str(uuid.uuid4())
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{task_id}.mp4')
    video.save(video_path)
    
    # 处理视频
    try:
        results = process_video(video_path, prompt)
        return jsonify({
            'success': True,
            'task_id': task_id,
            'frames': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    # 尝试初始化模型
    init_sam3()
    
    # 启动 Flask
    app.run(host='0.0.0.0', port=8094, debug=True)