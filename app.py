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
from translate import Translator
from dotenv import load_dotenv
from modelscope import snapshot_download

# 加载 .env 文件中的环境变量
load_dotenv()

# ==========================================
# 全局参数配置 (Global Parameters)
# ==========================================
# 允许上传的视频扩展名
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'} # 使用位置: app.py L150
# 模型名称
SAM3_MODEL_NAME = "facebook/sam3" # 使用位置: app.py L43
# Hugging Face Token (用于访问受限模型)
# 请在项目根目录创建 .env 文件，并添加 HF_TOKEN=your_token
HF_TOKEN = os.getenv("HF_TOKEN") # 使用位置: app.py L51, L52
# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 使用位置: app.py L54
# 项目根目录下的存储路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 使用位置: app.py L35
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads') # 使用位置: app.py L36
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'results') # 使用位置: app.py L37
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'sam3') # 使用位置: app.py L43

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 全局变量
sam_processor = None # 使用位置: app.py L42
sam_model = None # 使用位置: app.py L44
translator = None # 使用位置: app.py L90 (翻译器)
tasks = {} # 使用位置: app.py L165 (任务状态存储)

def init_sam3():
    """初始化 SAM3 模型与处理器"""
    global sam_processor, sam_model, translator
    try:
        print(f"正在从 ModelScope 加载 SAM3 模型: {SAM3_MODEL_NAME} ...")
        # 从 ModelScope 下载模型 (国内直连)
        model_dir = snapshot_download(SAM3_MODEL_NAME, cache_dir=MODEL_DIR)
        # 加载本地模型
        sam_processor = AutoProcessor.from_pretrained(model_dir, token=HF_TOKEN)
        sam_model = AutoModelForMaskGeneration.from_pretrained(model_dir, token=HF_TOKEN).to(DEVICE)
        # 初始化翻译器 (中转英)
        translator = Translator(from_lang="zh", to_lang="en")
        print("SAM3 模型及翻译器加载成功")
        return True
    except Exception as e:
        print(f"SAM3 模型加载失败: {e}")
        return False

def extract_frames(video_path, target_fps=16):
    """从视频中提取帧并压缩到目标帧率
    
    Args:
        video_path: 视频文件路径
        target_fps: 目标帧率，默认为16
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0:
        fps = 30
    
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frames.append(rgb_frame)
    
    cap.release()
    
    if not original_frames:
        return [], fps, (width, height)
    
    total_frames = len(original_frames)
    
    if total_frames <= target_fps:
        return original_frames, fps, (width, height)
    
    frame_indices = np.linspace(0, total_frames - 1, target_fps, dtype=int)
    sampled_frames = [original_frames[i] for i in frame_indices]
    
    new_fps = fps * (target_fps / total_frames)
    
    return sampled_frames, new_fps, (width, height)

def generate_masks(image_np, prompt=None):
    """根据提示词生成 masks 并提取元数据 (Bounding Box, Area, Score)"""
    if sam_model is None or sam_processor is None:
        return []
    
    image_pil = Image.fromarray(image_np)
    
    try:
        # 如果有中文，翻译成英文 (SAM3 只支持英文文本提示)
        if prompt and any('\u4e00' <= char <= '\u9fff' for char in prompt):
            try:
                prompt = translator.translate(prompt)
                print(f"提示词已翻译为: {prompt}")
            except Exception as e:
                print(f"翻译失败: {e}")

        # 文本提示使用 text 参数 (SAM3 processor 的正确参数名)
        if prompt and prompt.strip():
            inputs = sam_processor(images=image_pil, text=prompt, return_tensors="pt").to(DEVICE)
        else:
            # 自动模式参数修正：需要提供 points_per_side 来生成点网格
            inputs = sam_processor(
                images=image_pil, 
                return_tensors="pt",
                points_per_side=32, # 自动全图分割的点密度
            ).to(DEVICE)
            
        with torch.no_grad():
            outputs = sam_model(**inputs)
            
        # 使用官方后处理函数，更准确且能处理尺寸缩放
        try:
            results_list = sam_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )
            # 获取第一个 batch 的结果
            res = results_list[0]
            masks_np = res["masks"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            bboxes = res["boxes"].cpu().numpy()
        except Exception as pe:
            print(f"官方后处理失败，使用备选手动处理: {pe}")
            # 备选手动处理逻辑 (包含之前的 0.5 阈值修正)
            masks = outputs.get('pred_masks') or outputs.get('masks')
            if masks is None: return []
            if masks.ndim == 4: masks = masks[0]
            masks_np = (torch.sigmoid(masks) > 0.5).cpu().numpy()
            scores = outputs.get('iou_scores') or outputs.get('scores')
            if scores is not None and scores.ndim > 1: scores = scores[0].cpu().numpy()
            bboxes = None

        valid_results = []
        for i, m in enumerate(masks_np):
            # 确保 mask 是 2D (H, W)
            if m.ndim == 3:
                m = m.any(axis=0)
                
            area = int(m.sum())
            if area > 200: # 过滤噪点
                # 计算 Bounding Box (XYXY)
                if bboxes is not None:
                    bbox = [int(x) for x in bboxes[i]]
                else:
                    pos = np.where(m)
                    if pos[0].size > 0:
                        bbox = [int(np.min(pos[1])), int(np.min(pos[0])), int(np.max(pos[1])), int(np.max(pos[0]))]
                    else:
                        bbox = [0, 0, 0, 0]
                
                valid_results.append({
                    "mask": m,
                    "bbox": bbox,
                    "area": area,
                    "score": float(scores[i]) if scores is not None else 1.0
                })
                
        return valid_results[:5]
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
    """后台处理视频任务，并将结果和元数据保存在任务专用文件夹下"""
    try:
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['current_action'] = '正在读取视频帧...'
        
        # 建立任务文件夹
        video_filename = os.path.basename(video_path)
        task_dir_name = f"{os.path.splitext(video_filename)[0]}_{task_id}"
        task_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_dir_name)
        os.makedirs(task_dir, exist_ok=True)
        
        frames_rgb, fps, size = extract_frames(video_path)
        tasks[task_id]['total_frames'] = len(frames_rgb)
        processed_frames_bgr = []
        all_segmentation_data = [] # 存储每帧的分割元数据
        
        for idx, frame_rgb in enumerate(frames_rgb):
            tasks[task_id]['current_action'] = f'正在处理第 {idx+1}/{len(frames_rgb)} 帧...'
            tasks[task_id]['processed_frames'] = idx
            tasks[task_id]['progress'] = (idx / len(frames_rgb)) * 80
            tasks[task_id]['log'] = f'正在对帧 {idx} 进行 {prompt if prompt else "自动"} 分割...'
            
            mask_results = generate_masks(frame_rgb, prompt)
            frame_data = {"frame_index": idx, "masks": []}
            
            # 绘制结果并收集元数据
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            for m_res in mask_results:
                mask_bool = m_res["mask"]
                bbox = m_res["bbox"]
                area = m_res["area"]
                score = m_res["score"]
                
                # 收集元数据
                frame_data["masks"].append({
                    "bbox": bbox,
                    "area": area,
                    "score": score
                })
                
                # 获取随机颜色并确保是 list of ints
                color = [int(c) for c in np.random.randint(0, 255, 3)]
                
                # 调整 mask 尺寸以匹配原图 (如果模型输出了缩小的 mask)
                if mask_bool.shape != (size[1], size[0]):
                    mask_bool = cv2.resize(mask_bool.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)

                # 应用半透明颜色叠加
                mask_overlay = frame_bgr.copy()
                mask_overlay[mask_bool] = color
                cv2.addWeighted(mask_overlay, 0.4, frame_bgr, 0.6, 0, frame_bgr)
            
            processed_frames_bgr.append(frame_bgr)
            all_segmentation_data.append(frame_data)

        tasks[task_id]['current_action'] = '正在合成视频及保存元数据...'
        tasks[task_id]['progress'] = 90
        
        # 1. 保存标注后的视频
        output_video_filename = f'labeled_{task_id}.mp4'
        output_video_path = os.path.join(task_dir, output_video_filename)
        combine_frames_to_video(processed_frames_bgr, output_video_path, fps, size)
        
        # 2. 保存分割数据 (JSON)
        data_filename = 'segmentation_data.json'
        data_path = os.path.join(task_dir, data_filename)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                "task_id": task_id,
                "prompt": prompt,
                "frames": all_segmentation_data
            }, f, indent=4, ensure_ascii=False)
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        # 返回相对路径供前端预览 (Flask 静态路由)
        tasks[task_id]['video_url'] = f'/outputs/{task_dir_name}/{output_video_filename}'
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

@app.route('/outputs/<path:path>')
def serve_output(path):
    return send_from_directory(app.config['OUTPUT_FOLDER'], path)

if __name__ == '__main__':
    init_sam3()
    app.run(host='0.0.0.0', port=8095)
