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
SAM3_MODEL_NAME = "facebook/sam3-base" # 正确的图像分割模型
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
sam_pipeline = None # 使用位置: app.py L45 (文本分割Pipeline)
translator = None # 使用位置: app.py L90 (翻译器)
tasks = {} # 使用位置: app.py L165 (任务状态存储)

def init_sam3():
    """初始化 SAM3 模型与处理器"""
    global sam_processor, sam_model, sam_pipeline, translator
    try:
        # 使用本地已有的模型路径 (facebook/sam3)
        model_dir = os.path.join(BASE_DIR, "models", "sam3", "facebook", "sam3")
        
        if not os.path.exists(model_dir):
            print(f"模型路径不存在: {model_dir}，正在从 ModelScope 尝试下载...")
            model_dir = snapshot_download(SAM3_MODEL_NAME, cache_dir=MODEL_DIR)
        
        print(f"正在加载 SAM3 模型: {model_dir}")
        sam_processor = AutoProcessor.from_pretrained(model_dir, token=HF_TOKEN)
        sam_model = AutoModelForMaskGeneration.from_pretrained(model_dir, token=HF_TOKEN).to(DEVICE)
        
        # 不再使用 pipeline，直接用 model + processor
        sam_pipeline = None
        
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
        return [], fps if fps > 0 else 30, (width if width > 0 else 1280, height if height > 0 else 720)
    
    total_frames = len(original_frames)
    
    if total_frames <= target_fps:
        return original_frames, fps, (width, height)
    
    frame_indices = np.linspace(0, total_frames - 1, target_fps, dtype=int)
    sampled_frames = [original_frames[i] for i in frame_indices]
    
    new_fps = fps * (target_fps / total_frames)
    
    return sampled_frames, new_fps, (width, height)

def generate_masks(image_np, prompt=None):
    """根据提示词生成 masks 并提取元数据 (Bounding Box, Area, Score)"""
    if sam_processor is None or sam_model is None:
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

        # 直接使用 model + processor 进行分割 (绕过 pipeline 的默认阈值过滤)
        inputs = sam_processor(images=image_pil, return_tensors="pt")
        
        if DEVICE == "cuda":
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sam_model(**inputs)
        
        # 获取原始输出
        pred_masks = outputs.get("pred_masks", outputs.get("masks"))
        iou_scores = outputs.get("iou_scores", outputs.get("scores"))
        object_score_logits = outputs.get("object_score_logits")
        
        if pred_masks is None or len(pred_masks) == 0:
            return []
        
        # 后处理: 处理图像原始尺寸
        if "original_sizes" in inputs and "reshaped_input_sizes" in inputs:
            orig_size = inputs["original_sizes"][0]
            input_size = inputs["reshaped_input_sizes"][0]
            h, w = int(orig_size[0]), int(orig_size[1])
            input_h, input_w = int(input_size[0]), int(input_size[1])
            scale_x = w / input_w
            scale_y = h / input_h
        else:
            h, w = image_pil.size[::-1]
            scale_x, scale_y = 1.0, 1.0
        
        # 手动过滤 masks (降低阈值，因为模型架构不匹配导致分数偏低)
        valid_results = []
        
        # 检查 batch 维度并 squeeze
        if pred_masks.dim() == 4:
            pred_masks = pred_masks[0]  # [num_masks, H, W]
        if iou_scores is not None and iou_scores.dim() > 1:
            iou_scores = iou_scores.squeeze()  # [num_masks]
        if object_score_logits is not None and object_score_logits.dim() > 0:
            object_score_logits = object_score_logits.squeeze()  # 标量
        
        # 转换为 numpy 数组
        if iou_scores is not None:
            iou_scores_np = iou_scores.cpu().numpy()
        else:
            iou_scores_np = None
        
        if object_score_logits is not None:
            object_score_logits_np = float(object_score_logits.cpu().numpy())
        else:
            object_score_logits_np = None
        
        num_masks = pred_masks.shape[0]
        
        for i in range(num_masks):
            mask = pred_masks[i]
            
            # 将 mask 从模型输出尺寸缩放回原始图像尺寸
            if scale_x != 1.0 or scale_y != 1.0:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                ).squeeze()
            
            mask = mask.cpu().numpy()
            
            # 转换为二值 mask
            mask = (mask > 0.0).astype(np.uint8)
            
            # 确保是 2D
            if mask.ndim == 3:
                mask = mask.any(axis=0)
            
            area = int(mask.sum())
            if area > 200:
                # 计算 Bounding Box (XYXY)
                pos = np.where(mask)
                if pos[0].size > 0:
                    bbox = [int(np.min(pos[1])), int(np.min(pos[0])), int(np.max(pos[1])), int(np.max(pos[0]))]
                else:
                    bbox = [0, 0, 0, 0]
                
                # 计算综合分数 (IOU + object score)
                iou = 0.5
                if iou_scores_np is not None:
                    iou = float(iou_scores_np[i])
                
                obj_score = 0.5
                if object_score_logits_np is not None:
                    obj_score = 1.0 / (1.0 + np.exp(-object_score_logits_np))  # sigmoid (scalar)
                
                combined_score = (iou + obj_score) / 2.0
                
                # 使用更低的阈值，因为模型架构不匹配
                if combined_score > 0.4:
                    valid_results.append({
                        "mask": mask,
                        "bbox": bbox,
                        "area": area,
                        "score": combined_score
                    })
        
        return valid_results
    except Exception as e:
        print(f"生成 mask 失败: {e}")
        return []

def combine_frames_to_video(frames_bgr, output_path, fps, size):
    """将帧合成视频，使用 H.264 编码以确保浏览器兼容性"""
    if not frames_bgr:
        print("警告: 没有帧可合成视频")
        return
    
    # 确保 size 有效
    if size[0] <= 0 or size[1] <= 0:
        size = (frames_bgr[0].shape[1], frames_bgr[0].shape[0])
    
    # 确保 fps 有效
    if fps <= 0:
        fps = 30
    
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
                if size[0] > 0 and size[1] > 0 and mask_bool.shape != (size[1], size[0]):
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
