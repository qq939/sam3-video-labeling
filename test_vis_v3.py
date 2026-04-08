import unittest
import numpy as np
import cv2
import os
import torch

class TestSAM3Visualization(unittest.TestCase):
    def test_mask_overlay_logic(self):
        """测试 mask 叠加逻辑是否产生可见变化"""
        # 创建一个纯黑图像
        size = (640, 480)
        frame_bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # 创建一个简单的矩形 mask
        mask_bool = np.zeros((size[1], size[0]), dtype=bool)
        mask_bool[100:200, 100:200] = True
        
        # 验证面积是否超过新阈值 200
        self.assertGreater(mask_bool.sum(), 200, "测试 Mask 面积应超过 200 像素")
        
        # 模拟 app.py 中的叠加逻辑
        color = [0, 255, 0] # 绿色
        mask_overlay = frame_bgr.copy()
        mask_overlay[mask_bool] = color
        cv2.addWeighted(mask_overlay, 0.4, frame_bgr, 0.6, 0, frame_bgr)
        
        # 检查 mask 区域是否有颜色变化
        roi = frame_bgr[150, 150]
        self.assertTrue(np.any(roi > 0), "Mask 叠加区域不应为纯黑")
        self.assertEqual(roi[1], int(255 * 0.4), "绿色通道值应符合权重计算")
        print("Mask 叠加可视化逻辑验证通过")

    def test_video_writer_codec(self):
        """测试视频编码器是否能成功创建文件"""
        output_path = "test_codec.mp4"
        fps = 24
        size = (640, 480)
        
        # 尝试使用 avc1
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        if not out.isOpened():
            print("avc1 失败，尝试 mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        self.assertTrue(out.isOpened(), "无法打开任何视频编码器")
        
        # 写入一帧
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        out.write(frame)
        out.release()
        
        self.assertTrue(os.path.exists(output_path), "视频文件未生成")
        os.remove(output_path)
        print("视频编码器及文件生成验证通过")

if __name__ == '__main__':
    unittest.main()
