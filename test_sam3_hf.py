import sys
import torch
import unittest
from unittest.mock import MagicMock
import signal
import numpy as np

# 增加超时机制
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("测试执行超时")

class TestSAM3HF(unittest.TestCase):
    def test_sam3_pipeline_import(self):
        """测试 transformers 是否支持 SAM3 相关组件"""
        try:
            from transformers import Sam3Processor, Sam3Model, Sam3TrackerProcessor, Sam3TrackerModel
            print("成功从 transformers 导入 SAM3 组件")
        except ImportError as e:
            self.fail(f"无法从 transformers 导入 SAM3 组件: {e}")

    def test_app_generate_masks_mock(self):
        """测试 app.py 中的 generate_masks 函数"""
        from app import generate_masks
        import app
        
        # Mock sam_model 和 sam_processor
        app.sam_model = MagicMock()
        app.sam_processor = MagicMock()
        
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        # 模拟模型输出
        mock_output = MagicMock()
        mock_output.pred_masks = torch.from_numpy(np.zeros((1, 1, 100, 100))).float()
        mock_output.get.return_value = None
        app.sam_model.return_value = mock_output
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        masks = generate_masks(test_image)
        
        # 由于我们设置了 area > 200，全零的 mask 应该被过滤掉
        self.assertEqual(len(masks), 0)
        print("app.generate_masks Mock 测试通过 (空面积过滤)")


if __name__ == "__main__":
    unittest.main()
