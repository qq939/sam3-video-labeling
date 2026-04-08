import unittest
import json
import time
from app import app, tasks
import io

class TestVideoFlow(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        """测试主页加载"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SAM3', response.data)

    def test_upload_and_status_flow(self):
        """测试上传视频并获取状态的流程"""
        # 模拟上传一个空视频文件（由于后端 extract_frames 会报错，我们需要 mock 掉处理逻辑或提供真实视频）
        # 这里我们主要测试 Flask 路由逻辑
        data = {
            'video': (io.BytesIO(b"fake video content"), 'test.mp4'),
            'prompt': 'test prompt'
        }
        
        response = self.app.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        
        res_data = json.loads(response.data)
        self.assertIn('task_id', res_data)
        task_id = res_data['task_id']
        
        # 检查状态接口
        response = self.app.get(f'/status/{task_id}')
        self.assertEqual(response.status_code, 200)
        status_data = json.loads(response.data)
        self.assertIn('status', status_data)
        self.assertIn(status_data['status'], ['pending', 'processing', 'completed', 'failed'])

    def test_invalid_status_id(self):
        """测试无效的任务 ID"""
        response = self.app.get('/status/non-existent-id')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()
