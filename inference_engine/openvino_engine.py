"""
OpenVINO 推理引擎实现
"""

import os
import numpy as np
from typing import Dict, Any, Optional

from inference_engine.base_engine import InferenceEngineBase


class OpenVINOEngine(InferenceEngineBase):
    """基于 OpenVINO 的推理引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 OpenVINO 引擎
        
        Args:
            config: 配置字典，额外支持:
                - device: 推理设备，默认 'CPU'
                - cache_dir: 模型缓存目录
        """
        super().__init__(config)
        self.device = config.get('device', 'CPU')
        self.cache_dir = config.get('cache_dir', None)
        
        self._core: Optional[Any] = None
        self._compiled_model: Optional[Any] = None
        self._infer_request: Optional[Any] = None
    
    def load(self):
        """加载 OpenVINO 模型"""
        import openvino as ov
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"OpenVINO model not found: {self.model_path}")
        
        try:
            # 创建 OpenVINO Core
            self._core = ov.Core()
            
            # 设置缓存
            if self.cache_dir:
                self._core.set_property({'CACHE_DIR': self.cache_dir})
            
            # 读取模型
            ov_model = self._core.read_model(self.model_path)
            
            # 处理动态形状
            input_layer = ov_model.input(0)
            input_partial_shape = input_layer.partial_shape
            
            if input_partial_shape.is_dynamic and self.input_size is not None:
                static_shape = [1, self.input_size]
                ov_model.reshape({input_layer: static_shape})
                print(f"[OpenVINOEngine] Reshaped dynamic input to static: {static_shape}")
            
            # 编译模型
            compile_config = {}
            if self.cache_dir:
                compile_config['CACHE_DIR'] = self.cache_dir
            
            self._compiled_model = self._core.compile_model(ov_model, self.device, compile_config)
            
            # 创建推理请求
            self._infer_request = self._compiled_model.create_infer_request()
            
            # 打印模型信息
            compiled_input = self._compiled_model.input(0)
            compiled_output = self._compiled_model.output(0)
            print(f"[OpenVINOEngine] Model loaded: {self.model_path}")
            print(f"[OpenVINOEngine] Input shape: {compiled_input.shape}, Output shape: {compiled_output.shape}")
            
            self.is_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVINO model: {e}")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行 OpenVINO 推理
        
        Args:
            input_data: 输入数据，shape (batch_size, input_size)
            
        Returns:
            输出数据，shape (batch_size, output_size)
        """
        if self._infer_request is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            input_data = input_data.astype(np.float32)
            
            # 设置输入
            input_tensor = ov.Tensor(array=input_data, shared_memory=True)
            self._infer_request.set_input_tensor(input_tensor)
            
            # 执行推理
            self._infer_request.infer()
            
            # 获取输出
            output_tensor = self._infer_request.get_output_tensor()
            return output_tensor.data.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"OpenVINO inference failed: {e}")
    
    def unload(self):
        """释放 OpenVINO 资源"""
        if self._infer_request is not None:
            del self._infer_request
            self._infer_request = None
        if self._compiled_model is not None:
            del self._compiled_model
            self._compiled_model = None
        if self._core is not None:
            del self._core
            self._core = None
        self.is_loaded = False
        print("[OpenVINOEngine] Model unloaded")


# 延迟导入 openvino，避免模块未安装时的导入错误
def _import_ov():
    import openvino as ov
    return ov

# 在模块级别导入，使 infer 方法中的 ov.Tensor 可用
try:
    ov = _import_ov()
except ImportError:
    ov = None