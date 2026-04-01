"""
PyTorch 推理引擎实现
"""

import os
import numpy as np
from typing import Dict, Any, Optional

from inference_engine.base_engine import InferenceEngineBase


class PyTorchEngine(InferenceEngineBase):
    """基于 PyTorch 的推理引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 PyTorch 引擎
        
        Args:
            config: 配置字典，额外支持:
                - device: 推理设备，默认 'cpu' (支持 'cpu', 'cuda')
                - jit_trace: 是否使用 JIT trace 优化
        """
        super().__init__(config)
        self.device = config.get('device', 'cpu').lower()
        self.jit_trace = config.get('jit_trace', False)
        
        self._model: Optional[Any] = None
        self._torch_device: Optional[Any] = None
    
    def load(self):
        """加载 PyTorch 模型"""
        import torch
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PyTorch model not found: {self.model_path}")
        
        try:
            # 设置设备
            if self.device == 'cuda' and not torch.cuda.is_available():
                print("[PyTorchEngine] CUDA not available, falling back to CPU")
                self.device = 'cpu'
            
            self._torch_device = torch.device(self.device)
            
            # 加载模型
            model_dict = torch.load(self.model_path, map_location=self._torch_device, weights_only=False)
            
            # 支持不同的保存格式
            if isinstance(model_dict, dict):
                if 'model' in model_dict:
                    self._model = model_dict['model']
                elif 'state_dict' in model_dict:
                    raise ValueError("State dict only loading not supported. "
                                   "Please save the full model using torch.save(model, path)")
                else:
                    self._model = model_dict
            else:
                self._model = model_dict
            
            # 如果是 JIT traced 模型
            if isinstance(self._model, torch.jit.ScriptModule):
                self.jit_trace = True
            
            # 设置为评估模式
            if hasattr(self._model, 'eval'):
                self._model.eval()
            
            # 移动到目标设备
            if hasattr(self._model, 'to'):
                self._model = self._model.to(self._torch_device)
            
            self.is_loaded = True
            print(f"[PyTorchEngine] Model loaded: {self.model_path} on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行 PyTorch 推理
        
        Args:
            input_data: 输入数据，shape (batch_size, input_size)
            
        Returns:
            输出数据，shape (batch_size, output_size)
        """
        import torch
        
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # 转换为 tensor
            input_tensor = torch.from_numpy(input_data).float().to(self._torch_device)
            
            # 推理
            with torch.no_grad():
                if self.jit_trace:
                    output_tensor = self._model.forward(input_tensor)
                else:
                    output_tensor = self._model(input_tensor)
            
            # 转换回 numpy
            if isinstance(output_tensor, (list, tuple)):
                output_tensor = output_tensor[0]
            
            if hasattr(output_tensor, 'cpu'):
                output_np = output_tensor.cpu().numpy()
            else:
                output_np = np.array(output_tensor)
            
            return output_np.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"PyTorch inference failed: {e}")
    
    def unload(self):
        """释放 PyTorch 资源"""
        import torch
        
        if self._model is not None:
            del self._model
            self._model = None
        self._torch_device = None
        self.is_loaded = False
        
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[PyTorchEngine] Model unloaded")