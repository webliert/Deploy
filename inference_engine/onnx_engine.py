"""
ONNX Runtime 推理引擎实现
"""

import os
import numpy as np
from typing import Dict, Any, Optional

from inference_engine.base_engine import InferenceEngineBase


class ONNXEngine(InferenceEngineBase):
    """基于 ONNX Runtime 的推理引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 ONNX 引擎
        
        Args:
            config: 配置字典，额外支持:
                - intra_op_num_threads: 算子内部并行线程数
                - inter_op_num_threads: 算子间并行线程数
                - enable_mem_pattern: 启用内存模式
                - enable_mem_reuse: 启用内存重用
                - graph_optimization_level: 图优化级别
        """
        super().__init__(config)
        self.intra_op_num_threads = config.get('intra_op_num_threads', 1)
        self.inter_op_num_threads = config.get('inter_op_num_threads', 1)
        self.enable_mem_pattern = config.get('enable_mem_pattern', False)
        self.enable_mem_reuse = config.get('enable_mem_reuse', True)
        self.graph_optimization_level = config.get('graph_optimization_level', 'ORT_ENABLE_ALL')
        
        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None
    
    def load(self):
        """加载 ONNX 模型"""
        import onnxruntime as ort
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        try:
            # 配置 SessionOptions
            options = ort.SessionOptions()
            
            # 设置图优化级别
            opt_level_map = {
                'ORT_ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                'ORT_ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                'ORT_DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            }
            options.graph_optimization_level = opt_level_map.get(
                self.graph_optimization_level, 
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            
            # 设置线程数
            options.intra_op_num_threads = self.intra_op_num_threads
            options.inter_op_num_threads = self.inter_op_num_threads
            
            # 内存优化配置
            options.enable_mem_pattern = self.enable_mem_pattern
            options.enable_mem_reuse = self.enable_mem_reuse
            
            # 创建推理会话
            self._session = ort.InferenceSession(
                self.model_path, 
                options, 
                providers=['CPUExecutionProvider']
            )
            
            # 获取输入名称
            self._input_name = self._session.get_inputs()[0].name
            
            # 验证输入输出大小
            input_shape = self._session.get_inputs()[0].shape
            output_shape = self._session.get_outputs()[0].shape
            
            if self.input_size is not None:
                expected_size = input_shape[-1] if len(input_shape) > 1 else input_shape[0]
                if expected_size != self.input_size:
                    print(f"[ONNXEngine] Warning: configured input_size {self.input_size} "
                          f"but model expects {expected_size}")
            
            if self.output_size is not None:
                expected_size = output_shape[-1] if len(output_shape) > 1 else output_shape[0]
                if expected_size != self.output_size:
                    print(f"[ONNXEngine] Warning: configured output_size {self.output_size} "
                          f"but model outputs {expected_size}")
            
            self.is_loaded = True
            print(f"[ONNXEngine] Model loaded: {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行 ONNX 推理
        
        Args:
            input_data: 输入数据，shape (batch_size, input_size)
            
        Returns:
            输出数据，shape (batch_size, output_size)
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            input_data = input_data.astype(np.float32)
            outputs = self._session.run(None, {self._input_name: input_data})
            return outputs[0].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}")
    
    def unload(self):
        """释放 ONNX 资源"""
        if self._session is not None:
            del self._session
            self._session = None
            self._input_name = None
            self.is_loaded = False
            print("[ONNXEngine] Model unloaded")