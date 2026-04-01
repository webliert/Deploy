"""
推理引擎基类
定义统一的推理接口，所有具体引擎需继承此类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class InferenceEngineBase(ABC):
    """推理引擎抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化推理引擎
        
        Args:
            config: 配置字典，应包含:
                - model_path: 模型文件路径
                - input_size: 输入张量大小
                - output_size: 输出张量大小
                - dtype: 数据类型，默认 'float32'
        """
        self.config = config
        self.model_path = config.get('model_path', '')
        self.input_size = config.get('input_size', None)
        self.output_size = config.get('output_size', None)
        self.dtype = config.get('dtype', 'float32')
        self.is_loaded = False
        self._session: Optional[Any] = None
    
    @abstractmethod
    def load(self):
        """
        加载模型到内存
        子类必须实现此方法
        
        Raises:
            FileNotFoundError: 模型文件不存在
            RuntimeError: 模型加载失败
        """
        raise NotImplementedError
    
    @abstractmethod
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行单次推理
        
        Args:
            input_data: 输入数据，shape 应为 (1, input_size) 或 (batch_size, input_size)
            
        Returns:
            输出数据，shape 为 (batch_size, output_size)
            
        Raises:
            RuntimeError: 模型未加载或推理失败
        """
        raise NotImplementedError
    
    @abstractmethod
    def unload(self):
        """
        释放模型资源
        子类必须实现此方法
        """
        raise NotImplementedError
    
    def __enter__(self):
        """支持上下文管理器"""
        if not self.is_loaded:
            self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时释放资源"""
        self.unload()
        return False
    
    def __del__(self):
        """析构时释放资源"""
        if self.is_loaded:
            self.unload()