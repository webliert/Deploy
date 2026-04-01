"""
推理引擎工厂
根据配置创建对应的推理引擎实例
"""

from enum import Enum
from typing import Dict, Any

from inference_engine.base_engine import InferenceEngineBase


class EngineType(Enum):
    """推理引擎类型枚举"""
    ONNX = "onnx"
    OPENVINO = "openvino"
    PYTORCH = "pytorch"


class EngineFactory:
    """推理引擎工厂类"""
    
    _engine_registry = {}
    
    @classmethod
    def register_engine(cls, engine_type: EngineType, engine_class):
        """注册新的引擎类型"""
        cls._engine_registry[engine_type] = engine_class
    
    @classmethod
    def _get_engine_class(cls, engine_type: EngineType):
        """获取引擎类（延迟导入）"""
        if engine_type not in cls._engine_registry:
            if engine_type == EngineType.ONNX:
                from inference_engine.onnx_engine import ONNXEngine
                cls._engine_registry[engine_type] = ONNXEngine
            elif engine_type == EngineType.OPENVINO:
                from inference_engine.openvino_engine import OpenVINOEngine
                cls._engine_registry[engine_type] = OpenVINOEngine
            elif engine_type == EngineType.PYTORCH:
                from inference_engine.pytorch_engine import PyTorchEngine
                cls._engine_registry[engine_type] = PyTorchEngine
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
        
        return cls._engine_registry[engine_type]
    
    @classmethod
    def create_engine(
        cls,
        engine_type: EngineType,
        model_path: str,
        input_size: int = None,
        output_size: int = None,
        dtype: str = 'float32',
        **extra_config
    ) -> InferenceEngineBase:
        """
        创建推理引擎实例
        """
        config = {
            'model_path': model_path,
            'input_size': input_size,
            'output_size': output_size,
            'dtype': dtype,
            **extra_config
        }
        
        engine_class = cls._get_engine_class(engine_type)
        engine = engine_class(config)
        engine.load()
        
        return engine
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> InferenceEngineBase:
        """
        从配置字典创建推理引擎
        """
        engine_config = config.get('engine', {})
        model_config = config.get('model', {})
        
        type_str = engine_config.get('type', 'onnx').lower()
        try:
            engine_type = EngineType(type_str)
        except ValueError:
            raise ValueError(f"Unsupported engine type: {type_str}")
        
        merged_config = {**engine_config, **model_config}
        
        engine_class = cls._get_engine_class(engine_type)
        engine = engine_class(merged_config)
        engine.load()
        
        return engine
    
    @classmethod
    def get_supported_engines(cls) -> list:
        """获取支持的引擎类型列表"""
        return [e.value for e in EngineType]