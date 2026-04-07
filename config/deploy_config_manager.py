"""
Deploy 统一配置管理器
管理所有配置：机器人选择、运行参数、策略、推理引擎
"""
from __future__ import annotations

import os
import yaml
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


class DeployConfig:
    """
    统一的配置数据类，提供类型安全的配置访问
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    # ============================================================
    # 机器人配置
    # ============================================================
    @property
    def robot_profile(self) -> str:
        """获取机器人profile名称"""
        return self._config.get("robot", {}).get("profile", "tienkung2_lite")
    
    # ============================================================
    # 运行参数
    # ============================================================
    @property
    def dt(self) -> float:
        """控制周期"""
        return self._config.get("runtime", {}).get("dt", 0.01)
    
    @property
    def sim(self) -> bool:
        """是否仿真模式"""
        return self._config.get("runtime", {}).get("sim", False)
    
    @property
    def debug(self) -> bool:
        """是否调试模式"""
        return self._config.get("runtime", {}).get("debug", False)
    
    @property
    def control_tool(self) -> str:
        """控制工具类型"""
        return self._config.get("runtime", {}).get("control_tool", "keyboard")
    
    # ============================================================
    # 策略配置
    # ============================================================
    @property
    def active_policy(self) -> str:
        """当前激活的策略"""
        return self._config.get("policies", {}).get("active", "walk_amp")
    
    def get_available_policies(self) -> Dict[str, Dict[str, Any]]:
        """获取所有可用的策略配置"""
        return self._config.get("policies", {}).get("available", {})
    
    def is_policy_enabled(self, policy_name: str) -> bool:
        """检查指定策略是否启用"""
        policy = self.get_available_policies().get(policy_name, {})
        return policy.get("enabled", False)
    
    def get_policy_class(self, policy_name: str) -> Optional[str]:
        """获取策略类名"""
        policy = self.get_available_policies().get(policy_name, {})
        return policy.get("class")
    
    # ============================================================
    # 推理引擎配置
    # ============================================================
    @property
    def inference_engine_config(self) -> Dict[str, Any]:
        """获取推理引擎配置"""
        return self._config.get("inference_engine", {})
    
    @property
    def engine_type(self) -> str:
        """默认引擎类型"""
        return self.inference_engine_config.get("type", "onnx")
    
    # ============================================================
    # 控制器配置
    # ============================================================
    def get_controller_config(self) -> Dict[str, Any]:
        """获取当前控制器的配置"""
        tool = self.control_tool
        return self._config.get(tool, {})
    
    # ============================================================
    # 机器人接口配置
    # ============================================================
    @property
    def robot_interface_config(self) -> Dict[str, Any]:
        return self._config.get("robot_interface", {})
    
    @property
    def clip_actions(self) -> bool:
        return self.robot_interface_config.get("clip_actions", False)
    
    @property
    def disable_joints(self) -> bool:
        return self.robot_interface_config.get("disable_joints", False)
    
    @property
    def arms_control_status(self) -> List[str]:
        return self.robot_interface_config.get("arms_control_status", [])
    
    # ============================================================
    # IMU 配置
    # ============================================================
    @property
    def xsense_roll_offset(self) -> float:
        return self._config.get("imu", {}).get("xsense_data_roll_offset", 0.0)


class DeployConfigManager:
    """
    部署配置管理器
    职责：从统一配置文件加载所有配置，提供便捷的访问接口
    """
    
    _instance: Optional[DeployConfigManager] = None
    
    def __init__(self, config_path: str = None, base_dir: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 统一配置文件路径（相对于base_dir或绝对路径）
            base_dir: 项目根目录
        """
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
        
        self.base_dir = base_dir
        self.config_dir = os.path.join(base_dir, "config")
        
        # 确定配置文件路径
        if config_path is None:
            config_path = os.path.join(self.config_dir, "deploy_config.yaml")
        elif not os.path.isabs(config_path):
            config_path = os.path.join(self.config_dir, config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 加载主配置
        with open(config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)
        
        self.config = DeployConfig(self._raw_config)
        
        # 加载机器人profile
        self._robot_profile: Dict[str, Any] = {}
        self._load_robot_profile()
        
        # 策略配置缓存
        self._policy_configs: Dict[str, Dict[str, Any]] = {}
        
        # 关节索引映射
        self._joint_index_map: Dict[str, int] = {}
        self._index_joint_map: Dict[int, str] = {}
        self._build_joint_maps()
        
        print(f"[DeployConfigManager] Loaded config from: {config_path}")
        print(f"[DeployConfigManager] Robot: {self.config.robot_profile}")
        print(f"[DeployConfigManager] Active policy: {self.config.active_policy}")
        print(f"[DeployConfigManager] Sim: {self.config.sim}, Debug: {self.config.debug}")
    
    @classmethod
    def get_instance(cls, config_path: str = None) -> DeployConfigManager:
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
    @classmethod
    def reset(cls):
        """重置单例（用于测试）"""
        cls._instance = None
    
    # ============================================================
    # 内部方法
    # ============================================================
    
    def _load_robot_profile(self):
        """加载机器人profile"""
        profile_name = self.config.robot_profile
        profile_path = os.path.join(self.config_dir, "robots", f"{profile_name}.yaml")
        
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Robot profile not found: {profile_path}")
        
        with open(profile_path, 'r') as f:
            self._robot_profile = yaml.safe_load(f)
        
        print(f"[DeployConfigManager] Loaded robot profile: {profile_name}")
    
    def _build_joint_maps(self):
        """构建关节名称和索引的双向映射"""
        joint_xml = self._robot_profile.get("joint_xml", [])
        for idx, name in enumerate(joint_xml):
            self._joint_index_map[name] = idx
            self._index_joint_map[idx] = name
    
    def _dict_to_array(self, param_dict: Dict[str, float], joint_names: List[str]) -> np.ndarray:
        """将参数字典转换为numpy数组"""
        result = np.zeros(len(joint_names), dtype=np.float32)
        for i, name in enumerate(joint_names):
            if name in param_dict:
                result[i] = param_dict[name]
        return result
    
    # ============================================================
    # 机器人参数访问
    # ============================================================
    
    @property
    def robot_name(self) -> str:
        return self._robot_profile.get("robot_name", "Unknown")
    
    @property
    def motor_num(self) -> int:
        return self._robot_profile.get("motor_num", 0)
    
    @property
    def floating_base_dof(self) -> int:
        return self._robot_profile.get("floating_base_dof", 6)
    
    @property
    def whole_joint_num(self) -> int:
        return self.motor_num + self.floating_base_dof
    
    def get_joint_xml(self) -> List[str]:
        """获取 XML/MuJoCo 中的关节顺序"""
        return self._robot_profile.get("joint_xml", [])
    
    def get_joint_lab(self) -> List[str]:
        """获取策略/实验室使用的关节顺序"""
        joint_lab = self._robot_profile.get("joint_lab", [])
        if not joint_lab:
            return self.get_joint_xml()
        return joint_lab
    
    # 保持向后兼容
    def get_joint_order(self) -> List[str]:
        """获取关节顺序（向后兼容，等同于 get_joint_xml）"""
        return self.get_joint_xml()
    
    def get_joint_index(self, joint_name: str) -> int:
        return self._joint_index_map.get(joint_name, -1)
    
    def get_joint_name(self, index: int) -> str:
        return self._index_joint_map.get(index, "")
    
    # ============================================================
    # 策略参数获取
    # ============================================================
    
    def get_robot_params_for_policy(self, joint_names: List[str]) -> Dict[str, np.ndarray]:
        """
        获取策略所需的机器人参数
        
        Args:
            joint_names: 策略使用的关节名称列表
            
        Returns:
            包含kp, kd, zero_pos, ct_scale的字典
        """
        return {
            "kp": self._dict_to_array(self._robot_profile.get("kp", {}), joint_names),
            "kd": self._dict_to_array(self._robot_profile.get("kd", {}), joint_names),
            "zero_pos": self._dict_to_array(self._robot_profile.get("zero_pos", {}), joint_names),
            "ct_scale": self._dict_to_array(self._robot_profile.get("ct_scale", {}), joint_names),
        }
    
    def get_zero_pos(self) -> np.ndarray:
        """
        获取所有关节的零位偏移
        
        Returns:
            零位偏移数组
        """
        joint_order = self.get_joint_order()
        return self._dict_to_array(self._robot_profile.get("zero_pos", {}), joint_order)
    
    def get_ct_scale(self) -> np.ndarray:
        """
        获取所有关节的电流到力矩的转换系数
        
        Returns:
            ct_scale数组
        """
        joint_order = self.get_joint_order()
        return self._dict_to_array(self._robot_profile.get("ct_scale", {}), joint_order)
    
    def get_joints_by_group(self, group: str) -> List[str]:
        """根据分组获取关节列表"""
        parts = group.split(".")
        joints_config = self._robot_profile.get("joints", {})
        
        result = joints_config
        for part in parts:
            if isinstance(result, dict):
                result = result.get(part, [])
            else:
                return []
        
        return result if isinstance(result, list) else []
    
    def get_all_joints_in_groups(self, groups: List[str]) -> List[str]:
        """获取多个分组的所有关节"""
        result = []
        for group in groups:
            result.extend(self.get_joints_by_group(group))
        return result
    
    # ============================================================
    # 便捷方法
    # ============================================================
    
    def has_waist(self) -> bool:
        """是否有腰部关节"""
        return self._robot_profile.get("robot_specific", {}).get("has_waist", False)
    
    def has_ankle_parallel(self) -> bool:
        """是否有并联脚踝"""
        return self._robot_profile.get("ankle_parallel", {}).get("enabled", False)
    
    def get_ankle_parallel_params(self) -> Dict[str, np.ndarray]:
        """获取并联脚踝参数"""
        ankle_config = self._robot_profile.get("ankle_parallel", {})
        return {
            "kp_p": np.array(ankle_config.get("kp_p", [15.0, 15.0, 15.0, 15.0]), dtype=np.float32),
            "kd_p": np.array(ankle_config.get("kd_p", [1.25, 1.25, 1.25, 1.25]), dtype=np.float32),
        }
    
    def get_joint_limits(self, joint_name: str = None) -> Dict[str, Any]:
        """获取关节限位"""
        limits = self._robot_profile.get("joint_limits", {})
        if joint_name:
            return limits.get(joint_name, {})
        return limits
    
    
    def create_id_map(self):
        """创建电机ID映射"""
        from Robot.joint_id_map import create_id_map as _create_id_map
        
        profile_name = self.config.robot_profile
        id_map = _create_id_map(profile_name)
        
        print(f"[DeployConfigManager] Created joint_id_map for: {profile_name}")
        print(f"[DeployConfigManager]   - leg_motor_nums: {id_map.leg_motor_nums}")
        print(f"[DeployConfigManager]   - waist_motor_nums: {id_map.waist_motor_nums}")
        print(f"[DeployConfigManager]   - arm_motor_nums: {id_map.arm_motor_nums}")
        print(f"[DeployConfigManager]   - whole_motor_nums: {id_map.whole_motor_nums}")
        
        return id_map


# ============================================================
# 便捷函数
# ============================================================

def get_deploy_config(config_path: str = None) -> DeployConfigManager:
    """获取统一配置管理器实例"""
    return DeployConfigManager.get_instance(config_path)