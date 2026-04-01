"""
Robot Configuration Manager
统一管理机器人配置参数，支持多机器人适配
"""
from __future__ import annotations

import os
import yaml
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


class RobotConfigManager:
    """
    机器人配置管理器
    职责：加载、管理和提供机器人配置参数
    同时支持从主配置文件和机器人profile文件加载配置
    """
    
    _instances: Dict[str, RobotConfigManager] = {}  # 支持多配置文件的单例
    
    def __new__(cls, config_path: str, base_dir: str = None):
        """单例模式（基于配置文件路径）"""
        # 标准化配置文件路径作为单例key
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(os.path.dirname(current_dir))
        
        # 处理相对路径
        if not os.path.isabs(config_path):
            # 如果以 config/ 开头，相对于项目根目录
            if config_path.startswith('config/'):
                full_config_path = os.path.join(base_dir, config_path)
            else:
                # 否则相对于 config 目录
                full_config_path = os.path.join(base_dir, 'config', config_path)
        else:
            full_config_path = config_path
        
        config_key = os.path.abspath(full_config_path)
        
        if config_key not in cls._instances:
            cls._instances[config_key] = super().__new__(cls)
        
        return cls._instances[config_key]
    
    def __init__(self, config_path: str, base_dir: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如 'tienkung2_lite_config.yaml' 或 'config/tienkung2_lite_config.yaml'
            base_dir: 项目根目录
        """
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        if base_dir is None:
            # 默认从当前文件的上两级目录作为项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
        
        self.base_dir = base_dir
        self.profile_dir = os.path.join(base_dir, 'config', 'robot_profiles')
        self.main_config: Dict[str, Any] = {}  # 主配置文件
        self.profile: Dict[str, Any] = {}      # robot profile 文件
        self._joint_index_map: Dict[str, int] = {}
        self._index_joint_map: Dict[int, str] = {}
        
        # 运行时配置
        self.sim: bool = False
        self.debug: bool = False
        
        # 加载主配置文件并自动加载profile
        self._load_config_from_path(config_path)
        
        self._initialized = True
    
    def _load_config_from_path(self, config_path: str) -> None:
        """
        从配置文件路径加载配置
        
        Args:
            config_path: 配置文件路径（相对于项目根目录或config目录）
        """
        # 处理相对路径
        if not os.path.isabs(config_path):
            # 如果以 config/ 开头，相对于项目根目录
            if config_path.startswith('config/'):
                full_config_path = os.path.join(self.base_dir, config_path)
            else:
                # 否则相对于 config 目录
                full_config_path = os.path.join(self.base_dir, 'config', config_path)
        else:
            full_config_path = config_path
        
        if not os.path.exists(full_config_path):
            raise FileNotFoundError(f"Config file not found: {full_config_path}")
        
        # 加载主配置文件
        with open(full_config_path, 'r') as f:
            self.main_config = yaml.safe_load(f)
        
        # 同步运行时配置
        self.sim = self.main_config.get('sim', False)
        self.debug = self.main_config.get('debug', False)
        
        print(f"[RobotConfigManager] Loaded main config: {config_path}")
        print(f"[RobotConfigManager] sim={self.sim}, debug={self.debug}")
        
        # 从主配置中读取 robot_profile 并加载对应的 profile 文件
        robot_profile = self.main_config.get('robot_profile')
        if robot_profile:
            self.load_profile(robot_profile)
        else:
            raise ValueError(f"No 'robot_profile' field found in config file: {config_path}")
    
    @classmethod
    def get_instance(cls, config_path: str = None) -> RobotConfigManager:
        """获取单例实例"""
        if config_path is None:
            # 如果没有提供配置路径，返回任意一个已存在的实例
            if cls._instances:
                return next(iter(cls._instances.values()))
            raise RuntimeError("RobotConfigManager not initialized. Call with config_path first.")
        return cls(config_path)
    
    def load_profile(self, profile_name: str) -> None:
        """
        加载机器人profile
        
        Args:
            profile_name: profile名称，如 'tienkung2_lite' 或 'tienkung3_dex'
        """
        # 记录当前加载的profile名称
        self._current_profile = profile_name
        
        profile_path = os.path.join(self.profile_dir, f"{profile_name}_profile.yaml")
        
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Robot profile not found: {profile_path}")
        
        with open(profile_path, 'r') as f:
            self.profile = yaml.safe_load(f)
        
        # 构建关节索引映射
        self._build_joint_maps()
        
        print(f"[RobotConfigManager] Loaded profile: {self.profile.get('robot_name', profile_name)}")
        print(f"[RobotConfigManager] Motor num: {self.motor_num}")
    
    def _build_joint_maps(self) -> None:
        """构建关节名称和索引的双向映射"""
        joint_order = self.profile.get('joint_order', [])
        for idx, name in enumerate(joint_order):
            self._joint_index_map[name] = idx
            self._index_joint_map[idx] = name
    
    @property
    def robot_name(self) -> str:
        """机器人名称"""
        return self.profile.get('robot_name', 'Unknown')
    
    @property
    def motor_num(self) -> int:
        """电机数量"""
        return self.profile.get('motor_num', 0)
    
    @property
    def floating_base_dof(self) -> int:
        """浮动基座自由度"""
        return self.profile.get('floating_base_dof', 6)
    
    @property
    def dt(self) -> float:
        """控制周期"""
        return self.profile.get('dt', 0.0025)
    
    @property
    def whole_joint_num(self) -> int:
        """总关节数（电机 + 浮动基座）"""
        return self.motor_num + self.floating_base_dof
    
    def get_joint_index(self, joint_name: str) -> int:
        """获取关节索引"""
        return self._joint_index_map.get(joint_name, -1)
    
    def get_joint_name(self, index: int) -> str:
        """根据索引获取关节名称"""
        return self._index_joint_map.get(index, "")
    
    def get_joint_order(self) -> List[str]:
        """获取关节顺序列表"""
        return self.profile.get('joint_order', [])
    
    def _dict_to_array(self, param_dict: Dict[str, float], joint_names: List[str] = None) -> np.ndarray:
        """
        将参数字典转换为numpy数组
        
        Args:
            param_dict: 参数字典，如 {'l_hip_roll': 700.0, ...}
            joint_names: 指定关节顺序，如果为None则使用joint_order
            
        Returns:
            参数数组
        """
        if joint_names is None:
            joint_names = self.get_joint_order()
        
        result = np.zeros(len(joint_names), dtype=np.float32)
        for i, name in enumerate(joint_names):
            if name in param_dict:
                result[i] = param_dict[name]
        return result
    
    def get_zero_pos(self, joint_names: List[str] = None) -> np.ndarray:
        """
        获取零位位置
        
        Args:
            joint_names: 指定关节列表，如果为None则返回所有关节
            
        Returns:
            零位位置数组
        """
        zero_pos_dict = self.profile.get('zero_pos', {})
        return self._dict_to_array(zero_pos_dict, joint_names)
    
    def get_ct_scale(self, joint_names: List[str] = None) -> np.ndarray:
        """
        获取电流转力矩系数
        
        Args:
            joint_names: 指定关节列表
            
        Returns:
            ct_scale数组
        """
        ct_scale_dict = self.profile.get('ct_scale', {})
        return self._dict_to_array(ct_scale_dict, joint_names)
    
    def get_kp(self, joint_names: List[str] = None) -> np.ndarray:
        """
        获取Kp增益
        
        Args:
            joint_names: 指定关节列表
            
        Returns:
            Kp数组
        """
        kp_dict = self.profile.get('kp', {})
        return self._dict_to_array(kp_dict, joint_names)
    
    def get_kd(self, joint_names: List[str] = None) -> np.ndarray:
        """
        获取Kd增益
        
        Args:
            joint_names: 指定关节列表
            
        Returns:
            Kd数组
        """
        kd_dict = self.profile.get('kd', {})
        return self._dict_to_array(kd_dict, joint_names)
    
    def get_joint_limits(self, joint_name: str = None) -> Dict[str, Any]:
        """
        获取关节限位
        
        Args:
            joint_name: 关节名称，如果为None则返回所有关节限位
            
        Returns:
            关节限位字典
        """
        limits = self.profile.get('joint_limits', {})
        if joint_name:
            return limits.get(joint_name, {})
        return limits
    
    def get_joint_limits_array(self, joint_names: List[str] = None) -> tuple:
        """
        获取关节限位数组形式
        
        Args:
            joint_names: 指定关节列表
            
        Returns:
            (min_array, max_array) 元组
        """
        if joint_names is None:
            joint_names = self.get_joint_order()
        
        limits = self.profile.get('joint_limits', {})
        min_arr = np.zeros(len(joint_names), dtype=np.float32)
        max_arr = np.zeros(len(joint_names), dtype=np.float32)
        
        for i, name in enumerate(joint_names):
            if name in limits:
                min_arr[i] = limits[name].get('min', 0.0)
                max_arr[i] = limits[name].get('max', 0.0)
        
        return min_arr, max_arr
    
    def get_params_for_policy(self, joint_names: List[str]) -> Dict[str, np.ndarray]:
        """
        获取策略所需的参数子集
        
        Args:
            joint_names: 策略使用的关节名称列表
            
        Returns:
            包含kp, kd, zero_pos, ct_scale等参数的字典
        """
        return {
            'kp': self.get_kp(joint_names),
            'kd': self.get_kd(joint_names),
            'zero_pos': self.get_zero_pos(joint_names),
            'ct_scale': self.get_ct_scale(joint_names),
        }
    
    def get_joints_by_group(self, group: str) -> List[str]:
        """
        根据分组获取关节列表
        
        Args:
            group: 分组名称，如 'legs.left', 'arms.right', 'waist'
            
        Returns:
            关节名称列表
        """
        parts = group.split('.')
        joints_config = self.profile.get('joints', {})
        
        result = joints_config
        for part in parts:
            if isinstance(result, dict):
                result = result.get(part, [])
            else:
                return []
        
        return result if isinstance(result, list) else []
    
    def get_all_joints_in_groups(self, groups: List[str]) -> List[str]:
        """
        获取多个分组的所有关节
        
        Args:
            groups: 分组列表，如 ['legs.left', 'legs.right']
            
        Returns:
            合并后的关节名称列表
        """
        result = []
        for group in groups:
            result.extend(self.get_joints_by_group(group))
        return result
    
    def has_ankle_parallel(self) -> bool:
        """是否有并联脚踝"""
        return self.profile.get('ankle_parallel', {}).get('enabled', False)
    
    def has_waist(self) -> bool:
        """是否有腰部电机"""
        return self.profile.get('robot_specific', {}).get('has_waist', False)
    
    def get_ankle_parallel_params(self) -> Dict[str, np.ndarray]:
        """获取并联脚踝参数"""
        ankle_config = self.profile.get('ankle_parallel', {})
        return {
            'kp_p': np.array(ankle_config.get('kp_p', [15.0, 15.0, 15.0, 15.0]), dtype=np.float32),
            'kd_p': np.array(ankle_config.get('kd_p', [1.25, 1.25, 1.25, 1.25]), dtype=np.float32),
        }
    
    # ============================================================
    # 主配置文件加载方法（用于 robot_interface）
    # ============================================================
    
    def load_main_config(self, config_file_name: str = None, robot_profile: str = None) -> None:
        """
        加载主配置文件
        
        Args:
            config_file_name: 配置文件名称，如 'tienkung2_lite_config.yaml'
            robot_profile: 机器人profile名称，如果提供则自动查找对应的配置文件
        """
        if config_file_name is None and robot_profile is not None:
            # 自动生成配置文件名
            config_file_name = f"{robot_profile}_config.yaml"
        
        if config_file_name is None:
            raise ValueError("Either config_file_name or robot_profile must be provided")
        
        config_path = os.path.join(self.base_dir, config_file_name)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Main config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.main_config = yaml.safe_load(f)
        
        # 同步运行时配置
        self.sim = self.main_config.get('sim', False)
        self.debug = self.main_config.get('debug', False)
        
        print(f"[RobotConfigManager] Loaded main config: {config_file_name}")
        print(f"[RobotConfigManager] sim={self.sim}, debug={self.debug}")
    
    @property
    def robot_interface_config(self) -> Dict[str, Any]:
        """获取 robot_interface 配置部分"""
        return self.main_config.get('robot_interface', {})
    
    @property
    def imu_config(self) -> Dict[str, Any]:
        """获取 IMU 配置部分"""
        return self.main_config.get('imu', {})
    
    @property
    def clip_actions(self) -> bool:
        """是否裁剪动作到关节限位"""
        return self.robot_interface_config.get('clip_actions', False)
    
    @property
    def disable_joints(self) -> bool:
        """是否禁用所有关节"""
        return self.robot_interface_config.get('disable_joints', False)
    
    @property
    def xsense_roll_offset(self) -> float:
        """IMU Roll 角度偏移"""
        return self.imu_config.get('xsense_data_roll_offset', 0.0)
    
    @property
    def waist_control_status(self) -> List[str]:
        """腰部控制状态列表"""
        return self.robot_interface_config.get('waist_control_status', [])
    
    @property
    def legs_control_status(self) -> List[str]:
        """腿部控制状态列表"""
        return self.robot_interface_config.get('legs_control_status', [])
    
    @property
    def arms_control_status(self) -> List[str]:
        """手臂控制状态列表"""
        return self.robot_interface_config.get('arms_control_status', [])
    
    @property
    def left_arm_only_status(self) -> List[str]:
        """仅左手臂控制状态列表"""
        return self.robot_interface_config.get('left_arm_only_status', [])
    
    @property
    def right_arm_only_status(self) -> List[str]:
        """仅右手臂控制状态列表"""
        return self.robot_interface_config.get('right_arm_only_status', [])
    
    @property
    def ankle_kp_p(self) -> np.ndarray:
        """并联脚踝 Kp 参数"""
        params = self.get_ankle_parallel_params()
        return params['kp_p']
    
    @property
    def ankle_kd_p(self) -> np.ndarray:
        """并联脚踝 Kd 参数"""
        params = self.get_ankle_parallel_params()
        return params['kd_p']
    
    @property
    def ankle_parallel_enabled(self) -> bool:
        """是否启用并联脚踝"""
        return self.has_ankle_parallel()
    
    def get_config_for_robot_interface(self) -> Dict[str, Any]:
        """
        获取 robot_interface 所需的所有配置
        
        Returns:
            包含所有 robot_interface 相关配置的字典
        """
        return {
            # 运行时模式
            'sim': self.sim,
            'debug': self.debug,
            'floating_base_dof': self.floating_base_dof,
            
            # robot_interface 配置
            'clip_actions': self.clip_actions,
            'disable_joints': self.disable_joints,
            
            # 控制状态
            'waist_control_status': self.waist_control_status,
            'legs_control_status': self.legs_control_status,
            'arms_control_status': self.arms_control_status,
            'left_arm_only_status': self.left_arm_only_status,
            'right_arm_only_status': self.right_arm_only_status,
            
            # 电机参数
            'zero_pos': self.get_zero_pos(),
            'ct_scale': self.get_ct_scale(),
            
            # IMU
            'xsense_roll_offset': self.xsense_roll_offset,
            
            # 关节限位
            'joint_limits': self.get_joint_limits(),
            
            # 并联脚踝
            'ankle_kp_p': self.ankle_kp_p,
            'ankle_kd_p': self.ankle_kd_p,
        }
    
    def create_id_map(self):
        """
        创建电机ID映射（使用统一的 joint_id_map 模块）
        
        Returns:
            BodyServoIdMap实例
        """
        # 使用统一的工厂函数，根据 robot_name 创建对应的 ID 映射
        from Robot.joint_id_map import create_id_map as _create_id_map
        
        id_map = _create_id_map(self._current_profile)
        
        print(f"[RobotConfigManager] Created joint_id_map for: {self._current_profile}")
        print(f"[RobotConfigManager]   - leg_motor_nums: {id_map.leg_motor_nums}")
        print(f"[RobotConfigManager]   - waist_motor_nums: {id_map.waist_motor_nums}")
        print(f"[RobotConfigManager]   - arm_motor_nums: {id_map.arm_motor_nums}")
        print(f"[RobotConfigManager]   - whole_motor_nums: {id_map.whole_motor_nums}")
        
        return id_map


def get_robot_config_manager(config_path: str = None) -> RobotConfigManager:
    """
    工厂函数：获取机器人配置管理器实例
    
    Args:
        config_path: 配置文件路径，如 'tienkung2_lite_config.yaml' 或 'config/tienkung2_lite_config.yaml'
        
    Returns:
        RobotConfigManager实例
    """
    if config_path:
        return RobotConfigManager(config_path)
    return RobotConfigManager.get_instance()
