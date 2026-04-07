"""
Common Module
提供机器人控制系统的常用工具类和函数
"""

# 控制命令相关
from common.command_control.joystick_control import (
    ControlFlag,
    JoystickHumanoid,
    YUNZHUOMap,
    YUNZHUOFlag,
)
from common.command_control.xbox_control import XBOXController
from common.command_control.stdin_keyboard_control import KeyboardController

# 基础数学函数
from common.BasicFunction import (
    rot_x,
    rot_y,
    rot_z,
    euler_xyz_to_matrix,
    clip_vector,
    clip_scalar,
    gait_phase,
    fifth_poly,
    LowPassFilter,
)

# 工具类
from common.peekqueue import PeekableQueue

# 配置管理
from config.deploy_config_manager import DeployConfigManager, get_deploy_config
from common.robot_data import RobotData

# 别名：get_robot_config_manager 等价于 get_deploy_config
get_robot_config_manager = get_deploy_config

__all__ = [
    # 控制命令
    "ControlFlag",
    "JoystickHumanoid",
    "YUNZHUOMap",
    "YUNZHUOFlag",
    "XBOXController",
    "KeyboardController",
    # 基础数学函数
    "rot_x",
    "rot_y",
    "rot_z",
    "euler_xyz_to_matrix",
    "clip_vector",
    "clip_scalar",
    "gait_phase",
    "fifth_poly",
    "LowPassFilter",
    # 工具类
    "PeekableQueue",
    # 配置管理
    "DeployConfigManager",
    "get_deploy_config",
    "RobotData",
]