"""
Robot Interface Base
机器人接口抽象基类定义
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from common import RobotData, ControlFlag
    from FSM.fsm_base import FSMStateName
    from rclpy.node import Node


class RobotInterface(ABC):
    """机器人接口抽象基类

    定义机器人接口的抽象方法，所有具体实现都应继承此类
    """

    def __init__(self, robot_data: 'RobotData'):
        """
        初始化机器人接口

        Args:
            robot_data: 机器人数据对象，包含机器人状态信息
        """
        self.robot_data_ = robot_data

    @abstractmethod
    def init(self, node: 'Node'):
        """初始化接口

        Args:
            node: ROS2节点，用于创建发布者和订阅者
        """
        pass

    @abstractmethod
    def update_robot_data(self, flag: 'ControlFlag', time_passed: float):
        """更新机器人状态

        从传感器读取数据并更新robot_data

        Args:
            flag: 控制标志
            time_passed: 经过的时间（秒）
        """
        pass

    @abstractmethod
    def send_motor_commands(self, flag: 'ControlFlag'):
        """发布电机控制命令

        将控制命令转换为电机命令并发布

        Args:
            flag: 控制标志
        """
        pass

    @abstractmethod
    def update_fsm(self, current_state: Optional['FSMStateName'] = None):
        """更新FSM接口

        Args:
            current_state: 当前FSM状态，如果为None则不更新
        """
        pass
