"""
FSM State Implementations - Stop State
"""

import os
import numpy as np
import yaml

from typing import Optional

from FSM.fsm_base import FSMState, FSMStateName
from common import ControlFlag, RobotData, get_robot_config_manager


class FSMStateStop(FSMState):
    """停止状态实现"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.STOP
        
        # 获取配置管理器
        self.config_mgr = get_robot_config_manager()
        self.motor_num_ = self.config_mgr.motor_num
        
        # 从策略本地配置文件加载
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "stop.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        
        # 获取策略使用的关节组
        joint_groups = policy_config.get('joint_groups', 
            ['legs.left', 'legs.right', 'arms.left', 'arms.right', 'waist'])
        
        # 检查机器人是否有腰部关节
        if not self.config_mgr.has_waist():
            joint_groups = [g for g in joint_groups if g != 'waist']
        
        joint_seq = self.config_mgr.get_all_joints_in_groups(joint_groups)
        
        # 从配置管理器获取参数
        params = self.config_mgr.get_robot_params_for_policy(joint_seq)
        self.kp_pos_ = params['kp']
        self.kd_pos_ = params['kd']
        
        self.action_num_ = self.motor_num_
        
        # Initialize vectors
        self.hold_position_ = np.zeros(self.motor_num_)
        
        print(f"[FSMStateStop] Motor num: {self.motor_num_}")

    def on_enter(self):
        """进入停止状态"""
        self.hold_position_ = self.robot_data_.q_a_[-self.motor_num_:].copy()
        print("[FSMStateStop] Enter stop state")

    def run(self, flag: ControlFlag):
        """运行停止状态"""
        if self.robot_data_ is None:
            return
        self.robot_data_.q_d_[-self.motor_num_:] = self.hold_position_
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0.0
        self.robot_data_.tau_d_[-self.motor_num_:] = 0.0
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_

    def on_exit(self):
        """退出停止状态"""
        self.hold_position_.fill(0.0)
        print("[FSMStateStop] Exit stop position control state")

    def check_transition(self, flag: ControlFlag) -> Optional[FSMStateName]:
        """检查状态转换"""
        cmd = flag.fsm_state_command
        transitions = {
            "gotoSTOP": FSMStateName.STOP,
            "gotoWALKAMP": FSMStateName.WALKAMP,
            "gotoZERO": FSMStateName.ZERO,
            "gotoBEYONDZERO": FSMStateName.BEYONDZERO,
        }
        return transitions.get(cmd, None)
