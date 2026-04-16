"""
FSM State Implementations - Zero State
"""
import os
from typing import Optional

import numpy as np
import yaml

from FSM.fsm_base import FSMState, FSMStateName
from common import ControlFlag, RobotData, get_robot_config_manager


class FSMStateZero(FSMState):
    """零位状态实现"""
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.ZERO
        self.q_factor_ = 0.0
        
        # 获取配置管理器
        self.config_mgr = get_robot_config_manager()
        self.motor_num_ = self.config_mgr.motor_num
        
        # 从策略本地配置文件加载
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "zero.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        
        # 获取策略使用的关节组
        joint_groups = policy_config.get('joint_groups', ['legs.left', 'legs.right', 'arms.left', 'arms.right'])
        joint_seq = self.config_mgr.get_all_joints_in_groups(joint_groups)
        
        # 从配置管理器获取参数
        params = self.config_mgr.get_robot_params_for_policy(joint_seq)
        self.zero_positions_ = params['zero_pos']
        self.kp_pos_ = params['kp']
        self.kd_pos_ = params['kd']
        self.zero_positions_height_ = self.zero_positions_.copy()
        
        # 策略控制参数
        self.interp_step_ = float(policy_config.get("interp_step", 0.001))
        self.interp_max_ = float(policy_config.get("interp_max", 0.9))
        self.action_num_ = self.motor_num_
        
        print(f"[FSMStateZero] Motor num: {self.motor_num_}")
        
        self.zero_positions = np.zeros(self.motor_num_)

    def on_enter(self):
        print("[FSMStateZero] Enter zero state")
        self.q_factor_ = 0.0

    def run(self, flag: ControlFlag):
        if self.robot_data_ is None:
            return
        q_est = self.robot_data_.q_a_[-self.motor_num_:]  # numpy数组切片
        if getattr(flag, 'height_control', False):
            self.zero_positions = self.zero_positions_height_
        else:
            self.zero_positions = self.zero_positions_
        if self.q_factor_ < self.interp_max_:
            pos_cmd = (1.0 - self.q_factor_) * q_est + self.q_factor_ * self.zero_positions
            self.q_factor_ = min(self.q_factor_ + self.interp_step_, self.interp_max_)
        else:
            pos_cmd = self.zero_positions
        self.robot_data_.q_d_[-self.motor_num_:] = pos_cmd
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0
        self.robot_data_.tau_d_[-self.motor_num_:] = 0
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_

    def on_exit(self):
        print("[FSMStateZero] Exit zero position control state")

    def check_transition(self, flag: ControlFlag) -> Optional[FSMStateName]:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDZERO":
            return FSMStateName.BEYONDZERO
        elif flag.fsm_state_command == "gotoHOMIE":
            return FSMStateName.HOMIE
        else:
            return None  # 无状态转换