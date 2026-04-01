"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np

from FSM.fsm_base import FSMState, FSMStateName
from common import ControlFlag, RobotData
from config.robot_config_manager import RobotConfigManager
import yaml
import os


class FSMStateStop(FSMState):
    """停止状态实现 - 与C++版本完全一致"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.STOP
        
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "stop.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        # 获取机器人配置管理器
        try:
            self.config_manager = RobotConfigManager.get_instance()
            self.motor_num_ = self.config_manager.motor_num
            print(f"[FSMStateStop] Using robot profile: {self.config_manager.robot_name}")
            
            # 获取策略使用的关节组（包括腰部）
            joint_groups = policy_config.get('joint_groups', 
                ['legs.left', 'legs.right', 'arms.left', 'arms.right', 'waist'])
            
            # 检查机器人是否有腰部关节
            if not self.config_manager.has_waist():
                # 如果没有腰部关节，移除waist组
                joint_groups = [g for g in joint_groups if g != 'waist']
            
            joint_seq = self.config_manager.get_all_joints_in_groups(joint_groups)
            
            # 从配置管理器获取参数
            params = self.config_manager.get_params_for_policy(joint_seq)
            self.kp_pos_ = params['kp']
            self.kd_pos_ = params['kd']
            
        except Exception as e:
            print(f"[FSMStateStop] Failed to get config manager, using defaults: {e}")
            self.config_manager = None
            self.action_num_ = policy_config.get("actions_size", 20)
            self.motor_num_ = policy_config.get("motor_num", 20)
            self.kp_pos_ = np.array(policy_config.get("kp_pos", [700.0] * self.motor_num_), dtype=float)
            self.kd_pos_ = np.array(policy_config.get("kd_pos", [20.0] * self.motor_num_), dtype=float)
        
        self.action_num_ = self.motor_num_
        
        # Initialize vectors
        self.hold_position_ = np.zeros(self.motor_num_)
        
        print(f"[FSMStateStop] Motor num: {self.motor_num_}")

    def on_enter(self):
        """进入停止状态 - 与C++版本完全一致"""
        # Store the last motor positions as hold positions (equivalent to tail(motor_num_))
        self.hold_position_ = self.robot_data_.q_a_[-self.motor_num_:].copy()
        print("[FSMStateStop] Enter stop state")

    def run(self, flag: ControlFlag):
        """运行停止状态 - 与C++版本完全一致"""
        if self.robot_data_ is None:
            return
        print(f"""[FSMStateStop] Holding position: {self.hold_position_}""")
        # Enforce the hold position for every frame (equivalent to tail(motor_num_))
        self.robot_data_.q_d_[-self.motor_num_:] = self.hold_position_
        # Set desired joint velocities to zero
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0.0
        # Set desired torques to zero
        self.robot_data_.tau_d_[-self.motor_num_:] = 0.0

        # Set proportional and derivative gains
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_

    def on_exit(self):
        """退出停止状态 - 与C++版本完全一致"""
        self.hold_position_.fill(0.0)
        print("[FSMStateStop] Exit stop position control state")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoWALKAMP_OV":
            return FSMStateName.WALKAMP_OV
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDZERO":
            return FSMStateName.BEYONDZERO
        else:
            return None  # 无状态转换
