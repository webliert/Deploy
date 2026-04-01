"""
FSM State Implementations
Concrete implementations of different FSM states
"""
import numpy as np
from FSM.fsm_base import FSMState, FSMStateName
from common import ControlFlag, RobotData
from config.robot_config_manager import RobotConfigManager
import os
import yaml


class FSMStateZero(FSMState):
    """零位状态实现（完整C++逻辑移植）"""
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.ZERO
        self.q_factor_ = 0.0
        
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "zero.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        # 获取机器人配置管理器
        try:
            self.config_manager = RobotConfigManager.get_instance()
            self.motor_num_ = self.config_manager.motor_num
            print(f"[FSMStateZero] Using robot profile: {self.config_manager.robot_name}")
            
            # 获取策略使用的关节组
            joint_groups = policy_config.get('joint_groups', ['legs.left', 'legs.right', 'arms.left', 'arms.right'])
            joint_seq = self.config_manager.get_all_joints_in_groups(joint_groups)
            
            # 从配置管理器获取参数
            params = self.config_manager.get_params_for_policy(joint_seq)
            self.zero_positions_ = params['zero_pos']
            self.kp_pos_ = params['kp']
            self.kd_pos_ = params['kd']
            
            # 零位高度（如果有特殊配置则使用，否则与zero_positions相同）
            self.zero_positions_height_ = self.zero_positions_.copy()
            
        except Exception as e:
            print(f"[FSMStateZero] Failed to get config manager, using defaults: {e}")
            self.config_manager = None
            self.action_num_ = policy_config.get("actions_size", 20)
            self.motor_num_ = policy_config.get("motor_num", 20)
            self.zero_positions_ = np.array(policy_config.get("zero_positions", [0.0] * self.motor_num_), dtype=float)
            self.zero_positions_height_ = np.array(policy_config.get("zero_positions_height", self.zero_positions_.tolist()), dtype=float)
            self.kp_pos_ = np.array(policy_config.get("kp_pos", [700.0] * self.motor_num_), dtype=float)
            self.kd_pos_ = np.array(policy_config.get("kd_pos", [20.0] * self.motor_num_), dtype=float)
        
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
