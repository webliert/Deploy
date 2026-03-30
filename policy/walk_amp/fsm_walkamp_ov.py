"""
FSM State Implementations - OpenVINO Version
OpenVINO-based inference for 20-joint humanoid robot
"""

import numpy as np
import openvino as ov

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
from common.BasicFunction import clip_vector, gait_phase
import os
import yaml
from scipy.spatial.transform import Rotation


class FSMStateWALKAMPOV(FSMState):
    """WALKAMP OpenVINO 策略状态实现"""

    def _reset_internal_state(self):
        """把所有随时间变化的内部状态重置成初始值"""
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)

        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.is_first_step_ = True

        # 期望关节 / 期望速度 / 力矩重置为"初始姿态"
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        self.robot_data_.q_d_[base:base + len(self.joint_xml)] = self.joint_pos_array
        self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
        self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "walk_amp_ov.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        # Load configuration
        self.action_num_ = policy_config.get('actions_size')
        self.motor_num_ = policy_config.get('motor_num')
        self.dt_ = policy_config.get('dt')

        # Size configuration
        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist')
        self.obs_size_ = size_config.get('observations_size')

        # Control configuration
        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale')
        self.decimation_ = control_config.get('decimation')

        # Normalization configuration
        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})

        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 100.0)

        # Initialize buffers and actions
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)

        # Flags
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.timer_gait_ = 0.0

        # gait parameters
        self.gait_cycle = 0.85
        self.left_phase_ratio = 0.38
        self.right_phase_ratio = 0.38
        self.left_theta_offset = 0.38
        self.right_theta_offset = 0.88

        self.is_first_step_ = True

        # Initialize OpenVINO session
        self.model_path = os.path.join(current_dir, "model", policy_config["model_path"])
        self._init_openvino_session()

        self.joint_seq = None
        self.joint_pos_array_seq = None
        self.action_scale = None
        self.stiffness_array_seq = None
        self.damping_array_seq = None

        joint_names = policy_config.get('joint_names')
        print(f"[DEBUG] joint_names from config: {joint_names}")
        if joint_names is None:
            raise ValueError("[FSMStateWALKAMPOV] Missing 'joint_names' in walk_amp_ov.yaml")

        self.joint_seq = list(joint_names)

        if self.action_scale_ is None:
            raise ValueError("[FSMStateWALKAMPOV] Missing 'control.action_scale' in walk_amp_ov.yaml")

        if np.isscalar(self.action_scale_):
            self.action_scale = np.full(len(self.joint_seq), float(self.action_scale_), dtype=np.float32)
        else:
            self.action_scale = np.array(self.action_scale_, dtype=np.float32)

        if len(self.action_scale) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMPOV] control.action_scale length {len(self.action_scale)} does not match joint count {len(self.joint_seq)}"
            )

        init_state_config = policy_config.get('init_state', {})
        default_joint_angles = init_state_config.get('default_joint_angles')
        if default_joint_angles is None:
            raise ValueError("[FSMStateWALKAMPOV] Missing 'init_state.default_joint_angles' in walk_amp_ov.yaml")

        self.joint_pos_array_seq = np.array(default_joint_angles, dtype=np.float32)
        if len(self.joint_pos_array_seq) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMPOV] init_state.default_joint_angles length {len(self.joint_pos_array_seq)} does not match joint count {len(self.joint_seq)}"
            )

        gains_config = policy_config.get('gains', {})
        kp_values = gains_config.get('kp')
        kd_values = gains_config.get('kd')
        if kp_values is None or kd_values is None:
            raise ValueError("[FSMStateWALKAMPOV] Missing 'gains.kp' or 'gains.kd' in walk_amp_ov.yaml")

        self.stiffness_array_seq = np.array(kp_values, dtype=np.float32)
        self.damping_array_seq = np.array(kd_values, dtype=np.float32)

        if len(self.stiffness_array_seq) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMPOV] gains.kp length {len(self.stiffness_array_seq)} does not match joint count {len(self.joint_seq)}"
            )
        if len(self.damping_array_seq) != len(self.joint_seq):
            raise ValueError(
                f"[FSMStateWALKAMPOV] gains.kd length {len(self.damping_array_seq)} does not match joint count {len(self.joint_seq)}"
            )

        # joint_xml: 29 joints in mujoco XML order
        self.joint_xml = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ]

        # Map from lab 20-joint order to mujoco XML 29-joint order
        self.lab2mj = []
        for name in self.joint_seq:
            if name not in self.joint_xml:
                raise ValueError(f"[FSMStateWALKAMPOV] joint '{name}' from walk_amp_ov.yaml not found in joint_xml!")
            self.lab2mj.append(self.joint_xml.index(name))
        self.lab2mj = np.array(self.lab2mj, dtype=int)

        # Build 29-length arrays for mujoco
        n_mj = len(self.joint_xml)
        self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        self.stiffness_array = np.zeros(n_mj, dtype=np.float32)
        self.damping_array = np.zeros(n_mj, dtype=np.float32)

        for lab_idx, mj_idx in enumerate(self.lab2mj):
            self.joint_pos_array[mj_idx] = self.joint_pos_array_seq[lab_idx]
            self.stiffness_array[mj_idx] = self.stiffness_array_seq[lab_idx]
            self.damping_array[mj_idx] = self.damping_array_seq[lab_idx]

        # Set other parameters
        self.kps_lab = self.stiffness_array_seq
        self.kds_lab = self.damping_array_seq
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

        self.filtered_x_speed = 0

    def _init_openvino_session(self):
        """初始化 OpenVINO 推理会话"""
        try:
            self.ov_core = ov.Core()
            ov_model = self.ov_core.read_model(self.model_path)
            
            # 处理动态形状 - 设置静态输入形状
            input_layer = ov_model.input(0)
            input_partial_shape = input_layer.partial_shape
            
            # 检查是否有动态维度
            if input_partial_shape.is_dynamic:
                print(f"[FSMStateWALKAMPOV] Model has dynamic input shape: {input_partial_shape}")
                # 设置静态形状 [batch_size, obs_size * num_hist]
                static_shape = [1, self.obs_size_ * self.num_hist_]
                ov_model.reshape({input_layer: static_shape})
                print(f"[FSMStateWALKAMPOV] Reshaped to static: {static_shape}")
            
            self.ov_compiled_model = self.ov_core.compile_model(ov_model, "CPU")
            self.ov_infer_request = self.ov_compiled_model.create_infer_request()

            print(f"[FSMStateWALKAMPOV] OpenVINO model loaded successfully: {self.model_path}")

            # Print input/output info
            compiled_input = self.ov_compiled_model.input(0)
            compiled_output = self.ov_compiled_model.output(0)
            print(f"[FSMStateWALKAMPOV] Input shape: {compiled_input.shape}, Output shape: {compiled_output.shape}")
        except Exception as e:
            print(f"[FSMStateWALKAMPOV] Failed to load OpenVINO model: {e}")
            self.ov_compiled_model = None

    def on_enter(self):
        """进入WALKAMP状态"""
        self._reset_internal_state()
        print("[FSMStateWALKAMPOV] enter")
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self._warmup_inference_counter = 0
        self.timer_gait_ = 0.0

    def run(self, flag: ControlFlag):
        """运行WALKAMP状态"""
        gait = gait_phase(
            self.timer_gait_,
            self.gait_cycle,
            self.left_theta_offset,
            self.right_theta_offset,
            self.left_phase_ratio,
            self.right_phase_ratio,
        ).astype(np.float32)

        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:
            self.compute_observation(flag, gait)
            self.compute_actions()

            # lab 顺序目标角 20 维
            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab

            # 拿当前 mj 顺序的关节角
            target_dof_pos_mj = self.robot_data_.get_joint_pos().copy()

            # 只更新 20 个受控 DOF
            target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab
            commanded_pos = target_dof_pos_mj

            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            self.robot_data_.q_d_[base:base + len(self.joint_xml)] = commanded_pos
            self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

            self.last_actions_[:] = self.actions_

        self.timer_gait_ += self.dt_
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array

    def compute_observation(self, flag: ControlFlag, gait):
        """计算观测 - 75维单帧"""
        roll, pitch, yaw = (
            float(self.robot_data_.imu_data_[2]),
            float(self.robot_data_.imu_data_[1]),
            float(self.robot_data_.imu_data_[0]),
        )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0., 0., -1.], dtype=np.float32))

        x_speed_command, y_speed_command, yaw_speed_command = self.robot_data_.get_walk_cmd()
        command = np.concatenate([
            np.array([x_speed_command, y_speed_command, yaw_speed_command], dtype=np.float32),
        ])

        ang_vel = self.robot_data_.get_angular_velocity()

        q_mj = self.robot_data_.get_joint_pos()
        dq_mj = self.robot_data_.get_joint_vel()

        # 只取 20 个受控关节，变成 lab 顺序
        qj = q_mj[self.lab2mj]
        dqj = dq_mj[self.lab2mj]

        qj = qj - self.default_angles_lab

        # Observation = ang_vel(3) + gravity(3) + command(3) + q(20) + dq(20) + action(20) + gait(6) = 75
        proprio = np.concatenate([
            ang_vel,               # 3 elements
            gravity_init,          # 3 elements
            command,               # 3 elements
            qj,                    # 20 elements
            dqj,                   # 20 elements
            self.last_actions_,    # 20 elements
            gait                   # 6 elements
        ])

        # History buffer management
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                start_idx = i * self.obs_size_
                end_idx = start_idx + self.obs_size_
                self.proprio_hist_buf_[start_idx:end_idx] = proprio
            self.is_first_obs_ = False
        else:
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift_size:] = proprio

        # Clip observations
        self.observations_ = np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)

    @staticmethod
    def euler_to_quaternion_scipy(roll, pitch, yaw, degrees=False):
        r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
        q_xyzw = r.as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    @staticmethod
    def quat_rotate_inverse_numpy(q_xyzw, v):
        q_w = q_xyzw[3]
        q_v = q_xyzw[:3]
        a = v * (2.0 * q_w * q_w - 1.0)
        b = np.cross(q_v, v) * (2.0 * q_w)
        c = q_v * (2.0 * np.dot(q_v, v))
        return a - b + c

    def compute_actions(self):
        """使用 OpenVINO 进行推理"""
        if self.ov_compiled_model is None:
            return

        try:
            # Prepare input tensor
            input_data = self.observations_.reshape(1, -1).astype(np.float32)

            # OpenVINO inference
            input_tensor = ov.Tensor(array=input_data, shared_memory=True)
            self.ov_infer_request.set_input_tensor(input_tensor)
            self.ov_infer_request.infer()
            output_tensor = self.ov_infer_request.get_output_tensor()
            output_data = output_tensor.data[0]

            # Extract and clip actions
            for i in range(self.action_num_):
                self.actions_[i] = np.clip(output_data[i], -self.clip_act_, self.clip_act_)

            if self.is_first_action_:
                print("[FSMStateWALKAMPOV] First Observation:")
                for i in range(self.obs_size_):
                    print(f"{self.observations_[i]:.6f} ", end="")
                print()
                self.is_first_action_ = False

        except Exception as e:
            print(f"[FSMStateWALKAMPOV] OpenVINO inference error: {e}")

    def on_exit(self):
        """退出WALKAMP状态"""
        print("[FSMStateWALKAMPOV] exit")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoWALKAMP_OV":
            return FSMStateName.WALKAMP_OV
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        else:
            return None