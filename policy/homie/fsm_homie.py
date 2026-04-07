"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np

from typing import Optional
from FSM.fsm_base import FSMState, FSMStateName
from common import ControlFlag, RobotData, clip_vector, gait_phase, get_robot_config_manager
from inference_engine import EngineFactory, EngineType, InferenceEngineBase
import os
import yaml
from scipy.spatial.transform import Rotation

class FSMStateHOMIE(FSMState):
    """WALKAMP策略状态实现"""
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        # 获取当前文件目录（policy/homie/）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 从策略本地配置文件加载（实现策略与机器人配置的解耦）
        config_path = os.path.join(current_dir, "config", "homie.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[FSMStateHOMIE] Cannot find config file: {config_path}")
        
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        
        print(f"[FSMStateHOMIE] Loaded policy config from: {config_path}")

        # 从策略配置获取参数
        self.dt_ = policy_config.get('dt')

        # Size configuration
        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist')
        self.obs_size_ = size_config.get('observations_size')

        # Control configuration
        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale', 0.25)
        self.decimation_ = control_config.get('decimation', 1)

        # Normalization configuration
        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})

        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 100.0)

        # 获取机器人配置管理器
        try:
            self.config_manager = get_robot_config_manager()
            self.motor_num_ = self.config_manager.motor_num
            self.dt_ = self.config_manager.config.dt
            print(f"[FSMStateHOMIE] Using robot profile: {self.config_manager.robot_name}")
            print(f"[FSMStateHOMIE] Motor num from profile: {self.motor_num_}")
            print(f"[FSMStateHOMIE] dt from main config: {self.dt_}")
        except Exception as e:
            print(f"[FSMStateHOMIE] Failed to get config manager, using defaults: {e}")
            self.motor_num_ = 20
            self.config_manager = None


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

        # 从 config_manager 获取关节列表，如果不可用则使用回退逻辑
        if self.config_manager:
            self.joint_lab = self.config_manager.get_joint_lab()
            self.joint_seq = list(self.joint_lab)
            print(f"[FSMStateHOMIE] Joints from config manager: {len(self.joint_seq)}")
        else:
            # 回退：使用硬编码的关节列表
            print("[FSMStateHOMIE] Config manager not available, using fallback joint list")
            self.joint_lab = [
                "hip_roll_l_joint", "hip_roll_r_joint",
                "shoulder_pitch_l_joint", "shoulder_pitch_r_joint",
                "hip_pitch_l_joint", "hip_pitch_r_joint",
                "shoulder_roll_l_joint", "shoulder_roll_r_joint",
                "hip_yaw_l_joint", "hip_yaw_r_joint",
                "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
                "knee_pitch_l_joint", "knee_pitch_r_joint",
                "elbow_pitch_l_joint", "elbow_pitch_r_joint",
                "ankle_pitch_l_joint", "ankle_pitch_r_joint",
                "ankle_roll_l_joint", "ankle_roll_r_joint",
            ]
            self.joint_seq = list(self.joint_lab)
            print(f"[FSMStateHOMIE] Fallback joints count: {len(self.joint_seq)}")

        if self.config_manager:
            # 从配置管理器获取参数（kp, kd, zero_pos）
            params = self.config_manager.get_robot_params_for_policy(self.joint_seq)
            self.stiffness_array_seq = params['kp']
            self.damping_array_seq = params['kd']
            self.joint_pos_array_seq = params['zero_pos']
        else:
            # 回退：从策略配置文件读取（兼容旧方式）
            gains_config = policy_config.get('gains', {})
            self.stiffness_array_seq = np.array(gains_config.get('kp', []), dtype=np.float32)
            self.damping_array_seq = np.array(gains_config.get('kd', []), dtype=np.float32)

            init_state_config = policy_config.get('init_state', {})
            self.joint_pos_array_seq = np.array(init_state_config.get('default_joint_angles', []), dtype=np.float32)

        # action_scale可以是标量或数组
        if np.isscalar(self.action_scale_):
            self.action_scale = np.full(len(self.joint_seq), float(self.action_scale_), dtype=np.float32)
        else:
            self.action_scale = np.array(self.action_scale_[:len(self.joint_seq)], dtype=np.float32)

        self.action_num_ = len(self.joint_seq)
        print(f"[FSMStateHOMIE] Joint count: {len(self.joint_seq)}")

        # Initialize inference engine from config (must be after action_num_ is set)
        self._init_inference_engine(policy_config, current_dir)

        # Initialize buffers and actions
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)

        # warm_start_time from policy config
        warm_start_time = policy_config.get('warm_start_time', 0.3)
        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if warm_start_time > 0 and step > 0:
            self._warm_start_steps = max(1, int(warm_start_time / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0

        # joint_xml: 从 config_manager 获取关节顺序（与 mujoco XML 一致）
        if self.config_manager:
            self.joint_xml = self.config_manager.get_joint_xml()
            print(f"[FSMStateHOMIE] Joint XML from config manager: {len(self.joint_xml)} joints")
        else:
            # 回退：使用硬编码的 joint_xml
            self.joint_xml = [
                "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint",
                "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
                "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint",
                "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
                "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
                "elbow_pitch_l_joint",
                "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
                "elbow_pitch_r_joint",
            ]
            print(f"[FSMStateHOMIE] Joint XML fallback count: {len(self.joint_xml)}")

        # Map from lab joint order to mujoco XML joint order
        self.lab2mj = []
        for name in self.joint_seq:
            if name not in self.joint_xml:
                print(f"[FSMStateHOMIE] Warning: joint '{name}' not found in joint_xml, skipping")
                continue
            self.lab2mj.append(self.joint_xml.index(name))
        self.lab2mj = np.array(self.lab2mj, dtype=int)

        # Build arrays for mujoco
        n_mj = len(self.joint_xml)
        self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        self.stiffness_array = np.zeros(n_mj, dtype=np.float32)
        self.damping_array = np.zeros(n_mj, dtype=np.float32)

        for lab_idx, mj_idx in enumerate(self.lab2mj):
            if lab_idx < len(self.joint_pos_array_seq):
                self.joint_pos_array[mj_idx] = self.joint_pos_array_seq[lab_idx]
            if lab_idx < len(self.stiffness_array_seq):
                self.stiffness_array[mj_idx] = self.stiffness_array_seq[lab_idx]
            if lab_idx < len(self.damping_array_seq):
                self.damping_array[mj_idx] = self.damping_array_seq[lab_idx]

        # Set other parameters
        self.kps_lab = self.stiffness_array_seq
        self.kds_lab = self.damping_array_seq
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

        self.filtered_x_speed = 0

    def _reset_internal_state(self):
        """把所有随时间变化的内部状态重置成初始值"""

        # 1) 清空 obs / hist / actions
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)

        # 2) 标志位重置
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.is_first_step_ = True

        # 3) 期望关节 / 期望速度 / 力矩重置为“初始姿态”
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        # # 期望角 = 初始角
        self.robot_data_.q_d_[base:base + len(self.joint_xml)] = self.joint_pos_array
        # # 期望速度 = 0
        self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
        # # 期望力矩 = 0（位置控制）
        self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

    def _init_inference_engine(self, policy_config: dict, current_dir: str):
        """初始化推理引擎 - 根据 engine_type 条件判断读取对应引擎配置"""
        # 获取引擎类型
        engine_type = policy_config.get('engine_type', 'onnx').lower()
        
        # 获取模型通用配置
        model_config = policy_config.get('model', {})
        
        # 根据 engine_type 条件判断，只读取当前选中引擎的配置
        engine_config = {'type': engine_type}
        
        # 预检查模型文件存在性
        def check_model_file(path: str) -> bool:
            if not os.path.exists(path):
                print(f"\033[91m[FSMStateHOMIE] ERROR: Model file does NOT exist: {path}\033[0m")
                return False
            if os.path.getsize(path) == 0:
                print(f"\033[91m[FSMStateHOMIE] ERROR: Model file is empty: {path}\033[0m")
                return False
            return True
        
        if engine_type == 'onnx':
            # 读取 ONNX 配置块
            onnx_cfg = policy_config.get('onnx', {})
            if not onnx_cfg:
                raise ValueError("[FSMStateHOMIE] Missing 'onnx' config block in homie.yaml")
            model_file = os.path.join(current_dir, "model", onnx_cfg.get('model_path', ''))
            check_model_file(model_file)
            engine_config['model_path'] = model_file
            engine_config['intra_op_num_threads'] = onnx_cfg.get('intra_op_num_threads', 1)
            engine_config['inter_op_num_threads'] = onnx_cfg.get('inter_op_num_threads', 1)
            engine_config['enable_mem_pattern'] = onnx_cfg.get('enable_mem_pattern', False)
            engine_config['enable_mem_reuse'] = onnx_cfg.get('enable_mem_reuse', True)
            engine_config['graph_optimization_level'] = onnx_cfg.get(
                'graph_optimization_level', 'ORT_ENABLE_ALL')
            
        elif engine_type == 'openvino':
            # 读取 OpenVINO 配置块
            ov_cfg = policy_config.get('openvino', {})
            if not ov_cfg:
                raise ValueError("[FSMStateHOMIE] Missing 'openvino' config block in homie.yaml")
            model_file = os.path.join(current_dir, "model", ov_cfg.get('model_path', ''))
            check_model_file(model_file)
            engine_config['model_path'] = model_file
            engine_config['device'] = ov_cfg.get('device', 'CPU')
            engine_config['cache_dir'] = ov_cfg.get('cache_dir', None)
            
        elif engine_type == 'pytorch':
            # 读取 PyTorch 配置块
            pt_cfg = policy_config.get('pytorch', {})
            if not pt_cfg:
                raise ValueError("[FSMStateHOMIE] Missing 'pytorch' config block in homie.yaml")
            model_file = os.path.join(current_dir, "model", pt_cfg.get('model_path', ''))
            check_model_file(model_file)
            engine_config['model_path'] = model_file
            engine_config['device'] = pt_cfg.get('device', 'cpu')
            engine_config['jit_trace'] = pt_cfg.get('jit_trace', False)
            
        else:
            raise ValueError(f"[FSMStateHOMIE] Unsupported engine type: {engine_type}")
        
        # 合并通用配置，构建完整的引擎配置
        full_config = {
            'engine': engine_config,
            'model': {
                'input_size': self.obs_size_ * self.num_hist_,
                'output_size': self.action_num_,
                'dtype': model_config.get('dtype', 'float32'),
            }
        }
        
        try:
            self.inference_engine_: InferenceEngineBase = EngineFactory.create_from_config(full_config)
            print(f"[FSMStateHOMIE] Inference engine loaded: {engine_type}")
        except Exception as e:
            print(f"[FSMStateHOMIE] Failed to load inference engine: {e}")
            self.inference_engine_ = None

    def on_enter(self):
        """进入WALKAMP状态"""
        self._reset_internal_state()
        print("[FSMStateHOMIE] enter")
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self._warmup_inference_counter = 0
        self.timer_gait_ = 0.0
        self._last_engine_warn_time = 0.0

        # 检查推理引擎状态
        if not hasattr(self, 'inference_engine_') or self.inference_engine_ is None:
            print("\033[91m[FSMStateHOMIE] WARNING: Inference engine was NOT loaded successfully!\033[0m")
            print("\033[91m[FSMStateHOMIE] Policy will NOT run! Check model path and engine config.\033[0m")
        else:
            # 重新加载推理引擎（如果之前被unload了）
            if not self.inference_engine_.is_loaded:
                try:
                    self.inference_engine_.load()
                    print("[FSMStateHOMIE] Inference engine reloaded on enter")
                except Exception as e:
                    print(f"\033[91m[FSMStateHOMIE] Failed to reload inference engine: {e}\033[0m")
        
        if self.inference_engine_ and self.inference_engine_.is_loaded:
            print(f"[FSMStateHOMIE] ✅ Inference engine ready, running normally")

        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)

    def run(self, flag: ControlFlag):
        """运行WALKAMP状态 - 与C++版本完全一致"""
        # Only run policy inference every decimation_ steps
        gait = gait_phase(
            self.timer_gait_,
            self.gait_cycle,
            self.left_theta_offset,
            self.right_theta_offset,
            self.left_phase_ratio,
            self.right_phase_ratio,
        ).astype(np.float32)

        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:

            # print(f"[FSMStateHOMIE] Gait phase: {gait}")
            self.compute_observation(flag,gait)
            self.compute_actions()

            # lab 顺序目标角 motion_num 维
            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab

            # 拿一份当前 mj 顺序的关节角（或你原来用的 default 也行）
            target_dof_pos_mj = self.robot_data_.get_joint_pos().copy()

            # 只更新 motion_num 个受控 DOF
            target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab
            commanded_pos = target_dof_pos_mj
            if self._warm_start_steps > 0 and self._warmup_inference_counter < self._warm_start_steps:
                self._warmup_inference_counter += 1
                blend = self._warmup_inference_counter / float(self._warm_start_steps)
                commanded_pos = (1.0 - blend) * self._warm_start_pose + blend * target_dof_pos_mj

            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            self.robot_data_.q_d_[base:base + len(self.joint_xml)] = commanded_pos

            self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

            self.last_actions_[:] = self.actions_


        self.timer_gait_ += self.dt_
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array

    def compute_observation(self, flag: ControlFlag, gait):
        roll, pitch, yaw = (
                        float(self.robot_data_.imu_data_[2]),
                        float(self.robot_data_.imu_data_[1]),
                        float(self.robot_data_.imu_data_[0]),
                    )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init   = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0.,0.,-1.], dtype=np.float32))
        

        x_speed_command, y_speed_command, yaw_speed_command = self.robot_data_.get_walk_cmd()
        new_filtered_x_speed = 1 * x_speed_command + (1 - 1) * self.filtered_x_speed
        change = new_filtered_x_speed - self.filtered_x_speed
        change = np.clip(change, -0.005, 0.005)
        self.filtered_x_speed = self.filtered_x_speed + change
        command = np.concatenate([
            np.array([
                x_speed_command,
                y_speed_command,
                yaw_speed_command,
            ], dtype=np.float32),
        ])

        ang_vel = self.robot_data_.get_angular_velocity()
        q_mj = self.robot_data_.get_joint_pos()   # mj 顺序，长度 29
        dq_mj = self.robot_data_.get_joint_vel()

        # 只取 motion_num 个受控关节，变成 lab 顺序
        qj = q_mj[self.lab2mj]
        dqj = dq_mj[self.lab2mj]

        qj = qj - self.default_angles_lab


        # Observation = ang_vel(3) + gravity(3) + command(3) + q(motion_num) + dq(motion_num) + action(motion_num) + gait(6)= 84
        proprio = np.concatenate([
            ang_vel ,              # 3 elements
            gravity_init,
            command,
            qj,
            dqj,
            self.last_actions_,
            gait
        ])

        # History buffer management exactly like C++
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                start_idx = i * self.obs_size_
                end_idx = start_idx + self.obs_size_
                self.proprio_hist_buf_[start_idx:end_idx] = proprio
            self.is_first_obs_ = False
        else:
            # Shift history: head((num_hist-1)*obs_size) = tail((num_hist-1)*obs_size)
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift_size:] = proprio

        # Clip observations exactly like C++
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
        if not hasattr(self, 'inference_engine_') or self.inference_engine_ is None:
            # 每5秒输出一次警告防止刷屏
            if self.robot_data_.time_now_ - self._last_engine_warn_time > 5.0:
                print(f"\033[91m[FSMStateHOMIE] WARNING: No inference engine! Cannot compute actions at t={self.robot_data_.time_now_:.1f}s\033[0m")
                self._last_engine_warn_time = self.robot_data_.time_now_
            return

        try:
            # Prepare input tensor
            input_data = self.observations_.reshape(1, -1).astype(np.float32)

            # Inference using unified engine interface
            output_data = self.inference_engine_.infer(input_data)[0]

            # Extract and clip actions exactly like C++
            for i in range(self.action_num_):
                self.actions_[i] = np.clip(output_data[i], -self.clip_act_, self.clip_act_)

            if self.is_first_action_:
                print("[FSMStateHOMIE] First Observation:")
                for i in range(self.obs_size_):
                    print(f"{self.observations_[i]:.6f} ", end="")
                print()
                self.is_first_action_ = False

        except Exception as e:
            print(f"[FSMStateHOMIE] Inference error: {e}")

    def on_exit(self):
        """退出WALKAMP状态"""
        print("[FSMStateHOMIE] exit")
        # 释放推理引擎
        if hasattr(self, 'inference_engine_') and self.inference_engine_ is not None:
            try:
                self.inference_engine_.unload()
            except Exception as e:
                print(f"[FSMStateHOMIE] failed to unload inference engine: {e}")
        # 关掉 obs 日志文件（如果存在）
        obs_log_file = getattr(self, "obs_log_file", None)
        if obs_log_file is not None:
            try:
                obs_log_file.flush()
                obs_log_file.close()
                obs_log_path = getattr(self, "obs_log_path", "unknown")
                print(f"[FSMStateHOMIE] obs log saved to {obs_log_path}")
            except Exception as e:
                print(f"[FSMStateHOMIE] failed to close obs log: {e}")
            self.obs_log_file = None

    def check_transition(self, flag: ControlFlag) -> Optional[FSMStateName]:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        else:
            return None  # 无状态转换
