"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np

from typing import Optional
from FSM.fsm_base import FSMState, FSMStateName
from common import ControlFlag, RobotData, get_robot_config_manager
from inference_engine import EngineFactory, EngineType, InferenceEngineBase
from config.deploy_config_manager import DeployConfigManager
import os
import yaml
from scipy.spatial.transform import Rotation

class FSMStateHOMIE(FSMState):
    """HOMIE策略状态实现 - Actor-Critic网络推理
    
    训练配置:
        - 观测维度: 62 (commands(3)*scale + height_cmd(1) + ang_vel(3) + gravity(3) + dof_pos(20) + dof_vel(20) + actions(12))
        - 动作维度: 12 (仅腿部关节)
        - 历史长度: 6
        - commands_scale: [2.0, 2.0, 0.5]
    """
    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        config_path = os.path.join(current_dir, "config", "homie.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[FSMStateHOMIE] Cannot find config file: {config_path}")
        
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        
        print(f"[FSMStateHOMIE] ===============================================")
        print(f"[FSMStateHOMIE] ============ __init__ Start ===========")
        print(f"[FSMStateHOMIE] ===============================================")

        print(f"\n[FSMStateHOMIE] Config file loaded from: {config_path}")

        # ========== Size Config ==========
        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist', 6)
        self.obs_size_ = size_config.get('observations_size', 62)
        print(f"\n[FSMStateHOMIE] [Size Config]")
        print(f"  - num_hist_: {self.num_hist_}")
        print(f"  - obs_size_: {self.obs_size_}")

        # ========== Control Config ==========
        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale', 0.25)
        self.decimation_ = control_config.get('decimation', 1)
        print(f"\n[FSMStateHOMIE] [Control Config]")
        print(f"  - action_scale_: {self.action_scale_}")
        print(f"  - decimation_: {self.decimation_}")

        # ========== Normalization Config ==========
        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})
        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 100.0)
        print(f"\n[FSMStateHOMIE] [Normalization Config]")
        print(f"  - clip_obs_: {self.clip_obs_}")
        print(f"  - clip_act_: {self.clip_act_}")
        
        # ========== Special Config (from yaml) ==========
        special_config = policy_config.get('special', {})
        yaml_kp = special_config.get('kp', np.zeros(20, dtype=np.float32))
        yaml_kd = special_config.get('kd', np.zeros(20, dtype=np.float32))
        yaml_zero_pos = special_config.get('zero_pos', np.zeros(20, dtype=np.float32))
        print(f"\n[FSMStateHOMIE] [Special Config from yaml]")
        print(f"  - yaml_kp (from special.kp): {yaml_kp}")
        print(f"  - yaml_kd (from special.kd): {yaml_kd}")
        print(f"  - yaml_zero_pos (from special.zero_pos): {yaml_zero_pos}")

        # 观测缩放因子（与训练配置一致）
        self.commands_scale = np.array([2.0, 2.0, 0.5], dtype=np.float32)
        self.ang_vel_scale_ = 0.5   # obs_scales.ang_vel
        self.dof_vel_scale_ = 0.05   # obs_scales.dof_vel
        self.dof_pos_scale_ = 1.0   # obs_scales.dof_pos
        print(f"\n[FSMStateHOMIE] [Observation Scale]")
        print(f"  - commands_scale: {self.commands_scale}")
        print(f"  - ang_vel_scale_: {self.ang_vel_scale_}")
        print(f"  - dof_vel_scale_: {self.dof_vel_scale_}")
        print(f"  - dof_pos_scale_: {self.dof_pos_scale_}")

        # ========== Robot Config Manager ==========
        print(f"\n[FSMStateHOMIE] [Robot Config Manager]")
        try:
            self.config_manager = get_robot_config_manager()
            print(f"  - config_manager acquired successfully")
        except Exception as e:
            print(f"  - Failed to get config manager: {e}")
        
        if self.config_manager:
            self.motor_num_ = self.config_manager.motor_num
            self.dt_ = self.config_manager.config.dt
            self.joint_xml = self.config_manager.get_joint_xml()
            n_mj = len(self.joint_xml)
            
            # NOTE: 使用从yaml读取的kp/kd/zero_pos，而不是零数组
            # 如果yaml中没有配置，则使用零数组
            self.stiffness_array = np.array(yaml_kp, dtype=np.float32) if len(yaml_kp) >= n_mj else np.zeros(n_mj, dtype=np.float32)
            self.damping_array = np.array(yaml_kd, dtype=np.float32) if len(yaml_kd) >= n_mj else np.zeros(n_mj, dtype=np.float32)
            self.default_joint_pos = np.array(yaml_zero_pos, dtype=np.float32) if len(yaml_zero_pos) >= n_mj else np.zeros(n_mj, dtype=np.float32)
            
            print(f"  - robot_name: {self.config_manager.robot_name}")
            print(f"  - motor_num_: {self.motor_num_}")
            print(f"  - dt_: {self.dt_}")
            print(f"  - joint_xml ({len(self.joint_xml)} joints): {self.joint_xml}")
            print(f"  - stiffness_array shape: {self.stiffness_array.shape}, values: {self.stiffness_array}")
            print(f"  - damping_array shape: {self.damping_array.shape}, values: {self.damping_array}")
            print(f"  - default_joint_pos shape: {self.default_joint_pos.shape}, values: {self.default_joint_pos}")
        else:
            # 没有config_manager时使用yaml默认值
            self.stiffness_array = np.array(yaml_kp, dtype=np.float32)
            self.damping_array = np.array(yaml_kd, dtype=np.float32)
            self.default_joint_pos = np.array(yaml_zero_pos, dtype=np.float32)
            self.motor_num_ = len(self.stiffness_array)
            self.dt_ = 0.01
            self.joint_xml = [f"joint_{i}" for i in range(self.motor_num_)]
            print(f"  - Using yaml defaults (no config_manager)")
            print(f"  - motor_num_: {self.motor_num_}")
            print(f"  - dt_: {self.dt_}")

        self.action_num_ = 12
        self.leg_joint_indices = np.arange(12, dtype=int)
        self.arm_joint_indices = np.arange(12, 20, dtype=int)
        print(f"\n[FSMStateHOMIE] [Action Config]")
        print(f"  - action_num_: {self.action_num_}")
        print(f"  - leg_joint_indices: {self.leg_joint_indices}")
        print(f"  - arm_joint_indices: {self.arm_joint_indices}")
        print(f"  - joint_xml length: {len(self.joint_xml)}")
        
        if np.isscalar(self.action_scale_):
            self.action_scale = np.full(self.action_num_, float(self.action_scale_), dtype=np.float32)
        else:
            self.action_scale = np.array(self.action_scale_[:self.action_num_], dtype=np.float32)
        print(f"  - action_scale: {self.action_scale}")

        # ========== Inference Engine ==========
        self._init_inference_engine(policy_config, current_dir)

        # ========== State Buffers ==========
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)
        print(f"\n[FSMStateHOMIE] [State Buffers]")
        print(f"  - observations_ shape: {self.observations_.shape}")
        print(f"  - proprio_hist_buf_ shape: {self.proprio_hist_buf_.shape}")
        print(f"  - last_actions_ shape: {self.last_actions_.shape}")
        print(f"  - actions_ shape: {self.actions_.shape}")
        print(f"  - _warm_start_pose shape: {self._warm_start_pose.shape}")

        warm_start_time = policy_config.get('warm_start_time', 0.3)
        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if warm_start_time > 0 and step > 0:
            self._warm_start_steps = max(1, int(warm_start_time / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0
        print(f"\n[FSMStateHOMIE] [Warm Start Config]")
        print(f"  - warm_start_time: {warm_start_time}")
        print(f"  - step (decimation * dt): {step}")
        print(f"  - _warm_start_steps: {self._warm_start_steps}")

        self.is_first_obs_ = True
        self.is_first_action_ = True
        self._last_engine_warn_time = 0.0
        self.filtered_x_speed = 0

        print(f"\n[FSMStateHOMIE] ===============================================")
        print(f"[FSMStateHOMIE] ============ __init__ End ===========")
        print(f"[FSMStateHOMIE] ===============================================")
        print(f"[FSMStateHOMIE] Summary:")
        print(f"  - Total input size: {self.obs_size_ * self.num_hist_}")
        print(f"  - Total output size: {self.action_num_}")
        print(f"  - Motor num: {self.motor_num_}")

    def _reset_internal_state(self):
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self._warmup_inference_counter = 0
        
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        self.robot_data_.q_d_[base:base + len(self.joint_xml)] = self.default_joint_pos
        self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
        self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

    def _init_inference_engine(self, policy_config: dict, current_dir: str):
        engine_type = policy_config.get('engine_type', 'onnx').lower()
        model_config = policy_config.get('model', {})
        engine_config = {'type': engine_type}
        
        def check_model_file(path: str) -> bool:
            if not os.path.exists(path):
                print(f"\033[91m[FSMStateHOMIE] ERROR: Model file does NOT exist: {path}\033[0m")
                return False
            if os.path.getsize(path) == 0:
                print(f"\033[91m[FSMStateHOMIE] ERROR: Model file is empty: {path}\033[0m")
                return False
            return True
        
        if engine_type == 'onnx':
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
            engine_config['graph_optimization_level'] = onnx_cfg.get('graph_optimization_level', 'ORT_ENABLE_ALL')
            
        elif engine_type == 'openvino':
            ov_cfg = policy_config.get('openvino', {})
            if not ov_cfg:
                raise ValueError("[FSMStateHOMIE] Missing 'openvino' config block in homie.yaml")
            model_file = os.path.join(current_dir, "model", ov_cfg.get('model_path', ''))
            check_model_file(model_file)
            engine_config['model_path'] = model_file
            engine_config['device'] = ov_cfg.get('device', 'CPU')
            engine_config['cache_dir'] = ov_cfg.get('cache_dir', None)
            
        elif engine_type == 'pytorch':
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
        self._reset_internal_state()
        print("[FSMStateHOMIE] enter")
        self._last_engine_warn_time = 0.0

        if not hasattr(self, 'inference_engine_') or self.inference_engine_ is None:
            print("\033[91m[FSMStateHOMIE] WARNING: Inference engine NOT loaded!\033[0m")
        else:
            if not self.inference_engine_.is_loaded:
                try:
                    self.inference_engine_.load()
                    print("[FSMStateHOMIE] Inference engine reloaded")
                except Exception as e:
                    print(f"\033[91m[FSMStateHOMIE] Failed to reload inference engine: {e}\033[0m")
                    
        if self.inference_engine_ and self.inference_engine_.is_loaded:
            print(f"[FSMStateWALKAMP] ✅ Inference engine ready, running normally")
        
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)

    def run(self, flag: ControlFlag):
        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:
            self.compute_observation(flag)
            self.compute_actions()
            
            target_pos = self.robot_data_.get_joint_pos().copy()
            target_pos[self.leg_joint_indices] = (
                self.actions_ * self.action_scale + self.default_joint_pos[self.leg_joint_indices]
            )
            commanded_pos = target_pos
            if self._warm_start_steps > 0 and self._warmup_inference_counter < self._warm_start_steps:
                self._warmup_inference_counter += 1
                blend = self._warmup_inference_counter / float(self._warm_start_steps)
                commanded_pos = (1.0 - blend) * self._warm_start_pose + blend * target_pos
            
            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            
            # DEBUG: 打印目标位置
            if self.is_first_action_:
                print(f"[DEBUG] target_pos[leg]: {target_pos[self.leg_joint_indices]}")
                print(f"[DEBUG] commanded_pos[leg]: {commanded_pos[self.leg_joint_indices]}")
                print(f"[DEBUG] base index: {base}, motor_num: {self.motor_num_}")
                print(f"[DEBUG] joint_xml length: {len(self.joint_xml)}")
                print(f"[DEBUG] stiffness_array[:6]: {self.stiffness_array[:6]}")
                self.is_first_action_ = False
            
            self.robot_data_.q_d_[base:base + len(self.joint_xml)] = commanded_pos
            self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0
            
            self.last_actions_[:] = self.actions_
        
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array

    def compute_observation(self, flag: ControlFlag):
        roll, pitch, yaw = (
                        float(self.robot_data_.imu_data_[2]),
                        float(self.robot_data_.imu_data_[1]),
                        float(self.robot_data_.imu_data_[0]),
                    )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init   = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0.,0.,-1.], dtype=np.float32))
        
        # print(f"[DIAG] imu_data raw: {self.robot_data_.imu_data_}")
        # print(f"[DIAG] roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
        # print(f"[DIAG] gravity_init: {gravity_init}")
        
        # gravity = self.robot_data_.get_project_gravity()      # 测试好像有问题
        
        # walk_cmd = self.robot_data_.get_walk_cmd()
        # command = (walk_cmd * self.commands_scale).astype(np.float32) #测试好像有问题
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
        command = (command * self.commands_scale).astype(np.float32)
        height_cmd = np.array([getattr(flag, 'height_cmd', getattr(flag, 'walk_height_command', 0.9))], dtype=np.float32)
        
        # 应用观测缩放因子（与训练配置一致）
        ang_vel = self.robot_data_.get_angular_velocity() * self.ang_vel_scale_
        
        q_mj = self.robot_data_.get_joint_pos()
        dq_mj = self.robot_data_.get_joint_vel() * self.dof_vel_scale_
        
        # DEBUG: 打印关节数据维度
        # print(f"[DEBUG] q_mj shape: {q_mj.shape}, dq_mj shape: {dq_mj.shape}")
        # print(f"[DEBUG] q_mj[:6]: {q_mj[:6]}")
        # print(f"[DEBUG] dq_mj[:6]: {dq_mj[:6]}")
        
        qj = (q_mj - self.default_joint_pos) * self.dof_pos_scale_
        
        proprio = np.concatenate([
            command,
            height_cmd,
            ang_vel,
            gravity_init,
            qj,
            dq_mj,
            self.last_actions_,
        ])

        # DEBUG: 打印 proprio 维度
        # print(f"[DEBUG] proprio shape: {proprio.shape}, expected: {self.obs_size_}")

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
            if self.robot_data_.time_now_ - self._last_engine_warn_time > 5.0:
                print(f"\033[91m[FSMStateHOMIE] WARNING: No inference engine! Cannot compute actions at t={self.robot_data_.time_now_:.1f}s\033[0m")
                self._last_engine_warn_time = self.robot_data_.time_now_
            return

        # print(f"[DIAG] observations_[:10]: {self.observations_[:10]}")
        # print(f"[DIAG] observations_[-15:]: {self.observations_[-15:]}")
        # print(f"[DIAG] observations_ min/max: {self.observations_.min():.4f}/{self.observations_.max():.4f}")
    
        try:
            input_data = self.observations_.reshape(1, -1).astype(np.float32)
            output_data = self.inference_engine_.infer(input_data)[0]

            # DEBUG: 打印推理输出
            # print(f"[DEBUG] Infer output shape: {output_data.shape}, values: {output_data[:6]}")

            for i in range(self.action_num_):
                self.actions_[i] = np.clip(output_data[i], -self.clip_act_, self.clip_act_)

            # DEBUG: 打印动作输出
            # print(f"[DEBUG] Actions: {self.actions_}")

            if self.is_first_action_:
                print("[FSMStateHOMIE] First Observation:")
                for i in range(self.obs_size_):
                    print(f"[FSMStateHOMIE]  the {i}th observation is {self.observations_[i]:.6f} ", end="")
                print()
                self.is_first_action_ = False
                
            # print(f"[FSMStateHOMIE] Actions computed at t={self.robot_data_.time_now_:.2f}s: {self.actions_}")

        except Exception as e:
            print(f"[FSMStateHOMIE] Inference error: {e}")

    def on_exit(self):
        print("[FSMStateHOMIE] exit")
        if hasattr(self, 'inference_engine_') and self.inference_engine_ is not None:
            try:
                self.inference_engine_.unload()
            except Exception as e:
                print(f"[FSMStateHOMIE] failed to unload inference engine: {e}")
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
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoHOMIE":
            return FSMStateName.HOMIE
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoWALKAMP":
            return FSMStateName.WALKAMP
        else:
            return None
