"""
Robot Interface Implementation
机器人接口具体实现
"""
from __future__ import annotations
import queue
from common import PeekableQueue, RobotData, ControlFlag
import yaml
import os
from typing import Dict, Any, Optional

import numpy as np
# ROS messages
from bodyctrl_msgs.msg import MotorStatusMsg, CmdMotorCtrl, MotorCtrl, Imu, MotorStatus
import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import transforms3d as t3d
import functools
import time
import math
from sptlib_python import funcSPTrans as FuncSPTrans

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from FSM.fsm_base import FSMStateName
from config.robot_config_manager import RobotConfigManager

# 导入基类
from Robot.robot_inferface_base import RobotInterface


def timing_decorator(func):
    """
    装饰器：记录函数执行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # print(f"[TIMING] {func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper


class RobotInterfaceImpl(RobotInterface):
    """机器人接口具体实现"""

    def __init__(self, robot_data: RobotData, config_path: str = 'tienkung3_dex_config.yaml'):
        super().__init__(robot_data)
        self.initialized = False
        self.node = None
        self.config_path = config_path

        # 初始化配置管理器（传入配置文件路径，自动加载profile）
        self.config_manager = RobotConfigManager(config_path)

        # 从配置管理器获取关键参数
        self.sim = self.config_manager.sim
        self.debug = self.config_manager.debug
        self.floating_base_dof = self.config_manager.floating_base_dof

        # ID映射（延迟到init方法中通过配置管理器动态创建）
        self.id_map = None

        # 消息队列
        self.queue_leg_motor_state = PeekableQueue(maxsize=1)
        self.queue_arm_motor_state = PeekableQueue(maxsize=1)
        self.queue_waist_motor_state = PeekableQueue(maxsize=1)
        self.queue_imu_xsens = PeekableQueue(maxsize=1)
        self.queue_walk_cmd = PeekableQueue(maxsize=1)

        self.whole_joint_nums = 0

        # 临时变量用于优化计算
        self.temp_q = None
        # 预分配另一个用于存储中间计算的临时数组
        self._temp_zero_cnt = None

        # 电机控制参数
        self.motor_dir = None
        self.zero_cnt = None
        self.zero_offset = None

        # 添加标志位，用于跟踪是否是首次接收数据
        self.first_leg_data_received = False
        self.first_arm_data_received = False
        self.first_waist_data_received = False

        # 关节限位
        self.joint_limits = {}
        self.joint_pos_threshold = math.pi

        # 串并联转换器
        self.fun_s2p = FuncSPTrans()

        # 串并联转换相关变量（延迟初始化）
        self.left_ankle_indices = None
        self.right_ankle_indices = None
        self.q_a_p = np.zeros(4)  # 并联关节位置
        self.qdot_a_p = np.zeros(4)  # 并联关节速度
        self.tor_a_p = np.zeros(4)  # 并联关节力矩
        self.ankle_kp_p = np.zeros(4)  # 并联关节刚度
        self.ankle_kd_p = np.zeros(4)  # 并联关节阻尼

        # TF相关属性
        self.tf_buffer = None
        self.tf_listener = None
        
        # ROS publishers and subscribers
        self.pub_leg_motor_cmd = None
        self.pub_arm_motor_cmd = None
        self.pub_waist_motor_cmd = None
        self.sub_leg_state = None
        self.sub_arm_state = None
        self.sub_waist_state = None

        # 当前机器人所处状态
        self.current_state: FSMStateName = FSMStateName.STOP

    def update_fsm(self, current_state: Optional[FSMStateName] = None):
        """更新FSM接口"""
        if current_state is not None:
            self.current_state = current_state

    def _apply_config_from_manager(self):
        """从配置管理器应用配置到当前实例"""
        # 运行时模式
        self.sim = self.config_manager.sim
        self.debug = self.config_manager.debug
        self.floating_base_dof = self.config_manager.floating_base_dof
        
        # robot_interface 配置
        self.clip_actions = self.config_manager.clip_actions
        self.disable_joints_ = self.config_manager.disable_joints
        
        # 加载控制状态
        self._load_control_status()
        
        # 电机参数
        self.zero_pos = self.config_manager.get_zero_pos()
        self.ct_scale = self.config_manager.get_ct_scale()
        
        # IMU
        self.xsense_roll_offset = self.config_manager.xsense_roll_offset
        
        # 关节限位
        self._load_joint_limits()
        
        # 并联脚踝参数
        self.ankle_kp_p = self.config_manager.ankle_kp_p
        self.ankle_kd_p = self.config_manager.ankle_kd_p

    def _load_control_status(self):
        """从配置管理器加载控制状态"""
        # 字符串命令到枚举值的映射
        state_to_FSMState = {
            "STOP": FSMStateName.STOP,
            "ZERO": FSMStateName.ZERO,
            "WALKAMP": FSMStateName.WALKAMP,
            "WALKAMP_OV": FSMStateName.WALKAMP_OV,
        }
        self.waist_control_status = [state_to_FSMState[state] for state in self.config_manager.waist_control_status]
        self.legs_control_status = [state_to_FSMState[state] for state in self.config_manager.legs_control_status]
        self.arms_control_status = [state_to_FSMState[state] for state in self.config_manager.arms_control_status]
        self.left_arm_only_status = [state_to_FSMState[state] for state in self.config_manager.left_arm_only_status]
        self.right_arm_only_status = [state_to_FSMState[state] for state in self.config_manager.right_arm_only_status]

    def _load_joint_limits(self):
        """从配置管理器加载关节限位值"""
        # 从配置管理器获取关节限位信息
        joint_limits_config = self.config_manager.get_joint_limits()

        if not joint_limits_config:
            error_msg = "No joint_limits section in config"
            # self.node.get_logger().error(error_msg)
            print(error_msg)
            raise ValueError(error_msg)
        else:
            # 从配置中加载限位值
            for joint_name, limits in joint_limits_config.items():
                if 'min' in limits and 'max' in limits:
                    self.joint_limits[joint_name] = {
                        'min': float(limits['min']),
                        'max': float(limits['max'])
                    }
                else:
                    error_msg = f"Invalid limits for joint {joint_name}"
                    print(error_msg)
                    # self.node.get_logger().error(error_msg)
                    raise ValueError(error_msg)

            # self.node.get_logger().info(f"Loaded joint limits from config manager")
            # print(f"Loaded joint limits from config manager: {self.joint_limits}")
        
        # 记录加载的限位值
        # for joint_name, limits in self.joint_limits.items():
        #     self.node.get_logger().debug(
        #         f"Joint {joint_name}: [{limits['min']}, {limits['max']}]")
        # print("-" * 30 + '关节限位值' + '-' * 30)
        # print(self.joint_limits)

    def init(self, node: Node):
        """初始化接口"""
        self.node = node
        self.initialized = True

        # 从配置管理器应用配置
        self._apply_config_from_manager()
        
        # 动态创建ID映射（通过配置管理器加载对应的joint_id_map模块）
        self.id_map = self.config_manager.create_id_map()
        
        # 初始化依赖id_map的变量
        self.whole_joint_nums = self.id_map.whole_motor_nums + self.floating_base_dof
        self.temp_q = np.empty(self.id_map.whole_motor_nums)
        self._temp_zero_cnt = np.empty(self.id_map.whole_motor_nums)
        self.motor_dir = np.ones(self.id_map.whole_motor_nums)
        self.zero_cnt = np.zeros(self.id_map.whole_motor_nums)
        self.zero_offset = np.zeros(self.id_map.whole_motor_nums)
        
        # 预计算ID到限位的映射
        self.id_to_limits = {}
        for joint_name, limits in self.joint_limits.items():
            index = self.id_map.get_index_by_name(joint_name)
            if index >= 0:
                motor_id = self.id_map.get_id_by_index(index)
                self.id_to_limits[motor_id] = limits
        
        # 初始化串并联转换关节索引（根据id_map动态计算）
        # 注意：这里假设踝关节在关节顺序中的位置是相对于腿部的索引
        # 左踝关节: 4 (pitch), 5 (roll) 
        # 右踝关节: 10 (pitch), 11 (roll)
        self.left_ankle_indices = np.array([4, 5]) + self.floating_base_dof
        self.right_ankle_indices = np.array([10, 11]) + self.floating_base_dof
        
        # 初始化ROS接口
        self._init_ros_interfaces()
        
        node.get_logger().info("Robot interface initialized")

    def _init_ros_interfaces(self):
        """初始化ROS接口"""
        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.RELIABLE,
            # history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # 发布者
        self.pub_leg_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/leg/cmd_ctrl', qos_profile)
        self.pub_arm_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/arm/cmd_ctrl', qos_profile)
        self.pub_waist_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/waist/cmd_ctrl', qos_profile)

        # 订阅者
        self.sub_leg_state = self.node.create_subscription(
            MotorStatusMsg, '/leg/status', self._leg_motor_status_callback,
            qos_profile)
        self.sub_arm_state = self.node.create_subscription(
            MotorStatusMsg, '/arm/status', self._arm_motor_status_callback,
            qos_profile)
        self.sub_waist_state = self.node.create_subscription(
            MotorStatusMsg, '/waist/status', self._waist_motor_status_callback,
            qos_profile)

        #（非电机相关）
        self.sub_imu_xsens = self.node.create_subscription(
            Imu, '/imu/status', self._imu_status_callback, qos_profile)


    # @timing_decorator
    def get_imu_data(self):
        """处理传感器数据（非电机）"""
        # 处理IMU数据
        while True:
            try:
                msg = self.queue_imu_xsens.peek()
                self.robot_data_.imu_data_[0] = msg.euler.yaw
                self.robot_data_.imu_data_[1] = msg.euler.pitch
                self.robot_data_.imu_data_[2] = msg.euler.roll
                self.robot_data_.imu_data_[3] = msg.angular_velocity.x
                self.robot_data_.imu_data_[4] = msg.angular_velocity.y
                self.robot_data_.imu_data_[5] = msg.angular_velocity.z
                self.robot_data_.imu_data_[6] = msg.linear_acceleration.x
                self.robot_data_.imu_data_[7] = msg.linear_acceleration.y
                self.robot_data_.imu_data_[8] = msg.linear_acceleration.z
                break
            except queue.Empty:
                time.sleep(0.0001)

    def update_robot_state(self):
        """读取电机状态数据更新为机器人状态数据"""
        # 处理腿部电机状态
        while True:
            try:
                msg = self.queue_leg_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                    print(f"Time difference: {time_diff_ms} ms")
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_leg_data_received)
                break
            except queue.Empty:
                time.sleep(0.0001)

        # 处理手臂电机状态
        while True:
            try:
                msg = self.queue_arm_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                    print(f"Time difference: {time_diff_ms} ms")
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_arm_data_received)

                break
            except queue.Empty:
                time.sleep(0.0001)

        # 处理腰部电机状态
        while True:
            try:
                msg = self.queue_waist_motor_state.peek()
                if self.debug:
                    current_time = self.node.get_clock().now().to_msg()
                    msg_time = msg.header.stamp
                    time_diff = (current_time.sec -
                                 msg_time.sec) * 1000000000 + (
                                     current_time.nanosec - msg_time.nanosec)
                    time_diff_ms = time_diff / 1000000.0
                    print(f"Time difference: {time_diff_ms} ms")
                for status in msg.status:
                    self.motor_state_to_robot_state(
                        status, self.first_waist_data_received)
                break
            except queue.Empty:
                time.sleep(0.0001)

    def motor_state_to_robot_state(self, status, received_flag: bool):
        index = self.id_map.get_index_by_id(status.name)
        if index >= 0:
            robotdata_index = index + self.floating_base_dof  # 偏移到完整关节数组中
            # 直接赋值
            self.robot_data_.q_a_[robotdata_index] = status.pos
            self.robot_data_.q_dot_a_[robotdata_index] = status.speed
            self.robot_data_.tau_a_[
                robotdata_index] = status.current * self.ct_scale[min(
                    index,
                    len(self.ct_scale) - 1)]
            self.robot_data_.temperature_a[
                robotdata_index - self.floating_base_dof] = status.temperature

            self.robot_data_.q_a_[robotdata_index] = (
                status.pos - self.zero_pos[index]
            ) * self.motor_dir[index] + self.zero_offset[index]
            if self.robot_data_.q_a_[robotdata_index] > math.pi:
                self.zero_cnt[index] = -1.0
            elif self.robot_data_.q_a_[robotdata_index] < -math.pi:
                self.zero_cnt[index] = 1.0

            self.robot_data_.q_a_[
                robotdata_index] += self.zero_cnt[index] * 2.0 * math.pi
            self.robot_data_.q_dot_a_[robotdata_index] *= self.motor_dir[index]
            self.robot_data_.tau_a_[robotdata_index] *= self.motor_dir[index]

            if not received_flag or abs(
                    self.robot_data_.q_a_[robotdata_index] -
                    self.robot_data_.q_a_last[robotdata_index]
            ) > self.joint_pos_threshold:
                if received_flag:
                    self.node.get_logger().warn(
                        f"Joint {index} error detected")
                    self.robot_data_.q_a_[
                        robotdata_index] = self.robot_data_.q_a_last[
                            robotdata_index]
                    self.robot_data_.q_dot_a_[
                        robotdata_index] = self.robot_data_.qdot_a_last[
                            robotdata_index]
                    self.robot_data_.tau_a_[
                        robotdata_index] = self.robot_data_.tor_a_last[
                            robotdata_index]
                else:
                    # 首次接收数据，更新标志位
                    received_flag = True
            self.robot_data_.q_a_last[robotdata_index] = self.robot_data_.q_a_[
                robotdata_index]
            self.robot_data_.qdot_a_last[
                robotdata_index] = self.robot_data_.q_dot_a_[robotdata_index]
            self.robot_data_.tor_a_last[
                robotdata_index] = self.robot_data_.tau_a_[robotdata_index]

    def update_sensor_states(self):
        # 获取Imu数据
        self.get_imu_data()
        # 添加IMU偏移
        self.robot_data_.imu_data_[2] += self.xsense_roll_offset

    def update_robot_data(self, flag: ControlFlag, time_passed: float):
        # 更新传感器数据
        self.update_sensor_states()
        # 更新电机状态数据
        self.update_robot_state()
        # 更新机器人控制命令
        self.update_robot_cmd(flag)
        # 脚踝并联转串联
        self.backup_serial_cmd()
        if not self.sim:
            #真机
            self.ankle_parallel_to_serial()

        # 更新robot_data 时间戳
        self.robot_data_.time_now_ = time_passed

    def backup_serial_cmd(self):
        self.robot_data_.q_d_s_ = self.robot_data_.q_d_.copy()

    def _check_and_clip_joint_limits_fast(
            self, cmd_name: int, position: float) -> tuple[bool, float]:
        """
        快速检查并修正关节位置限位（避免重复查询）
        """
        if not self.sim and self.clip_actions:
            limit = self.id_to_limits[cmd_name]
            clipped_pos = np.clip(position, limit["min"], limit["max"])
            return clipped_pos == position, clipped_pos
        else:
            return True, position

    @timing_decorator
    def convert_to_motor_commands(self):
        """将机器人控制命令转换为电机控制命令"""
        # 使用切片操作一次性复制数据，避免逐个元素赋值
        q_d_reordered = self.robot_data_.q_d_[self.floating_base_dof:]
        qdot_d_reordered = self.robot_data_.q_dot_d_[self.floating_base_dof:]
        tor_d_reordered = self.robot_data_.tau_d_[self.floating_base_dof:]

        # 计算 q_d_reordered - self.zero_offset
        np.subtract(q_d_reordered, self.zero_offset, out=self.temp_q)
        # 计算 self.zero_cnt * 2.0 * self.pi
        np.multiply(self.zero_cnt, 2.0 * math.pi, out=self._temp_zero_cnt)
        # 计算 q_d_reordered - self.zero_offset - self.zero_cnt * 2.0 * self.pi
        np.subtract(self.temp_q, self._temp_zero_cnt, out=self.temp_q)
        # 计算 (q_d_reordered - self.zero_offset - self.zero_cnt * 2.0 * self.pi) * self.motor_dir
        np.multiply(self.temp_q, self.motor_dir, out=self.temp_q)
        # 计算最终结果 (q_d_reordered - self.zero_offset - self.zero_cnt * 2.0 * self.pi) * self.motor_dir + self.zero_pos
        np.add(self.temp_q,
               self.zero_pos,
               out=self.robot_data_.q_d_[self.floating_base_dof:])

        # qdot_d和tor_d的计算也可以向量化
        np.multiply(qdot_d_reordered,
                    self.motor_dir,
                    out=self.robot_data_.q_dot_d_[self.floating_base_dof:])
        np.multiply(tor_d_reordered,
                    self.motor_dir,
                    out=self.robot_data_.tau_d_[self.floating_base_dof:])

        # 如果关节被禁用
        if self.disable_joints_:
            self.robot_data_.joint_kp_p_.fill(0.0)
            self.robot_data_.joint_kd_p_.fill(0.0)
            self.robot_data_.tau_d_.fill(0.0)
            self.node.get_logger().warn("Joints disabled!")

    def publish_motor_commands(self, flag: ControlFlag):
        # flag_fsm_command = flag.fsm_state_command
        current_state = self.current_state
        # 发布腿部控制命令
        if self.legs_control_status == [] or current_state in self.legs_control_status:
            leg_msg = CmdMotorCtrl()
            leg_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.leg_motor_nums):
                cmd = MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(self.robot_data_.joint_kp_p_[i])
                cmd.kd = float(self.robot_data_.joint_kd_p_[i])
                cmd.pos = float(self.robot_data_.q_d_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[i + self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[i + self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    print(
                        f"Joint (id: {cmd.name}) position {cmd.pos} is out of limits"
                    )
                leg_msg.cmds.append(cmd)
            self.pub_leg_motor_cmd.publish(leg_msg)

        # 只在特定模式下控制腰部
        if current_state in self.waist_control_status:
            # 腰部控制命令
            waist_msg = CmdMotorCtrl()
            waist_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.waist_motor_nums):
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums
                cmd.name = self.id_map.get_id_by_index(
                    motor_idx)  # 12 -> 33(yaw)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    print(
                        f"Joint (id: {cmd.name}) position {cmd.pos} is out of limits"
                    )
                waist_msg.cmds.append(cmd)
            # print(f'waist_msg {waist_msg}')
            self.pub_waist_motor_cmd.publish(waist_msg)

        # 只在特定模式下控制手臂
        if current_state in self.arms_control_status:
            # 手臂控制命令
            arm_msg = CmdMotorCtrl()
            arm_msg.header.stamp = self.node.get_clock().now().to_msg()
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7, self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(self.robot_data_.joint_kp_p_[motor_idx])
                cmd.kd = float(self.robot_data_.joint_kd_p_[motor_idx])
                cmd.pos = float(self.robot_data_.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    self.robot_data_.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    self.robot_data_.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    print(
                        f"Joint (id: {cmd.name}) position {cmd.pos} is out of limits"
                    )
                arm_msg.cmds.append(cmd)
            # print(f'arm_msg {arm_msg}')
            self.pub_arm_motor_cmd.publish(arm_msg)

    @timing_decorator
    def send_motor_commands(self, flag: ControlFlag):
        """发布电机控制命令"""
        if not self.initialized:
            return
        if not self.sim:
            #真机
            self.ankle_serial_to_parallel()
        self.convert_to_motor_commands()
        self.publish_motor_commands(flag)

    # ROS回调函数
    def _leg_motor_status_callback(self, msg):
        """腿部电机状态回调"""
        try:
            self.queue_leg_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_leg_motor_state.get_nowait()  # 移除旧数据
                self.queue_leg_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _arm_motor_status_callback(self, msg):
        """手臂电机状态回调"""
        try:
            self.queue_arm_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_arm_motor_state.get_nowait()  # 移除旧数据
                self.queue_arm_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _waist_motor_status_callback(self, msg):
        """腰部电机状态回调"""
        try:
            self.queue_waist_motor_state.put_nowait(msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_waist_motor_state.get_nowait()  # 移除旧数据
                self.queue_waist_motor_state.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _imu_status_callback(self, msg):
        """IMU状态回调"""
        try:
            self.queue_imu_xsens.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_imu_xsens.get_nowait()  # 移除旧数据
                self.queue_imu_xsens.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略


    def ankle_parallel_to_serial(self):
        # 串并联转换：并转串 (类似C++版本中的处理)
        # 提取左右脚两个踝关节（并联关节）
        q_a_p = np.zeros(4)  # 并联关节角度（实际）
        qdot_a_p = np.zeros(4)  # 并联关节速度（实际）
        tor_a_p = np.zeros(4)  # 并联关节力矩（实际）
        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        tor_a_s = np.zeros(4)  # 串联关节力矩（实际）

        q_a_p[:2] = self.robot_data_.q_a_[
            self.left_ankle_indices]  # 左脚踝关节 (pitch, roll)
        q_a_p[2:] = self.robot_data_.q_a_[
            self.right_ankle_indices]  # 右脚踝关节 (pitch, roll)

        qdot_a_p[:2] = self.robot_data_.q_dot_a_[self.left_ankle_indices]
        qdot_a_p[2:] = self.robot_data_.q_dot_a_[self.right_ankle_indices]

        tor_a_p[:2] = self.robot_data_.tau_a_[self.left_ankle_indices]
        tor_a_p[2:] = self.robot_data_.tau_a_[self.right_ankle_indices]

        self.q_a_p = q_a_p.copy()
        self.qdot_a_p = qdot_a_p.copy()
        self.tor_a_p = tor_a_p.copy()
        if self.debug:
            print("-" * 20 + "并转串联前" + "-" * 20)
            print("q_a_p:", q_a_p)
            print("qdot_a_p:", qdot_a_p)
            print("tor_a_p:", tor_a_p)

        # 计算并转串（正运动学）
        self.fun_s2p.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        self.fun_s2p.calcFK()
        self.fun_s2p.calcIK()

        success, q_a_s, qdot_a_s, tor_a_s = self.fun_s2p.get_s_state()

        if self.debug:
            print("-" * 20 + "并转串联后" + "-" * 20)
            print("q_a_s:", q_a_s)
            print("qdot_a_s:", qdot_a_s)
            print("tor_a_s:", tor_a_s)

        # 用串联关节值替换原来的并联关节值
        self.robot_data_.q_a_[self.left_ankle_indices] = q_a_s[:2]  # 左脚踝关节串联值
        self.robot_data_.q_a_[self.right_ankle_indices] = q_a_s[2:]  # 右脚踝关节串联值

        self.robot_data_.q_dot_a_[self.left_ankle_indices] = qdot_a_s[:2]
        self.robot_data_.q_dot_a_[self.right_ankle_indices] = qdot_a_s[2:]

        self.robot_data_.tau_a_[self.left_ankle_indices] = tor_a_s[:2]
        self.robot_data_.tau_a_[self.right_ankle_indices] = tor_a_s[2:]

    def ankle_serial_to_parallel(self):
        # 串转并：将串联关节命令转换为并联关节命令（类似C++版本）
        # 提取踝关节两关节的串联命令
        q_d_p = np.zeros(4)  # 并联关节角度（期望）
        qdot_d_p = np.zeros(4)  # 并联关节速度（期望）
        tor_d_p = np.zeros(4)  # 并联关节力矩（期望）
        q_d_s = np.zeros(4)  # 串联关节角度（期望）
        qdot_d_s = np.zeros(4)  # 串联关节速度（期望）
        tor_d_s = np.zeros(4)  # 串联关节力矩（期望）

        q_d_s[:2] = self.robot_data_.q_d_[self.left_ankle_indices]  # 左脚踝关节串联命令
        q_d_s[2:] = self.robot_data_.q_d_[
            self.right_ankle_indices]  # 右脚踝关节串联命令

        qdot_d_s[:2] = self.robot_data_.q_dot_d_[self.left_ankle_indices]
        qdot_d_s[2:] = self.robot_data_.q_dot_d_[self.right_ankle_indices]

        tor_d_s[:2] = self.robot_data_.tau_d_[self.left_ankle_indices]
        tor_d_s[2:] = self.robot_data_.tau_d_[self.right_ankle_indices]

        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        q_a_s[:2] = self.robot_data_.q_a_[self.left_ankle_indices]  # 左脚踝关节串联值
        q_a_s[2:] = self.robot_data_.q_a_[self.right_ankle_indices]  # 右脚踝关节串联值
        qdot_a_s[:2] = self.robot_data_.q_dot_a_[
            self.left_ankle_indices]  # 左脚踝关节串联速度
        qdot_a_s[2:] = self.robot_data_.q_dot_a_[
            self.right_ankle_indices]  # 右脚踝关节串联速度

        kp = np.zeros(4)  # 串联关节刚度
        kd = np.zeros(4)  # 串联关节阻尼
        kp[:2] = self.robot_data_.joint_kp_p_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kp[2:] = self.robot_data_.joint_kp_p_[self.right_ankle_indices -
                                              self.floating_base_dof]
        kd[:2] = self.robot_data_.joint_kd_p_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kd[2:] = self.robot_data_.joint_kd_p_[self.right_ankle_indices -
                                              self.floating_base_dof]

        tor_d_s = kp * (q_d_s - q_a_s) + kd * (qdot_d_s - qdot_a_s)

        if self.debug:
            print("-" * 20 + "串转并联前" + "-" * 20)
            print("q_d_s:", q_d_s)
            print("qdot_d_s:", qdot_d_s)
            print("tor_d_s:", tor_d_s)

        # 串转并计算
        self.fun_s2p.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        self.fun_s2p.calc_joint_pos_ref()
        self.fun_s2p.calc_joint_tor_des()
        success, q_d_p, qdot_d_p, tor_d_p = self.fun_s2p.get_p_des()

        q_d_p = (tor_d_p - self.ankle_kd_p *
                 (qdot_d_p - self.qdot_a_p)) / self.ankle_kp_p + self.q_a_p

        if self.debug:
            print("-" * 20 + "串转并联后" + "-" * 20)
            print("q_d_p:", q_d_p)
            print("qdot_d_p:", qdot_d_p)
            print("tor_d_p:", tor_d_p)

        # 用并联关节命令覆盖原来的串联命令
        self.robot_data_.q_d_[self.left_ankle_indices] = q_d_p[:2]  # 左脚踝关节并联命令
        self.robot_data_.q_d_[self.right_ankle_indices] = q_d_p[2:]  # 右脚踝关节并联命令

        self.robot_data_.q_dot_d_[self.left_ankle_indices] = qdot_d_p[:2]
        self.robot_data_.q_dot_d_[self.right_ankle_indices] = qdot_d_p[2:]

        # self.robot_data_.tau_d_[self.left_ankle_indices] = tor_d_p[:2]
        # self.robot_data_.tau_d_[self.right_ankle_indices] = tor_d_p[2:]
        self.robot_data_.tau_d_[self.left_ankle_indices] = 0.0
        self.robot_data_.tau_d_[self.right_ankle_indices] = 0.0
        # 更新脚踝Kp,Kd
        self.robot_data_.joint_kp_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[:2]
        self.robot_data_.joint_kp_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[2:]
        self.robot_data_.joint_kd_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[:2]
        self.robot_data_.joint_kd_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[2:]

    def update_robot_cmd(self, flag: ControlFlag):
        """更新机器人控制命令"""
        if flag.enable:
            # 使用getattr设置默认值，避免AttributeError
            x_command = getattr(flag, 'x_speed_command')
            y_command = getattr(flag, 'y_speed_command')
            yaw_command = getattr(flag, 'yaw_speed_command')
            self.robot_data_.walk_cmd_ = [x_command, y_command, yaw_command]
        self.robot_data_.control_flag.fsm_state_command = flag.fsm_state_command

def get_robot_interface(robot_data: RobotData, config_path: str = 'tienkung3_dex_config.yaml') -> RobotInterface:
    """工厂函数，返回机器人接口实例
    
    Args:
        robot_data: 机器人数据对象
        config_path: 配置文件路径，如 'tienkung3_dex_config.yaml' 或 'config/tienkung3_dex_config.yaml'
        
    Returns:
        RobotInterface实例
    """
    return RobotInterfaceImpl(robot_data, config_path=config_path)
