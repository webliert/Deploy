"""
FSM Implementation
Complete FSM implementation with state management
"""
from typing import Dict
from .fsm_base import RobotFSM, FSMStateName
from common import RobotData, ControlFlag, get_deploy_config
import functools
import time

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
        # 注释掉打印，避免过多输出
        # print(f"[TIMING] {func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper

class RobotFSMImpl(RobotFSM):
    """机器人FSM具体实现"""

    def __init__(self, robot_data: RobotData, config: Dict):
        super().__init__(robot_data)
        self.config = config

        # 当前状态
        self.current_state = FSMStateName.STOP
        self.state_objects = {}

        # 初始化所有状态对象
        self._init_states()

        # 进入初始状态
        self.state_objects[self.current_state].on_enter()

    def _init_states(self):
        """初始化所有状态对象"""
        # 导入状态对象（按需延迟加载）
        # 这样只有启用的策略才会被加载
        
        # STOP 状态总是启用
        from policy.stop.fsm_stop import FSMStateStop
        self.state_objects[FSMStateName.STOP] = FSMStateStop(self.robot_data_)
        print(f"[RobotFSM] State STOP initialized")
        
        # ZERO 状态总是启用
        from policy.zero.fsm_zero import FSMStateZero
        self.state_objects[FSMStateName.ZERO] = FSMStateZero(self.robot_data_)
        print(f"[RobotFSM] State ZERO initialized")
        
        # 动态加载策略状态
        policy_state_map = {
            "walk_amp": ("WALKAMP", "policy.walk_amp.fsm_walkamp", "FSMStateWALKAMP"),
            "beyond_mimic": ("BEYOND_MIMIC", "policy.beyond_mimic.fsm_beyond_mimic", "FSMStateBeyondMimic"),
            "beyondzero": ("BEYONDZERO", "policy.beyondzero.fsm_beyondzero", "FSMStateBeyondZero"),
            "homie": ("HOMIE", "policy.homie.fsm_homie", "FSMStateHOMIE"),
        }
        
        try:
            config_manager = get_deploy_config()
            enabled_policies = config_manager.config.get_available_policies()
        except Exception:
            enabled_policies = {}
        
        for policy_name, (state_name_str, module_path, class_name) in policy_state_map.items():
            policy_config = enabled_policies.get(policy_name, {})
            if not policy_config.get("enabled", False):
                print(f"[RobotFSM] Policy {policy_name} is disabled, skipping")
                continue
            
            try:
                module = __import__(module_path, fromlist=[class_name])
                state_class = getattr(module, class_name)
                self.state_objects[getattr(FSMStateName, state_name_str)] = state_class(self.robot_data_)
                print(f"[RobotFSM] State {state_name_str} initialized")
            except Exception as e:
                print(f"[RobotFSM] Failed to initialize state {state_name_str}: {e}")
    @timing_decorator
    def run_fsm(self, flag: ControlFlag):
        """运行FSM"""
        # 检查状态转换
        current_state_obj = self.state_objects[self.current_state]
        next_state = current_state_obj.check_transition(flag)

        # 如果需要状态转换
        if next_state is not None and next_state != self.current_state:
            if next_state in self.state_objects:
                print(f"FSM transition: {self.current_state.name} -> {next_state.name}")

                # 退出当前状态
                current_state_obj.on_exit()

                # 切换到新状态
                self.current_state = next_state
                self.state_objects[self.current_state].on_enter()
            else:
                # 状态未实现（可能是被禁用的策略），静默忽略
                pass

        # 运行当前状态
        self.state_objects[self.current_state].run(flag)

    def get_current_state(self) -> FSMStateName:
        """获取当前FSM状态"""
        return self.current_state


def get_robot_fsm(robot_data: RobotData, config: Dict) -> RobotFSM:
    """工厂函数，返回机器人FSM实例"""
    return RobotFSMImpl(robot_data, config)
