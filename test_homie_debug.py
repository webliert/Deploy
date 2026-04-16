#!/usr/bin/env python3
"""测试HOMIE状态机的调试脚本"""

import sys
sys.path.insert(0, '/home/szxx/lxy_ws/Deploy')

import numpy as np
from common import RobotData, ControlFlag, get_robot_config_manager
from policy.homie.fsm_homie import FSMStateHOMIE

# 初始化配置管理器
config_manager = get_robot_config_manager()

# 创建RobotData
robot_data = RobotData()

# 创建HOMIE状态
homie_state = FSMStateHOMIE(robot_data)

# 进入状态
homie_state.on_enter()

# 创建ControlFlag
flag = ControlFlag()
flag.fsm_state_command = ""

# 运行几个周期
for i in range(5):
    robot_data.time_now_ = i * 0.01
    homie_state.run(flag)

print("\n" + "="*50)
print("测试完成")
print("="*50)
