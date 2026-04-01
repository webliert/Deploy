# 机器人配置管理系统

## 概述

本系统提供统一的机器人配置管理，支持多机器人适配。通过将机器人特有参数（如电机数量、关节配置、增益参数等）从策略代码中分离出来，实现了代码的通用化。

## 架构

```
config/
├── robot_profiles/           # 机器人配置文件
│   ├── tg22_profile.yaml    # TG22机器人（20电机）
│   ├── dex_profile.yaml     # Dex机器人（29电机）
│   └── README.md
├── tg22_config.yaml         # TG22运行配置
└── dex_config.yaml          # Dex运行配置

common/
└── robot_config_manager.py  # 配置管理器

policy/
├── walk_amp/config/         # 策略配置（仅策略特有参数）
├── zero/config/
└── stop/config/
```

## 文件说明

### 1. 机器人Profile文件 (`config/robot_profiles/`)

定义机器人的硬件参数：

```yaml
# tg22_profile.yaml 示例
robot_name: "TG22"
motor_num: 20
floating_base_dof: 6
dt: 0.0025

# 关节定义
joint_order:
  - l_hip_roll
  - l_hip_pitch
  # ... 更多关节

# 关节参数（按关节名索引）
zero_pos:
  l_hip_roll: 0.0
  l_hip_pitch: -0.5
  # ...

kp:
  l_hip_roll: 700.0
  # ...

kd:
  l_hip_roll: 20.0
  # ...
```

### 2. 策略配置文件 (`policy/*/config/`)

仅包含策略特有参数，机器人参数通过配置管理器获取：

```yaml
# walk_amp_ov.yaml 示例
model_path: "walk_lite_40700.xml"
dt: 0.0025

# 引用robot_profile中的关节组
joint_groups:
  - "legs.left"
  - "legs.right"
  - "arms.left"
  - "arms.right"

control:
  action_scale: 0.25
  decimation: 1
```

### 3. 运行配置文件 (`config/`)

运行时的控制参数配置：

```yaml
# dex_config.yaml
robot_profile: "dex"
sim: false
control_tool: keyboard

keyboard:
  initial_height: 0.90
  max_forward_speed: 1.5
```

## 使用方法

### 添加新机器人

1. 在 `config/robot_profiles/` 下创建新的profile文件：

```bash
cp tg22_profile.yaml new_robot_profile.yaml
```

2. 修改机器人参数：

```yaml
robot_name: "NewRobot"
motor_num: 25  # 修改电机数量
# ...
```

3. 创建对应的运行配置文件：

```yaml
# new_robot_config.yaml
robot_profile: "new_robot"
# ...
```

### 在代码中使用

```python
from common.robot_config_manager import get_robot_config_manager

# 初始化配置管理器
config_manager = get_robot_config_manager('tg22')

# 获取参数
motor_num = config_manager.motor_num
zero_pos = config_manager.get_zero_pos()
kp = config_manager.get_kp()
kd = config_manager.get_kd()

# 获取指定关节组的参数
joint_groups = ['legs.left', 'legs.right']
joints = config_manager.get_all_joints_in_groups(joint_groups)
params = config_manager.get_params_for_policy(joints)
```

## 优势

1. **单一数据源**：机器人参数只在profile文件中定义，避免重复
2. **易于扩展**：添加新机器人只需创建新的profile文件
3. **策略通用**：同一策略可适配不同机器人
4. **维护简单**：修改参数只需改一处

## 关节组说明

关节组用于组织关节列表：

- `legs.left`：左腿关节
- `legs.right`：右腿关节
- `arms.left`：左臂关节
- `arms.right`：右臂关节
- `waist`：腰部关节

在策略配置中通过 `joint_groups` 引用这些组：

```yaml
joint_groups:
  - "legs.left"
  - "legs.right"
```

## 注意事项

1. profile文件中的关节名称必须与 `joint_order` 一致
2. 策略配置中的 `joint_groups` 必须在profile中定义
3. 运行配置文件会自动检测 `robot_profile` 字段并加载对应配置