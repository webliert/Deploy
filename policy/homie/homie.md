## Tienkung2 Lite 训练时 Actor-Critic 网络输入输出详细总结

根据代码分析，以下是Tienkung2 Lite训练时Actor和Critic网络的详细输入输出信息：

---

### 1. 观测空间维度配置 (tienkung2_lite_config.py)

```python
num_dofs = 20                    # 总自由度数量 (12腿部关节 + 8手臂关节)
num_actions = 12                 # 下半身动作数量（腿部控制，每条腿6个关节）
num_one_step_observations = 2 * num_dofs + 10 + num_actions = 62  # 单步观测维度
num_one_step_privileged_obs = num_one_step_observations + 3 = 65  # 单步特权观测维度
num_actor_history = 6            # Actor历史长度
num_critic_history = 1           # Critic历史长度
num_observations = 6 * 62 = 372   # Actor总输入维度
num_privileged_obs = 1 * 65 = 65 # Critic总输入维度
```

---

### 2. 单步观测内容 (legged_robot.py 第313-325行)

**Actor 单步观测 (62维)** 按顺序包含：

| 序号 | 内容                               | 维度 | 说明                                                 |
| ---- | ---------------------------------- | ---- | ---------------------------------------------------- |
| 1    | `commands[:, :3] * commands_scale` | 3    | 命令速度 (lin_vel_x, lin_vel_y, ang_vel_yaw)，带缩放 |
| 2    | `commands[:, 4]`                   | 1    | 目标高度命令                                         |
| 3    | `imu_ang_vel`                      | 3    | IMU角速度（机体坐标系）                              |
| 4    | `imu_projected_gravity`            | 3    | 投影重力向量                                         |
| 5    | `dof_pos - default_dof_pos`        | 20   | 关节位置偏差（相对于默认位置，包含腿和手臂）         |
| 6    | `dof_vel`                          | 20   | 关节速度（包含腿和手臂）                             |
| 7    | `actions[:, :12]`                  | 12   | 上一步的腿部动作（仅腿部，不含手臂）                 |

**Critic 特权观测 (65维)**：
- 在Actor观测(62维)基础上额外添加：
- `base_lin_vel` (3维) - 真实基座线速度（特权信息）

---

### 3. Actor网络详解

#### 输入 (him_actor_critic.py)

**总输入**: `obs_history` 历史观测序列，维度为 `(batch, 372)` 或 `(batch, actor_history_length * num_one_step_obs)`

**处理流程**:
1. **HIMEstimator编码器** 处理历史本体感知观测：
   - 输入: `obs_history[:, 0:372]` (完整的6步历史观测)
   - 输出: 
     - `vel` (3维): 估计的基座线速度
     - `dynamic_latent` (32维): 动力学潜在表示

2. **Terrain Encoder (地形编码器)** - 如果启用高度信息：
   - 输入: `obs_history[:, -(height_points + 62):]` (当前观测 + 高度点)
   - 输出: `terrain_latent` (32维): 地形潜在表示

3. **最终Actor MLP输入** (维度计算):
   ```
   如果使用高度信息:
   mlp_input = num_one_step_obs(62) + vel(3) + dynamic_latent(32) + terrain_latent(32) = 129维
   如果不使用高度信息:
   mlp_input = num_one_step_obs(62) + vel(3) + dynamic_latent(32) = 97维
   ```

#### 网络结构
```python
Actor MLP: [mlp_input_dim] -> [512, 256, 256] -> [num_actions=12]
```

#### 输出
- **动作均值** (action_mean): 12维，表示每个腿部关节的目标位置偏移
- **实际动作**: 通过正态分布采样 `Normal(action_mean, std).sample()`

---

### 4. Critic网络详解

#### 输入
**总输入**: `critic_observations` (65维)
- 与Actor不同，Critic只使用当前时刻的单步特权观测
- 包含真实线速度（特权信息）

#### 网络结构
```python
Critic MLP: [65] -> [512, 256, 256] -> [1]
```

#### 输出
- **价值估计** (value): 标量，表示当前状态的价值函数V(s)

---

### 5. HIMEstimator 详解

**目的**: 从历史观测中提取动力学隐变量

**输入**: 
- `obs_history[:, 0:actor_proprioceptive_obs_length]` (372维，6步历史观测)

**网络结构**:
```python
Encoder: [372] -> [256, 256] -> [35]  # 输出vel(3) + latent(32)
Target: [num_one_step_obs] -> [256, 256] -> [32]  # 用于对比学习
```

**输出**:
- `vel` (3维): 预测的基座线速度
- `dynamic_latent` (32维): 归一化后的动力学隐变量

---

### 6. 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tienkung2 Lite 训练数据流                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  obs_history (batch, 372)          privileged_obs (batch, 65)      │
│         │                                    │                       │
│         ▼                                    ▼                       │
│  ┌─────────────────┐                 ┌──────────────┐              │
│  │  HIMEstimator   │                 │   Critic     │              │
│  │  (Encoder)      │                 │   MLP        │              │
│  └────────┬────────┘                 └──────┬───────┘              │
│           │ vel(3), dynamic(32)             │                       │
│           ▼                                  ▼ value(1)             │
│  ┌─────────────────────────┐                  │                     │
│  │ concat:                 │                  │                      │
│  │ current_obs(62) +      │                  │                      │
│  │ vel(3) + dynamic(32) + │                  │                      │
│  │ terrain_latent(32)     │                  │                      │
│  └───────────┬─────────────┘                  │                      │
│              │                                │                      │
│              ▼                                │                      │
│      ┌──────────────┐                         │                      │
│      │ Actor MLP    │                         │                      │
│      └──────┬───────┘                         │                      │
│             │                                 │                      │
│             ▼ action(12)                      │                      │
│             └──────────────────────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```


根据配置文件 `HomieRL/legged_gym/legged_gym/envs/tienkung2_lite/tienkung2_lite_config.py`，`self.dof_pos` 的关节顺序如下（对应 URDF 中的 20 个关节）：

**关节顺序（索引 0-19）：**

| 索引 | 关节名称               | 默认角度 (rad) |
| ---- | ---------------------- | -------------- |
| 0    | hip_roll_l_joint       | 0.0            |
| 1    | hip_pitch_l_joint      | -0.5           |
| 2    | hip_yaw_l_joint        | 0.0            |
| 3    | knee_pitch_l_joint     | 1.0            |
| 4    | ankle_pitch_l_joint    | -0.5           |
| 5    | ankle_roll_l_joint     | 0.0            |
| 6    | hip_roll_r_joint       | -0.0           |
| 7    | hip_pitch_r_joint      | -0.5           |
| 8    | hip_yaw_r_joint        | 0.0            |
| 9    | knee_pitch_r_joint     | 1.0            |
| 10   | ankle_pitch_r_joint    | -0.5           |
| 11   | ankle_roll_r_joint     | 0.0            |
| 12   | shoulder_pitch_l_joint | 0.0            |
| 13   | shoulder_roll_l_joint  | 0.1            |
| 14   | shoulder_yaw_l_joint   | -0.0           |
| 15   | elbow_pitch_l_joint    | -0.3           |
| 16   | shoulder_pitch_r_joint | 0.0            |
| 17   | shoulder_roll_r_joint  | -0.1           |
| 18   | shoulder_yaw_r_joint   | 0.0            |
| 19   | elbow_pitch_r_joint    | -0.3           |

**`default_dof_pos` 的值：**
`tensor([ 0.0, -0.5, 0.0, 1.0, -0.5, 0.0, 0.0, -0.5, 0.0, 1.0, -0.5, 0.0, 0.0, 0.1, 0.0, -0.3, 0.0, -0.1, 0.0, -0.3])`

这个值在 `_init_buffers()` 方法中根据 URDF 文件中的关节名称顺序构建，通过 `self.cfg.init_state.default_joint_angles` 字典映射到 `self.default_dof_pos` 张量。


`self.commands_scale` 的值是 `tensor([2.0, 2.0, 0.5])`。

这个值在 `_init_buffers()` 方法中定义：

```python
self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
```

它是一个形状为 `(3,)` 的张量，包含三个元素：
- 第一个元素：`obs_scales.lin_vel` = 2.0（用于 x 轴线速度）
- 第二个元素：`obs_scales.lin_vel` = 2.0（用于 y 轴线速度）
- 第三个元素：`obs_scales.ang_vel` = 0.5（用于偏航角速度）

这个缩放因子用于在 `compute_observations()` 中对命令进行缩放：`self.commands[:, :3] * self.commands_scale`



根据 URDF 文件和配置文件，我整理了完整的关节信息：

## 一、关节顺序 (按 URDF 定义顺序)

| 索引 | 关节名称               | 类型 |
| ---- | ---------------------- | ---- |
| 0    | hip_roll_l_joint       | 左腿 |
| 1    | hip_pitch_l_joint      | 左腿 |
| 2    | hip_yaw_l_joint        | 左腿 |
| 3    | knee_pitch_l_joint     | 左腿 |
| 4    | ankle_pitch_l_joint    | 左腿 |
| 5    | ankle_roll_l_joint     | 左腿 |
| 6    | hip_roll_r_joint       | 右腿 |
| 7    | hip_pitch_r_joint      | 右腿 |
| 8    | hip_yaw_r_joint        | 右腿 |
| 9    | knee_pitch_r_joint     | 右腿 |
| 10   | ankle_pitch_r_joint    | 右腿 |
| 11   | ankle_roll_r_joint     | 右腿 |
| 12   | shoulder_pitch_l_joint | 左臂 |
| 13   | shoulder_roll_l_joint  | 左臂 |
| 14   | shoulder_yaw_l_joint   | 左臂 |
| 15   | elbow_pitch_l_joint    | 左臂 |
| 16   | shoulder_pitch_r_joint | 右臂 |
| 17   | shoulder_roll_r_joint  | 右臂 |
| 18   | shoulder_yaw_r_joint   | 右臂 |
| 19   | elbow_pitch_r_joint    | 右臂 |

---

## 二、完整参数表

| 索引 | 关节名称               | Kp [N·m/rad] | Kd [N·m·s/rad] | Default Pos [rad] | 控制方式 |
| ---- | ---------------------- | ------------ | -------------- | ----------------- | -------- |
| 0    | hip_roll_l_joint       | 700          | 10             | 0.0               | 力矩     |
| 1    | hip_pitch_l_joint      | 700          | 10             | **-0.5**          | 力矩     |
| 2    | hip_yaw_l_joint        | 500          | 5              | 0.0               | 力矩     |
| 3    | knee_pitch_l_joint     | 700          | 10             | **1.0**           | 力矩     |
| 4    | ankle_pitch_l_joint    | 30           | 2.5            | **-0.5**          | 力矩     |
| 5    | ankle_roll_l_joint     | 16.8         | 1.4            | 0.0               | 力矩     |
| 6    | hip_roll_r_joint       | 700          | 10             | 0.0               | 力矩     |
| 7    | hip_pitch_r_joint      | 700          | 10             | **-0.5**          | 力矩     |
| 8    | hip_yaw_r_joint        | 500          | 5              | 0.0               | 力矩     |
| 9    | knee_pitch_r_joint     | 700          | 10             | **1.0**           | 力矩     |
| 10   | ankle_pitch_r_joint    | 30           | 2.5            | **-0.5**          | 力矩     |
| 11   | ankle_roll_r_joint     | 16.8         | 1.4            | 0.0               | 力矩     |
| 12   | shoulder_pitch_l_joint | 60           | 3              | 0.0               | 位置     |
| 13   | shoulder_roll_l_joint  | 20           | 1.5            | 0.1               | 位置     |
| 14   | shoulder_yaw_l_joint   | 10           | 1              | 0.0               | 位置     |
| 15   | elbow_pitch_l_joint    | 10           | 1              | -0.3              | 位置     |
| 16   | shoulder_pitch_r_joint | 60           | 3              | 0.0               | 位置     |
| 17   | shoulder_roll_r_joint  | 20           | 1.5            | -0.1              | 位置     |
| 18   | shoulder_yaw_r_joint   | 10           | 1              | 0.0               | 位置     |
| 19   | elbow_pitch_r_joint    | 10           | 1              | -0.3              | 位置     |

---

## 三、观测向量 (Observation) 结构

单步观测维度：**62**

| 起始索引 | 结束索引 | 内容                                            | 维度 | 缩放因子             |
| -------- | -------- | ----------------------------------------------- | ---- | -------------------- |
| 0        | 3        | command[:3] (lin_vel_x, lin_vel_y, ang_vel_yaw) | 3    | × obs_scales.lin_vel |
| 3        | 4        | command[4] (height_cmd)                         | 1    | 无缩放               |
| 4        | 7        | imu_ang_vel (角速度)                            | 3    | × obs_scales.ang_vel |
| 7        | 10       | imu_projected_gravity (重力方向)                | 3    | 无缩放               |
| 10       | 30       | dof_pos - default_dof_pos (20个关节)            | 20   | × obs_scales.dof_pos |
| 30       | 50       | dof_vel (20个关节速度)                          | 20   | × obs_scales.dof_vel |
| 50       | 62       | actions[:12] (上一步动作)                       | 12   | 无缩放               |

**总观测维度：** 62 × 6 (历史长度) = **372**

---

## 四、缩放因子

```python
obs_scales.lin_vel = 2.0    # 需要查看基础配置
obs_scales.ang_vel = 0.5
obs_scales.dof_pos = 1.0
obs_scales.dof_vel = 0.05
action_scale = 0.25
```

---

## 五、部署时需要的数组格式

```python
# 只需要前12个关节 (腿部)
kps = [700, 700, 500, 700, 30, 16.8,   # 左腿
       700, 700, 500, 700, 30, 16.8]   # 右腿

kds = [10, 10, 5, 10, 2.5, 1.4,        # 左腿
       10, 10, 5, 10, 2.5, 1.4]        # 右腿

default_angles = [0.0, -0.5, 0.0, 1.0, -0.5, 0.0,  # 左腿
                  0.0, -0.5, 0.0, 1.0, -0.5, 0.0]  # 右腿
```
