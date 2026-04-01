"""
Robot Body ID Mapping
统一管理所有机器人的关节ID映射配置

通过 create_id_map(robot_name) 工厂函数创建对应机器人的ID映射
"""
from typing import Dict, List, Optional

# ============================================================
# 机器人映射配置字典
# ============================================================

ROBOT_ID_CONFIGS = {
    "tienkung3_dex": {
        "leg_ids": [51, 52, 53, 54, 55, 56,  # 左腿
                    61, 62, 63, 64, 65, 66],   # 右腿
        "leg_names": [
            "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
            "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll"
        ],
        "waist_ids": [33, 32, 31],
        "waist_names": ["waist_yaw", "waist_roll", "waist_pitch"],
        "arm_ids": [11, 12, 13, 14, 15, 16, 17,  # 左臂
                    21, 22, 23, 24, 25, 26, 27],   # 右臂
        "arm_names": [
            "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
            "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll",
            "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
            "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"
        ],
    },
    "tienkung2_lite": {
        "leg_ids": [51, 52, 53, 54, 55, 56,  # 左腿
                    61, 62, 63, 64, 65, 66],   # 右腿
        "leg_names": [
            "hip_roll_l_joint", "hip_pitch_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_roll_r_joint", "hip_pitch_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint"
        ],
        "waist_ids": [],
        "waist_names": [],
        "arm_ids": [11, 12, 13, 14,   # 左臂
                    21, 22, 23, 24],   # 右臂
        "arm_names": [
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint", "elbow_pitch_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint", "elbow_pitch_r_joint"
        ],
    },
}


# ============================================================
# BodyServoIdMap 类
# ============================================================

class BodyServoIdMap:
    """身体伺服电机ID映射"""

    def __init__(self, robot_name: str = "tienkung3_dex"):
        """
        初始化身体CAN ID映射
        
        Args:
            robot_name: 机器人名称，如 'tienkung3_dex' 或 'tienkung2_lite'
        """
        self.robot_name = robot_name
        self.id_to_index_map: Dict[int, int] = {}
        self.index_to_id_map: Dict[int, int] = {}
        self.name_to_index_map: Dict[str, int] = {}
        self.index_to_name_map: Dict[int, str] = {}
        self.leg_motor_nums = 0
        self.waist_motor_nums = 0
        self.arm_motor_nums = 0
        self.whole_motor_nums = 0

    def body_can_id_map_init(self, config: Optional[Dict] = None):
        """
        初始化身体CAN ID映射
        
        Args:
            config: 机器人配置字典，如果为None则使用默认配置
        """
        if config is None:
            if self.robot_name not in ROBOT_ID_CONFIGS:
                raise ValueError(
                    f"Unknown robot '{self.robot_name}'. "
                    f"Available robots: {list(ROBOT_ID_CONFIGS.keys())}"
                )
            config = ROBOT_ID_CONFIGS[self.robot_name]
        
        # 腿部关节映射
        leg_ids = config["leg_ids"]
        leg_names = config["leg_names"]
        self.leg_motor_nums = len(leg_ids)

        # 腰部关节映射
        waist_ids = config["waist_ids"]
        waist_names = config["waist_names"]
        self.waist_motor_nums = len(waist_ids)

        # 手臂关节映射
        arm_ids = config["arm_ids"]
        arm_names = config["arm_names"]
        self.arm_motor_nums = len(arm_ids)

        # 合并所有映射
        all_ids = leg_ids + waist_ids + arm_ids
        all_names = leg_names + waist_names + arm_names
        self.whole_motor_nums = len(all_ids)

        # 创建双向映射
        for index, (can_id, name) in enumerate(zip(all_ids, all_names)):
            self.id_to_index_map[can_id] = index
            self.index_to_id_map[index] = can_id
            self.name_to_index_map[name] = index
            self.index_to_name_map[index] = name

    def get_index_by_id(self, can_id: int) -> int:
        """根据CAN ID获取索引"""
        return self.id_to_index_map.get(can_id, -1)

    def get_id_by_index(self, index: int) -> int:
        """根据索引获取CAN ID"""
        return self.index_to_id_map.get(index, -1)

    def get_index_by_name(self, name: str) -> int:
        """根据关节名称获取索引"""
        return self.name_to_index_map.get(name, -1)

    def get_name_by_index(self, index: int) -> str:
        """根据索引获取关节名称"""
        return self.index_to_name_map.get(index, "")


# ============================================================
# 工厂函数
# ============================================================

def create_id_map(robot_name: str) -> BodyServoIdMap:
    """
    工厂函数：根据机器人名称创建ID映射
    
    Args:
        robot_name: 机器人名称，如 'tienkung3_dex' 或 'tienkung2_lite'
        
    Returns:
        BodyServoIdMap 实例，已初始化完成
    """
    id_map = BodyServoIdMap(robot_name)
    id_map.body_can_id_map_init()
    return id_map


def get_available_robots() -> List[str]:
    """获取所有可用的机器人名称"""
    return list(ROBOT_ID_CONFIGS.keys())
