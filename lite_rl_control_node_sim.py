"""
Lite RL Control Node (Simulation Mode)
使用统一配置管理器 (DeployConfigManager)
"""
import os
import sys

import rclpy

from rl_control_node import XMIGCSControlNode
from config.deploy_config_manager import DeployConfigManager

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LiteControlNode_sim(XMIGCSControlNode):
    """
    仿真控制节点，使用统一配置管理
    """
    
    def __init__(self, debug=False):
        # 使用统一配置管理器
        self.deploy_config = DeployConfigManager.get_instance()
        
        print(f"[LiteControlNode_sim] Using unified config")
        print(f"[LiteControlNode_sim] Robot: {self.deploy_config.robot_name}")
        print(f"[LiteControlNode_sim] Sim: {self.deploy_config.config.sim}")
        print(f"[LiteControlNode_sim] Active policy: {self.deploy_config.config.active_policy}")
        print(f"[LiteControlNode_sim] Available policies:")
        for name, policy in self.deploy_config.config.get_available_policies().items():
            status = "enabled" if policy.get("enabled") else "disabled"
            print(f"[LiteControlNode_sim]   - {name}: {status}")
        
        # 调用父类初始化
        super().__init__("deploy_config.yaml", debug)
        
        # 强制设置仿真模式
        self.robot_interface.sim = True
        
        print("[LiteControlNode_sim] Simulation mode initialized")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None
    shutdown_needed = True
    try:
        node = LiteControlNode_sim(debug=False)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "node" in locals() and node is not None:
            node.destroy_node()
        # 避免重复调用 shutdown
        if shutdown_needed:
            try:
                rclpy.shutdown()
            except Exception:
                pass  # 忽略已关闭的错误


if __name__ == "__main__":
    main()