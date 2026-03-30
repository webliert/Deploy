import os
import sys

import rclpy
import yaml

from lite_rl_control_node import LiteControlNode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LiteControlNode_sim(LiteControlNode):
    def __init__(self, debug=False):
        super().__init__(debug)
        print("Simulation mode enabled")
        self.robot_interface.sim = True

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None
    try:
        node = LiteControlNode_sim(debug=False)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if "node" in locals() and node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
