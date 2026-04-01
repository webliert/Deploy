import os
import sys

import rclpy
import yaml

from rl_control_node import XMIGCSControlNode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LiteControlNode_sim(XMIGCSControlNode):
    def __init__(self, config_file='tienkung2_lite_config.yaml', debug=False):
        super().__init__(config_file, debug)
        print("rewrite sim")

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
