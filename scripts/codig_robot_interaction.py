import sys
sys.path.insert(0, "/root/catkin_ws/src/codig_robot")

import rospy as ros
from sensor_msgs.msg import Joy
from scripts.codig_robot_interface import FrankaRolloutInterface


class CodigRealRobot():
    def __init__(self):
        ros.init_node("policy_node")
        self.robot = FrankaRolloutInterface()
        self.spacemouse_sub = ros.Subscriber(
            "/spacenav/joy",
            Joy,
            self.joy_callback,
            queue_size=1,
        )

    def joy_callback(self, msg):
        if msg.buttons[14] > 0 and not self.robot.EXECUTE:
            self.robot.EXECUTE = True
            print("Execution started.")
        elif msg.buttons[10] > 0 and self.robot.EXECUTE:
            self.robot.EXECUTE = False
            print("Execution stopped.")
        elif msg.buttons[11] > 0:
            # success 
            pass
        elif msg.buttons[12] > 0:
            # failure 
            pass


if __name__ == "__main__":
    CodigRealRobot()
    ros.spin()
