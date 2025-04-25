import sys
sys.path.insert(0, "/root/catkin_ws/src/codig_robot")

import rospy as ros
from sensor_msgs.msg import Joy
from scripts.codig_robot_interface import FrankaRolloutInterface
from pathlib import Path
import datetime


class CodigRealRobot():
    def __init__(self, task_name, policy_type, experiment_name):
        ros.init_node("policy_node")
        self.robot = FrankaRolloutInterface(task_name, policy_type)
        self.spacemouse_sub = ros.Subscriber(
            "/spacenav/joy",
            Joy,
            self.joy_callback,
            queue_size=1,
        )
        Path(f"logs/{task_name}/{policy_type}").mkdir(parents=True, exist_ok=True)
        self.log_file_path = f"logs/{task_name}/{policy_type}/{experiment_name}{datetime.datetime.now()}.log"
        self.log(f"checkpoint_path: {self.robot.checkpoint_path}")
        self.LOG_TOGGLE = True

    def log(self, log_text):
        with open(self.log_file_path, "a") as file:
            file.write(f"{log_text}\n")

    def joy_callback(self, msg):
        if msg.buttons[14] > 0 and not self.robot.EXECUTE:
            self.robot.EXECUTE = True
            print("Execution started.")
        elif msg.buttons[10] > 0 and self.robot.EXECUTE:
            self.robot.EXECUTE = False
            print("Execution stopped.")
        elif msg.buttons[6] > 0 and self.LOG_TOGGLE:
            # success 
            self.log(f"recorded success")
            self.LOG_TOGGLE = False
        elif msg.buttons[7] > 0 and self.LOG_TOGGLE:
            # failure 
            self.log(f"recorded failure")
            self.LOG_TOGGLE = False
        elif msg.buttons[6] == 0 and msg.buttons[7] == 0:
            self.LOG_TOGGLE = True


if __name__ == "__main__":
    CodigRealRobot(
        task_name="pour_in_bowl",
        policy_type="codig",
        experiment_name=""
        )
    ros.spin()
