#!/usr/bin/env python3

import rospy as ros
from sensor_msgs.msg import JointState, Joy, Image, PointCloud2
from std_msgs.msg import Int8, String
import message_filters
from threading import Thread, Lock
import rosbag
import argparse
from pathlib import Path
import os


class RecordBagNode():
    def __init__(self, bag_root, task, policy="latest"):
        self.bag_root = bag_root
        self.task = task
        self.bag_path = None

        assert policy in ["latest", "synced"]
        self.policy = policy
        self.demo_num = 1
        self.bag = None
        self.is_recording = False
        self.bag_lock = Lock()

        ros.init_node("bag_record_node")
        self.spacemouse_sub = ros.Subscriber(
            "/spacenav/joy",
            Joy,
            self.joy_callback,
            queue_size=1,
        )
        self.record_cmd_sub = ros.Subscriber(
            "/record_command",
            String,
            self.record_cmd_callback,
            queue_size=1,
        )

        front_cam_sub = message_filters.Subscriber("/front_cam/color/image_raw", Image)
        wrist_cam_sub = message_filters.Subscriber("/wrist_cam/color/image_raw", Image)
        front_pts_sub = message_filters.Subscriber("/front_cam/depth/color/points", PointCloud2)
        js_sub = message_filters.Subscriber("/joint_states", JointState)

        if self.policy == "synced":
            self.ats = message_filters.ApproximateTimeSynchronizer(
                [front_cam_sub, wrist_cam_sub, front_pts_sub, js_sub],
                queue_size=100,
                slop=0.1
            )
            self.ats.registerCallback(self.write_to_bag)
        elif self.policy == "latest":
            self.front_cam_cache = message_filters.Cache(front_cam_sub, 1)
            self.wrist_cam_cache = message_filters.Cache(wrist_cam_sub, 1)
            self.front_pts_cache = message_filters.Cache(front_pts_sub, 1)
            self.js_cache = message_filters.Cache(js_sub, 1)
            exec_thread = Thread(target=self.cache_aggregate)
            exec_thread.start()
    
    def _start_recording(self, folder_name):
        if self.is_recording:
            return
        self.bag_path = os.path.join(self.bag_root, folder_name)
        Path(self.bag_path).mkdir(parents=True, exist_ok=True)
        existing_files = os.listdir(self.bag_path)
        used_nums = []
        for fname in existing_files:
            if fname.startswith("demo_") and fname.endswith(".bag"):
                try:
                    num = int(fname[len("demo_"):-len(".bag")])
                    used_nums.append(num)
                except ValueError:
                    continue
        self.demo_num = max(used_nums) + 1 if used_nums else 1
        ros.loginfo(f"started recording demo: {self.demo_num} in {self.bag_path}")
        with self.bag_lock:
            self.bag = rosbag.Bag(os.path.join(self.bag_path, f"demo_{self.demo_num}.bag"), "w", compression="lz4")
        self.is_recording = True

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        with self.bag_lock:
            self.bag.close()
        ros.loginfo(f"saved demo: {self.demo_num}")
        self.demo_num += 1

    def joy_callback(self, msg):
        if msg.buttons[8] > 0 and not self.is_recording:
            # 3 button on space mouse
            self._start_recording(self.task)
        elif msg.buttons[9] > 0 and self.is_recording:
            # 4 button on space mouse
            self._stop_recording()
    
    def record_cmd_callback(self, msg):
        cmd = msg.data
        if cmd.startswith("start:"):
            folder_name = cmd[len("start:"):]
            self._start_recording(folder_name)
        elif cmd == "stop":
            self._stop_recording()
    
    def cache_aggregate(self):
        assert self.policy == "latest"
        r = ros.Rate(10)
        while not ros.is_shutdown():
            front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg = self.front_cam_cache.getLast(), \
                self.wrist_cam_cache.getLast(), self.front_pts_cache.getLast(), self.js_cache.getLast()
            self.write_to_bag(front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg)
            r.sleep()
    
    def write_to_bag(self, front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg):
        if self.is_recording:
            assert front_cam_msg is not None
            assert wrist_cam_msg is not None
            assert front_pts_msg is not None
            assert js_msg is not None
            with self.bag_lock:
                self.bag.write("/front_cam/color/image_raw", front_cam_msg, front_cam_msg.header.stamp)
                self.bag.write("/wrist_cam/color/image_raw", wrist_cam_msg, wrist_cam_msg.header.stamp)
                self.bag.write("/front_cam/depth/color/points", front_pts_msg, front_pts_msg.header.stamp)
                self.bag.write("/joint_states", js_msg, js_msg.header.stamp)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Codig demonstration recorder")
    parser.add_argument("--bag_root", default="./bags", type=str, help="bag root path")
    parser.add_argument("--task", default="task", type=str, help="name of the task")
    parser.add_argument("--policy", default="synced", type=str, help="time policy")
    args, _ = parser.parse_known_args()
    RecordBagNode(args.bag_root, args.task, policy=args.policy)
    ros.spin()
    