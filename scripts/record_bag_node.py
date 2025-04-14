#!/usr/bin/env python

import rospy as ros
from sensor_msgs.msg import JointState, Joy, Image, PointCloud2
import message_filters
from threading import Thread, Lock
import rosbag
import argparse
import os


class RecordBagNode():
    def __init__(self, root_path, policy="latest"):
        self.root_path = root_path
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
            self.front_cam_cache = message_filters.Cache(front_cam_sub, 100)
            self.wrist_cam_cache = message_filters.Cache(wrist_cam_sub, 100)
            self.front_pts_cache = message_filters.Cache(front_pts_sub, 100)
            self.js_cache = message_filters.Cache(js_sub, 100)
            exec_thread = Thread(target=self.cache_aggregate)
            exec_thread.start()
    
    def joy_callback(self, msg):
        if msg.buttons[8] > 0 and not self.is_recording:
            ros.loginfo(f"started recording demo: {self.demo_num}")
            self.bag = rosbag.Bag(os.path.join(self.root_path, f"demo_{self.demo_num}.bag"), "w", compression="lz4")
            self.is_recording = True
        elif msg.buttons[9] > 0 and self.is_recording:
            self.is_recording = False
            with self.bag_lock:
                self.bag.close()
            ros.loginfo(f"saved demo: {self.demo_num}")
            self.demo_num += 1
            self.bag = None
    
    def cache_aggregate(self):
        assert self.policy == "latest"
        r = ros.Rate(20)
        while not ros.is_shutdown():
            if self.is_recording:
                front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg = self.front_cam_cache.getLast(), \
                    self.wrist_cam_cache.getLast(), self.front_pts_cache.getLast(), self.js_cache.getLast()
                with self.bag_lock:
                    self.write_to_bag(front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg)
            r.sleep()
    
    def write_to_bag(self, front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg):
        assert front_cam_msg is not None
        assert wrist_cam_msg is not None
        assert front_pts_msg is not None
        assert js_msg is not None
        if self.bag is not None:
            self.bag.write("/front_cam/color/image_raw", front_cam_msg, front_cam_msg.header.stamp)
            self.bag.write("/wrist_cam/color/image_raw", wrist_cam_msg, wrist_cam_msg.header.stamp)
            self.bag.write("/front_cam/depth/color/points", front_pts_msg, front_pts_msg.header.stamp)
            self.bag.write("/joint_states", js_msg, js_msg.header.stamp)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Codig demonstration recorder")
    parser.add_argument("--bag_root", default="./bags", type=str, help="bag root path")
    args, _ = parser.parse_known_args()
    RecordBagNode(args.bag_root)
    ros.spin()
    