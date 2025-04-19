import sys
sys.path.insert(0, "/root/codig")

import time
import cv2
from cv_bridge import CvBridge
import numpy as np
from hydra import initialize, compose
import numpy as np
from utils.misc import HOME, set_seed
from policy_rollout.rollout import PolicyRollout

import rospy as ros
from threading import Lock, Thread
import actionlib
import message_filters
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState, PointCloud2, Image
from control_msgs.msg import FollowJointTrajectoryActionGoal
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
import argparse
from scipy.interpolate import CubicSpline


class FrankaGripperInterface():
    def __init__(self, numb_duration=2.0, grasp_threshold=(0.039, 0.038)):
        self.numb_duration = numb_duration
        self.grasp_threshold = grasp_threshold
        self.grasp_action_client = actionlib.SimpleActionClient(
            'franka_gripper/grasp',
            GraspAction
        )
        self.grasp_action_client.wait_for_server()
        self.move_action_client = actionlib.SimpleActionClient(
            'franka_gripper/move',
            MoveAction
        )
        self.move_action_client.wait_for_server()
        self.is_grasped = False
        self.gripper_close_time, self.gripper_open_time = ros.Time.now(), ros.Time.now()

    def handle_grasp(self, width_pred):
        if self.is_grasped:
            self.gripper_open_time = ros.Time.now()
        if not self.is_grasped and \
            ros.Time.now()-self.gripper_open_time > ros.Duration(self.numb_duration) and \
                width_pred < self.grasp_threshold[0]:
            grap_goal = GraspGoal()
            grap_goal.width = 0.0
            grap_goal.epsilon.inner = 0.08
            grap_goal.epsilon.outer = 0.08
            grap_goal.speed = 0.1
            grap_goal.force = 5.0
            self.grasp_action_client.send_goal(grap_goal)
            self.is_grasped = True
            self.gripper_close_time = ros.Time.now()
        if self.is_grasped and \
            ros.Time.now()-self.gripper_close_time > ros.Duration(self.numb_duration) and \
                width_pred > self.grasp_threshold[1]:
            move_goal = MoveGoal()
            move_goal.width = 0.08
            move_goal.speed = 0.1
            self.move_action_client.send_goal(move_goal)
            self.is_grasped = False


class FrankaRolloutInterface(PolicyRollout):
    """Rollout Interface for Franka FR3 robot"""

    def __init__(self):
        with initialize(version_base=None, config_path="../../../../codig/config"):
            self.roll_cfg = compose(config_name="rr_mm_rollout", overrides=["++ckpt_tag=mma1_block"])
        super().__init__(self.roll_cfg)
        set_seed(self.roll_cfg.seed)

        K = 1
        self.dt = 0.033/K + 0.001

        self.online_jpos_publisher = ros.Publisher(
            "/effort_joint_trajectory_controller/follow_joint_trajectory/goal",
            FollowJointTrajectoryActionGoal,
            queue_size=1,
        )
        self.reset_publisher = ros.Publisher("/policy/reset", Empty, queue_size=1)

        self.bridge = CvBridge()
        self.data_lock = Lock()
        self.msg_data_dict = dict()
        self.joint_names = ["fr3_joint1", 
                   "fr3_joint2", 
                   "fr3_joint3", 
                   "fr3_joint4", 
                   "fr3_joint5", 
                   "fr3_joint6", 
                   "fr3_joint7", 
                   "fr3_finger_joint1", 
                   "fr3_finger_joint2"]
        
        front_cam_sub = message_filters.Subscriber("/front_cam/color/image_raw", Image)
        wrist_cam_sub = message_filters.Subscriber("/wrist_cam/color/image_raw", Image)
        front_pts_sub = message_filters.Subscriber("/front_cam/depth/color/points", PointCloud2)
        js_sub = message_filters.Subscriber("/joint_states", JointState)
    
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [front_cam_sub, wrist_cam_sub, front_pts_sub, js_sub],
            queue_size=10,
            slop=0.1
        )
        self.ats.registerCallback(self.data_callback)

        self.gripper_interface = FrankaGripperInterface()

        self.EXECUTE = False
        self.current_rollout_step = 0
        self.data_dict = dict(cam_obs_front_img=np.zeros((1,self.roll_cfg.n_obs_steps,3,96,96), np.float32),
                              cam_obs_wrist_img=np.zeros((1,self.roll_cfg.n_obs_steps,3,96,96), np.float32),
                              rgb_points=np.zeros((1,self.roll_cfg.n_obs_steps,96,96,6), np.float32),
                              rob_obs_history=np.zeros((1,self.roll_cfg.n_obs_steps,9), np.float32))
        exec_thread = Thread(target=self.rollout)
        exec_thread.start()
    
    def data_callback(self, front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg):
        with self.data_lock:
            self.msg_data_dict["front_img"] = front_cam_msg
            self.msg_data_dict["wrist_img"] = wrist_cam_msg
            self.msg_data_dict["rgb_points"] = front_pts_msg
            self.msg_data_dict["joint_states"] = js_msg

    def get_latest_observation(self):
        with self.data_lock:
            front_cam_msg, wrist_cam_msg, front_pts_msg, js_msg = self.msg_data_dict["front_img"], \
            self.msg_data_dict["wrist_img"], self.msg_data_dict["rgb_points"], self.msg_data_dict["joint_states"]

        front_img = self.bridge.imgmsg_to_cv2(front_cam_msg, desired_encoding='bgr8')
        front_img = cv2.resize(front_img, (96, 96), interpolation=cv2.INTER_LINEAR)
        front_img = cv2.rotate(front_img, cv2.ROTATE_180)
        front_img = np.moveaxis(front_img, -1, -3)[...,::-1] / 255.0 # TODO: need to refactor
        self.data_dict["front_img"] = np.roll(self.data_dict["front_img"], shift=-1, axis=1)
        self.data_dict["front_img"][0,-1,...] = front_img

        wrist_img = self.bridge.imgmsg_to_cv2(wrist_cam_msg, desired_encoding='bgr8')
        wrist_img = cv2.resize(wrist_img, (96, 96), interpolation=cv2.INTER_LINEAR)
        wrist_img = np.moveaxis(wrist_img, -1, -3)[...,::-1] / 255.0 # TODO: need to refactor
        self.data_dict["wrist_img"] = np.roll(self.data_dict["wrist_img"], shift=-1, axis=1)
        self.data_dict["wrist_img"][0,-1,...] = wrist_img

        np_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('b0', '<f4'), ('rgb', '<f4')]
        np_pc = np.frombuffer(front_pts_msg.data, dtype=np_dtype)
        points = np.expand_dims(np.hstack((np.expand_dims(np_pc['x'],-1), np.expand_dims(np_pc['y'], -1), np.expand_dims(np_pc['z'],-1))), 0)
        points = points.reshape((front_pts_msg.height,front_pts_msg.width,3))[65:405,190:640]
        rgb = np.frombuffer(np.ascontiguousarray(np_pc['rgb']).data, dtype=np.uint8)
        rgb = np.expand_dims(rgb,0).reshape(front_pts_msg.height*front_pts_msg.width,4)[:,:3]
        rgb = np.expand_dims(rgb,0).reshape(front_pts_msg.height,front_pts_msg.width,3)[65:405,190:640]
        rgb = rgb.astype(np.float32) / 255.0
        rgb_points = np.concatenate((rgb, points), axis=-1)
        rgb_points = cv2.resize(rgb_points, (96, 96), interpolation=cv2.INTER_LINEAR)
        self.data_dict["rgb_points"] = np.roll(self.data_dict["rgb_points"], shift=-1, axis=1)
        self.data_dict["rgb_points"][0,-1,...] = rgb_points

        joint_pos = np.asarray(js_msg.position)
        self.data_dict["rob_obs_history"] = np.roll(self.data_dict["rob_obs_history"], shift=-1, axis=1)
        self.data_dict["rob_obs_history"][0,-1,...] = joint_pos
    
    def postprocess_trajectory(traj, num_queries, k_size=4):
        # fit a fine cubic spline for interpolation
        fit_array_x = np.arange(traj.shape[0]) / traj.shape[0]
        fit_array_x_query = np.linspace(0, fit_array_x[-1], num_queries)
        cs = CubicSpline(fit_array_x, traj)
        spline_traj = np.array([cs(x) for x in fit_array_x_query])

        # smoothen the trajectory for execution
        smoothened_traj = np.copy(spline_traj)
        kernel = np.ones(k_size)/k_size
        for dim in range(spline_traj.shape[1]):
            smoothened_traj[k_size//2:-(k_size//2)+(1-k_size%2), dim] = \
                np.convolve(spline_traj[:,dim], kernel, 'valid')
        
        return fit_array_x_query, smoothened_traj
    
    def rollout(self):
        while True: # rollout infinitely
            if self.EXECUTE:
                if self.current_rollout_step < self.roll_cfg.max_traj_len + self.roll_cfg.n_obs_steps:
                    self.get_latest_observation()
                    self.current_rollout_step += 1
                    if self.current_rollout_step > self.roll_cfg.n_obs_steps:
                        act_h = self.get_model_pred(**self.data_dict).squeeze(0).cpu().numpy()
                        for h_idx in range(self.roll_cfg.rollout_steps):
                            next_js_goal = FollowJointTrajectoryActionGoal()
                            next_js_goal.goal.trajectory.joint_names = self.joint_names[:-2]
                            point = JointTrajectoryPoint(positions=act_h[h_idx][:-2])
                            point.time_from_start = ros.Duration.from_sec(self.dt)
                            next_js_goal.goal.trajectory.points.append(point)
                            next_js_goal.goal.goal_time_tolerance = ros.Duration.from_sec(0.5)
                            self.online_jpos_publisher.publish(next_js_goal)
                            self.gripper_interface.handle_grasp(act_h[h_idx][-1])
                            ros.sleep(self.dt)
                            self.current_rollout_step += 1
                else:
                    self.EXECUTE = False
                    self.current_rollout_step = 0
            else:
                self.current_rollout_step = 0    
