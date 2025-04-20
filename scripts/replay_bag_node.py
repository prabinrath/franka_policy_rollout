#!/usr/bin/env python3

import rospy as ros
import actionlib
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
import argparse
import rosbag
from scipy.interpolate import CubicSpline
import numpy as np


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

def replay_bag(args):
    ros.init_node("replay_bag_node")
    online_command_publisher = ros.Publisher(
            "/effort_joint_trajectory_controller/follow_joint_trajectory/goal",
            FollowJointTrajectoryActionGoal,
            queue_size=1,
        )
    reset_publisher = ros.Publisher("/policy/reset", Empty, queue_size=1)
    grasp_action_client = actionlib.SimpleActionClient(
        'franka_gripper/grasp',
        GraspAction
    )
    grasp_action_client.wait_for_server()
    move_action_client = actionlib.SimpleActionClient(
        'franka_gripper/move',
        MoveAction
    )
    move_action_client.wait_for_server()

    joints_track = []
    gripper_track = []
    joint_names = None
    with rosbag.Bag(args.demo_bag, 'r') as bag:
        for topic, msg, _ in bag.read_messages():
            if "joint_states" in topic:
                if joint_names is None:
                    joint_names = msg.name
                joints_track.append(msg.position)
            if "gripper_state" in topic:
                gripper_track.append(msg.data)
    joints_track_np = np.array(joints_track)
    _, joints_track_np = postprocess_trajectory(joints_track_np, joints_track_np.shape[0])
    dt = 0.1

    move_goal = MoveGoal()
    move_goal.width = 0.08
    move_goal.speed = 0.1
    move_action_client.send_goal(move_goal)
    move_action_client.wait_for_result(ros.Duration(5))
    is_grasped = False
    gripper_close_time, gripper_open_time = ros.Time.now(), ros.Time.now()
    for js, gs in zip(joints_track_np, gripper_track):
        next_js_goal = FollowJointTrajectoryActionGoal()
        next_js_goal.goal.trajectory.joint_names = joint_names[:-2]
        point = JointTrajectoryPoint(positions=js[:-2])
        point.time_from_start = ros.Duration.from_sec(dt)
        next_js_goal.goal.trajectory.points.append(point)
        next_js_goal.goal.goal_time_tolerance = ros.Duration.from_sec(0.5)
        online_command_publisher.publish(next_js_goal)
        if is_grasped:
            gripper_open_time = ros.Time.now()
        if not is_grasped and ros.Time.now()-gripper_open_time > ros.Duration(2.0) and gs > 0:
            grap_goal = GraspGoal()
            grap_goal.width = 0.0
            grap_goal.epsilon.inner = 0.08
            grap_goal.epsilon.outer = 0.08
            grap_goal.speed = 0.1
            grap_goal.force = 5.0
            grasp_action_client.send_goal(grap_goal)
            is_grasped = True
            gripper_close_time = ros.Time.now()
        if is_grasped and ros.Time.now()-gripper_close_time > ros.Duration(2.0) and gs < 1:
            move_goal = MoveGoal()
            move_goal.width = 0.08
            move_goal.speed = 0.1
            move_action_client.send_goal(move_goal)
            is_grasped = False
        ros.sleep(dt)
    ros.sleep(2)
    reset_publisher.publish(Empty())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bag post processor")
    parser.add_argument("--demo_bag", default="bags/test_gripper/demo_1_latest.bag", type=str, help="path to bag")
    args, _ = parser.parse_known_args()

    replay_bag(args)