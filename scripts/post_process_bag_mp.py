import cv2
from cv_bridge import CvBridge
from collections import defaultdict
from multiprocessing import Manager, Pool
import numpy as np
import h5py
import argparse
import rosbag
import glob
import os


def single_process(idx, path, args, lock):
    data_track = defaultdict(list)
    bridge = CvBridge()

    with rosbag.Bag(path, 'r') as bag:
        for topic, msg, _ in bag.read_messages():
            if "image_raw" in topic:
                if "front" in topic:
                    image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                    image = cv2.rotate(image, cv2.ROTATE_180)
                    data_track["front_img"].append(np.expand_dims(image,0))
                if "wrist" in topic:
                    image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
                    data_track["wrist_img"].append(np.expand_dims(image,0))
            if "points" in topic:
                np_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('b0', '<f4'), ('rgb', '<f4')]
                np_pc = np.frombuffer(msg.data, dtype=np_dtype)
                points = np.expand_dims(np.hstack((np.expand_dims(np_pc['x'],-1), np.expand_dims(np_pc['y'], -1), np.expand_dims(np_pc['z'],-1))), 0)
                points = points.reshape((msg.height,msg.width,3))[65:405,190:640]
                rgb = np.frombuffer(np.ascontiguousarray(np_pc['rgb']).data, dtype=np.uint8)
                rgb = np.expand_dims(rgb,0).reshape(msg.height*msg.width,4)[:,:3]
                rgb = np.expand_dims(rgb,0).reshape(msg.height,msg.width,3)[65:405,190:640]
                rgb = rgb.astype(np.float32)/255.0
                rgb_points = np.concatenate((rgb, points), axis=-1)
                rgb_points = cv2.resize(rgb_points, (args.width * args.pc_density, args.height * args.pc_density), interpolation=cv2.INTER_LINEAR)
                data_track["rgb_points"].append(np.expand_dims(rgb_points,0))
            if "joint_states" in topic:
                js = np.vstack((np.asarray(msg.position), np.asarray(msg.velocity), np.asarray(msg.effort)))
                data_track["joint_states"].append(np.expand_dims(js,0))

    if args.down_sample_factor > 1:
        total_len = len(data_track["front_img"])
        indices = np.linspace(0, total_len - 1, total_len//args.down_sample_factor).astype(int)
        for key in data_track:
            data_track[key] = np.vstack(data_track[key])
            data_track[key] = data_track[key][indices]
    else:
        for key in data_track:
            data_track[key] = np.vstack(data_track[key])

    with lock:
        with h5py.File(os.path.join(args.dataset_path, f"{args.task}.h5"), "a") as file:
            if "data" not in file.keys():
                data = file.create_group(f"data")
            else:
                data = file["data"]
            demo = data.create_group(f"demo_{idx}")
            for key in data_track:
                demo.create_dataset(key, data=data_track[key], compression="lzf")
    print(f"Processed {path.split('/')[-1]}")

def postprocess_bag(args):
    bag_path = os.path.join(args.bag_root, args.task)
    paths_track = []
    for path_tuple in enumerate(glob.glob(f"{bag_path}/*.bag")):
        paths_track.append(path_tuple)
        if len(paths_track) == args.batch_size:
            with Manager() as manager:
                lock = manager.Lock()
                assignments = [(idx, path, args, lock) for idx, path in paths_track]
                with Pool() as pool:
                    pool.starmap(single_process, assignments)
            paths_track.clear()
    if len(paths_track) > 0:
        with Manager() as manager:
            lock = manager.Lock()
            assignments = [(idx, path, args, lock) for idx, path in paths_track]
            with Pool() as pool:
                pool.starmap(single_process, assignments)
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bag post processor")
    parser.add_argument("--bag_root", default="../codig_robot/bags/", type=str, help="bag root path")
    parser.add_argument("--task", default="block_pick", type=str, help="task name")
    parser.add_argument("--dataset_path", default="./dataset", type=str, help="dataset path")
    parser.add_argument("--height", default=96, type=int, help="image height")
    parser.add_argument("--width", default=96, type=int, help="image width")
    parser.add_argument("--pc_density", default=1, type=int, help="pc density multiplier")
    parser.add_argument("--batch_size", default=1, type=int, help="multi processing batch size")
    parser.add_argument("--down_sample_factor", default=3, type=int, help="time step downsampling factor")
    args, _ = parser.parse_known_args()

    postprocess_bag(args)