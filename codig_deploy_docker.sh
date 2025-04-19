xhost +
docker run --privileged --rm -it --gpus all --network=host -e DISPLAY=$DISPLAY -e ROS_MASTER_URI=http://10.42.0.1:11311/ -e ROS_IP=10.42.0.87 -v /tmp/.X11-unix:/tmp/.X11-unix -v $1:/root/catkin_ws/src/codig_robot -v $2:/root/codig prabinrath/codig_deploy:latest bash
