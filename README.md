FastSAM ROS with Docker
=======================

This ROS package wraps the [Official FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) repo. Further, [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) is used to build an image with all the necessary dependencies. Then, [`docker compose`](https://docs.docker.com/compose/install/) can be used to create a container to launch a ROS node. Note that the `docker-compose.yml` file shares the container network with the host network---thus, the node will be able to communicate with other ROS nodes on the host machine.

**Tested on:**
- Docker version 23.0.0, build e92dd87
- NVIDIA Container Runtime Hook version 1.12.0, commit: 62bd015475656ef795cb0d59cc4030a6bd4a9526
- NVIDIA GTX 1070 8GB
- Ubuntu 20.04 / ROS Noetic

## Getting Started

Install `docker`, `docker-compose`, `nvidia-docker` (i.e., `nvidia-container-toolkit`). Note that you do not need CUDA installed on your host machine, but you do need an NVIDIA driver installed on the host for your particular NVIDIA GPU.


```bash
$ git clone --recurse-submodules https://github.com/mit-acl/fastsam_ros # get FastSAM as submodule
$ cd fastsam_ros
$ docker compose build
$ docker compose up
```

## Running inside a namespace

You can use `docker compose run` to set the `ROS_NAMESPACE` environment variable:

```bash
docker compose run -e ROS_NAMESPACE=robot/camera fastsam_ros roslaunch fastsam_ros detect.launch
```

## Webcam demo

1. Install your favorite ROS web camera package. We will use [`usb_cam`](http://wiki.ros.org/usb_cam): `sudo apt install ros-noetic-usb-cam`.
2. Start a `roscore`.
3. Start the USB camera (with remapping): `rosrun usb_cam usb_cam_node usb_cam/image_raw:=image_raw`
4. Start `fastsam_ros` via `docker compose up`


### Credits

- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- [Meta's Segment Anything](https://github.com/facebookresearch/segment-anything)
- [YOLOv7 ROS](https://github.com/mit-acl/yolov7_ros)