# LiLi-OM (LIvox LiDAR-Inertial Odometry and Mapping)
## -- Towards High-Performance Solid-State-LiDAR-Inertial Odometry and Mapping
This is the code repository of LiLi-OM, a real-time tightly-coupled LiDAR-inertial odometry and mapping system for solid-state LiDAR (Livox Horizon) and conventional LiDARs (e.g., Velodyne). It has two variants as shown in the folder: 

- LiLi-OM, for [Livox Horizon](https://www.livoxtech.com/de/horizon) with a newly proposed feature extraction module,
- LiLi-OM-ROT, for conventional LiDARs of spinning mechanism with feature extraction module similar to [LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM).

Both variants exploit the same backend module, which is proposed to directly fuse LiDAR and (preintegrated) IMU measurements based on a keyframe-based sliding window optimization. Detailed information can be found in our paper shown below.

## Cite
Thank you for citing out [*LiLi-OM* paper](https://arxiv.org/pdf/2010.13150.pdf) if you use any of this code: 
```
@article{arXiv20_Li,
 author = {Kailai Li and Meng Li and Uwe D. Hanebeck},
 journal = {arXiv preprint arXiv:2010.13150},
 month = {October},
 title = {Towards High-Performance Solid-State-LiDAR-Inertial Odometry and Mapping},
 url = {https://arxiv.org/abs/2010.13150},
 year = {2020}
}
```

## Data sets
We provide data sets recorded by Livox Horizon (10 Hz) and Xsens MTi-670 (200 Hz)

[Download Link](https://isas-server.iar.kit.edu/lidardataset/) 

## Dependency

System dependencies (tested on Ubuntu 18.04/20.04)
- [ROS](http://wiki.ros.org/noetic/Installation) (tested with Melodic/Noetic)
- [gtsam](https://gtsam.org/) (GTSAM 4.0)
- [ceres](http://ceres-solver.org/installation.html) (Ceres Solver 2.0)

In ROS workspce: 
- [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver) (ROS driver for Livox Horizon)


## Compilation
Compile with [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/index.html):
```
cd ~/catkin_ws/src
git clone https://github.com/KIT-ISAS/lili-om
cd ..
catkin build livox_ros_driver
catkin build lili_om
catkin build lili_om_rot
```
## Usasge
1. Run a launch file for lili_om or lili_om_rot 
2. Play bag file

- Example for running lili_om (Livox Horizon):
```
roslaunch lili_om run_fr_iosb.launch
rosbag play FR_IOSB_Long.bag -r 1.0 --clock --pause
```
- Example for running lili_om_rot (spinning LiDAR like the Velodyne HDL-64E in FR_IOSB data set):
```
roslaunch lili_om_rot run_fr_iosb.launch
rosbag play FR_IOSB_Long_64.bag -r 1.0 --clock --pause
```

## Contributors
Meng Li (Email: [limeng1523@outlook.com](limeng1523@outlook.com))

[Kailai Li](https://isas.iar.kit.edu/Staff_Li.php) (Email: [kailai.li@kit.edu](kailai.li@kit.edu))
