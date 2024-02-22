# BEVL-Navigation

## Introduction
This repository is using [Intel RealSense D435i](https://www.intelrealsense.com/depth-camera-d435i/) as a RGBD camera. The camera is mounted on a head strap that can be worn on the head. This repository combines object detecting algorithm [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and object tracking algorithm [STVIR/pysot](https://github.com/STVIR/pysot) to detect doors and keep tracking it. Optionally, an Arduino Motor system can provide feedback with "Forward", "Left", "Right", and "Stop" commands to subjects.

## Publication
We published the study using this repository in EMBC 2022. [Paper Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9871117)
```
@inproceedings{jeganathan2022integration,
  title={Integration of artificial vision with non-visual peripheral cues to guide mobility},
  author={Jeganathan, V Swetha E and Lin, Chien Erh and Son, Hojun and Krishnagiri, Divya S and Wei, Yumou and Weiland, James D},
  booktitle={2022 44th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={5136--5139},
  year={2022},
  organization={IEEE}
}
```

## install and build
```
git clone --recursive https://github.com/chienerh/BEVL-Navigation.git
cd BEVL-Navigation/
```
### create conda environment and install the requirements
Note: realsense is only available in python 3.6, not python 3.7

```
conda create --name door python=3.6
conda activate door
pip install -r requirements.txt
```
### pysot and pytorch-ssd packages set up
```
cp -r pytorch-ssd/vision ../../anaconda3/envs/door/lib/python3.6/site-packages/vision
```
### librealsense set up
See [RealSense/librealsense page](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python).

## How to use
```
cd src/
```
If you want to run RealSense + Door algorithm,
```
python realsense_door.py
```
If you want to run RealSense + Door algorithm + Arduino Motor Feedback,
```
sudo python3 arduino_door.py
```
If you want to run RealSense + Door algorithm + Arduino Motor Feedback without forward command and use argus feedback instead,
```
sudo python3 rs_arduino_door.py
```

## Result
### Speed
While running on GTX 1660 Ti Laptop, it can reach to 10 FPS.
### Door detection
### Door tracking

## Combine with ORB-SLAM2
To combine our system with [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2), place the files in orb-slam2 folder to designated location in [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).
* Replace `ORB-SLAM2/CMakeLists.txt`
* `ORB-SLAM2/detectntrack.py`
* `ORB-SLAM2/Examples/RGB-D/RealSense.yaml`
* `ORB-SLAM2/Examples/RGB-D/rgbd_realsense.cc`
* `ORB-SLAM2/Examples/RGB-D/rgbd_rs_door.cc`
build ORB-SLAM2 again.
```
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3.6 -DPYTHON_INCLUDE_DIRS=/usr/include/python3.6 -DPYTHON_LIBRARY=/usr/lib/python3.6/config/libpython3.6.so
make -j
```
For running RealSense + ORB-SLAM2,
```
./Examples/RGB-D/rgbd_realsense
```
For running RealSense + ORB-SLAM2 + Door algorithm,
```
./Examples/RGB-D/rgbd_rs_door
```
