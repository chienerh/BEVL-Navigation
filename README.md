# BEVL-Navigation

## Introduction
This repository is using [Intel RealSense D435i](https://www.intelrealsense.com/depth-camera-d435i/) as a RGBD camera. The camera is mounted on a head strap that can be worn on the head. This repository combines object detecting algorithm [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and object tracking algorithm [STVIR/pysot](https://github.com/STVIR/pysot) to detect doors and keep tracking it. Optionally, an Arduino Motor system can provide feedback with "Forward", "Left", "Right", and "Stop" commands to subjects.

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
cp -r pysot/toolkit ../../anaconda3/envs/door/lib/python3.6/site-packages/toolkit
cp -r pysot/pysot ../../anaconda3/envs/door/lib/python3.6/site-packages/pysot
cp -r pytorch-ssd/vision ../../anaconda3/envs/door/lib/python3.6/site-packages/vision
cd pysot
python setup.py build_ext --inplace
cd ..
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
python arduino_door.py
```

## Result
### Speed
While running on GTX 1660 Ti Laptop, it can reach to 10 FPS.
### Door detection
### Door tracking
