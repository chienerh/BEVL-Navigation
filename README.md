# BEVL-Navigation

## install and build
```
git clone --recursive https://github.com/chienerh/BEVL-Navigation.git
cd BEVL-Navigation
```
### create conda environment and install the requirements

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
