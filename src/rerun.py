import numpy as np
import glob
import cv2
import time
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Rerun Realsense Door')
    parser.add_argument('--folder', type=str, help='folder name of the data you want to rerun',
                        default='data/test/')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path_rgb = os.path.join(args.folder, 'RGB')
    path_depth = os.path.join(args.folder, 'Depth')
    files_rgb = sorted(glob(path_rgb))
    files_depth = sorted(glob(path_depth))

    # read txt from log file



if __name__ == '__main__':
    main()


