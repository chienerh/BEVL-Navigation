from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import time
import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file', default='pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    parser.add_argument('--snapshot', type=str, help='model name', default='pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
    parser.add_argument('--video_name', default='data/EPSON/20190214_clip/20190214_trim.mp4', type=str,
                        help='videos or image files')
    args = parser.parse_args() 
    
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)


if __name__ == '__main__':
    main()