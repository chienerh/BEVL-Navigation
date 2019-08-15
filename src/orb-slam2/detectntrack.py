from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
import time

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

config = '../BEVL-Navigation/pysot/experiments/siamrpn_alex_dwxcorr/config.yaml'
snapshot = '../BEVL-Navigation/pysot/experiments/siamrpn_alex_dwxcorr/model.pth'
video_name = ''
net_type = 'vgg16-ssd'
model_path = '../BEVL-Navigation/pytorch-ssd/models/20190622/vgg16-ssd-Epoch-30-Loss-3.0707082748413086.pth'
label_path = '../BEVL-Navigation/pytorch-ssd/models/20190622/open-images-model-labels.txt'

# load config
cfg.merge_from_file(config)
cfg.CUDA = torch.cuda.is_available()
device = torch.device('cuda' if cfg.CUDA else 'cpu')

# Predictor
print('loading predictor...')
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
net = create_vgg_ssd(num_classes, is_test=True)
net.load(model_path)
predictor = create_vgg_ssd_predictor(net, candidate_size=200)

# Tracker
print('loading tracker...')
model = ModelBuilder()
model.load_state_dict(torch.load(snapshot, map_location=lambda storage, loc: storage.cpu()))
model.eval().to(device)
tracker = build_tracker(model)

# Initialization for Functions
initialized = False
track_time = 0

# Initialization for Detect & Tracking
pred_boxes = None
# pred_labels = []
bbox = []  # x1, y1, x2, y2
depth_value = None

# Initialization for Feedback
bbox_limit = [233, 178, 407, 302, 124, 174]
command = None
last_cmd = None
timer = time.time()

# Set up display
cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('argus', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('output', 700, 540)
cv2.moveWindow('RealSense', 0, 0)
cv2.moveWindow('depth', 700, 0)
cv2.moveWindow('argus', 0, 540)
frame_shape = (640, 480)  # (1280, 720)(640, 480)
frame_shape_hw = (frame_shape[1], frame_shape[0])
frame_output = np.zeros(frame_shape_hw)
frame_showing = np.zeros(frame_shape_hw)
frame_depth = np.zeros(frame_shape_hw)
frame_argus = np.zeros(frame_shape_hw)


def reset():
    global frame_output, frame_argus, depth_value, last_cmd, command, bbox, pred_boxes
    frame_output = np.zeros(frame_shape_hw)
    frame_argus = np.zeros(frame_shape_hw)
    depth_value = None
    last_cmd = command
    command = None
    bbox = None
    pred_boxes = []


def detectntrack(image, depth_image=None):
    global frame_showing, frame_depth, pred_boxes, initialized, track_time, frame_shape, frame_shape_hw
    reset()

    # Change input image format
    image = image.reshape(frame_shape[1], frame_shape[0], 3)
    frame_showing = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (480, 640, 3)

    if depth_image is not None:
        # frame_shape_hw = depth_image.shape
        # frame_shape = (frame_shape_hw[1], frame_shape_hw[0])
        frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=20), cv2.COLORMAP_JET)

    pred_boxes, labels, probs = predictor.predict(image, 10, 0.6)

    pred_labels = []
    for i in range(pred_boxes.size(0)):
        box = pred_boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        pred_labels.append(class_names[labels[i]])
        cv2.rectangle(frame_showing, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        cv2.putText(frame_showing, label,
                    (box[0] + 15, box[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # font scale
                    (255, 0, 255),
                    2)  # line type

    if len(pred_boxes) > 0:  # Object detected
        detected(image)
    else:  # No object is detected
        if track_time > 50:
            lose_track()
        elif initialized:
            keep_tracking(image)
        # else: do nothing
    show_result(depth_image)
    argus2()
    key = show_img()

    return 1


def detected(image):
    # Object detected. Initialize tracker
    global initialized, track_time, tracker, bbox
    boxes = pred_boxes.detach().cpu().numpy()  # boxes=[x1, y1, x2, y2]
    bbox_tracker = [int(boxes[0, 0]), int(boxes[0, 1]), int(boxes[0, 2] - boxes[0, 0]),
                    int(boxes[0, 3] - boxes[0, 1])]  # bbox=[x1, y1, w, h]
    tracker.init(image, bbox_tracker)
    bbox = [int(boxes[0, 0]), int(boxes[0, 1]), int(boxes[0, 2]), int(boxes[0, 3])]  # bbox=[x1, y1, x2, y2]
    initialized = True
    track_time = 0


def keep_tracking(image):
    # Already initialized a tracker. Keep tracking
    global track_time, bbox
    outputs = tracker.track(image)
    bbox = list(map(int, outputs['bbox']))  # bbox=[x1, y1, w, h]
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # bbox=[x1, y1, x2, y2]
    track_time += 1


def lose_track():
    # loss track of the object
    global initialized, track_time
    initialized = False
    track_time = 0


def show_result(depth_image):
    global frame_showing, frame_depth, frame_output, depth_value
    if bbox:
        cv2.rectangle(frame_showing, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      (0, 255, 0), 3)
        if depth_image is not None:
            depth_value = get_mean_depth(depth_image)
            if depth_value is not None and not np.isnan(depth_value):
                cv2.putText(frame_showing, f"{depth_value:.2f} meters away",
                            (bbox[0] + 15, bbox[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,  # font scale
                            (255, 0, 255),
                            2)  # line type
                frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.rectangle(frame_depth, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              (0, 255, 0), 3)
        x1, y1 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
        x2, y2 = min(int(bbox[2]), frame_shape[0]), min(int(bbox[3]), frame_shape[1])
        frame_output[y1:y2, x1:x2] = 255


def argus2():
    global frame_argus, frame_showing, command
    frame_argus = np.zeros((bbox_limit[4], bbox_limit[5]))
    # Add Argus FOV box on frame_showing
    cv2.rectangle(frame_showing, (bbox_limit[0], bbox_limit[1]),
                  (bbox_limit[2], bbox_limit[3]),
                  (255, 0, 0), 3)

    if bbox:
        # check if bbox_argus is in bbox_limit
        if bbox[0] < bbox_limit[0] and bbox[2] < bbox_limit[0]:
            bbox_argus = None
            command = 'Left'
        elif bbox[0] > bbox_limit[2] and bbox[2] > bbox_limit[2]:
            bbox_argus = None
            command = 'Right'
        elif bbox[1] < bbox_limit[1] and bbox[3] < bbox_limit[1]:
            bbox_argus = None
            command = 'Up'
        elif bbox[1] > bbox_limit[3] and bbox[3] > bbox_limit[3]:
            bbox_argus = None
            command = 'Down'
        else:
            bbox_argus = [max(bbox_limit[0], bbox[0]), max(bbox_limit[1], bbox[1]),
                          min(bbox_limit[2], bbox[2]),
                          min(bbox_limit[3], bbox[3])]
            command = 'Forward'

        if depth_value:
            if depth_value <= 2.0:
                command = 'Stop'

        print('%s        \r' % command, end="")

        # Add bbox on Argus
        if bbox_argus:
            cv2.rectangle(frame_showing, (bbox_argus[0], bbox_argus[1]),
                          (bbox_argus[2], bbox_argus[3]),
                          (0, 0, 255), 3)

            # Output for displaying on ArgusII
            x1 = max(bbox_argus[0] - bbox_limit[0], 0)
            y1 = max(bbox_argus[1] - bbox_limit[1], 0)
            x2 = min(bbox_argus[2] - bbox_limit[0], bbox_limit[5])
            y2 = min(int(bbox_argus[3] - bbox_limit[1]), bbox_limit[4])
            frame_argus[y1:y2, x1:x2] = 255


def get_mean_depth(depth_image):
    if depth_image is not None:
        roi = depth_image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        mean_depth = np.NaN if np.all(roi != roi) else np.nanmean(roi)
        if not np.isnan(mean_depth):
            return mean_depth
        else:
            return None
    else:
        return None


def show_img():
    global frame_argus
    frame_argus = cv2.resize(frame_argus, (640, 480))
    cv2.imshow('output', frame_output)
    cv2.imshow('RealSense', frame_showing)
    cv2.imshow('depth', frame_depth)
    cv2.imshow('argus', frame_argus)
    key = cv2.waitKey(1)

    return key


def cmd_to_arduino(arduino):
    global timer
    if command == 'Left':
        arduino.write('2'.encode())
    elif command == 'Right':
        arduino.write('3'.encode())
    elif command == 'Forward':
        arduino.write('4'.encode())
    elif command == 'Stop':
        arduino.write('0'.encode())
    timer = time.time()


def give_cmd(arduino):
    if command is not None:
        print('%s        \r' % command, end="")
        if time.time() - timer > 1.0:
            cmd_to_arduino(arduino)
        else:
            if last_cmd is not None:
                if last_cmd is not command:
                    cmd_to_arduino(arduino)
    else:
        print('%s        \r' % 'no command', end="")
