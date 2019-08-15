from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import time
import cv2
import torch
import numpy as np
import serial

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

import pyrealsense2 as rs

torch.set_num_threads(1)

# Global
stat_time = []
initialized = False
track_time = 0
first_cmd = True

def parse_args():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
    parser.add_argument('--snapshot', type=str, help='model name',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/model.pth')
    parser.add_argument('--video_name', default='', type=str,
                        help='videos or image files')
    parser.add_argument('--net_type', type=str, help='model name',
                        default='vgg16-ssd')
    parser.add_argument('--model_path', type=str, help='model name',
                        default='pytorch-ssd/models/20190622/vgg16-ssd-Epoch-30-Loss-3.0707082748413086.pth')
    parser.add_argument('--label_path', type=str, help='model name',
                        default='pytorch-ssd/models/20190622/open-images-model-labels.txt')

    args = parser.parse_args()
    return args


def prepare_predictor(net_type, model_path, label_path):
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)
    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(num_classes, is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(num_classes, is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
    elif net_type == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(num_classes, is_test=True)
    else:
        raise ValueError("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

    net.load(model_path)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        raise ValueError("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

    return class_names, predictor


def do_object_detection(orig_image, predictor, class_names):
    detect_img = orig_image.copy()
    image = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)  # (480, 640, 3)
    boxes, labels, probs = predictor.predict(image, 10, 0.6)

    pred_labels = []
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        pred_labels.append(class_names[labels[i]])
        cv2.rectangle(detect_img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        cv2.putText(detect_img, label,
                    (box[0] + 15, box[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # font scale
                    (255, 0, 255),
                    2)  # line type
    return boxes, pred_labels, detect_img


def prepare_tracker(device, model_path):
    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(model_path,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    return tracker


def show_result(frame, bbox, depth_frame):
    """
    :param frame:
    :param bbox: x1, y1, x2, y2
    :param depth_frame:
    :return: frame_showing, frame_output, frame_depth, depth
    """
    frame_showing = frame.copy()
    cv2.rectangle(frame_showing, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]),
                  (0, 255, 0), 3)
    frame_depth = np.zeros_like(frame)
    depth = None
    if depth_frame is not None:
        depth = get_mean_depth(depth_frame, bbox)
        if depth is not None and not np.isnan(depth):
            cv2.putText(frame_showing, f"{depth:.2f} meters away",
                        (bbox[0] + 15, bbox[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # font scale
                        (255, 0, 255),
                        2)  # line type
            frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(frame_depth, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 3)
    x1, y1, x2, y2 = max(int(bbox[0]), 0), max(int(bbox[1]), 0), min(int(bbox[2]), frame.shape[1]), min(
        int(bbox[3]), frame.shape[0])
    frame_output = np.zeros_like(frame)
    frame_output[y1:y2, x1:x2] = 255
    return frame_showing, frame_output, frame_depth, depth


def get_mean_depth(depth_frame, bbox):
    if depth_frame is not None:
        roi = depth_frame[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        mean_depth = np.NaN if np.all(roi != roi) else np.nanmean(roi) / 1e3
        if not np.isnan(mean_depth):
            return mean_depth
            # print(f"{mean_depth} meters away")
        else:
            return None
    else:
        return None


def calculate_bbox_limit(frame_shape):
    # horizontal and vertical field of view of Argus II
    argus_fov_h = 19
    argus_fov_v = 11

    # horizontal and vertical field of view of Realsense
    rs_fov_h = 69.4
    rs_fov_v = 42.5

    # Find pixel range for Argus II
    w, h = frame_shape
    h_argus = int(h * argus_fov_v / rs_fov_v)
    w_argus = int(w * argus_fov_h / rs_fov_h)

    # Four corners of Argus II field of view
    top = (h // 2) - (h_argus // 2)
    bottom = (h // 2) + (h_argus // 2)
    left = (w // 2) - (w_argus // 2)
    right = (w // 2) + (w_argus // 2)
    bbox_limit = [left, top, right, bottom, bottom-top, right-left]

    return bbox_limit


def argus2(frame_showing, bbox_limit, bbox=None):
    h_argus, w_argus = bbox_limit[4], bbox_limit[5]
    frame_argus = np.zeros((h_argus, w_argus))
    command = None
    # Add Argus FOV box on frame_showing
    cv2.rectangle(frame_showing, (bbox_limit[0], bbox_limit[1]),
                  (bbox_limit[2], bbox_limit[3]),
                  (255, 0, 0), 3)

    if bbox:
        # check if bbox_argus is in bbox_limit
        if bbox[0] < bbox_limit[0] and bbox[2] < bbox_limit[0]:
            # LEFT
            bbox_argus = None
            command = 'Left'
        elif bbox[0] > bbox_limit[2] and bbox[2] > bbox_limit[2]:
            # RIGHT
            bbox_argus = None
            command = 'Right'
        elif bbox[1] < bbox_limit[1] and bbox[3] < bbox_limit[1]:
            # UP
            bbox_argus = None
            command = 'Up'
        elif bbox[1] > bbox_limit[3] and bbox[3] > bbox_limit[3]:
            # DOWN
            bbox_argus = None
            command = 'Down'
        else:
            bbox_argus = [max(bbox_limit[0], bbox[0]), max(bbox_limit[1], bbox[1]), min(bbox_limit[2], bbox[2]),
                          min(bbox_limit[3], bbox[3])]
            command = 'Forward'

        # Add bbox on Argus
        if bbox_argus:
            cv2.rectangle(frame_showing, (bbox_argus[0], bbox_argus[1]),
                          (bbox_argus[2], bbox_argus[3]),
                          (0, 0, 255), 3)

            # Output for displaying on ArgusII
            x1, y1, x2, y2 = max(bbox_argus[0]-bbox_limit[0], 0), max(bbox_argus[1]-bbox_limit[1], 0), min(
                bbox_argus[2]-bbox_limit[0], w_argus), min(int(bbox_argus[3]-bbox_limit[1]), h_argus)
            frame_argus[y1:y2, x1:x2] = 255

    return frame_showing, frame_argus, command


def give_cmd(depth, command, arduino):
    global first_cmd

    if depth:
        if depth <= 1.5:
            command = 'Stop'

    if command is not None:
        print('%s    \r' % command, end="")
        if first_cmd or arduino.read().decode("utf-8") == '9':
            if command == 'Left':
                arduino.write('2'.encode())
            elif command == 'Right':
                arduino.write('3'.encode())
            elif command == 'Forward':
                arduino.write('4'.encode())
            elif command == 'Stop':
                arduino.write('0'.encode())
            first_cmd = False
    else:
        print('%s    \r' % 'no command', end="")
    return first_cmd


def setup_realsense():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    frame_shape = [640, 480]  # [1280, 720] [640, 480]
    config = rs.config()
    config.enable_stream(rs.stream.depth, frame_shape[0], frame_shape[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, frame_shape[0], frame_shape[1], rs.format.bgr8, 30)

    bbox_limit = calculate_bbox_limit(frame_shape)  # bbox_limit=[x1, y1, x2, y2, h, w]

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)

    cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('argus', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('output', 700, 540)
    cv2.moveWindow('RealSense', 0, 0)
    cv2.moveWindow('depth', 700, 0)
    cv2.moveWindow('argus', 0, 540)

    return pipeline, bbox_limit, align, profile


def setup_pipline():
    args = parse_args()

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print('device:', device)

    class_names, predictor = prepare_predictor(args.net_type, args.model_path, args.label_path)
    tracker = prepare_tracker(device, args.snapshot)

    return class_names, predictor, tracker


def get_frame(pipeline, align):
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        return 0, None, None

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return 1, depth_image, color_image


def main():
    class_names, predictor, tracker = setup_pipline()
    pipeline, bbox_limit, align, profile = setup_realsense()
    print('Finished preparing, start streaming')

    try:
        global track_time, initialized, first_cmd
        # connect to arduino
        arduino = serial.Serial("/dev/rfcomm1", 9600)
        while True:
            got_frame, depth_image, color_image = get_frame(pipeline, align)
            if got_frame != 1:
                continue

            boxes, labels, detect_img = do_object_detection(color_image, predictor, class_names)
            depth = None
            # Object detected
            if boxes.size(0) > 0:
                boxes = boxes.detach().cpu().numpy()  # boxes=[x1, y1, x2, y2]
                bbox_tracker = [int(boxes[0, 0]), int(boxes[0, 1]), int(boxes[0, 2] - boxes[0, 0]),
                                int(boxes[0, 3] - boxes[0, 1])]  # bbox=[x1, y1, w, h]
                tracker.init(color_image, bbox_tracker)
                bbox = [int(boxes[0, 0]), int(boxes[0, 1]), int(boxes[0, 2]), int(boxes[0, 3])]  # bbox=[x1, y1, x2, y2]
                frame_showing, frame_output, frame_depth, depth = show_result(detect_img, bbox, depth_image)
                frame_showing, frame_argus, command = argus2(frame_showing, bbox_limit, bbox)
                initialized = True
                track_time = 0
            else:
                # No object is detected
                if track_time > 50:
                    # loss track of the object
                    initialized = False
                    track_time = 0
                    frame_output = np.zeros_like(color_image)
                    frame_showing = color_image.copy()
                    frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    frame_showing, frame_argus, command = argus2(frame_showing, bbox_limit)
                elif initialized:
                    # Already initialized a tracker. Keep tracking
                    outputs = tracker.track(color_image)
                    bbox = list(map(int, outputs['bbox']))  # bbox=[x1, y1, w, h]
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # bbox=[x1, y1, x2, y2]
                    frame_showing, frame_output, frame_depth, depth = show_result(color_image, bbox, depth_image)
                    frame_showing, frame_argus, command = argus2(frame_showing, bbox_limit, bbox)
                    track_time += 1
                else:
                    # No tracker is initialized. Show the whole frame.
                    frame_output = np.zeros_like(color_image)
                    frame_showing = color_image.copy()
                    frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    frame_showing, frame_argus, command = argus2(frame_showing, bbox_limit)

            first_cmd = give_cmd(depth, command, arduino)
            frame_argus = cv2.resize(frame_argus, (640, 480))

            cv2.imshow('output', frame_output)
            cv2.imshow('RealSense', frame_showing)
            cv2.imshow('depth', frame_depth)
            cv2.imshow('argus', frame_argus)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                print('Exited the program by pressing q')
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
