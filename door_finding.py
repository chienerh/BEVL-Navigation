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
import glob

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


def parse_args():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
    parser.add_argument('--snapshot', type=str, help='model name',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/model.pth')
    parser.add_argument('--video_name', default='', type=str,
                        help='videos or image files')
    parser.add_argument('--net_type', type=str, help='model name',
                        default='mb1-ssd')
    parser.add_argument('--model_path', type=str, help='model name',
                        default='pytorch-ssd/models/20190618/mb1-ssd-Epoch-99-Loss-2.7762672106424966.pth')
    parser.add_argument('--label_path', type=str, help='model name',
                        default='pytorch-ssd/models/20190618/open-images-model-labels.txt')

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


class FrameReader(object):

    def __init__(self, video_name) -> None:
        self.video_name = video_name
        self._init_iters()

    def __iter__(self):
        self._init_iters()
        return self

    def __next__(self):
        return self.iter_next()

    def _init_iters(self):
        if not self.video_name:
            # RealSense
            self.frame_iter = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 1280x720 640, 360
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 1280x720 640, 480

            # Start streaming
            profile = self.frame_iter.start(config)

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            clipping_distance_in_meters = 1  # 1 meter
            self.clipping_distance = clipping_distance_in_meters / depth_scale

            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)

        elif self.video_name.endswith('avi') or \
                self.video_name.endswith('mp4'):
            self.frame_iter = cv2.VideoCapture(self.video_name)
        else:
            images = glob.glob(os.path.join(self.video_name, '*.jp*'))
            self.frame_iter = sorted(images,
                                     key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def iter_next(self):
        if isinstance(self.frame_iter, rs.pyrealsense2.pipeline):
            # RealSense
            # Get frameset of color and depth
            frames = self.frame_iter.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            while not aligned_depth_frame or not color_frame:
                frames = self.frame_iter.wait_for_frames()
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image
        elif isinstance(self.frame_iter, cv2.VideoCapture()):
            # Read from video
            ret, frame = self.frame_iter.read()
            if not ret:
                raise StopIteration
            return frame
        return cv2.imread(next(self.frame_iter))


def show_result(frame, bbox, depth_frame):
    frame_showing = frame.copy()
    cv2.rectangle(frame_showing, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]),
                  (0, 255, 0), 3)
    frame_depth = np.zeros_like(frame)
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
    x1, y1, x2, y2 = max(int(bbox[0]), 0), max(int(bbox[1]), 0), min(int(bbox[0] + bbox[2]), frame.shape[1]), min(
        int(bbox[1] + bbox[3]), frame.shape[0])
    frame_output = np.zeros_like(frame)
    frame_output[y1:y2, x1:x2] = 255
    return frame_showing, frame_output, frame_depth


def get_mean_depth(depth_frame, bbox):
    if depth_frame is not None:
        roi = depth_frame[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        mean_depth = np.NaN if np.all(roi != roi) else np.nanmean(roi)/ 1e3
        if not np.isnan(mean_depth):
            print(f"{mean_depth} meters away")
        else:
            mean_depth = None
    else:
        mean_depth = None
    return mean_depth


def main():
    args = parse_args()

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print('device:', device)

    class_names, predictor = prepare_predictor(args.net_type, args.model_path, args.label_path)
    tracker = prepare_tracker(device, args.snapshot)

    print('Finished preparing, start streaming')
    frame_reader = FrameReader(args.video_name)

    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    stat_time = []
    initialized = False
    track_time = 0

    try:
        for frames in frame_reader:
            if isinstance(frames, tuple):
                colour_frame, depth_frame = frames
            else:
                colour_frame = frames
                depth_frame = None

            cur_time = time.time()
            boxes, labels, detect_img = do_object_detection(colour_frame, predictor, class_names)
            # Object detected
            if boxes.size(0) > 0:
                boxes = boxes.detach().cpu().numpy()
                bbox = [int(boxes[0, 0]), int(boxes[0, 1]), int(boxes[0, 2]), int(boxes[0, 3])]
                tracker.init(colour_frame, bbox)
                frame_showing, frame_output, frame_depth = show_result(detect_img, bbox, depth_frame)
                initialized = True
                track_time = 0
            else:
                # No object is detected
                if track_time > 20:
                    # loss track of the object
                    initialized = False
                    track_time = 0
                    frame_output = np.zeros_like(colour_frame)
                    frame_showing = colour_frame.copy()
                    frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                elif initialized:
                    # Already initialized a tracker. Keep tracking
                    outputs = tracker.track(colour_frame)
                    bbox = list(map(int, outputs['bbox']))
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    frame_showing, frame_output, frame_depth = show_result(colour_frame, bbox, depth_frame)
                    track_time += 1
                    # print('track_time:', track_time)
                else:
                    # No tracker is initialized. Show the whole frame.
                    frame_output = np.zeros_like(colour_frame)
                    frame_showing = colour_frame.copy()
                    frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('output', frame_output)
            cv2.imshow(video_name, frame_showing)
            cv2.imshow('depth', frame_depth)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                print('Exited the program by pressing q')
                cv2.destroyAllWindows()
                break

            stat_time.append(time.time() - cur_time)
            # print('iteration time = ', time.time() - cur_time)
        if len(stat_time) > 0:
            print('Average iteration time =', np.average(stat_time))

    finally:
        if isinstance(frame_reader.frame_iter, rs.pyrealsense2.pipeline):
            frame_reader.frame_iter.stop()


if __name__ == '__main__':
    main()
