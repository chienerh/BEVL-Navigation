from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
# import torch
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


class Navigation:
    def __init__(self, det_net_type, det_model_path, det_label_path, trk_model_path, trk_config,
                 frame_shape=(640, 480)):
        # Initialization for RealSense
        self.frame_shape = frame_shape
        self.frame_shape_hw = (self.frame_shape[1], self.frame_shape[0])
        self.pipeline = None
        self.align = None
        self.depth_value = None
        self.depth_image = None
        self.color_image = None

        # Initialization for Display
        self.frame_output = np.zeros(self.frame_shape_hw)
        self.frame_showing = np.zeros(self.frame_shape_hw)
        self.frame_depth = np.zeros(self.frame_shape_hw)
        self.frame_argus = np.zeros(self.frame_shape_hw)

        # Initialization for Functions
        self.initialized = False
        self.track_time = 0

        # Initialization for Detect & Tracking
        self.predictor = None
        self.class_names = []
        self.detect_img = None
        self.pred_boxes = None
        self.pred_labels = []
        self.tracker = None
        self.bbox = []  # x1, y1, x2, y2

        # Initialization for Feedback
        self.bbox_limit = []  # bbox_limit=[x1, y1, x2, y2, h, w]
        self.command = None
        self.last_cmd = None
        self.timer = time.time()

        # Set up
        self.setup_pipeline(det_net_type, det_model_path, det_label_path, trk_model_path, trk_config)
        self.setup_realsense()

    def reset(self):
        self.frame_output = np.zeros(self.frame_shape_hw)
        self.depth_value = None
        self.depth_image = None
        self.color_image = None
        self.command = None
        self.bbox = None
        self.pred_boxes = []

    def prepare_predictor(self, net_type, model_path, label_path):
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        num_classes = len(self.class_names)
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
            self.predictor = create_vgg_ssd_predictor(net, candidate_size=200)
        elif net_type == 'mb1-ssd':
            self.predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
        elif net_type == 'mb1-ssd-lite':
            self.predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
        elif net_type == 'mb2-ssd-lite':
            self.predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
        elif net_type == 'sq-ssd-lite':
            self.predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
        else:
            raise ValueError("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")

    def prepare_tracker(self, device, model_path):
        import torch
        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(model_path,
                                         map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # build tracker
        self.tracker = build_tracker(model)

    def setup_pipeline(self, det_net_type, det_model_path, det_label_path, trk_model_path, trk_config):
        # load config
        import torch
        cfg.merge_from_file(trk_config)
        cfg.CUDA = torch.cuda.is_available()
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        print('device:', device)

        self.prepare_predictor(det_net_type, det_model_path, det_label_path)
        self.prepare_tracker(device, trk_model_path)

    def setup_realsense(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.frame_shape[0], self.frame_shape[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.frame_shape[0], self.frame_shape[1], rs.format.bgr8, 30)

        self.bbox_limit = self.calculate_bbox_limit()  # bbox_limit=[x1, y1, x2, y2, h, w]

        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.pipeline.start(config)

        cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('argus', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('output', 700, 540)
        cv2.moveWindow('RealSense', 0, 0)
        cv2.moveWindow('depth', 700, 0)
        cv2.moveWindow('argus', 0, 540)

    def calculate_bbox_limit(self):
        # horizontal and vertical field of view of Argus II
        argus_fov_h = 19
        argus_fov_v = 11

        # horizontal and vertical field of view of Realsense
        rs_fov_h = 69.4
        rs_fov_v = 42.5

        # Find pixel range for Argus II
        w, h = self.frame_shape
        h_argus = int(h * argus_fov_v / rs_fov_v)
        w_argus = int(w * argus_fov_h / rs_fov_h)

        # Four corners of Argus II field of view
        top = (h // 2) - (h_argus // 2)
        bottom = (h // 2) + (h_argus // 2)
        left = (w // 2) - (w_argus // 2)
        right = (w // 2) + (w_argus // 2)
        bbox_limit = [left, top, right, bottom, bottom - top, right - left]

        return bbox_limit

    def do_object_detection(self):
        image = cv2.cvtColor(self.frame_showing, cv2.COLOR_BGR2RGB)  # (480, 640, 3)
        self.pred_boxes, labels, probs = self.predictor.predict(image, 10, 0.6)

        self.pred_labels = []
        for i in range(self.pred_boxes.size(0)):
            box = self.pred_boxes[i, :]
            label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"
            self.pred_labels.append(self.class_names[labels[i]])
            cv2.rectangle(self.frame_showing, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(self.frame_showing, label,
                        (box[0] + 15, box[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # font scale
                        (255, 0, 255),
                        2)  # line type

    def get_frame(self):
        self.last_cmd = self.command
        self.reset()
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return False

        self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())
        self.frame_showing = self.color_image.copy()

        return True

    def show_img(self):
        self.frame_argus = cv2.resize(self.frame_argus, (640, 480))
        cv2.imshow('output', self.frame_output)
        cv2.imshow('RealSense', self.frame_showing)
        cv2.imshow('depth', self.frame_depth)
        cv2.imshow('argus', self.frame_argus)
        key = cv2.waitKey(1)

        return key

    def detected(self):
        # Object detected. Initialize tracker
        boxes = self.pred_boxes.detach().cpu().numpy()  # boxes=[x1, y1, x2, y2]
        bbox_tracker = [int(boxes[0, 0]), int(boxes[0, 1]), int(boxes[0, 2] - boxes[0, 0]),
                        int(boxes[0, 3] - boxes[0, 1])]  # bbox=[x1, y1, w, h]
        self.tracker.init(self.color_image, bbox_tracker)
        self.bbox = [int(boxes[0, 0]), int(boxes[0, 1]), int(boxes[0, 2]), int(boxes[0, 3])]  # bbox=[x1, y1, x2, y2]
        self.show_result()
        self.argus2()
        self.initialized = True
        self.track_time = 0

    def no_tracker(self):
        # No tracker is initialized. Show the whole frame.
        self.frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.argus2()

    def keep_tracking(self):
        # Already initialized a tracker. Keep tracking
        outputs = self.tracker.track(self.color_image)
        bbox = list(map(int, outputs['bbox']))  # bbox=[x1, y1, w, h]
        self.bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # bbox=[x1, y1, x2, y2]
        self.show_result()
        self.argus2()
        self.track_time += 1

    def lose_track(self):
        # loss track of the object
        self.frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.argus2()
        self.initialized = False
        self.track_time = 0

    def argus2(self):
        self.frame_argus = np.zeros((self.bbox_limit[4], self.bbox_limit[5]), np.uint8)
        # Add Argus FOV box on frame_showing
        cv2.rectangle(self.frame_showing, (self.bbox_limit[0], self.bbox_limit[1]),
                      (self.bbox_limit[2], self.bbox_limit[3]),
                      (255, 0, 0), 3)

        if self.bbox:
            # check if bbox_argus is in bbox_limit
            if self.bbox[0] < self.bbox_limit[0] and self.bbox[2] < self.bbox_limit[0]:
                bbox_argus = None
                self.command = 'Left'
            elif self.bbox[0] > self.bbox_limit[2] and self.bbox[2] > self.bbox_limit[2]:
                bbox_argus = None
                self.command = 'Right'
            elif self.bbox[1] < self.bbox_limit[1] and self.bbox[3] < self.bbox_limit[1]:
                bbox_argus = None
                self.command = 'Up'
            elif self.bbox[1] > self.bbox_limit[3] and self.bbox[3] > self.bbox_limit[3]:
                bbox_argus = None
                self.command = 'Down'
            else:
                bbox_argus = [max(self.bbox_limit[0], self.bbox[0]), max(self.bbox_limit[1], self.bbox[1]),
                              min(self.bbox_limit[2], self.bbox[2]),
                              min(self.bbox_limit[3], self.bbox[3])]
                self.command = 'Forward'

            # Add bbox on Argus
            if bbox_argus:
                cv2.rectangle(self.frame_showing, (bbox_argus[0], bbox_argus[1]),
                              (bbox_argus[2], bbox_argus[3]),
                              (0, 0, 255), 3)

                # Output for displaying on ArgusII
                x1, y1, x2, y2 = max(bbox_argus[0] - self.bbox_limit[0], 0), max(bbox_argus[1] - self.bbox_limit[1],
                                                                                 0), min(
                    bbox_argus[2] - self.bbox_limit[0], self.bbox_limit[5]), min(
                    int(bbox_argus[3] - self.bbox_limit[1]), self.bbox_limit[4])
                area_ratio = ((x2-x1)*(y2-y1)) / (self.bbox_limit[4] * self.bbox_limit[5])
                if area_ratio > 0.5:
                    self.frame_argus[y1:y2, x1:x2] = 255 - area_ratio * 128
                else:
                    self.frame_argus[y1:y2, x1:x2] = 255

    def show_result(self):
        cv2.rectangle(self.frame_showing, (self.bbox[0], self.bbox[1]),
                      (self.bbox[2], self.bbox[3]),
                      (0, 255, 0), 3)
        if self.depth_image is not None:
            self.depth_value = self.get_mean_depth()
            if self.depth_value is not None and not np.isnan(self.depth_value):
                cv2.putText(self.frame_showing, f"{self.depth_value:.2f} meters away",
                            (self.bbox[0] + 15, self.bbox[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,  # font scale
                            (255, 0, 255),
                            2)  # line type
                self.frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03),
                                                     cv2.COLORMAP_JET)
                cv2.rectangle(self.frame_depth, (self.bbox[0], self.bbox[1]),
                              (self.bbox[2], self.bbox[3]),
                              (0, 255, 0), 3)
        x1, y1, x2, y2 = max(int(self.bbox[0]), 0), max(int(self.bbox[1]), 0), min(int(self.bbox[2]),
                                                                                   self.frame_shape[0]), min(
            int(self.bbox[3]), self.frame_shape[1])
        self.frame_output[y1:y2, x1:x2] = 255

    def get_mean_depth(self):
        if self.depth_image is not None:
            roi = self.depth_image[self.bbox[0]: self.bbox[2], self.bbox[1]: self.bbox[3]]
            mean_depth = np.NaN if np.all(roi != roi) else np.nanmean(roi) / 1e3
            if not np.isnan(mean_depth):
                return mean_depth
            else:
                return None
        else:
            return None

    def cmd_to_arduino(self, arduino):
        if self.command == 'Left':
            arduino.write('2'.encode())
        elif self.command == 'Right':
            arduino.write('3'.encode())
        elif self.command == 'Forward':
            arduino.write('4'.encode())
        elif self.command == 'Stop':
            arduino.write('0'.encode())
        self.timer = time.time()

    def give_cmd(self, arduino):
        if self.command is not None:
            print('%s        \r' % self.command, end="")
            if time.time() - self.timer > 1.0:
                self.cmd_to_arduino(arduino)
            else:
                if self.last_cmd is not None:
                    if self.last_cmd is not self.command:
                        self.cmd_to_arduino(arduino)
        else:
            print('%s        \r' % 'no command', end="")

    def detect_n_track(self):
        if len(self.pred_boxes) > 0:  # Object detected
            self.detected()
        else:  # No object is detected
            if self.track_time > 50:
                self.lose_track()
            elif self.initialized:
                self.keep_tracking()
            else:
                self.no_tracker()
        if self.depth_value:
            if self.depth_value <= 2.0:
                self.command = 'Stop'
        print('%s        \r' % 'no command', end="")
