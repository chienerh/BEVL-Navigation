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

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
    parser.add_argument('--snapshot', type=str, help='model name',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/model.pth')
    parser.add_argument('--video_name', default='data/EPSON/20190214_clip/20190214_trim.mp4', type=str,
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
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.5)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    # cv2.imshow('annotated', orig_image)
    return boxes


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

    def iter_next(self):
        if isinstance(self.frame_iter, cv2.VideoCapture):
            ret, frame = self.frame_iter.read()
            # print(f"FPS: {self.frame_iter.get(cv2.CAP_PROP_FPS)}")
            if not ret:
                raise StopIteration
            return frame

        return cv2.imread(next(self.frame_iter))

    def _init_iters(self):
        if not self.video_name:
            self.frame_iter = cv2.VideoCapture(0)
            # warmup
            for i in range(5):
                self.frame_iter.read()

        elif self.video_name.endswith('avi') or \
                self.video_name.endswith('mp4'):
            self.frame_iter = cv2.VideoCapture(self.video_name)
        else:
            images = glob.iglob(os.path.join(self.video_name, '*.jp*'))
            self.frame_iter = sorted(images,
                                     key=lambda x: int(x.split('/')[-1].split('.')[0]))


def show_result(frame, bbox):
    frame_showing = frame.copy()
    cv2.rectangle(frame_showing, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  (0, 255, 0), 3)
    x1, y1, x2, y2 = max(int(bbox[0]), 0), max(int(bbox[1]), 0), min(int(bbox[0] + bbox[2]), frame.shape[1]), min(
        int(bbox[1] + bbox[3]), frame.shape[0])
    frame_output = np.zeros_like(frame)
    frame_output[y1:y2, x1:x2] = 255
    final_frame = cv2.hconcat((frame_output, frame_showing))
    return final_frame


def main():
    args = parse_args()

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    class_names, predictor = prepare_predictor(args.net_type, args.model_path, args.label_path)
    tracker = prepare_tracker(device, args.snapshot)
    frame_reader = FrameReader(args.video_name)

    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    stat_time = []

    # Stops when the predictor first detects a door
    boxes = torch.zeros(0, 4)
    frame = None
    while boxes.size(0) == 0:
        try:
            frame = next(frame_reader)
        except StopIteration:
            break
        boxes = do_object_detection(frame, predictor, class_names)

    boxes = boxes.detach().cpu().numpy()
    tracker.init(frame, boxes[0])

    for frame in frame_reader:
        cur_time = time.time()
        boxes = do_object_detection(frame, predictor, class_names)
        # Object detected
        if boxes.size(0) > 0:
            boxes = boxes.detach().cpu().numpy()
            bbox = [boxes[0, 0], boxes[0, 1], boxes[0, 2] - boxes[0, 0], boxes[0, 3] - boxes[0, 1]]
            tracker.init(frame, bbox)
            final_frame = show_result(frame, bbox)
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame_showing = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                final_frame = cv2.hconcat((frame, frame_showing))
            else:
                bbox = list(map(int, outputs['bbox']))
                final_frame = show_result(frame, bbox)

        cv2.imshow(video_name, final_frame)
        keyPressed = cv2.waitKey(1) & 0xff

        if keyPressed == 27 or keyPressed == 1048603:
            print('exited the program by pressing ESC')
            break  # esc to quit

        stat_time.append(time.time() - cur_time)
        # print('iteration time = ', time.time() - cur_time)
    print('average iteration time =', np.average(stat_time))


if __name__ == '__main__':
    main()
