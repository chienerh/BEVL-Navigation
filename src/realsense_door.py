from Navigation import Navigation

import cv2
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Realsense Door')
    parser.add_argument('--trk_config', type=str, help='tracking config file',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
    parser.add_argument('--trk_model', type=str, help='tracking model name',
                        default='pysot/experiments/siamrpn_alex_dwxcorr/model.pth')
    parser.add_argument('--video_name', default='', type=str,
                        help='videos or image files')
    parser.add_argument('--det_net_type', type=str, help='detection model type',
                        default='vgg16-ssd')
    parser.add_argument('--det_model', type=str, help='detection model name',
                        default='pytorch-ssd/models/20190622/vgg16-ssd-Epoch-30-Loss-3.0707082748413086.pth')
    parser.add_argument('--det_label', type=str, help='detection label path',
                        default='pytorch-ssd/models/20190622/open-images-model-labels.txt')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    nav = Navigation(args.det_net_type, args.det_model, args.det_label, args.trk_model,
                     args.trk_config)  # Navigation((w,h))
    try:
        while True:
            start_time = time.time()
            if not nav.get_frame():
                continue

            nav.do_object_detection()
            nav.detect_n_show()
            key = nav.show_img()
            if key & 0xFF == ord('q') or key == 27:
                print('Exited the program by pressing q')
                break

            # print("FPS: ", 1/(time.time()-start_time))

    finally:
        cv2.destroyAllWindows()
        nav.pipeline.stop()


if __name__ == '__main__':
    main()
