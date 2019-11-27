import numpy as np
import cv2
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Rerun Realsense Door')
    parser.add_argument('--folder', type=str, help='folder name of the data you want to rerun',
                        default='2019-10-22 16:10:32.848331')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path_rgb = os.path.join('data', args.folder, 'RGB')
    path_depth = os.path.join('data', args.folder, 'Depth')
    path_log = os.path.join('data', args.folder, args.folder + '.log')
    frame_shape = (640, 480)
    cv2.namedWindow('output', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('argus', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('output', 1400, 0)
    cv2.moveWindow('RealSense', 0, 0)
    cv2.moveWindow('depth', 700, 0)
    cv2.moveWindow('argus', 570, 280)
    # read text from log file
    with open(path_log, "r") as filestream:
        next(filestream)  # skip first row
        for line in filestream:
            # get log and change format
            bbox = [0, 0, 0, 0]
            currentline = line.split(", ")
            if len(currentline) == 5:
                [timestamp, frame_id, _, depth_value, command] = currentline
            else:
                [timestamp, frame_id, bbox[0], bbox[1], bbox[2], bbox[3], depth_value, command] = currentline
                bbox[0] = int(bbox[0].replace('[', ''))
                bbox[1] = int(bbox[1])
                bbox[2] = int(bbox[2])
                bbox[3] = int(bbox[3].replace(']', ''))
            if depth_value == 'None':
                depth_value = None
            else:
                depth_value = float(depth_value)
            command = command.replace(')', '')
            # read image
            image_rgb = cv2.imread(os.path.join(path_rgb, 'frame'+str(frame_id)+'.jpg'))
            image_depth = np.load(os.path.join(path_depth, 'frame'+str(frame_id)+'.npy'))
            frame_showing = image_rgb.copy()
            frame_depth = cv2.applyColorMap(cv2.convertScaleAbs(image_depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(frame_showing, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 3)
            cv2.putText(frame_showing, command,
                        (270, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # font scale
                        (0, 128, 255),
                        2)  # line type
            if image_depth is not None:
                if depth_value is not None:
                    cv2.putText(frame_showing, f"{depth_value:.2f} meters away",
                                (bbox[0] + 15, bbox[1] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,  # font scale
                                (255, 0, 255),
                                2)  # line type
                    cv2.rectangle(frame_depth, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]),
                                  (0, 255, 0), 3)
            x1, y1, x2, y2 = max(int(bbox[0]), 0), max(int(bbox[1]), 0), min(int(bbox[2]), frame_shape[0]), min(int(bbox[3]), frame_shape[1])
            # frame_output[y1:y2, x1:x2] = 255
            # cv2.imshow('output', frame_output)
            cv2.imshow('RealSense', frame_showing)
            cv2.imshow('depth', frame_depth)
            # cv2.imshow('argus', frame_argus)
            key = cv2.waitKey(50)
            if key & 0xFF == ord('q') or key == 27:
                print('Exited the program by pressing q')
                break


if __name__ == '__main__':
    main()


