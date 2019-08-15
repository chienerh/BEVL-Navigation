from Navigation import Navigation

import cv2
import serial


def main():
    nav = Navigation()
    try:
        # connect to arduino
        arduino = serial.Serial("/dev/rfcomm0", 9600)
        while True:
            if nav.get_frame() == 0:
                continue

            nav.do_object_detection()
            nav.detect_n_track()
            nav.give_cmd(arduino)

            key = nav.show_img()
            if key & 0xFF == ord('q') or key == 27:
                print('Exited the program by pressing q')
                break

    finally:
        cv2.destroyAllWindows()
        nav.pipeline.stop()


if __name__ == '__main__':
    main()