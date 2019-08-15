from Navigation import Navigation

import cv2
import time


def main():
    nav = Navigation()  # Navigation((w,h))
    try:
        noskip = True
        while True:
            start_time = time.time()
            if nav.get_frame() == 0:
                continue

            # Try to skip frames to make it faster
            if noskip:
                nav.do_object_detection()
                # noskip = False
            else:
                noskip = True
            nav.detect_n_track()

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
