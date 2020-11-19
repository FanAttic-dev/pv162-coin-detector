import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
import math

BASE_DIR_PATH = "../IMAGES/Coins/CZK"
CLASS = "10"
LONGER_EDGE_SIZE = 1024
WINDOW_NAME = 'FindCircles'
COIN_SIZE = 180


def nothing(_):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('MinDist', WINDOW_NAME, 100, 800, nothing)
    cv2.createTrackbar('CannyHigh', WINDOW_NAME, 400, 800, nothing)
    cv2.createTrackbar('AccTh', WINDOW_NAME, 75, 200, nothing)


def init_mpl():
    mpl.rcParams['toolbar'] = 'None'
    if 's' in mpl.rcParams['keymap.save']:
        mpl.rcParams['keymap.save'].remove('s')


def resize_image(img_color):
    w, h = img_color.shape[:2]
    longer_edge = max(w, h)
    scale_factor = LONGER_EDGE_SIZE / longer_edge
    new_size = (int(h * scale_factor), int(w * scale_factor))
    return cv2.resize(img_color, new_size)


def detect_circles(img_color, show_edges=False):
    # check dimensions
    if max(img_color.shape[:2]) > LONGER_EDGE_SIZE:
        raise Exception('image too large')

    while True:
        # get params
        min_dist = cv2.getTrackbarPos('MinDist', WINDOW_NAME)
        canny_high = cv2.getTrackbarPos('CannyHigh', WINDOW_NAME)
        acc_threshold = cv2.getTrackbarPos('AccTh', WINDOW_NAME)

        if show_edges:
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img_gray, canny_high/2, canny_high)
            cv2.namedWindow("CannyWindow", cv2.WINDOW_NORMAL)
            cv2.imshow("CannyWindow", edges)

        # find circles
        circles = find_circles_in_image(img_color, min_dist, canny_high, acc_threshold)

        # visualize
        img_color_circles = draw_circles_into_image(img_color, circles)
        cv2.imshow(WINDOW_NAME, img_color_circles)

        key = cv2.waitKey(1) % 256
        if key == ord('s'):
            extract_circles(img_color, circles)

        if (key == ord('q')) or (key == ord('n')) or (key == ord('p')):
            break

    return key


def find_circles_in_image(img_color, min_dist, canny_high, acc_threshold, show_edges=True):
    # to grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # blur
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 1.5)

    return cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.5, minDist=min_dist, param1=canny_high, param2=acc_threshold)


def draw_circles_into_image(img_color, circles):
    if circles is None:
        return img_color

    img_with_circles = img_color.copy()

    circles = np.uint16(np.round(circles))
    for (x, y, r) in circles[0, :]:
        # outer circle
        cv2.circle(img_with_circles, (x, y), r, (0, 255, 0), 2)
        # circle center
        cv2.circle(img_with_circles, (x, y), 2, (0, 0, 255), 3)

    return img_with_circles


def save_circles(rois):
    print(len(rois))
    plt.close('all')


def extract_circles(img_color, circles):
    rois = []
    circles_count = len(circles[0])

    # init figure
    fig_size = math.ceil(math.sqrt(circles_count))
    fig = plt.figure()

    # create button
    axsave = plt.axes([0.89, 0.01, 0.1, 0.075])
    bsave = Button(axsave, 'Save')
    bsave.on_clicked(lambda _: save_circles(rois))
    fig.canvas.mpl_connect('key_release_event', lambda event: save_circles(rois) if event.key == 's' else nothing(0))

    for i in range(fig_size * fig_size):
        if i >= circles_count:
            break

        x, y, r = circles[0][i]
        fig.add_subplot(fig_size, fig_size, i + 1)
        roi = img_color[int(y-r):int(y+r), int(x-r):int(x+r)]
        rois.append(roi)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    plt.show()


def process_class(class_name):
    class_path = BASE_DIR_PATH + "/" + class_name
    class_orig_dir_path = class_path + "/original"
    class_image_names = os.listdir(class_orig_dir_path)

    i = 0
    while i < len(class_image_names):
        image_name = class_image_names[i]
        image_path = class_orig_dir_path + "/" + image_name

        # open image
        img = cv2.imread(image_path)

        # resize image
        img = resize_image(img)

        # detect circles
        key = detect_circles(img, show_edges=True)

        if key == ord('p'):
            i -= 1
            continue

        if key == ord('q'):
            break

        # by pressing 'n', continue
        i += 1


if __name__ == '__main__':
    init_mpl()
    create_trackbars()

    process_class(CLASS)

    cv2.destroyAllWindows()
