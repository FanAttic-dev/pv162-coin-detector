import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
import math

TRAIN_DIR_PATH = "../IMAGES/Coins/CZK"
ORIGINAL_DIR_PATH = "../IMAGES/Coins/CZK_test"
CLASS = "1"
LONGER_EDGE_SIZE = 1024
WINDOW_NAME = 'FindCircles'
COIN_SIZE = 180
EQUALIZE_HIST = True


class Image:
    def __init__(self, image_name, dir_path, orig_dir_path, resize_image=True, equalize_hist=True):
        self.name = image_name
        self.dir_path = dir_path
        self.orig_dir_path = orig_dir_path
        self.img_color = cv2.imread(orig_dir_path + "/" + image_name)

        if resize_image:
            self.resize()

        clahe_clip_limit = cv2.getTrackbarPos('ClaheClipLimit', WINDOW_NAME)
        if equalize_hist and clahe_clip_limit > 0:
            hsv = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(10, 10))
            value = clahe.apply(value)
            hsv = cv2.merge((hue, saturation, value))
            self.img_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.rois = []

    def resize(self):
        w, h = self.img_color.shape[:2]
        longer_edge = max(w, h)
        scale_factor = LONGER_EDGE_SIZE / longer_edge
        new_size = (int(h * scale_factor), int(w * scale_factor))
        self.img_color = cv2.resize(self.img_color, new_size)


def nothing(_):
    pass


def create_trackbars():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('MinDist', WINDOW_NAME, 100, 800, nothing)
    cv2.createTrackbar('CannyHigh', WINDOW_NAME, 400, 800, nothing)
    cv2.createTrackbar('AccTh', WINDOW_NAME, 75, 200, nothing)
    cv2.createTrackbar('ClaheClipLimit', WINDOW_NAME, 2, 60, nothing)


def init_mpl():
    mpl.rcParams['toolbar'] = 'None'
    if 's' in mpl.rcParams['keymap.save']:
        mpl.rcParams['keymap.save'].remove('s')


def detect_circles(img, show_edges=False):
    # check dimensions
    if max(img.img_color.shape[:2]) > LONGER_EDGE_SIZE:
        raise Exception('image too large')

    if show_edges:
        cv2.namedWindow("CannyWindow", cv2.WINDOW_NORMAL)

    while True:
        # get params
        min_dist = cv2.getTrackbarPos('MinDist', WINDOW_NAME)
        canny_high = cv2.getTrackbarPos('CannyHigh', WINDOW_NAME)
        acc_threshold = cv2.getTrackbarPos('AccTh', WINDOW_NAME)

        if show_edges:
            edges = cv2.Canny(img.img_gray, canny_high / 2, canny_high)
            cv2.imshow("CannyWindow", edges)

        # find circles
        circles = find_circles_in_image(img, min_dist, canny_high, acc_threshold)

        # visualize
        img_color_circles = draw_circles_into_image(img, circles)
        cv2.imshow(WINDOW_NAME, img_color_circles)

        key = cv2.waitKey(1) % 256
        if key == ord('s'):
            extract_circles(img, circles)

        if (key == ord('q')) or (key == ord('n')) or (key == ord('p')):
            break

    return key


def find_circles_in_image(img, min_dist, canny_high, acc_threshold, show_edges=True):
    # to grayscale
    img_gray = cv2.cvtColor(img.img_color, cv2.COLOR_BGR2GRAY)

    # blur
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 1.5)

    return cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.5, minDist=min_dist, param1=canny_high, param2=acc_threshold)


def draw_circles_into_image(img_color, circles):
    if circles is None:
        return img_color

    img_with_circles = img_color.img_color.copy()

    circles = np.uint16(np.round(circles))
    for (x, y, r) in circles[0, :]:
        # outer circle
        cv2.circle(img_with_circles, (x, y), r, (0, 255, 0), 2)
        # circle center
        cv2.circle(img_with_circles, (x, y), 2, (0, 0, 255), 3)

    return img_with_circles


def save_circles(img):
    for i in range(len(img.rois)):
        name, extension = os.path.splitext(img.name)
        path = img.dir_path + "/" + name + "_" + str(i+1) + extension
        print("Saving into " + path)
        cv2.imwrite(path, img.rois[i])

    print()
    plt.close()


def extract_circles(img_color, circles):
    circles_count = len(circles[0])

    # init figure
    fig_size = math.ceil(math.sqrt(circles_count))
    fig = plt.figure()

    # create button
    axsave = plt.axes([0.89, 0.01, 0.1, 0.075])
    bsave = Button(axsave, 'Save')
    bsave.on_clicked(lambda _: save_circles(img_color))
    fig.canvas.mpl_connect('key_release_event', lambda event: save_circles(img_color) if event.key == 's' else nothing(0))

    for i in range(fig_size * fig_size):
        if i >= circles_count:
            break

        x, y, r = circles[0][i]
        fig.add_subplot(fig_size, fig_size, i + 1)
        roi = img_color.img_color[int(y - r):int(y + r), int(x - r):int(x + r)]
        img_color.rois.append(roi)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    plt.show(block=False)


def process_class(class_name):
    class_dir_path = TRAIN_DIR_PATH + "/" + class_name
    class_orig_dir_path = ORIGINAL_DIR_PATH + "/original/" + class_name

    i = 0
    class_image_names = os.listdir(class_orig_dir_path)
    while i < len(class_image_names):
        image_name = class_image_names[i]
        img = Image(image_name, class_dir_path, class_orig_dir_path, resize_image=True, equalize_hist=EQUALIZE_HIST)

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
    # init GUI
    init_mpl()
    create_trackbars()

    # load, visualize and save a class
    process_class(CLASS)

    cv2.destroyAllWindows()
