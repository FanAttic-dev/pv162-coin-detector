import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage import io
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter

LONGER_EDGE_SIZE = 1024
WINDOW_NAME = 'FindCircles'
COIN_SIZE = 180


def nothing(x):
    pass


def create_main_window():
    # create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('MinDist', WINDOW_NAME, 100, 800, nothing)
    cv2.createTrackbar('CannyHigh', WINDOW_NAME, 150, 800, nothing)
    cv2.createTrackbar('AccTh', WINDOW_NAME, 75, 200, nothing)


def find_circles_in_image(img_color, min_dist, canny_high, acc_threshold, show_edges=True):
    # to grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # blur
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 1.5)

    if show_edges:
        edges = cv2.Canny(img_gray, canny_high/2, canny_high)
        cv2.imshow("CannyWindow", edges)

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


def save_circles(img_color, circles):
    cv2.namedWindow("roi")

    circles_count = len(circles[0])
    for i in range(circles_count):
        x, y, r = circles[0][i]
        cv2.imshow("roi", img_color[int(y-r):int(y+r), int(x-r):int(x+r)])

        # TODO save

        # proceed
        key = cv2.waitKey(0) % 256

        if key == ord('q'):
            break

    cv2.destroyWindow("roi")


def detect_circles(window_name, img_color, show_edges=False):
    if max(img.shape[:2]) > LONGER_EDGE_SIZE:
        raise Exception('image too large')

    while True:
        min_dist = cv2.getTrackbarPos('MinDist', window_name)
        canny_high = cv2.getTrackbarPos('CannyHigh', window_name)
        acc_threshold = cv2.getTrackbarPos('AccTh', window_name)

        circles = find_circles_in_image(img_color, min_dist, canny_high, acc_threshold, show_edges=show_edges)

        img_color_circles = draw_circles_into_image(img_color, circles)

        cv2.imshow(window_name, img_color_circles)

        key = cv2.waitKey(1) % 256

        if key == ord('s'):
            save_circles(img_color, circles)
        elif key == ord('q'):
            print("q")
            break


def resize_image(img_color):
    w, h = img_color.shape[:2]
    longer_edge = max(w, h)
    scale_factor = LONGER_EDGE_SIZE / longer_edge
    new_size = (int(h * scale_factor), int(w * scale_factor))
    return cv2.resize(img_color, new_size)


if __name__ == '__main__':
    create_main_window()

    # open image
    many_coins_path = "../IMAGES/Coins/CZK/1/original/IMG_20201118_155626.jpg"
    img = cv2.imread(many_coins_path)

    # resize image
    img = resize_image(img)

    # detect circles
    detect_circles(WINDOW_NAME, img)

    cv2.destroyAllWindows()

