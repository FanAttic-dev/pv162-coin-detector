# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage import io
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter

LONGER_EDGE_SIZE = 1024

def nothing(x):
    pass


def video_capture(window_name, show_edges=False):
    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()

        min_dist = cv2.getTrackbarPos('MinDist', window_name)
        canny_high = cv2.getTrackbarPos('CannyHigh', window_name)
        acc_threshold = cv2.getTrackbarPos('AccTh', window_name)

        circles = find_circles(frame, min_dist, canny_high, acc_threshold, show_edges=show_edges)
        frame_with_circles = draw_circles(frame, circles)

        cv2.imshow(window_name, frame_with_circles)

        key = cv2.waitKey(1)

        if key == ord('s'):
            print("process circles")
            break
        elif key == ord('q'):
            break

    vid.release()


def find_circles(img_color, min_dist, canny_high, acc_threshold, show_edges=True):
    # to grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # blur
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 1.5)

    if show_edges:
        edges = cv2.Canny(img_gray, canny_high/2, canny_high)
        cv2.imshow("CannyWindow", edges)

    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.5, minDist=min_dist, param1=canny_high, param2=acc_threshold)

    return circles


def draw_circles(img, circles):
    if circles is None:
        return img

    img_with_circles = img.copy()

    # draw circle
    circles = np.uint16(np.round(circles))
    for (x, y, r) in circles[0, :]:
        # draw the outer circle
        cv2.circle(img_with_circles, (x, y), r, (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img_with_circles, (x, y), 2, (0, 0, 255), 3)

    return img_with_circles


def process_circles(img_color, circles):
    print(circles.shape)

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


def detect_circles(window_name, img_path, show_edges=False):
    # open image
    img_color = cv2.imread(img_path)

    # resize
    w, h = img_color.shape[:2]
    longer_edge = max(w, h)
    scale_factor = LONGER_EDGE_SIZE / longer_edge
    cv2.resize(img_color, (int(w * scale_factor), int(h * scale_factor)))

    print(img_color.shape)

    while True:
        min_dist = cv2.getTrackbarPos('MinDist', window_name)
        canny_high = cv2.getTrackbarPos('CannyHigh', window_name)
        acc_threshold = cv2.getTrackbarPos('AccTh', window_name)

        circles = find_circles(img_color, min_dist, canny_high, acc_threshold, show_edges=show_edges)

        img_color_circles = draw_circles(img_color, circles)

        cv2.imshow(window_name, img_color_circles)

        key = cv2.waitKey(1) % 256

        if key == ord('s'):
            process_circles(img_color, circles)
        elif key == ord('q'):
            print("q")
            break


if __name__ == '__main__':
    # create window
    window_name = 'FindCircles'
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('MinDist', window_name, 100, 800, nothing)
    cv2.createTrackbar('CannyHigh', window_name, 150, 800, nothing)
    cv2.createTrackbar('AccTh', window_name, 75, 200, nothing)

    # static image
    many_coins_path = "../IMAGES/Coins/Test/euro_coins_many.jpg"
    detect_circles(window_name, many_coins_path)

    # video capture
    #video_capture(window_name)

    cv2.destroyAllWindows()

