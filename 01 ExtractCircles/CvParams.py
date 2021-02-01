import os
import cv2

IMG_PATH = "/home/atti/Desktop/IMG_20210201_134815.jpg"
PREVIEW_WINDOW = "Display window"
CANNY_WINDOW = "Canny window"


def load_image(path):
    return cv2.imread(path)


def bgr2hsv(img_color):
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    return cv2.split(hsv)


def hsv2bgr(hue, saturation, value):
    hsv = cv2.merge((hue, saturation, value))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def equalize_hist(img_gray):
    clipLimit = cv2.getTrackbarPos('ClaheClipLimit', PREVIEW_WINDOW)
    claheSize = max(cv2.getTrackbarPos('ClaheSize', PREVIEW_WINDOW), 1)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(claheSize, claheSize))

    return clahe.apply(img_gray)


def nothing(_):
    pass


def canny(img_gray):
    canny_high = cv2.getTrackbarPos('CannyHigh', CANNY_WINDOW)
    return cv2.Canny(img_gray, canny_high / 2, canny_high)


def blur(img_gray):
    sigma = cv2.getTrackbarPos('GaussSigma', PREVIEW_WINDOW)
    size = 0
    return cv2.GaussianBlur(img_gray, (size, size), sigma)


if __name__ == '__main__':
    # GUI
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('ClaheClipLimit', PREVIEW_WINDOW, 2, 60, nothing)
    cv2.createTrackbar('ClaheSize', PREVIEW_WINDOW, 10, 100, nothing)
    cv2.createTrackbar('GaussSigma', PREVIEW_WINDOW, 2, 60, nothing)

    cv2.namedWindow(CANNY_WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('CannyHigh', CANNY_WINDOW, 200, 800, nothing)

    # load img
    while True:
        img_color = load_image(IMG_PATH)

        hue, saturation, value = bgr2hsv(img_color)

        if cv2.getTrackbarPos('ClaheClipLimit', PREVIEW_WINDOW) > 0:
            value = equalize_hist(value)

        value = blur(value)

        img_color_eq = hsv2bgr(hue, saturation, value)

        edges = canny(value)


        cv2.imshow(PREVIEW_WINDOW, img_color_eq)
        cv2.imshow(CANNY_WINDOW, edges)

        key = cv2.waitKey(1) % 256

        if key == ord('q'):
            break
