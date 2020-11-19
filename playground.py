import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img1 = cv2.imread("../IMAGES/Coins/CZK/1/original/IMG_20201118_155626.jpg")
    #img2 = cv2.imread("../IMAGES/Coins/CZK/1/original/IMG_20201118_155817.jpg")
    cv2.namedWindow("I", cv2.WINDOW_NORMAL)
    cv2.imshow("I", img1)
    plt.imshow(img1)
    plt.show()

    cv2.destroyAllWindows()
