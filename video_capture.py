import cv2

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
