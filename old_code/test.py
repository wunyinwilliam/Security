import time
import cv2
import numpy as np

white_pixels_last_frame = 0
WHITE_PIXELS_DIFFERENCE_THRESHOLD = 50000
WHITE_PIXELS_MIN_THRESHOLD = 100

cap = cv2.VideoCapture('0deg_lighton_andy.mp4')
ret, frame = cap.read()

while cap.isOpened():
    start = time.time()
    frame2 = frame.copy()
    ret, frame = cap.read()
    frame1 = frame.copy()
    motiondiff = cv2.absdiff(frame1, frame2)
    # cv2.imshow("motiondiff", motiondiff)
    gray = cv2.cvtColor(motiondiff, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("blur", blur)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    dilate = cv2.dilate(thresh, None, iterations=3)
    # cv2.imshow("dilate", dilate)

    # Get the number of white pixels (variation compared to last frame) of this frame
    white_pixels_this_frame = np.sum(dilate == 255)
    # print('number of white pixels: ', white_pixels_this_frame)

    # Skip this frame if number of white pixels is too small
    if white_pixels_this_frame < WHITE_PIXELS_MIN_THRESHOLD:
        continue

    # Calculate the difference of white pixels between this frame and last frame
    difference_white_pixels = white_pixels_this_frame - white_pixels_last_frame

    # Skip this frame if the difference is too large (i.e. false alarm)
    if difference_white_pixels > WHITE_PIXELS_DIFFERENCE_THRESHOLD:
        continue

    # Save number of white pixels to next frame
    white_pixels_last_frame = white_pixels_this_frame

    end = time.time()
    second = end - start
    fps = 1 / second
    cv2.putText(frame, 'fps: ' + format(fps, '.4f'), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyWindow()
