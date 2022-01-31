import numpy as np
import cv2
import cv2 as cv2
import pandas as pd
import time
import pickle
import jetson.inference
import jetson.utils
import argparse
import sys
import math

drawing = False  # true if mouse is pressed
ix, iy = -1, -1
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())
parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")

opt = parser.parse_known_args()[0]

input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)


def draw_boundary(frame, boxes):
    for box in boxes:
        mask = np.full((box[3]-box[1], box[2]-box[0], 3), 0).astype(np.uint8)
        frame[box[1]:box[3], box[0]:box[2]] = mask

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, draw_img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
            cv2.rectangle(draw_img, (ix, iy), (x, y), (0, 0, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
        cv2.rectangle(draw_img, (ix, iy), (x, y), (0, 0, 0), -1)
        cv2.imwrite('draw_mask.png', draw_img)


img = cv2.cvtColor(jetson.utils.cudaToNumpy(input.Capture()), cv2.COLOR_BGR2RGB)
# img = cv2.imread("Background.png", cv2.IMREAD_UNCHANGED)
height, width, _ = img.shape
edge_pixel = 20
edges = [[0, 0, edge_pixel, height-1],
         [0, 0, width-1, edge_pixel],
         [0, height-edge_pixel, width-1, height-1],
         [width-edge_pixel, 0, width-1, height-1]]


draw_img = np.zeros((height, width, 3), np.uint8)
draw_img[:] = (255, 255, 255)
draw_boundary(draw_img, edges)
cv2.imwrite('draw_mask.png', draw_img)

cv2.namedWindow('Draw Masks')
cv2.setMouseCallback('Draw Masks', draw_circle)

while 1:
    cv2.imshow('Draw Masks', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
