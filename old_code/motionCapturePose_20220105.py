import numpy as np
import math
import cv2 as cv2
#import mediapipe as mp
#mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
#mp_pose = mp.solutions.pose
import jetson.inference
import jetson.utils

import argparse
import sys
import pandas as pd

import time
import pickle
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
font = jetson.utils.cudaFont()
body_language_class = "Initial"
########################################
# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.",
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="densenet121-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)
if opt.network == "densenet121-body":
    sys.argv.append("--network=densenet121-body")

net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

BOXSIZE_BASE = 3000
windowName = "DEMO"
BOXSIZE_MAX = 30000
POSESIZE_MARGIN = 10
POSESIZE_MARGIN_HEIGHT = 50
white_pixels_last_frame = 0
WHITE_PIXELS_DIFFERENCE_THRESHOLD = 50000
WHITE_PIXELS_MIN_THRESHOLD = 100

SHOW_FPS = False
USE_MODEL_PREDICTION = False

###################################################
# Optical flow
# def optical_flow(one, two):
#    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
#    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
#    hsv = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))
#    # set saturation
#    hsv[:, :, 1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:, :, 1]
#    # obtain dense optical flow paramters
#    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
#                                        pyr_scale=0.5, levels=1, winsize=15,
#                                        iterations=2,
#                                        poly_n=5, poly_sigma=1.1, flags=0)
#    # convert from cartesian to polar
#    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#    # hue corresponds to direction
#    hsv[:, :, 0] = ang * (180 / np.pi / 2)
#    # value corresponds to magnitude
#    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#    # convert HSV to int32's
#    hsv = np.asarray(hsv, dtype=np.float32)
#    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#   return rgb_flow


#####################################################


def on_change(val):
    global BOXSIZE_BASE
    BOXSIZE_BASE = int(val)
    imageCopy = frame1.copy()

#    cv.putText(imageCopy, str(val), (0, imageCopy.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.imshow(windowName, imageCopy)

# cap = cv2.VideoCapture("rtsp://user:Password155@192.168.50.101:554/Streaming/Channels/103/?transportmode=unicast")
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('D:/temp/handtrack/pose/0deg_lighton_andy.mp4')
cap = cv2.VideoCapture('0deg_lighton_andy.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
cv2.imshow(windowName, frame)
# cv2.createTrackbar('Threshold', windowName, 5000, MaxboxSize, on_change)

if SHOW_FPS:
    start = time.time()

#with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True, model_complexity=0,  enable_segmentation=False, smooth_segmentation=False) as pose:
if(1):
    while cap.isOpened():
        if SHOW_FPS:
            end = time.time()
            second = end - start
            fps = 1 / second
            start = time.time()
        try:
            frame2 = frame.copy()
            ret, frame = cap.read()
            frame1 = frame.copy()
        except:
            print("No frame received")
        else:

            # Calculate the per-element absolute difference between two frames
            motiondiff = cv2.absdiff(frame1, frame2)
    #        cv2.imshow("motiondiff", motiondiff)

            # Convert the image from BGR to grayscale
            gray = cv2.cvtColor(motiondiff, cv2.COLOR_BGR2GRAY)
    #        cv2.imshow("gray", gray)

            # Blur an image using Gaussian filter
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #        cv2.imshow("blur", blur)

            # If the pixel value is smaller than the threshold (20), it is set to 0.
            # Otherwise, it is set to a maximum value (255)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            # cv2.imshow("thresh", thresh)

            # Dilates an image by using a specific structuring element
            dilate = cv2.dilate(thresh, None, iterations=3)
            # cv2.imshow("dilate", dilate)

            # Get the number of white pixels (variation compared to last frame) of this frame
            white_pixels_this_frame = np.sum(dilate == 255)
            # print('number of white pixels: ', white_pixels_this_frame)

            # Skip this frame if number of white pixels is too small
            if white_pixels_this_frame < WHITE_PIXELS_MIN_THRESHOLD:
                if SHOW_FPS:
                    cv2.putText(frame, 'fps: ' + format(fps, '.4f'), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.imshow(windowName, frame1)
                continue

            # Calculate the difference of white pixels between this frame and last frame
            difference_white_pixels = white_pixels_this_frame - white_pixels_last_frame

            # Skip this frame if the difference is too large (i.e. false alarm)
            if difference_white_pixels > WHITE_PIXELS_DIFFERENCE_THRESHOLD:
                if SHOW_FPS:
                    cv2.putText(frame1, 'fps: ' + format(fps, '.4f'), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                cv2.imshow(windowName, frame1)
                continue

            # Save number of white pixels to next frame
            white_pixels_last_frame = white_pixels_this_frame

            # Find contours in a binary image
            # cv2.RETR_TREE: retrieves all of the contours and reconstructs a full hierarchy of nested contours
            # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
            contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                # Calculate the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image
                (x, y, w, h) = cv2.boundingRect(contour)

                # Scale box size according to y-coordinate (if the box is nearer to the bottom, the box size scale larger)
                boxSize_scale = int(BOXSIZE_BASE * (1 + 3 * y/cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

                # Skip this contour if the contour area is smaller than the scaled box size
                if cv2.contourArea(contour) < boxSize_scale:
                    continue

                # Skip this contour if the contour area is larger than the max box size
                elif cv2.contourArea(contour) > BOXSIZE_MAX:
                    continue

    #####################################
                print("##########################################")
                print("x     y     w     h     cv2.contourArea(contour) boxSize_scale")
                print("{0:<5} {1:<5} {2:<5} {3:<5} {4:<24} {5:<13}".format(x, y, w, h, cv2.contourArea(contour), boxSize_scale))

                # Add the margin to the box (If it exceed the screen, set to marginal value)
                if(y-POSESIZE_MARGIN > 0):
                    new_y = y-POSESIZE_MARGIN
                else:
                    new_y = 1

                if(x-POSESIZE_MARGIN > 0):
                    new_x = x-POSESIZE_MARGIN
                else:
                    new_x = 1

                if(y + h + POSESIZE_MARGIN + POSESIZE_MARGIN_HEIGHT < cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                    box_yh = int(y+h+POSESIZE_MARGIN + POSESIZE_MARGIN_HEIGHT)
                else:
                    box_yh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-1)

                if(x + w + POSESIZE_MARGIN < cap.get(cv2.CAP_PROP_FRAME_WIDTH)):
                    box_xw = int(x+w+POSESIZE_MARGIN)
                else:
                    box_xw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)-1)

                print("new_x new_y box_xw box_yh")
                print("{0:<5} {1:<5} {2:<6} {3:<6}".format(new_x, new_y, box_xw, box_yh))

                # Extract the image's height and width according to the box
                image = frame1[new_y: box_yh, new_x: box_xw]
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]

    #####################################
    #posenet
                # Copy the image to CUDA memory
                cuda_mem = jetson.utils.cudaFromNumpy(image)
                poses = net.Process(cuda_mem, overlay=opt.overlay)
                posearray = [0]*18
                body_language_class = "No Action"
    #            print("len(poses): ", len(poses))
                for pose in poses[0:1]:
    #                print("Inside Pose")
                    for Keypoint in pose.Keypoints:
                        if(Keypoint.ID == 5):       # Left Shoulder
                            posearray[0] = Keypoint.x/w
                            posearray[1] = Keypoint.y/h
                            posearray[2] = 1
                        elif(Keypoint.ID == 6):     # Right Shoulder
                            posearray[3] = Keypoint.x/w
                            posearray[4] = Keypoint.y/h
                            posearray[5] = 1
                        elif(Keypoint.ID == 7):     # Left Elbow
                            posearray[6] = Keypoint.x/w
                            posearray[7] = Keypoint.y/h
                            posearray[8] = 1
                        elif(Keypoint.ID == 8):     # Right Elbow
                            posearray[9] = Keypoint.x/w
                            posearray[10] = Keypoint.y/h
                            posearray[11] = 1
                        elif(Keypoint.ID == 9):     # Left Wrist
                            posearray[12] = Keypoint.x/w
                            posearray[13] = Keypoint.y/h
                            posearray[14] = 1
                        elif(Keypoint.ID == 10):    # Right Wrist
                            posearray[15] = Keypoint.x/w
                            posearray[16] = Keypoint.y/h
                            posearray[17] = 1
                if(sum(posearray)>0):
#                    print("posearray")
#                    print(posearray)
                    if USE_MODEL_PREDICTION:
                        pose_row = list(np.array(posearray).flatten())
                        row = pose_row
                        X = pd.DataFrame([row])
                        body_language_class = model.predict(X)[0]
                    else:
                        if posearray[13] < posearray[1]:
                            body_language_class = "ThrowL"
                        if posearray[16] < posearray[4]:
                            body_language_class = "ThrowR"
                    print("body_language_class: ", body_language_class)
    #               body_language_prob = model.predict_proba(X)[0]
    #               print(body_language_class, body_language_prob)
    #                    body_language_class = "Test"
    #            font.OverlayText(cuda_mem,cuda_mem.width, cuda_mem.height, "{:s}".format(str(body_language_class)),5,5,font.White)

                image = jetson.utils.cudaToNumpy(cuda_mem)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #            body_language_class = "Test"


    #######################################3
    ######################################
    # find sign
    #            gray_sign = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #            th, threshed_sign = cv2.threshold(
    #                gray_sign, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #            cv2.imshow("threshed",  threshed_sign)
    #            if cv2.waitKey(0) == 27:
    #                break
    #            contours_sign, _ = cv2.findContours(
    #                threshed_sign, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #            contours_sign, _ = cv2.findContours(
    #                threshed_sign, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #           for contour_sign in contours_sign:
    #                (x_sign, y_sign, w_sign, h_sign) = cv2.boundingRect(contour_sign)

    ######################################


    ##################################
    # Increase Sharpest
    #            kernel = np.array([[0, -1, 0],
    #                               [-1, 5, -1],
    #                               [0, -1, 0]])
    #            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #            image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    #######################################
    #            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    ####################################################3
    #            image.flags.writeable = False
    #            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #            results = pose.process(image)

            # Draw the pose annotation on the image.
    #            image.flags.writeable = True
    #            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #            mp_drawing.draw_landmarks(
    #                image,
    #                results.pose_landmarks,
    #                mp_pose.POSE_CONNECTIONS,
    #                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    #            image = cv2.resize(image, (box_xw, box_yh),
    #                                interpolation=cv2.INTER_AREA)
    #            cv2.rectangle(image, (x_sign, y_sign),
    #                          (w_sign, h_sign), (200, 200, 0), 2)

    #####################################

                frame1[new_y: box_yh, new_x: box_xw] = image
                # cv2.imshow("pose", image)
                # if cv2.waitKey(0) == 27:
                #     break
    #            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #            cv2.drawContours(frame1, contour, -1, (200, 0, 128), 2)
                cv2.rectangle(frame1, (new_x, new_y),
                            (box_xw, box_yh), (200, 0, 0), 2)
                frame1=cv2.putText(frame1, body_language_class, (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    ############################################################

    #        cv.putText(frame1, "Status: {}".format('Movement'),(10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    #        cv2.rectangle(frame1, (0, 0), (int(math.sqrt(boxSize)),
    #                                       int(math.sqrt(boxSize))), (100, 100, 255), 2)
    #       cv2.rectangle(frame1, (0, 0), (200,400), (100, 100, 255), 2)
            if SHOW_FPS:
                cv2.putText(frame1, 'fps: ' + format(fps, '.4f'), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv2.imshow(windowName, frame1)
    #    cv.createTrackbar('slider', windowName, 700, 2000, on_change)

    #    frame1 = frame2
    #    ret, frame2 = cap.read()
            if cv2.waitKey(1) == 27:
                break
cap.release()
cv2.destroyWindow()
