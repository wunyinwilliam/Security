import cv2
import csv
import os
import numpy as numpy
import jetson.inference
import jetson.utils

import argparse
import sys
import time
import pickle 
import numpy as np

posearray = [0]*18
num_coords = len(posearray)
landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val),'y{}'.format(val),'v{}'.format(val)]
#with open ('posecoords.csv', mode='w', newline='') as f:
#    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    csv_writer.writerow(landmarks)
font = jetson.utils.cudaFont()
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

net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val),'y{}'.format(val),'v{}'.format(val)]
#with open ('posecoords.csv', mode='w', newline='') as f:
#    csv_writer = csv.writer(f, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
#    csv_writer.writerow(landmarks)
#######################################################3

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture('D:/temp/handtrack/pose/0deg_lighton_andy.mp4')
#cap = cv2.VideoCapture('0deg_lighton_andy.mp4')
#Idle, Gun, ThrowR, ThrowL, ArmsUp
class_name= "ThrowR"
MaxDataCollect = 2010
DataCollect = MaxDataCollect
IdleCnt=10
while cap.isOpened():
    ret,frame = cap.read()
    h,w=frame.shape[:2]
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    cuda_mem = jetson.utils.cudaFromNumpy(image)
    poses = net.Process(cuda_mem, overlay=opt.overlay)
    posearray = [0]*18
    for pose in poses:
        for Keypoint in pose.Keypoints:
            if(Keypoint.ID == 5):
                posearray[0] = Keypoint.x/w
                posearray[1] = Keypoint.y/h
                posearray[2] = 1
            elif(Keypoint.ID == 6):
                posearray[3] = Keypoint.x/w
                posearray[4] = Keypoint.y/h
                posearray[5] = 1
            elif(Keypoint.ID == 7):
                posearray[6] = Keypoint.x/w
                posearray[7] = Keypoint.y/h
                posearray[8] = 1
            elif(Keypoint.ID == 8):
                posearray[9] = Keypoint.x/w
                posearray[10] = Keypoint.y/h
                posearray[11] = 1
            elif(Keypoint.ID == 9):
                posearray[12] = Keypoint.x/w
                posearray[13] = Keypoint.y/h
                posearray[14] = 1
            elif(Keypoint.ID == 10):
                posearray[15] = Keypoint.x/w
                posearray[16] = Keypoint.y/h
                posearray[17] = 1
            if(sum(posearray)>0):
                print(posearray)
                pose_row = list(np.array(posearray).flatten())
                row =pose_row
                row.insert(0, class_name)
                if(DataCollect < MaxDataCollect-IdleCnt):
                    with open('posecoords.csv',mode='a',newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)
                DataCollect = DataCollect - 1
                font.OverlayText(cuda_mem,cuda_mem.width, cuda_mem.height, "{:s}".format(str(DataCollect)),5,5,font.Green)

    image = jetson.utils.cudaToNumpy(cuda_mem)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imshow("Check", image)
    if cv2.waitKey(10) == 27:
        break
    if DataCollect < 0:
        break
cap.release()
cv2.destroyAllWindows()
