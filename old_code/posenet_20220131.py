#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import numpy as np
import math
import cv2 as cv2
import pandas as pd
import time
import pickle 
import jetson.inference
import jetson.utils
import argparse
import sys
import math
actionBoxoffset = 30
y_offset = 100 # lower the detect y coz of the text display
##### MQTT Variable #######
boxSize = 3000 #min.boxsize
boxSizeCoeffx = 0.1 # linear coeff of min. box size
boxSizeCoeffy = 3 # linear coeff of min. box size
contoursMax = 10 #max number of contours
contoursThreshold = 25
#MQTT

import paho.mqtt.client as mqtt
########################################
#broker_address="192.168.50.160"
broker_address="127.0.0.1"
slave = "slave0"
########################################
subscribe_topic = slave + "/#"
#broker_address="iot.eclipse.org"
def on_message(client, userdata, message):
    global boxSize
    global boxSizeCoeffx
    global boxSizeCoeffy
    global contoursMax
    global contoursThreshold
    time.sleep(1)
#    print("received topic =",str(message.topic))
#    print("received message =",str(message.payload.decode("utf-8")))
    print("Message received-> " + message.topic + " " + str(message.payload.decode("utf-8")))  # 
    if(message.topic == (slave + "/boxSize")):
        boxSize = int(str(message.payload.decode("utf-8")))
    if(message.topic == (slave + "/boxSizeCoeffx")):
        boxSizeCoeffx = float(str(message.payload.decode("utf-8")))
    if(message.topic == (slave + "/boxSizeCoeffy")):
        boxSizeCoeffy = float(str(message.payload.decode("utf-8")))
    if(message.topic == (slave + "/contoursMax")):
        contoursMax = float(str(message.payload.decode("utf-8")))
    if(message.topic == (slave + "/contoursThreshold")):
        contoursThreshold = int(str(message.payload.decode("utf-8")))
print("creating new instance")
client = mqtt.Client(slave) #create new instance
client.on_message=on_message #attach function to callback
print("connecting to broker")
client.connect(broker_address) #connect to broker
client.loop_start() #start the loop
print("Subscribing to topic",subscribe_topic)
client.subscribe(subscribe_topic)

alive_cnt = 0
alive_cnt_max = 20
########################################

#with open('body_language.pkl', 'rb') as f:
#    model = pickle.load(f)
font = jetson.utils.cudaFont()
body_language_class = "Initial"

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
#parser.add_argument("--overlay", type=str, default="links,keypoints,box", help="pose overlay flags (e.g. --overlay=links,keypoints,box)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
#parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints,box)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--overlay", type=str, default="none", help="pose overlay flags (e.g. --overlay=links,keypoints,box)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
#img cuda space
#frame cv space
img = input.Capture()
frame = jetson.utils.cudaToNumpy(img)
frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
frame1 = frame
windowSizeX = int(img.width/4)
windowSizeY = int(img.height/4)

#MaxboxSize = 30000
MaxboxSize = windowSizeX*windowSizeY

# process frames until the user exits
movement_cnt = 0
action_cnt = 0
PTZx=0
PTZy=0
cuda_mem = jetson.utils.cudaAllocMapped(width=windowSizeX,
                                         height=windowSizeY,
                                         format=img.format)
#mask_img=cv2.read('',0) 
while True:
    print("#Frame Start&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    if(alive_cnt > alive_cnt_max):
        movement_topic = slave +"Out/movement"
        action_topic = slave +"Out/action"
        PTZx_topic = slave +"Out/PTZx"
        PTZy_topic = slave +"Out/PTZy"
        client.publish(movement_topic, payload=movement_cnt, qos=0, retain=False)
        client.publish(action_topic, payload=action_cnt, qos=0, retain=False)
        if(action_cnt !=0):
            client.publish(PTZx_topic, payload=PTZx, qos=0, retain=False)
            client.publish(PTZy_topic, payload=PTZy, qos=0, retain=False)
        alive_cnt = 0
        movement_cnt = 0
        action_cnt = 0
        PTZx=0
        PTZy=0        
    else:
        alive_cnt = alive_cnt + 1
    try:
        frame2 = frame.copy()
        img = input.Capture()
        frame = jetson.utils.cudaToNumpy(img)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame1 = frame
    except:
        print("skip frame")

    #CV stuff
    try:
        motiondiff = cv2.absdiff(frame1, frame2)
#        motiondiff = cv2.bitwise_and(motiondiff,motiondiff,mask=mask_img)
        gray = cv2.cvtColor(motiondiff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

#        erode_img = cv2.erode(thresh,iternations= 1)
#        dilate = cv2.dilate(erode_img, None, iterations=3)


        dilate = cv2.dilate(thresh, None, iterations=3)
#        dilate = cv2.dilate(thresh, np.ones((contoursThreshold,contoursThreshold),np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#        print("len(contours)")
#        print(len(contours))
        i =0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            boxSize_scale = int(boxSize * (1 + boxSizeCoeffx * x/img.width + boxSizeCoeffy * y/img.height))
            if cv2.contourArea(contour) < boxSize_scale:
                continue
            elif cv2.contourArea(contour) > MaxboxSize:
                continue
            if(i> contoursMax):
                continue
            i = i +1
############################################################
            print("#contour#########################################")
            movement_cnt = movement_cnt + 1
            new_x = x
            new_y = y
            if(new_x > img.width-windowSizeX):
                new_x = img.width-windowSizeX
            if(new_y > img.height-windowSizeY):
                new_y = img.height-windowSizeY
#            if(new_y < y_offset):
#                new_y = y_offset
            box_xw = new_x+windowSizeX
            box_yh = new_y+windowSizeY
            print("new_x, new_y, box_xw, box_yh")
            print(new_x, new_y, box_xw, box_yh)
            crop_roi = (new_x, new_y, box_xw, box_yh)
                                        
# crop the image to the ROI
            try:
                jetson.utils.cudaCrop(img, cuda_mem, crop_roi)
#            output.Render(cuda_mem)
            except:
                print("skip crop")

            h = cuda_mem.height
            w = cuda_mem.width
            poses = net.Process(cuda_mem, overlay=opt.overlay)
            body_language_class = "NoAction"
#            print("len(poses)")
#            print(len(poses))
#            for pose in poses[0:1]:
            for pose in poses:
                minX = w
                minY = h
                maxX = 0
                maxY = 0
                posearray = [0]*18
#                noseX = 0
#                noseY = 0
#                noseVisible = 0
                for Keypoint in pose.Keypoints:
                    if(minX > Keypoint.x):
                        minX = Keypoint.x
                    if(maxX < Keypoint.x):
                        maxX = Keypoint.x
                    if(minY > Keypoint.y):
                        minY = Keypoint.y
                    if(maxY < Keypoint.y):
                        maxY = Keypoint.y

#                    if(Keypoint.ID == 0): # nose
#                        noseX = Keypoint.x
#                        noseY = Keypoint.y
#                        noseVisible = 1
                    if(Keypoint.ID == 5): # Lshoulder
                        posearray[0] = Keypoint.x/w
                        posearray[1] = Keypoint.y/h #LshoulderY
                        posearray[2] = 1
                    elif(Keypoint.ID == 6): # Rshoulder
                        posearray[3] = Keypoint.x/w
                        posearray[4] = Keypoint.y/h #RshoulderY
                        posearray[5] = 1
                    elif(Keypoint.ID == 7):
                        posearray[6] = Keypoint.x/w
                        posearray[7] = Keypoint.y/h
                        posearray[8] = 1
                    elif(Keypoint.ID == 8):
                        posearray[9] = Keypoint.x/w
                        posearray[10] = Keypoint.y/h
                        posearray[11] = 1
                    elif(Keypoint.ID == 9): # Lwrist
                        posearray[12] = Keypoint.x/w
                        posearray[13] = Keypoint.y/h #LwristY
                        posearray[14] = 1
                    elif(Keypoint.ID == 10): # Rwrist
                        posearray[15] = Keypoint.x/w
                        posearray[16] = Keypoint.y/h #RwristY
                        posearray[17] = 1
#############################################################################
                if(minX > actionBoxoffset):
                    minX = minX - actionBoxoffset
                if(minY > actionBoxoffset):
                    minY = minY - actionBoxoffset
                if(maxX < w-actionBoxoffset):
                    maxX = maxX + actionBoxoffset
                if(maxY < w-actionBoxoffset):
                    maxY = maxY + actionBoxoffset
#if statement
                if(posearray[5] == 1 and posearray[17] == 1):
                    if(posearray[4] > posearray[16]):
                        body_language_class = "HandUp"
#                        jetson.utils.cudaDrawCircle(input, (int(posearray[15] * w) ,int(posearray[16] * h)), 25, (0,255,127,200))
#                        if(noseVisible ==1):
                        PTZx = int(posearray[15] * w)+new_x
                        PTZy = int(posearray[16] * h)+new_y
                        if(new_y > y_offset):
                            font.OverlayText(cuda_mem, cuda_mem.width, cuda_mem.height, "{:s}".format(body_language_class), int(posearray[15] * w) ,int(posearray[16] * h)-y_offset, font.Red, font.Gray40)
                        else:
                            font.OverlayText(cuda_mem, cuda_mem.width, cuda_mem.height, "{:s}".format(body_language_class), int(posearray[15] * w) ,int(posearray[16] * h)+1, font.Red, font.Gray40)
                        boxColor=(255,0,0,200) #red
                        jetson.utils.cudaDrawLine(cuda_mem, (minX,minY),(minX,maxY), boxColor,1)
                        jetson.utils.cudaDrawLine(cuda_mem, (minX,minY),(maxX,minY), boxColor,1)
                        jetson.utils.cudaDrawLine(cuda_mem, (minX,maxY),(maxX,maxY), boxColor,1)
                        jetson.utils.cudaDrawLine(cuda_mem, (maxX,minY),(maxX,maxY), boxColor,1)
                elif(posearray[2] == 1 and posearray[14] == 1):
                    if(posearray[1] > posearray[13]):
                        body_language_class = "HandUp"
#                        jetson.utils.cudaDrawCircle(input, (int(posearray[12] * w) ,int(posearray[13] * h)), 25, (0,255,127,200))
#                        if(noseVisible ==1):
                        PTZx = int(posearray[12] * w)+new_x
                        PTZy = int(posearray[13] * h)+new_y
                        if(new_y > y_offset):
                            font.OverlayText(cuda_mem, cuda_mem.width, cuda_mem.height, "{:s}".format(body_language_class), int(posearray[12] * w) ,int(posearray[13] * h)-y_offset, font.Red, font.Gray40)
                        else:
                            font.OverlayText(cuda_mem, cuda_mem.width, cuda_mem.height, "{:s}".format(body_language_class), int(posearray[12] * w) ,int(posearray[13] * h)+1, font.Red, font.Gray40)
                        boxColor=(255,0,0,200) #red
                        jetson.utils.cudaDrawLine(cuda_mem, (minX,minY),(minX,maxY), boxColor,1)
                        jetson.utils.cudaDrawLine(cuda_mem, (minX,minY),(maxX,minY), boxColor,1)
                        jetson.utils.cudaDrawLine(cuda_mem, (minX,maxY),(maxX,maxY), boxColor,1)
                        jetson.utils.cudaDrawLine(cuda_mem, (maxX,minY),(maxX,maxY), boxColor,1)
#############################################################################

#############################################################################
#classification
#            if(sum(posearray)>0):
#                pose_row = list(np.array(posearray).flatten())
#                row =pose_row
#                X = pd.DataFrame([row])
#                body_language_class = model.predict(X)[0]
#                body_language_prob = model.predict_proba(X)[0]
#############################################################################

                print("body_language_class")
                print(body_language_class)
            if(body_language_class != "NoAction"):
#                boxColor=(200,0,0,200) #red
#                jetson.utils.cudaDrawLine(cuda_mem, (1,1),(1,h-1), boxColor,1)
#                jetson.utils.cudaDrawLine(cuda_mem, (1,1),(w-1,1), boxColor,1)
#                jetson.utils.cudaDrawLine(cuda_mem, (1,h-1),(w-1,h-1), boxColor,1)
#                jetson.utils.cudaDrawLine(cuda_mem, (w-1,1),(w-1,h-1), boxColor,1)
#                font.OverlayText(cuda_mem, cuda_mem.width, cuda_mem.height, "{:s}".format(body_language_class), 5, 5, font.White, font.Gray40)
                action_cnt = action_cnt + 1
            else: #no action
                if(len(poses)>0): #no action but see pose
                    boxColor=(255,127,0,100) #orange
                    jetson.utils.cudaDrawLine(cuda_mem, (1,1),(1,h-1), boxColor,1)
                    jetson.utils.cudaDrawLine(cuda_mem, (1,1),(w-1,1), boxColor,1)
                    jetson.utils.cudaDrawLine(cuda_mem, (1,h-1),(w-1,h-1), boxColor,1)
                    jetson.utils.cudaDrawLine(cuda_mem, (w-1,1),(w-1,h-1), boxColor,1)
                else: # just move
                    boxColor=(0,127,0,50) #green
                    jetson.utils.cudaDrawLine(cuda_mem, (1,1),(1,h-1), boxColor,1)
                    jetson.utils.cudaDrawLine(cuda_mem, (1,1),(w-1,1), boxColor,1)
                    jetson.utils.cudaDrawLine(cuda_mem, (1,h-1),(w-1,h-1), boxColor,1)
                    jetson.utils.cudaDrawLine(cuda_mem, (w-1,1),(w-1,h-1), boxColor,1)
#            print("box printed")

            try:
            # compost the two images (the last two arguments are x,y coordinates in the output image)
                jetson.utils.cudaOverlay(cuda_mem, img, new_x, new_y)
#                jetson.utils.cudaOverlay(cuda_mem, img, 0,0)
            except:
                print("cudaOverlay out problem")
    except:
        print("skip memory error")


    #print("Before output display")

    try:
        boxSize_scalex = int(boxSize * (1 + boxSizeCoeffx))
        boxSize_scaley = int(boxSize * (1 + boxSizeCoeffy))
        boxSize_scalexy = int(boxSize * (1 + boxSizeCoeffx + boxSizeCoeffy))
        jetson.utils.cudaDrawRect(img, (1,1,int(math.sqrt(boxSize)),int(math.sqrt(boxSize))), (0,0,255,50))
        jetson.utils.cudaDrawRect(img, (1,img.height-int(math.sqrt(boxSize_scaley)),int(math.sqrt(boxSize_scaley)),img.height), (0,255,0,50))
        jetson.utils.cudaDrawRect(img, (img.width-int(math.sqrt(boxSize_scalex)),1,img.width,int(math.sqrt(boxSize_scalex))), (255,0,0,50))
        jetson.utils.cudaDrawRect(img, (img.width-int(math.sqrt(boxSize_scalexy)),img.height-int(math.sqrt(boxSize_scalexy)),img.width,img.height), (255,255,255,50))
    # render the image
        output.Render(img)
#    output.Render(cuda_mem)

    # update the title bar
        output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
#    net.PrintProfilerTimes()

    # exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break
    except:
        print("output display error")

    #print("After output display")
