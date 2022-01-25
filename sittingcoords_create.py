import cv2
import csv
import os
import numpy as numpy
posearray = [0]*6
num_coords = len(posearray)
landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val),'y{}'.format(val),'v{}'.format(val)]
with open ('posecoords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
