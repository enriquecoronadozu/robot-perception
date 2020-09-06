#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Get an image. Display it and save it using PIL."""

import qi
import argparse
import sys
import time
import numpy
import cv2
import vision_definitions
import threading
import nep

from naoqi import ALProxy
from naoqi_sensors.vision_definitions import kYUV422ColorSpace, kYUVColorSpace, kRGBColorSpace, kBGRColorSpace, kDepthColorSpace, kRawDepthColorSpace


ip  = "192.168.11.22"
resolution = 1

try:
    print ("IP: " + sys.argv[1])
    ip = sys.argv[1]
    print ("resolution: " + sys.argv[2])
    resolution = int(sys.argv[2])
except:
    pass

### Image format
##k960p = 3                # 1280*960
##k4VGA = 3                # 1280*960
##kVGA = 2                 # 640*480
##kQVGA = 1                # 320*240
##kQQVGA = 0               # 160*120

video_service = ALProxy("ALVideoDevice", ip, 9559)

node = nep.node("pepper_image")
pub = node.new_pub("robot_image", "image")

#video_service = session.service("ALVideoDevice")
#colorSpace = vision_definitions.kRGBColorSpace
colorSpace = 11

#colorSpace = vision_definitions.kBGRColorSpace
#colorSpace = vision_definitions.kRGBColorSpace

##img.header.frame_id = self.frame_id
##img.height = image[1]
##img.width = image[0]
##nbLayers = image[2]
##if image[3] == kYUVColorSpace:
##    encoding = "mono8"
##elif image[3] == kRGBColorSpace:
##    encoding = "rgb8"
##elif image[3] == kBGRColorSpace:
##        encoding = "bgr8"
##elif image[3] == kYUV422ColorSpace:
##    encoding = "yuv422" # this works only in ROS groovy and later
##elif image[3] == kDepthColorSpace or image[3] == kRawDepthColorSpace:
##    encoding = "16UC1"
##img.encoding = encoding
##img.step = img.width * nbLayers
##img.data = image[6]

##mono8: CV_8UC1, grayscale image
##mono16: CV_16UC1, 16-bit grayscale image
##bgr8: CV_8UC3, color image with blue-green-red color order
##rgb8: CV_8UC3, color image with red-green-blue color order
##bgra8: CV_8UC4, BGR color image with an alpha channel
##rgba8: CV_8UC4, RGB color image with an alpha channel

print "waiting service"
videoClient = video_service.subscribe("python_client", resolution, colorSpace, 15)
print "start"

while True:
    # image[6] contains the image data passed as an array of ASCII chars.
    start = time.time()
    dataImage = video_service.getImageRemote(videoClient)
    if(dataImage != None):

        end = time.time()
        elapsed = end - start
        #print ("time; " + str(elapsed))
        
        image = numpy.reshape(numpy.frombuffer(dataImage[6], dtype='%iuint8' % dataImage[2]), (dataImage[1], dataImage[0], dataImage[2]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #start = time.time()
        image = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
        pub.publish(image)
        #end = time.time()
        #elapsed = end - start
        #print ("enviada " + str(elapsed))
           


