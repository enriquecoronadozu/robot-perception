
import cv2
import numpy as np
import sys
import nep
import time


id_ = 0
x_resolution = 640
y_resolution = 480
flip = 0

try:
    print (sys.argv[1])
    id_ = int(sys.argv[1])
    print("Camera id:" + str(id_))
    x_resolution = int(sys.argv[2])
    print("X:" + str(x_resolution))
    y_resolution = int(sys.argv[3])
    print("Y:" + str(y_resolution))
    flip = int(sys.argv[4])
    print("flip:" + str(flip))
except:
    pass
    
node = nep.node('usb_camera')
pub_image = node.new_pub('robot_image','image')

video = cv2.VideoCapture(id_)
video.set(cv2.CAP_PROP_FRAME_WIDTH, x_resolution)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, y_resolution)

print('Start Video recording!')


while True:
    success, frame = video.read()
    # Display the resulting image
    if (x_resolution > 640):
        dim = (640, 480)
        frame = cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    if (flip == 1):
        frame = cv2.flip(frame,0)
    pub_image.publish(frame)
    time.sleep(.033)

video.release()
cv2.destroyAllWindows()
print("Camera turned off")

