import time
import cv2
import numpy as np
import os, sys
import pickle
import nep
import threading
import sharo

show_image = 1
try:
    print (sys.argv[1])
    show_image = int(sys.argv[1])
    print ("Show image: " + show_image)
except:
    pass


path_ =  "faces/"
node = nep.node('face_detection')
sub_image = node.new_sub('robot_image', 'image')
pub_position = node.new_pub('face_positions', 'json')

perception_face = sharo.BooleanPerception(node, "face_detection", "value", 1, 3)  # --------------- Sharo ------------------
frame = ""
bounding_boxes = ""


def thread_function(name):
    global sub_image, frame
    while True:
        s, img = sub_image.listen()
        if s:
            frame = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)


face_positions = threading.Thread(target=thread_function, args=(1,))
face_positions.start()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
model_filename = path_ + 'model/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(model_filename)
time.sleep(1)
people_detected_msg_sended = False
people_non_detected_msg_sended = False
n_detected = 0
non_detected = 0
start_detection =  time.time()
start_alone =  time.time()
print('Start!')


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)
    


main_face = [0,0]
nb_faces = 0
        
while True:
        newImage = frame.copy()
        close = 0
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        matrix = faceCascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30),)
        
        # n saves the number of face detected
        try:
            n,m =  matrix.shape
        except:
            n = 0

        if n > 0: 
            #boxes = [[int(matrix[0,0]), int(matrix[0,1]), int(matrix[0,0]) + int(matrix[0,2]), int(matrix[0,1]) + int(matrix[0,3])]]
            # If only 1 face that is the main face
            if n == 1:
                x_m = int(matrix[close,0]) + int(matrix[close,2]/2)
                y_m = int(matrix[close,1]) + int(matrix[close,3]/2)
                main_face = [x_m, y_m] # Center of face
            # If more than 1 face track the closest one to the main face
            else:

                dist = 10000000
                close = 0 # Index of closest face
                for i in range(n):
                    
                    x_m = int(matrix[i,0]) + int(matrix[i,2]/2)
                    y_m = int(matrix[i,1]) + int(matrix[i,3]/2)
                    dist_temp = ((main_face[0] - x_m)**2 + (main_face[1] - y_m)**2)**.5
                    if dist_temp < dist:
                        dist = dist_temp
                        close = i
                x_m = int(matrix[close,0]) + int(matrix[close,2]/2)
                y_m = int(matrix[close,1]) + int(matrix[close,3]/2)
                main_face = [x_m, y_m] # Center of face

            boxes = []
            for i in range(n):
                boxes.append([int(matrix[i,0]), int(matrix[i,1]), int(matrix[i,2]), int(matrix[i,3])])            
            #Send only main face
            bounding_boxes = np.array(boxes)
            end = time.time()
            # Only send one face
            center = {"x":boxes[close][0], "y":boxes[close][1]}
            size = {"w":boxes[close][2] , "h": boxes[close][3]}
            box = {"x1" : int(matrix[close,0]), "x2" : int(matrix[close,1]),"y1" : int(matrix[close,0] + matrix[i,2]), "y2" : int(matrix[close,1] + matrix[i,3])}
            pub_position.publish({"face": {"center":center, "size":size,"box":box}})
            nb_faces = bounding_boxes.shape[0]
            if(nb_faces > 0):
                perception_face.primitive_detected()      # --------------- Sharo ------------------
            else:
                pub_position.publish({"positions": []})
                perception_face.primitive_non_detected()   # --------------- Sharo ------------------
            
        else:
            perception_face.primitive_non_detected()       # --------------- Sharo ------------------
            nb_faces = 0

        
        if show_image == 1:
            if nb_faces > 0:
                for i in range(nb_faces):
                    if i == close:
                        draw_bounding_box(bounding_boxes[i], newImage, (0, 255, 0))
                    else:
                        draw_bounding_box(bounding_boxes[i], newImage, (0, 0, 200))
            cv2.imshow("Face Detection", newImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            




