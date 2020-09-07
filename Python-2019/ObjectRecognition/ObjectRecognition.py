import numpy as np
import threading
import time
import nep
import time
import cv2
import sys
from PIL import Image
import sharo

show_image = 1
model = "yolo3-320"

try:
    model = sys.argv[1]
    print ("Model:" + model)
    show_image = int(sys.argv[2])
    print ("Show image: " + str(show_image))
except:
    pass

path_models = "models/" #Path to models

# ------- Yolo parameters ------------
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold

# ------- TF parameters ------------
confThreshold = 0.5  #Confidence threshold
maskThreshold = 0.3  # Mask threshold

if model == "yolo3-320" or model == "yolo3-tiny":
    inpWidth = 320       #Width of network's input image
    inpHeight = 320      #Height of network's input image
    classesFile = path_models + "coco.names"
elif model == "yolo3-416":
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image
    classesFile = path_models + "coco.names"
else:
    inpWidth = 300       #Width of network's input image
    inpHeight = 300      #Height of network's input image
    classesFile = path_models + "coco_ms.names"
    

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print (classes)

# Lock primitive for securely accessing a shared variable
lock = threading.Lock()

# ---------- Nep -----------------

node = nep.node("object_recognition")
sub_img = node.new_sub("robot_image", "image")    # Set the topic and the configuration
object_per = sharo.ObjectPerception(node, classes)
print("usb source")

    
pub_image = node.new_pub('robot_image_recognition','image')
myImage = cv2.imread("x.jpg") # Temporal image

def thread_function(name):  # Get images as soon as possible
    global myImage
    while True:
        s, img = sub_img.listen()
        if s:
            lock.acquire()
            myImage = img.copy()
            lock.release()

get_images = threading.Thread(target=thread_function, args=(1,))
get_images.start()

# pascal-voc-classes.txt
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# BGR color
COLORS = (51,51,255)



def predict_detection(frame, net):
    '''
    Predict the objects present on the frame
    '''
    # Conversion to blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    # omImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
 
    # Detection and prediction with model
    net.setInput(blob)
    return net.forward()


# Remove the bounding boxes with  low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
 
    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
 
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    list_labels = []
    list_index = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)

        if confidences[i] > 0.5:
            label = classes[classIds[i]]
            position = [left + width/2, top + height/2]
            list_index.append(classIds[i])
            list_labels.append(label)
    
    object_per.manage_objects(list_index)
    
    #print(list_labels)
    #print(list_index)

            
        
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box yolo
def drawPred(classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))
     
    label = '%.2f' % conf
         
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
 
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

def draw_prediction(frame, detections, exploration_map, zone):
    '''
    Filters the predictions with a confidence threshold and draws these predictions
    '''

    (height, width) = frame.shape[:2]
  
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
    
 
        if confidence > 0.5:
            # Index of class label and bounding box are extracted
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])

            # Retreives corners of the bounding box
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)

           
                
            cv2.rectangle(frame, (startX, startY), (endX, endY),(255, 0, 0), 2)
            labelPosition = endY - 5
            cv2.putText(frame, label, (startX, labelPosition),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

    return frame


# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
     
    # Print a label of class.
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
     
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
 
    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom+1, left:right+1][mask]
 
    color = colors[classId%len(colors)]
    # Comment the above line and uncomment the two lines below to generate different instance colors
    #colorIndex = random.randint(0, len(colors)-1)
    #color = colors[colorIndex]
 
    frame[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
 
    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    im2, contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)

def load_training_set(filename):
    '''
    Load the data from a file and save them as a list
    '''
    training_set = []
    f = open(filename, "r")
    data = f.readlines()
    index = 0
    if f.mode == "r":
        for line in data:
            line = line.strip()
            training_set.append(line.split(","))
            # print((line.split(","))[-1])
        f.close()
    return training_set





exploration_map = [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]
zone = 1
n_frame = 0

# Set net and load model

print("[INFO] Loading model...")
if model == "ssd-mobilenet":
    net = cv2.dnn.readNetFromCaffe(path_models + "MobileNetSSD_deploy.prototxt",  path_models +  "MobileNetSSD_deploy.caffemodel")
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

elif model == "yolo3-tiny":
    modelConfiguration = path_models + "yolov3-tiny.cfg";
    modelWeights = path_models + "yolov3-tiny.weights";

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

elif model == "yolo3-320" :
    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = path_models + "yolov3.cfg";
    modelWeights = path_models +  "yolov3_320.weights";
        
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

    
elif model == "yolo3-tiny":
    modelConfiguration = path_models + "yolov3-tiny.cfg";
    modelWeights = path_models + "yolov3-tiny.weights";

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

elif model == "yolo3-416" :
    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = path_models + "yolov3.cfg";
    modelWeights = path_models +  "yolov3_416.weights";
        
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

elif model == "tensorflow":
    net = cv2.dnn.readNetFromTensorflow(path_models + 'MobileNetSSDV2.pb', path_models + 'MobileNetSSDV2.pbtxt')
    
    
 
    
print("[INFO] Start adquisition...")
    
    
while True:
        
    frame = myImage.copy()
    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    
    if model == "ssd-mobilenet":
        if (n_frame % 1) == 0:
            # Runs the detection on a frame and return bounding boxes and predicted labels
            detections = predict_detection(frame, net)
            frame = draw_prediction(frame, detections, exploration_map, zone)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            #print(detections)

    elif model == "tensorflow":
        detections = []
        rows, cols, channels = frame.shape
        
        # Use the given image as input, which needs to be blob(s).
        #blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
   
        net.setInput(blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
        out = net.forward()


        for detection in out[0,0]:

            classId = float(detection[1])
            score = float(detection[2])   # == confidence
            if score > 0.5:
                try:
                    clase = str(classes[int(classId)])
                    score_str = str(score)
                    #print("Clase: " + clase + " " + str(int(classId)-1))
                    #print("Score: " + score_str)
                    
                    left = detection[3] * cols
                    top = detection[4] * rows
                    right = detection[5] * cols
                    bottom = detection[6] * rows
             
                    #draw a red rectangle around detected objects
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                    labelPosition = int(bottom) - 5
                    label = '%s:%s' % (clase, score_str)
                    cv2.putText(frame, label, (int(left), labelPosition),cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
                    # Put efficiency information. The function getPerfProfile returns the 
                except:
                    pass

        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                

    else:

 
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
     
        # Sets the input to the network
        net.setInput(blob)
     
        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
         
        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)
 
        # Put efficiency information. The function getPerfProfile returns the 
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
 


    pub_image.publish(frame)

    if(show_image == 1):
        cv2.imshow("Object recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    n_frame += 1
   



    
