import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import nep
import threading
import sys
import time
import sharo


show_image = 1

try:
    show_image = int(sys.argv[1])
    print ("Show image: " + str(show_image))
except:
    pass


node = nep.node('emotion_recognition')
sub_image = node.new_sub('robot_image', 'image')
sub_position = node.new_sub('face_positions', 'json')
pub_emotion = node.new_pub('face_emotions', 'json')
myImage = cv2.imread("x.jpg") # Temporal image

classes = ["angry", "fear", "sad", "happy", "surprise", "neutral"]
object_per = sharo.ObjectPerception(node, classes)

def thread_function(name):  # Get images as soon as possible
    global myImage, sub_image
    while True:
        s, img = sub_image.listen()
        if s:
            myImage = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA)


    
get_images = threading.Thread(target=thread_function, args=(1,))
get_images.start()

# parameters for loading data and images
emotion_model_path = 'models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []
emotions_counter = []

while True:

  
    s, msg = sub_position.listen()
    if s:
        try:
            #bgr_image = video_capture.read()[1]
            bgr_image = myImage.copy()

            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            face_p = msg["face"]
            x1 = face_p["box"]["x1"]
            x2 = face_p["box"]["x2"]
            w = face_p["size"]["w"]
            h = face_p["size"]["h"]

            box = np.array([x1,x2,w,h])
            
            
            #for box in bounding_boxes:
            #print(box)
            x1, x2, y1, y2 = apply_offsets(box, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            pub_emotion.publish({"emotion":emotion_text, "probability":float(emotion_probability)})

            draw = True
            if emotion_text == 'angry':
                color = 1 * np.asarray((255, 0, 0))
                object_per.manage_objects([0])
            if emotion_text == 'fear':
                object_per.manage_objects([1])
                color = 1 * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                object_per.manage_objects([2])
                color = 1 * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                object_per.manage_objects([3])
                color = 1 * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                object_per.manage_objects([4])
                color = 1 * np.asarray((0, 255, 255))
            elif emotion_text == 'neutral':
                object_per.manage_objects([5])
                color = 1 * np.asarray((0, 0, 0))
            else:
                draw = False

            if draw:

                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(box, rgb_image, color)
                draw_text(box, rgb_image, emotion_mode,
                        color, 0, -45, 1, 1)

            if(show_image == 1):
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('window_frame', bgr_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except:
            pass


    else:
        pass
            

