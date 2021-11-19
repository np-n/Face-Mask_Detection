import cv2
import streamlit as st
from mtcnn import MTCNN
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.title("Webcam Live Feed")
cl1,cl2 = st.columns(2)
run = cl1.button('Run')
FRAME_WINDOW = st.image([])# Buffer for empty image

# label
labels = ('No Mask', 'Mask')

camera = cv2.VideoCapture(0)# Capturing the video
messages = ['Click on Run to Access your Webcam','Click on stop to quit Face Detection']
# Loading model
mask_model = tf.keras.models.load_model('./model/face_mask_vgg16.h5')

flag = True
stop = cl2.button('Stop')

if stop:
    flag = False

if run:
    st.error(messages[1])
    while run and flag:
        face_ = []
        prediction_ = []
        location_ = []

        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FRAME_WINDOW.image(frame)
        detector = MTCNN()
        output = detector.detect_faces(frame)
        # Considering multiple faces will be present in video
        for face in output:
            x,y,width,height = face['box']
            face_img = frame[y:y+height,x:x+width]
            face_img = cv2.resize(face_img,(128,128))
            face_.append(face_img)
            location_.append((x,y,x+width,y+height))

        conf = 0
        if len(face_)>0:
            face_ = np.array(face_,dtype='float32') # Ensuring that images are in array form
            prediction_ = mask_model.predict(face_)

            for (bbox,pred) in zip(location_,prediction_):
                (x1,y1,x2,y2) = bbox
                label = "Mask" if pred[0] > 0.5 else "No Mask"

                if label == "Mask":
                    conf = pred[0] * 100 # confidence percentage for mask

                else:
                    conf = (100-pred[0]*100)# Confidence percentage for non-mask
                

                color = (0,255,0) if label=="Mask" else (0,0,255)# green if mask else red
                label = "{}: {:.2f}%".format(label,conf)#label:confidence percentage

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Putting rectangle of bbox in frames
                cv2.rectangle(frame, (x1,y1-40), (x2,y1), color, -1)

                cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Putting text in live frames
        FRAME_WINDOW.image(frame)
    else:
        st.info(messages[0])
else:
    st.info(messages[0])

