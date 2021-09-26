import cv2
import tensorflow as tf
from mtcnn import MTCNN
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Capturing video
cap = cv2.VideoCapture(0)

# label
labels = ('No Mask', 'Mask')
detector = MTCNN()

# Loading model
mask_model = tf.keras.models.load_model('./face_mask_vgg16.h5')




while True:
    face_ = []
    prediction_ = []
    location_ = []
    # Reading image as frame from video
    ret, frame = cap.read()

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
    cv2.imshow('Image', frame)

    
    # To break/stop if 'q  or 'esc' is pressed
    if (cv2.waitKey(20) == ord('q')) or (cv2.waitKey(20) == 27) :
        break
cap.release()
cv2.destroyAllWindows()