import cv2
import tensorflow as tf
from mtcnn import MTCNN
import tensorflow as tf

# Capturing video
cap = cv2.VideoCapture(0)

# label
labels = ('No Mask', 'Mask')
detector = MTCNN()

# Loading model
mask_model = tf.keras.models.load_model('./face_mask_vgg16.h5')

while True:
    # Reading image as frame from video
    ret, frame = cap.read()
    output = detector.detect_faces(frame)

    # Considering multiple faces will be present in video
    for i in range(len(output)):
        x,y,width,height = output[i]['box']
        face_img = frame[y:y+height,x:x+width]
        face_img = cv2.resize(face_img,(128,128))
        prediction = mask_model.predict(face_img.reshape(-1,128,128,3))
        print(prediction[0][0])
        output_label = labels[int(prediction[0][0])]

        cv2.rectangle(frame,pt1=(x,y),pt2=(x+width,y+height),color=(0,0,255),thickness=3)
        cv2.putText(frame,output_label,(x+5,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_AA)

    # Showing frame to user
    cv2.imshow('Window',frame)
    
    # To break/stop if x is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()