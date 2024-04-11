import cv2
import numpy as np
import pickle
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
model = load_model('my_model.keras')

labels = {}
with open("labels_deep.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (200, 200))
        roi_gray_normalized = roi_gray_resized / 255.0
        roi_gray_expanded = np.expand_dims(roi_gray_normalized, axis=0)
        roi_gray_expanded = np.expand_dims(roi_gray_expanded, axis=3)  

        prediction = model.predict(roi_gray_expanded)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if confidence > 0.25:  
            name = labels[class_index]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name + " " + str(round(confidence, 2)), (x, y), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0, 0)  
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
