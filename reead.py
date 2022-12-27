

import cv2
import numpy as np

from tensorflow.keras import models



models = models.load_model('model-ducc10people_10epochs.h5')
face_detector = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_alt2.xml')

cam = cv2.VideoCapture(0)

while True:
    OK, frame =cam.read()
    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        listResult = ['MrChinh', 'MrPhuc', 'MrSam', 'MrsNgan', 'MrsThao', 'MrTrong', 'MrTuLong', 'MrVDDam', 'MrVuong', 'MrXuanBac']
        roi = cv2.resize(frame[y: y+h, x: x+w],(100, 100))
        result = np.argmax(models.predict(roi.reshape(-1, 100, 100, 3)))
        cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2)
        cv2.putText(frame, listResult[result], (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,25,255), 2)
            
    cv2.imshow('FRAME', frame)
    
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()