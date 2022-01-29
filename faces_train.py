import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton Jhon', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR ="C:\\Users\HRITHIK REDDY\Opencv\Face detect-recog\Faces"

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        paths = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(paths):
            img_path = os.path.join(paths,img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training Done-------------")

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#We will train the recognizer on the features and labels list
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save("Features",features)
np.save("Labels",labels)

