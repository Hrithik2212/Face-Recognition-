import numpy as np
import cv2 as cv
import os 
from frame_rescale import *


DIR = r'C:\Users\HRITHIK REDDY\Opencv\Face detect-recog\Faces'
people=[]
for person in os.listdir(DIR):
    people.append(person)


#We are importing the previously trained data 

haar_classifier=cv.CascadeClassifier('haar_face.xml')
features = np.load('Features.npy',allow_pickle=True)
labels= np.load('Labels.npy',allow_pickle=True)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

image=cv.imread(r'C:\Users\HRITHIK REDDY\Opencv\Face detect-recog\Validation\Maddona (5).jpg')
gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)


faces_rect=haar_classifier.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]
    image = cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
    gray = cv.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),4)
    label,confidence=face_recognizer.predict(faces_roi)

# print(labels)
print("The label= {} with a confidence {}".format(people[label],confidence))
print("Label index = {}".format(label))

print("\n\nDetection Done-----------")


# cv.imshow("Person",frame_rescale(image,2.4))
# cv.imshow("roi",frame_rescale(faces_roi,3))
# cv.waitKey(0)

