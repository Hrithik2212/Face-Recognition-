import cv2 as cv
capture = cv.VideoCapture(0)
while True:
    isTrue ,frame = capture.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier("C://Users/HRITHIK REDDY/Opencv/Face detect-recog/haar_face.xml")
    face_rect= haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
    for x,y,w,h in face_rect:
        detected_face= cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
    if(len(face_rect)==0):
        cv.imshow("Video",frame)
    else:
        cv.imshow("Video",detected_face)
    if cv.waitKey(1) & 0xFF==ord('d') :
        break
capture.release()
cv.destroyAllWindows


