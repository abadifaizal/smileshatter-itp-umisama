import numpy as np
import cv2

#library cascade call
faceCascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

#set camera size
cap.set(3,640) #set witdh
cap.set(4,480) #set height

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #face detector
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
    #eye detector
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(5,5),
            )
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0),2)
        
    #smile detector
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25,25),
        )
        for(xx,yy,ww,hh) in smile:
            cv2.rectangle(roi_color,(xx,yy),(xx+ww, yy+hh),(0,255,0),2)
        
        cv2.imshow('video',img)

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()