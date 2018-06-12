import numpy as np
import cv2


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


haar_face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascades/lbpcascade_frontalface.xml')
face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    img_copy = np.copy(colored_img)
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor, 5);
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy


test1 = cv2.imread('1.jpg')
faces_detected_img = detect_faces(lbp_face_cascade, test1)

cv2.imshow('Image', faces_detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
