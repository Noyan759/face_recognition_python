import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
# img = cv2.imread('shayan.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('noyan3.jpg')
i=0;
while 1:
    ret, img = cap.read()
	#img = cv2.imread('shayan.jpg', cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    print('waiting: ', i)
    k=cv2.waitKey(1)
    if k == 1:
        print('breaking\n')
        break
    # i=i+1
    # for x in range(30000000):
    #     y=1
    # name = ("noyanFace(%d).jpg" %i)
    # print(name)
    # cv2.imwrite("/faces/front/name", roi_gray)
    

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
        
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# scaleFactor=150/size(img,1)
# img=imresize(img, scaleFactor)
cv2.imshow('noyanFace3',roi_gray)
# cv2.imwrite('noyanFace3.jpg', roi_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()