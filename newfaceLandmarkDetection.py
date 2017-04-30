# print('Starting:')

import cv2
import dlib
import numpy as np
from math import sqrt
# print('importing done:')

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
# print('dlib files acquiring done:')

# print('Defining classes and functions:')
class TooManyFaces(Exception):
    tooManyFaces = 1
    # pass

class NoFaces(Exception):
    noFaces = 1
    # pass

def square(x):
    return x*x

# Feature array:
array=np.array([0,16,2,14,4,12,6,10,17,21,21,22,22,26,17,26,36,39,42,45,39,42,36,45,27,30,31,35,31,33,33,35,48,54,51,57,51,62,57,66,36,31,45,35,31,48,35,54,48,36,54,45,48,8,54,8,36,8,45,8])
s=(30,2)
featureCoordinates = np.zeros(s)
k=0
for i in range(30):
    for j in range(2):
        featureCoordinates[i][j]=array[k]
        k=k+1

def euclideanDistance(landmarks, i, j):
    x1 = landmarks[i].item(0)
    x2 = landmarks[j].item(0)
    y1 = landmarks[i].item(1)
    y2 = landmarks[j].item(1)
    return sqrt(square(x1-x2)+square(y1-y2))

def get_Landmarks(im):
    rects = detector(im, 1)

    if len(rects)>1:
        raise TooManyFaces
    if len(rects)==0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale = 0.4,
                    color = (0, 0, 255))
        cv2.circle(im, pos, 3, color = (0, 0, 255))
    return im

# print('classes and functions definition done:')

i=1
# print('detecting and annotating landmarks:')
# name = ("noyanFrontFace(%d).jpg" %i)
name = 'sohaibFace(3).jpg'
image = cv2.imread(name)
landmarks = get_Landmarks(image)

array=[]

def extractFeatures():
    c=1
    for j in range(30):
        p1=int(featureCoordinates[j][0])
        p2=int(featureCoordinates[j][1])
        ans=euclideanDistance(landmarks, p1, p2)
        print(c, p1, p2, ans)
        array.append(ans)
        c=c+1

# print('done')
# image_with_landmarks = annotate_landmarks(image, landmarks)
# print('landmarks detected and annotated:')

# print('Displaying processed image')
# cv2.imshow('Result', image_with_landmarks)
# cv2.imwrite('sohaibFace(3)_with_landmarks.jpg', image_with_landmarks)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print('Going well')