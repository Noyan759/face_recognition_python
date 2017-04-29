print('Starting:')

import cv2
import dlib
import numpy
from math import sqrt
print('importing done:')

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
print('dlib files acquiring done:')

print('Defining classes and functions:')
class TooManyFaces(Exception):
    tooManyFaces = 1
    # pass

class NoFaces(Exception):
    noFaces = 1
    # pass

def square(x):
    return x*x

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

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

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

print('classes and functions definition done:')

i=1
print('detecting and annotating landmarks:')
# name = ("noyanFrontFace(%d).jpg" %i)
name = 'noyanFace3.jpg'
image = cv2.imread(name)
landmarks = get_Landmarks(image)

c=0
array=[]
print('printing landmarks:')
j=0
while j<68:
    k=0
    while k<68:
        ans = euclideanDistance(landmarks, j, k)
        print(j, k, c)
        print(ans)
        array.append(ans)
        k=k+1
        c=c+1
    j=j+1

# print(array)    
# print(landmarks)
# ans = abs(landmarks[0]-landmarks[1])
# print(ans)

print('done')
image_with_landmarks = annotate_landmarks(image, landmarks)
print('landmarks detected and annotated:')

print('Displaying processed image')
cv2.imshow('Result', image_with_landmarks)
cv2.imwrite('noyanFace3_with_landmarks.jpg', image_with_landmarks)

cv2.waitKey(0)
cv2.destroyAllWindows()

print('Going well')