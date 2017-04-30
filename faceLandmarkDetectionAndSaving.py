print('Starting:')

import cv2
import dlib
import numpy
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
while i<=10:
    print('detecting and annotating landmarks:')
    name = ("musabFace(%d).jpg" %i)
    image = cv2.imread(name)
    landmarks = get_Landmarks(image)
    print('printing landmarks:')
    print(landmarks)
    print('done')
    image_with_landmarks = annotate_landmarks(image, landmarks)
    print('landmarks detected and annotated:')

    print('Displaying processed image')
    # cv2.imshow('Result', image_with_landmarks)
    savingName = ("musabFace(%d)_with_landmarks.jpg" %i)
    cv2.imwrite(savingName, image_with_landmarks)
    i=i+1

cv2.waitKey(0)
cv2.destroyAllWindows()

print('Going well')