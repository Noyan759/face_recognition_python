import cv2
import dlib
import numpy as np
from math import sqrt
import aNNv2 as aNN
import newfaceLandmarkDetection as fLD
import aNN as testaNN


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

pNetwork=aNN.network[2][0][0]

for iteration in range(100):  
    f=1 
    for picNo in range(20):
        name = ('p%d.jpg' %f)
        image = cv2.imread(name)
        landmarks = fLD.get_Landmarks(image)
        ans=fLD.extractFeatures(landmarks)

        for x in range(30):
            aNN.network[0][0][x]=ans[x]

        print(name, iteration)
        aNN.forwardPropogation()
        aNN.backPropogation(1)
        aNN.updateWeights()

        name = ('n%d.jpg' %f)
        image = cv2.imread(name)
        landmarks = fLD.get_Landmarks(image)
        ans=fLD.extractFeatures(landmarks)

        for x in range(30):
            aNN.network[0][0][x]=ans[x]

        print(name, iteration)
        aNN.forwardPropogation()
        aNN.backPropogation(0)
        aNN.updateWeights()

        f=f+1

print('\ntraining done.\n')

aNetwork=aNN.network[2][0][0]

name = 'noyanTest.jpg'
image = cv2.imread(name)
landmarks = fLD.get_Landmarks(image)
ans=fLD.extractFeatures(landmarks)
for x in range(30):
    aNN.network[0][0][x]=ans[x]
print('input layer updated.')
aNN.forwardPropogation()
positiveOutput=aNN.network[2][0][0]

name = 'musabTest.jpg'
image = cv2.imread(name)
landmarks = fLD.get_Landmarks(image)
ans=fLD.extractFeatures(landmarks)
for x in range(30):
    aNN.network[0][0][x]=ans[x]
print('input layer updated.')
aNN.forwardPropogation()
negativeOutput=aNN.network[2][0][0]

print('\nbefore training:\n')
print(pNetwork)
print('\nafter training:\n')
print(aNetwork)
print('\nafter testing positive:\n')
print(positiveOutput)
print('\nafter testing negative:\n')
print(negativeOutput)
