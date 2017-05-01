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
        eFA=fLD.extractFeatures(landmarks)
        ans=fLD.extractRatios(eFA)

        for x in range(30):
            aNN.network[0][0][x]=ans[x]
        
        print('name:')
        print(name, iteration)
        aNN.forwardPropogation()
        aNN.backPropogation(1)
        aNN.updateWeights()




        # name = ('n%d.jpg' %f)
        # image = cv2.imread(name)
        # landmarks = fLD.get_Landmarks(image)
        # ans=fLD.extractFeatures(landmarks)

        # for x in range(30):
        #     aNN.network[0][0][x]=ans[x]

        # print(name, iteration)
        # aNN.forwardPropogation()
        # aNN.backPropogation(0)
        # aNN.updateWeights()



        f=f+1
    err=aNN.network[2][0][0]-1
    print('error:')
    print(err)

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
noyanOutput=aNN.network[2][0][0]

name = 'n1.jpg'
image = cv2.imread(name)
landmarks = fLD.get_Landmarks(image)
ans=fLD.extractFeatures(landmarks)
for x in range(30):
    aNN.network[0][0][x]=ans[x]
print('input layer updated.')
aNN.forwardPropogation()
zeerakOutput=aNN.network[2][0][0]

print('\nbefore training:\n')
print(pNetwork)
print('\nafter training:\n')
print(aNetwork)
print('\nafter testing positive:\n')
print(noyanOutput)
print('\nafter testing negative:\n')
print(zeerakOutput)
