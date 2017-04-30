import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

array=np.array([0,16,2,14,4,12,6,10,17,21,21,22,22,26,17,26,36,39,42,45,39,42,36,45,27,30,31,35,31,33,33,35,48,54,51,57,51,62,57,66,36,31,45,35,31,48,35,54,48,36,54,45,48,8,54,8,36,8,45,8])

s=(30,2)
featureCoordinates = np.zeros(s)

k=0
for i in range(30):
    for j in range(2):
        featureCoordinates[i][j]=array[k]
        k=k+1

print(int(featureCoordinates[0][1]))