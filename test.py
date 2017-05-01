import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


# for iteration in range(100):  
#     f=1 
#     for picNo in range(20):
#         name = ('p%d.jpg' %f)
#         image = cv2.imread(name)
#         landmarks = fLD.get_Landmarks(image)
#         ans=fLD.extractFeatures(landmarks)

#         for x in range(30):
#             aNN.network[0][0][x]=ans[x]

#         print(name)
#         aNN.forwardPropogation()
#         aNN.backPropogation(1)
#         aNN.updateWeights()

#         name = ('n%d.jpg' %f)
#         image = cv2.imread(name)
#         landmarks = fLD.get_Landmarks(image)
#         ans=fLD.extractFeatures(landmarks)

#         for x in range(30):
#             aNN.network[0][0][x]=ans[x]

#         print(name)
#         aNN.forwardPropogation()
#         aNN.backPropogation(0)
#         aNN.updateWeights()

#         f=f+1        



# for picNo in range(20):
#     n=picNo+1
#     name = ('noyanFrontface(%d).jpg' %n)
#     image = cv2.imread(name)
#     landmarks = fLD.get_Landmarks(image)

#     print('hello1')
#     ans=fLD.extractFeatures(landmarks)

#     print('hello2')

#     for x in range(30):
#         aNN.network[0][0][x]=ans[x]

#     print('input layer updated.')

#     for i in range(100):
#         print(picNo, i)
#         aNN.forwardPropogation()
#         aNN.backPropogation()
#         aNN.updateWeights()
#         # print(aNN.weights)
#         # y = input("iteration(%d): " %i)