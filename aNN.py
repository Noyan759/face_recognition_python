import numpy as np

L=3
sl=np.array([3, 1])

# Initializing weights:
s1=(3,3)
# weightsL1 = np.full(s1, 0.01)
weightsL1 = np.random.rand(3,3)
s2=(1,3)
# weightsL2 = np.full(s2, 0.01)
weightsL2 = np.random.rand(1,3)

weights = np.array([weightsL1, weightsL2])/100

# # Input weights:
# for i in range(3):
#     for j in range(3):
#         weights[0][i][j]=float(input("input: "))
# for k in range(3):
#     weights[1][0][k]=float(input("input2: "))

# Initializing Network:
s1=(1,3)
inputLayer = np.zeros(s1)
for i in range(3):
    inputLayer[0][i]=i+1

s2=(1,3)
hiddenLayer = np.zeros(s2)

s3=(1,3) 
outputLayer = np.zeros(s3)

network = np.array([inputLayer, hiddenLayer, outputLayer])

# Initializing Error Matrix:
s1=(1,3)
errorLayer2 = np.zeros(s1)

s2=(1,3)
errorLayer3 = np.zeros(s2)

errorMatrix = np.array([errorLayer2, errorLayer3])

dOutput = 1


def sigmoid(x):
    return 1/(1 + np.exp(-x))

# k=0
# for i in range(3):
#     z=0
#     for j in range(3):
#         z=z+network[0][j]*weights[0][k]
#         k=k+1
#     network[1][i]=sigmoid(z)

# k=0
# for i in range(2):
#     z=0
#     for j in range(3):
#         z=z+network[1][j]*weights[1][k]
#         k=k+1
#     network[2][i]=sigmoid(z)

def elementMultiplication(a1, a2):
    a=a1
    for i in a1:
        a[i]=a1[i]*a2[i]
    return a

    

def forwardPropogation():
    for layer in range(L-1):
        k=0
        for i in range(sl[layer]):
            z=0
            for j in range(3):
                z=z+network[layer][0][j]*weights[layer][i][j]
                k=k+1
            network[layer+1][0][i]=sigmoid(z)  

def backPropogation():

    errorMatrix[1][0][0] = dOutput-network[2][0][0]
    for i in range(3):
        errorMatrix[0][0][i]=weights[1][0][i]*errorMatrix[1][0][0]*network[1][0][i]*(1-network[1][0][i])

    return 0

def updateWeights():
    for i in range(3):
        for j in range(3):
            weights[0][i][j]=weights[0][i][j]+network[0][0][j]*errorMatrix[0][0][i]

    for k in range(3):
        weights[1][0][k]=weights[1][0][k]+network[1][0][k]*errorMatrix[1][0][0]
    return 0


# print('\n\nnetwork: ')
# print(network)
# forwardPropogation()
# print('\nupdated_network: ')
# print(network)

# print('\n\nerrorMatrix: ')
# print(errorMatrix)
# backPropogation()
# print('\nupdated_errorMatrix: ')
# print(errorMatrix)

# print('\n\nweights: ')
# print(weights)
# updateWeights()
# print('\nupdated_weights: ')
# print(weights)

