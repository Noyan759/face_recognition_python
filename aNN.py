import numpy as np

L=3
sl=np.array([3, 1])

weightsL1 = np.array([[.1, .2, .1], [.0, .3, .2], [.2, .3, .1]])
weightsL2 = np.array([[.1, .2, .1]])
weights = np.array([weightsL1, weightsL2])

inputLayer = np.array([1, 2, 3])
hiddenLayer = np.array([0.0, 0.0, 0.0])
outputLayer = np.array([0.0])
network = np.array([inputLayer, hiddenLayer, outputLayer])

errorLayer2 = np.array([0.0, 0.0, 0.0])
errorLayer3 = np.array([0.0])
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
                z=z+network[layer][j]*weights[layer][i][j]
                k=k+1
            network[layer+1][i]=sigmoid(z)  

def backPropogation():

    errorMatrix[1] = dOutput-network[2]
    for i in range(3):
        errorMatrix[0][i]=weights[1][0][i]*errorMatrix[1]*network[1][i]*(1-network[1][i])

    return 0

def updateWeights():
    return 0

forwardPropogation()
backPropogation()
print(errorMatrix)

# x=np.array([[1,2]])
# print(weights.shape)
print('network: ')
print(network)
print('weights: ')
print(weights)
print('errorMatrix: ')
print(errorMatrix)
# print(np.transpose(weights[1]))

print(weights[1][0][0]*errorMatrix[1]*network[1][0]*(1-network[1][0]))