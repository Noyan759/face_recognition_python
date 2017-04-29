import numpy as np

L=3
sl=np.array([3, 2])
wieghtsL1 = np.array([.1, .2, .1, 0, .3, .1, .2, .3, .1])
wieghtsL2 = np.array([.1, .2, .1, 0, .3, .1])
wieghts = [wieghtsL1, wieghtsL2]
inputLayer = np.array([1, 2, 3])
hiddenLayer = np.array([0.0, 0.0, 0.0])
outputLayer = np.array([0.0, 0.0])
dOutput = 1

print(wieghts[0])
print(wieghts[1])

# network = np.matrix([[1, 2], [3, 4]])
network = [inputLayer, hiddenLayer, outputLayer]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# k=0
# for i in range(3):
#     z=0
#     for j in range(3):
#         z=z+network[0][j]*wieghts[0][k]
#         k=k+1
#     network[1][i]=sigmoid(z)

# k=0
# for i in range(2):
#     z=0
#     for j in range(3):
#         z=z+network[1][j]*wieghts[1][k]
#         k=k+1
#     network[2][i]=sigmoid(z)

for layer in range(L-1):
    k=0
    for i in range(sl[layer]):
        z=0
        for j in range(3):
            z=z+network[layer][j]*wieghts[layer][k]
            k=k+1
        network[layer+1][i]=sigmoid(z) 
        print(layer, i, network[layer+1][i])   

print(network[1])
print(network[2])
