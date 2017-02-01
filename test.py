
# coding: utf-8

# In[ ]:

import numpy as np
from network import network
import scipy.io as scio
import config
import matplotlib.pyplot as plt

mnist = scio.loadmat('mnist_big.mat')

trainExamples = 40
valExamples =  1000

trainList = [np.random.randint(0,60000) for i in range(trainExamples)]
valList = [np.random.randint(0,10000) for i in range(valExamples)]

trainLabel = np.asarray([[0 for i in range(config.numClasses)] for j in range(trainExamples)])
trainData = np.asarray([[0 for i in range(config.n_inputs)] for j in range(trainExamples)])

valLabel = np.asarray([0 for i in range(valExamples)])
valData = np.asarray([[0 for i in range(config.n_inputs)] for j in range(valExamples)])

j=0
for i in trainList:
    trainLabel[j,mnist['Y_train'][i]] = 1 
    trainData[j,:] = mnist['X_train'][i]
    j += 1

j=0
for i in valList:
    valLabel[j] = mnist['Y_test'][i] 
    valData[j,:] = mnist['X_test'][i]
    j += 1

trainData = trainData*1.0/255
valData = valData*1.0/255


# In[ ]:

net = network(1, [100], 'relu')


# In[ ]:

net.train(trainData, trainLabel, valData, valLabel, 10, 5,lr =0.0001, gradMethod = 'gdm')

