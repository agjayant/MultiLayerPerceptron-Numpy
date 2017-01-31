
# coding: utf-8

# In[ ]:

import numpy as np
from network import network
import scipy.io as scio
import config
import matplotlib.pyplot as plt

mnist = scio.loadmat('/home/jayant/CS771/hw2/PP1/mnist_big.mat')

trainExamples = 1000
valExamples =  100

trainLabel = np.asarray([[0 for i in range(10)] for j in range(trainExamples)])
for i in range(trainExamples):
    trainLabel[i,mnist['Y_train'][i]] = 1 
trainData = mnist['X_train'][0:trainExamples]

valData = mnist['X_test'][0:valExamples]
valLabel = mnist['Y_test'][0:valExamples]


# In[ ]:

net = network(2, [30,20], 'relu')


# In[ ]:

net.train(trainData, trainLabel, valData, valLabel, 100, 50, 0.0001)

