
# coding: utf-8

# In[ ]:

import numpy as np
from network import network
import scipy.io as scio
import config
import matplotlib.pyplot as plt

###################################################################
# Set the n_inputs and numClasses Params in config.py according 
# to trainData and trainLabel
###################################################################

trainData = [1,2,3,4,5]
trainLabel = [0,0,1]

assert(len(trainData) == config.n_inputs),'Set the Params in config.py'
assert(len(trainLabel) == config.numClasses),'Set the Params in config.py'

n_inputs = config.n_inputs
numClasses = config.numClasses


# In[ ]:

net = network(1, [3], 'tanh')


# In[ ]:

net.forward(trainData)
net.backward(trainData, trainLabel, 'mbsgd')


# In[ ]:

x=[]
for i in range(numClasses*net.nnod[0]+ net.nnod[0]*(n_inputs+1)):
    x.append(i)
y= []
z= []


# In[ ]:


epsi = 1e-04

for i in range(net.nnod[0]):
    for j in range(n_inputs+1):

        base = net.inW[i,j]

        net.inW[i,j] = base + epsi
        net.forward(trainData)
        one = net.trloss(trainData, trainLabel)
        
        net.inW[i,j] = base - epsi
        net.forward(trainData)
        two = net.trloss(trainData, trainLabel)
        
        net.inW[i,j] = base
        val1 = net.inError[i,j]
        val2 = (one-two)/(2*epsi)
        y.append(val1)
        z.append(val2)  


# In[ ]:

for i in range(numClasses):
    
    for j in range(net.nnod[0]):

        base = net.outW[i,j]

        net.outW[i,j] = base + epsi
        net.forward(trainData)
        one = net.trloss(trainData, trainLabel)
        
        net.outW[i,j] = base - epsi
        net.forward(trainData)
        two = net.trloss(trainData, trainLabel)
        
        net.outW[i,j] = base
        val1 = net.outError[i,j]
        val2 = (one-two)/(2*epsi)

        y.append(val1)
        z.append(val2)


# In[ ]:

plt.plot(x,np.asarray(y)-np.asarray(z))
plt.plot(x,z)
plt.show()

