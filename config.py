############################
## Training Parameters    ##
############################

trainExamples = 1000
valExamples = 100
batchSize = 10
maxIter = 500
lr = 0.0001

############################
## Network Parameters     ##
############################

'''
Architecture
'''

# Number of Hidden Layers
nhidden_layers = 2

# Number of nodes in each hidden layer
# Example: For 2 layers with 20 and 15 nodes each
# nnodes = [20,15]
nnodes = [20, 15]

#Activation Function : 'relu'
actfun = 'relu'

'''
Fixed
'''

#Input Size
n_inputs = 784

#Number of Classes :::: (10 for mnist)
numClasses = 10

#Network Initialisation
minW = -0.1
maxW =  0.1       # Network weights are initialised in range [minW,maxW]
initBias = 0.01   # Initial Bias Value for all layers
epsilon = 1e-08   # Smoothing term for AdaGrad
gamma = 0.9       # Momentum for GD with Momentum
