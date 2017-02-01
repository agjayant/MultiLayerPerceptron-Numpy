############################
## Network Parameters     ##
############################


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
