import numpy as np

def activation(x, func):

    if func == 'relu':

        def ReLU(y):
            return y if y>0 else 0
        ReLU = np.vectorize(ReLU)

        return ReLU(x)
    elif func == 'tanh':

        def tanh(y):
            return ( ( 1-np.exp( -2*y ) )/( 1 + np.exp( -2*y ) ) )
        tanh = np.vectorize(tanh)

        return tanh(x)

    else :
        assert(0 == 1), 'Invalid Activation Function'

def backActivate(error, inUnits, outUnits,  func):

    if func == 'relu':

        def ReLU(y):
            return 1 if y > 0 else 0

        ReLU = np.vectorize(ReLU)

        return error*ReLU(inUnits)
    elif func == 'tanh':

        def tanh(y):
            return 1-y*y
        tanh = np.vectorize(tanh)

        return error*tanh(outUnits)
    else :
        assert(0 == 1), 'Invalid Activation Function'



