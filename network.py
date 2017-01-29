import numpy as np
import activations

minW = -0.1
maxW = 0.1
n_inputs = 5      # input feature size
numClasses = 10     # number of classes in dataset
initBias = 0.01
lr == 0.001

class network:
    def __init__(self, nhidden_layers, nnodes, actfun):

        self.nhid = nhidden_layers
        self.nnod = nnodes
        self.actfn = actfun
        # initial weights and biases
        ############################

        # input weights: connecting input layer and first hidden layer
        self.inW =  np.random.uniform(minW,maxW,size=(nnodes, n_inputs +1 ) )
        self.inW[:,n_inputs] = initBias

        # hidden weights: connecting intermediate hidden layers
        if nhidden_layers > 1:
            self.hidW =  np.random.uniform(minW,maxW,size=(nhidden_layers - 1, nnodes,nnodes+1) )
            self.hidW[:,:,nnodes] = initBias


        # output weights: connecting last hidden layer and output layer
        self.outW =  np.random.uniform(minW,maxW,size=(numClasses,nnodes+1) )
        self.outW[:,nnodes] = initBias

        #each neuron has an input and an output
        # netUnits[2][5][0], netUnits[2][5][1] --> input, output of the 6th neuron in 3rd layer
        self.netUnits = [[[0 for inout in range(2)] for inlayer in range(nnodes)] for numlayers in range(nhidden_layers)]
        self.netUnits = np.array(self.netUnits)

        #backward error for each neuron
        #self.netErrors = [[0 for inlayer in range(nnodes)] for numlayers in range(nhidden_layers)]
        #self.netErrors = np.array(self.netErrors)

        # output of the network
        self.outNet = [0]* numClasses
        self.outProb = [0]* numClasses

    def forward(self, inputVal):

        assert (n_inputs == len(inputVal) ), 'Input Size Mismatch'

        # bias unit
        inputVal = np.array(inputVal)
        inputVal = np.append(inputVal,1)

        #### <<< TODO >>> ####
        assert (self.nhid > 0), "Atleast One Hidden Layer"

        # computation for first hidden layer
        self.netUnits[0,:,0] = np.dot(self.inW , inputVal)
        self.netUnits[0,:,1] = self.activate(self.netUnits[0,:,0])

        #computation for rest of the hidden layers
        if self.nhid > 1:

            for i in range(1,self.nhid):
                tempInp = np.copy(self.netUnits[i-1,:,1])
                tempInp = np.append(tempInp,1)
                self.netUnits[i,:,0]= np.dot(self.hidW[i-1,:,:],tempInp)
                self.netUnits[i,:,1]= self.activate(self.netUnits[i,:,0])

        #computation for the output layer
        tempInp= np.copy(self.netUnits[self.nhid-1,:,1])
        tempInp = np.append(tempInp,1)
        self.outNet = np.dot(self.outW,tempInp)

    def activate(self, inp):

        if self.actfn == 'relu':
            return activations.relu(inp)
        else:
            assert(0 == 1),'Invalid Activation Function'

    def predict(self):

        self.outProb = np.exp(self.outNet)/sum(np.exp(self.outNet))
        return self.outProb

    def update(self, w , grad):

        return w - lr*grad

    def backActivate(self, error, layer):

        if self.actfn == 'relu':

            def ReLU(x):
                return 1 if x > 0 else 0

            ReLU = np.vectorize(ReLU)

            mask = ReLU(self.netUnits[layer-1,:,0])
            return mask*error
        else:
            assert(0 == 1),'Invalid Activation Layer'

    def backward(self, label, inputVal):

        assert(len(label) == numClasses), "Size Mismatch : Target Label"
        assert(sum(label) == 1), "An image can not be of more than one digit at the same time!!"

        # gradient to be backpropagated for Sigmoid Cross Entropy Loss
        backError = -1 * (label * (1 + self.outProb) )

        # errors for the last hidden layer
        lastHidError = []

        for i in range(self.nnod):
            lastHidError.append( sum(backError * self.outW[:,i]) )

        # update output weights
        for i in range(numClasses):
            for j in range(self.nnod):
                self.outW[i,j] =  update(self.outW[i,j],backError[i]*self.netUnits[self.nhid-1,j,1] )

        #update output biases
        for i in range(numClasses):
            self.outW[i,self.nnod] = update(self.outW[i,self.nnod], backError[i])

        prevError = np.asarray(lastHidError)
        # error for the rest of the hidden layers
        for i in range(self.nhid-1):

            tempError = backActivate(prevError , self.nhid - i)
            prevError = []
            for j in range(self.nnod):
                prevError.append(sum(tempError*self.hidW[self.nhid - i -2, :, j]))

            prevError = np.asarray(prevError)

            # update hidden weights
            for k in range(self.nnod):
                for j in range(self.nnod):
                    self.hidW[ self.nhid - i-2,k,j ]= update(self.hidW[self.nhid -i-2,k,j],tempError[j]*self.netUnits[self.nhid-i-2,k,1] )

            # update hidden biases
            for k in range(self.nnod):
                self.hidW[self.nhid-i-2,k, self.nnod] = update(self.hidW[self.nhid-i-2,k, self.nnod],tempError[k] )


        tempError = backActivate(prevError, 1)

        #update input weights
        for i in range(self.nnod):
            for j in range(n_inputs):
                self.inW[i, j] = update(self.inW[i,j], tempError[i]*inputVal[j] )

        #update input biases
        for i in range(self.nnod):
            self.inW[i,n_inputs] = update(self.inW[i, n_inputs], tempError[i])








