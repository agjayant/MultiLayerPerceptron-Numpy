import numpy as np
import activations

minW = -0.1
maxW = 0.1
n_inputs = 5      # input feature size
numClasses = 10     # number of classes in dataset
initBias = 0.01

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

        # output probablities
        self.outNet = [0]* numClasses

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











