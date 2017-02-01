import numpy as np
import activations
import config

minW = config.minW
maxW = config.maxW
n_inputs = config.n_inputs          # input feature size
numClasses = config.numClasses
initBias = config.initBias
epsi = config.epsilon
gamma= config.gamma

class network:
    def __init__(self, nhidden_layers, nnodes, actfun):

        assert(nhidden_layers == len(nnodes)), "Invalid Input, len(nnodes) vs nhidden_layers"
        self.nhid = nhidden_layers
        self.nnod = nnodes
        self.actfn = actfun
        # initial weights and biases
        ############################

        # input weights: connecting input layer and first hidden layer
        self.inW =  np.random.uniform(minW,maxW,size=(nnodes[0], n_inputs +1 ) )
        self.inW[:,n_inputs] = initBias

        self.inError = np.asarray([[0 for i in range(n_inputs+1)] for j in range(nnodes[0])])
        self.inPastGrad = np.asarray([[0 for i in range(n_inputs+1)] for j in range(nnodes[0])])
        self.inDir = np.asarray([[0 for i in range(n_inputs+1)] for j in range(nnodes[0])])

        # hidden weights: connecting intermediate hidden layers
        if nhidden_layers > 1:
            self.hidW = []
            self.hidError = []
            self.hidPastGrad = []
            self.hidDir = []
            for i in range(nhidden_layers-1):
                temp =  np.random.uniform(minW,maxW,size=( nnodes[i+1],nnodes[i]+1) )
                self.hidW.append(temp)
                self.hidW[i][:,nnodes[i]] = initBias
                self.hidError.append(np.asarray([[0 for k in range(nnodes[i]+1)] for j in range(nnodes[i+1])]))
                self.hidPastGrad.append(np.asarray([[0 for k in range(nnodes[i]+1)] for j in range(nnodes[i+1])]))
                self.hidDir.append(np.asarray([[0 for k in range(nnodes[i]+1)] for j in range(nnodes[i+1])]))


        # output weights: connecting last hidden layer and output layer
        self.outW =  np.random.uniform(minW,maxW,size=(numClasses,nnodes[nhidden_layers-1]+1) )
        self.outW[:,nnodes[nhidden_layers-1]] = initBias

        #output weights accumulated Error
        self.outError = np.asarray([[0 for i in range(nnodes[nhidden_layers-1]+1)] for j in range(numClasses)])
        self.outPastGrad = np.asarray([[0 for i in range(nnodes[nhidden_layers-1]+1)] for j in range(numClasses)])
        self.outDir = np.asarray([[0 for i in range(nnodes[nhidden_layers-1]+1)] for j in range(numClasses)])

        #each neuron has an input and an output
        # netUnits[2][5,0], netUnits[2][5,1] --> input, output of the 6th neuron in 3rd layer
        self.netUnits = []
        for i in range(nhidden_layers):
            self.netUnits.append(np.asarray([[0 for inout in range(2)] for inlayer in range(nnodes[i])]))

        # output of the network
        self.outNet = np.asarray([0]* numClasses)
        self.outProb = np.asarray([0]* numClasses)

    def forward(self, inputVal):

        assert (n_inputs == len(inputVal) ), 'Input Size Mismatch'

        # bias unit
        inputVal = np.array(inputVal)
        inputVal = np.append(inputVal,1)

        assert (self.nhid > 0), "Atleast One Hidden Layer"

        # computation for first hidden layer
        self.netUnits[0][:,0] = np.dot(self.inW , inputVal)
        self.netUnits[0][:,1] = self.activate(self.netUnits[0][:,0])

        #computation for rest of the hidden layers
        if self.nhid > 1:

            for i in range(1,self.nhid):
                tempInp = np.copy(self.netUnits[i-1][:,1])
                tempInp = np.append(tempInp,1)
                self.netUnits[i][:,0]= np.dot(self.hidW[i-1],tempInp)
                self.netUnits[i][:,1]= self.activate(self.netUnits[i][:,0])

        #computation for the output layer
        tempInp= np.copy(self.netUnits[self.nhid-1][:,1])
        tempInp = np.append(tempInp,1)
        self.outNet = np.dot(self.outW,tempInp)
        self.outProb = np.exp(self.outNet)/sum(np.exp(self.outNet))

    def activate(self, inp):

        return activations.activation(inp, self.actfn)

    def predict(self, inputVal):

        self.forward(inputVal)
        return self.outProb.argmax()

    def update(self, batchSize,lr, gradMethod):

        if gradMethod == 'mbsgd':

            self.inW = self.inW - lr*self.inError/batchSize
            for i in range(self.nhid-1):
                self.hidW[i] = self.hidW[i] - lr*self.hidError[i]/batchSize
            self.outW = self.outW - lr*self.outError/batchSize

        elif gradMethod == 'adagrad':

            # update output weights
            for i in range(numClasses):
                for j in range(self.nnod[self.nhid - 1]):
                    self.outW[i,j] -= lr*self.outError[i,j]/(batchSize*np.sqrt(self.outPastGrad[i,j] + epsi))

            #update output biases
            for i in range(numClasses):
                self.outW[i,self.nnod[self.nhid-1]] -= lr*self.outError[i, self.nnod[self.nhid-1] ]/(batchSize*np.sqrt(self.outPastGrad[i,self.nnod[self.nhid-1] ]+ epsi))

            for i in range(self.nhid-1):
                # update hidden weights
                for k in range(self.nnod[self.nhid-i-2]):
                    for j in range(self.nnod[self.nhid-i-1]):
                        self.hidW[ self.nhid - i-2][j,k ] -= lr*self.hidError[self.nhid-i-2][j,k]/(batchSize*np.sqrt(self.hidPastGrad[self.nhid-i-2][j,k]+ epsi))

                # update hidden biases
                for k in range(self.nnod[self.nhid-i-1]):
                    self.hidW[self.nhid-i-2][k, self.nnod[self.nhid-i-2]] -= lr*self.hidError[self.nhid-i-2][k, self.nnod[self.nhid-i-2]]/(batchSize*np.sqrt(  self.hidPastGrad[self.nhid-i-2][k, self.nnod[self.nhid-i-2]]   + epsi))

            #update input weights
            for i in range(self.nnod[0]):
                for j in range(n_inputs):
                    self.inW[i, j] -= lr*self.inError[i,j]/(batchSize*np.sqrt( self.inPastGrad[i,j] + epsi ))

            #update input biases
            for i in range(self.nnod[0]):
                self.inW[i,n_inputs] -= lr*self.inError[i,n_inputs]/(batchSize*np.sqrt(self.inPastGrad[i,n_inputs] +epsi))

        elif gradMethod == 'gdm':

            # update output weights
            for i in range(numClasses):
                for j in range(self.nnod[self.nhid - 1]):
                    self.outDir[i,j] = gamma*self.outDir[i,j] + lr*self.outError[i,j]/batchSize
                    self.outW[i,j] -= self.outDir[i,j]

            #update output biases
            for i in range(numClasses):
                self.outDir[i,self.nnod[self.nhid-1]] = gamma*self.outDir[i,self.nnod[self.nhid-1]] + lr*self.outError[i, self.nnod[self.nhid-1] ]/(batchSize)
                self.outW[i,self.nnod[self.nhid-1]] -= self.outDir[i,self.nnod[self.nhid-1]]

            for i in range(self.nhid-1):
                # update hidden weights
                for k in range(self.nnod[self.nhid-i-2]):
                    for j in range(self.nnod[self.nhid-i-1]):
                        self.hidDir[ self.nhid - i-2][j,k ] = gamma*self.hidDir[ self.nhid - i-2][j,k ] + lr*self.hidError[self.nhid-i-2][j,k]/(batchSize)
                        self.hidW[ self.nhid - i-2][j,k ] -=  self.hidDir[ self.nhid - i-2][j,k ]

                # update hidden biases
                for k in range(self.nnod[self.nhid-i-1]):
                    self.hidDir[self.nhid-i-2][k, self.nnod[self.nhid-i-2]] = gamma*self.hidDir[self.nhid-i-2][k, self.nnod[self.nhid-i-2]] + lr*self.hidError[self.nhid-i-2][k, self.nnod[self.nhid-i-2]]/(batchSize)
                    self.hidW[self.nhid-i-2][k, self.nnod[self.nhid-i-2]] -=  self.hidDir[self.nhid-i-2][k, self.nnod[self.nhid-i-2]]

            #update input weights
            for i in range(self.nnod[0]):
                for j in range(n_inputs):
                    self.inDir[i, j] =gamma*self.inDir[i, j] + lr*self.inError[i,j]/(batchSize)
                    self.inW[i, j] -=  self.inDir[i, j]

            #update input biases
            for i in range(self.nnod[0]):
                self.inDir[i,n_inputs] = gamma*self.inDir[i,n_inputs]+ lr*self.inError[i,n_inputs]/(batchSize)
                self.inW[i,n_inputs] -=  self.inDir[i,n_inputs]


        else:
            assert(0==1),'Invalid Gradient Method'

    def backActivate(self, error, layer):

        return activations.backActivate( error, self.netUnits[layer-1][:,0], self.netUnits[layer-1][:,1] , self.actfn )

    def backward(self,inputVal,  label):

        assert(len(label) == numClasses), "Size Mismatch : Target Label"
        assert(sum(label) == 1 ), "An image can not have more than one class"

        # gradient to be backpropagated for Cross Entropy Loss
        backError = self.outProb - label

        # errors for the last hidden layer
        lastHidError = []

        for i in range(self.nnod[self.nhid - 1]):
            lastHidError.append( sum(backError * self.outW[:,i]) )

        # update output weights
        for i in range(numClasses):
            for j in range(self.nnod[self.nhid - 1]):
                self.outError[i,j] += backError[i]*self.netUnits[self.nhid-1][j,1]
                self.outPastGrad[i,j] += self.outError[i,j]**2

        #update output biases
        for i in range(numClasses):
            self.outError[i,self.nnod[self.nhid-1]] += backError[i]
            self.outPastGrad[i,self.nnod[self.nhid-1]] += self.outError[i,self.nnod[self.nhid-1]]**2

        prevError = np.asarray(lastHidError)

        # error for the rest of the hidden layers
        for i in range(self.nhid-1):

            tempError = self.backActivate(prevError , self.nhid - i)
            prevError = []
            for j in range(self.nnod[self.nhid-i-2]):
                prevError.append(sum(tempError*self.hidW[self.nhid - i -2][:, j]))

            prevError = np.asarray(prevError)
            # update hidden weights
            for k in range(self.nnod[self.nhid-i-2]):
                for j in range(self.nnod[self.nhid-i-1]):
                    self.hidError[ self.nhid - i-2][j,k ] += tempError[j]*self.netUnits[self.nhid-i-2][k,1]
                    self.hidPastGrad[ self.nhid - i-2][j,k ] += self.hidError[self.nhid-i-2][j,k]**2

            # update hidden biases
            for k in range(self.nnod[self.nhid-i-1]):
                self.hidError[self.nhid-i-2][k, self.nnod[self.nhid-i-2]] += tempError[k]
                self.hidPastGrad[self.nhid-i-2][k, self.nnod[self.nhid-i-2]] += self.hidError[self.nhid-i-2][k, self.nnod[self.nhid-i-2]]**2

        tempError = self.backActivate(prevError, 1)

        #update input weights
        for i in range(self.nnod[0]):
            for j in range(n_inputs):
                self.inError[i, j] +=  tempError[i]*inputVal[j]
                self.inPastGrad[i, j] += self.inError[i,j]**2

        #update input biases
        for i in range(self.nnod[0]):
            self.inError[i,n_inputs] += tempError[i]
            self.inPastGrad[i,n_inputs] += self.inError[i, n_inputs]**2

    def trloss(self, inData, inlabel):

        loss = 0
        for i in range(numClasses):
            loss = loss - inlabel[i]*np.log(self.outProb[i])

        return loss
    def validate(self, valData, valLabel):
        correct = 0
        for i in range(len(valData)):
            if self.predict(valData[i]) == valLabel[i]:
                correct += 1
        return correct*1.0/len(valData)

    def train(self, trainInput, trainLabel,valData, valLabel, batchSize, maxIter, lr=0.0001, gradMethod='mbsgd'):

        # assert(len(trainInput) >= batchSize), "Batch Size is greater than number of examples."
        assert(len(trainInput) == len(trainLabel)), "Size Mismatch, Training Data not equal to Training labels"
        assert(len(trainInput[0,:]) == n_inputs ), "Input Data Dimension Mismatch"
        assert(len(trainLabel[0,:]) == numClasses), "Target Label Dimension Mismatch"

        j = 0
        iterforEpoch = len(trainInput)/batchSize
        for i in range(maxIter):

            #zero out the accumulated gradients
            self.inError[:,:] = 0
            for k in range(self.nhid-1):
                self.hidError[k][:,:] = 0
            self.outError[:,:] = 0

            loss = 0
            for k in range(batchSize):

                batchInput = trainInput[(j+k)%len(trainInput),:]
                batchLabel = trainLabel[(j+k)%len(trainInput),:]

                #forward pass
                self.forward(batchInput)

                # backward pass: accumulate gradients
                self.backward(batchInput, batchLabel)

                loss = loss + self.trloss(batchInput, batchLabel)
            # update weights
            self.update(batchSize, lr, gradMethod )

            print 'Iteration ',i+1,'/',maxIter,'Training Loss = ', loss/batchSize

            if (i+1)%iterforEpoch == 0 :
                print 'Epoch ', i/iterforEpoch + 1 ,'Accuracy: ',self.validate(valData, valLabel)

            j = (j+batchSize)%len(trainInput)







