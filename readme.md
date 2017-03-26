
# Running:
    Please refer test.py
    
# config.py:
    Contains the Fixed Parameters for the network such as input Dimension, Number of Classes, Initial bias etc

# activations.py:
    Contains the Forward and Backward Pass Code for Activation Functions : ['relu', 'tanh']

# test.py:
    Testing and Training Script
    Gradient Method Options: ['mbsgd', 'adagrad', 'gdm']
    mbsgd: Mini Batch Stochastic Gradient Descent
    adagrad: AdaGrad
    gdm: Gradient Descent with Momentum

# network.py:
    
    init : Initialises the neurons, Weight values, Gradients [0], Directions for gdm, Past Gradient Squared Sum for Adagrad
    forward: Forward Pass Code
    predict: Predicts the Class given an input vector
    update: Given a Method updates the weights of the network using gradients etc
    backward: Backward pass (Accumulates Gradients )
    trloss: return the loss for a training example
    train : training function given training and validation data

# Other Information:
    Loss Function: Cross Entropy Loss
    Default training method: Mini Batch Stochastic gradient Descent
