# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:51:57 2020

@author: Michael
"""

import numpy as np
from math import exp
from random import random

###############################################################################
############################## FUNCTIONS ######################################

# use this function to create a weight vector and storage vector for a layer
def Layer(Input,output,Layer):
    o = []
    for i in np.arange(0,output):
        temp = []
        temp2 = []
        # Bias term adds an extra weight
        for j in np.arange(0,(Input+1)):
            # Generate the random weight
            temp.append(random())
            # generate the blank slate vector used for propagation later
            temp2.append(0)
        o.append(temp2)
        Layer.append(temp)
    return(Layer,o)

def neuron_output(input_signal, weights):
    output = 0
    # now we loop through the weights and compute the dot product against the 
    # input
    for i in range(len(weights)-1):
        output = output + input_signal[i] * weights[i]
    # the final weight is bias term, we so make sure include it
    output = output + weights[-1]
    return(output)
        
def Forward_Propagate(input_row, MLP, output):
    signal = input_row
    for i in np.arange(0,len(MLP)):
        # holds the input signal temporarily
        prop_inputs = []
        for j in np.arange(0,len(MLP[i])):
            # compute initial neuron output
            output[i][j] =(neuron_output(signal,MLP[i][j]))
            # now feed output through the activation function
            prop_inputs.append(sigmoid(output[i][j]))
        signal = prop_inputs
    return(signal,output)

# Sigmoid activation function
def sigmoid(x):
    return(1/(1 + exp(-x)))
# derivative of the activation function that we need to create the error function
def derivative(x):
    return(x*(1-x))
  
# back propagation algorithm.
def Back_Propagate(Expected_output, neuron_outputs, MLP):
    neuron_changes = []
    for i in reversed(np.arange(0,len(MLP))):
        errors = []
        # for the end of network
        if i == (len(MLP)-1):
            for j in np.arange(0,len(MLP[i])):
                errors.append(Expected_output[j] - neuron_outputs[i][j])
        else:
            for j in np.arange(0,len(MLP[i])):
                error = 0
                for k in np.arange(0,len(MLP[i+1])):
                    error = error + MLP[i][j]*neuron_changes[i+1][k]
        for j in np.arange(0,len(MLP[i])):
            neuron = MLP[i][j]
            neuron_changes.append = errors[j] * derivative(neuron_outputs[i][j])
    print('hi')
    


###############################################################################
#################                 SCRIPT                  #####################

# First Step: import training data/labels using numpy
#train_data = np.loadtxt(open("train_data.csv", "rb"), delimiter=",")
#train_labels = np.loadtxt(open("train_labels.csv", "rb"), delimiter=",")

# now that we have the data in matrix we can initialize our MLP network
nn = []

# I have decided to set the HL to be 15 neurons
H = 15
HL = []
OL = []
output = []

# We need to randomly initialize the sets of weights in between the input and hidden 
# layer and hidden layer and output layer using our layer function
#HL = Layer(784,H,HL)
HL,o1 = Layer(7,2,HL)
output.append(o1)
nn.append(HL)

# Repeat for the output layer
#OL = Layer(15,4,OL)
OL,o2 = Layer(7,3,OL)
output.append(o2)
nn.append(OL)


# Now we implement Forward propagation through the MLP. Remember the activation 
# for each neuron is simply just the sum of the weights(for that neuron)
# multiplied by the input. This gives us the weighted input signal

network = [[[0.13436424411240122, 0.8474337369372327, 0.763774618976614]],
		[[0.2550690257394217, 0.49543508709194095],[0.4494910647887381, 0.651592972722763]]]

row = [1,0,None]
signal_output, n_output = Forward_Propagate(row,network,output)

print(signal_output)
